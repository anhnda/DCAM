"""
csae_pca_baseline.py
====================
Build a DETERMINISTIC low-rank PCA reconstructor for layer activations and
package it so it drops straight into check_drop_csae_fixed.py.

WHY THIS EXISTS (the experiment)
--------------------------------
We established that the layer is a GENERIC, low-rank, shared feature space:
~98% of per-cell variance is within-class, the effective rank is a few
hundred, and class identity is a thin ~48-dim projection. If that picture is
right, an OVERCOMPLETE CSAE (D = C*8 = 8192 atoms) is solving a problem the
data does not have -- there is no overcomplete structure to find, which is
exactly why its atoms are seed-unstable.

The honest baseline for a generic low-rank space is its PRINCIPAL SUBSPACE:
project each per-cell activation onto the top-D PCA directions and back,

        x_hat = mu + V V^T (x - mu),      V = top-D eigenvectors of Cov.

No training, no seeds, no sparsity, no overcompleteness. Fully deterministic.

THE CLAIM THIS TESTS
--------------------
Run this PCA-D reconstructor through check_drop_csae_fixed.py and compare its
accuracy to the 8192-atom CSAE's (~68%). If PCA-200 matches CSAE-8192, the
overcomplete SAE bought NOTHING over a deterministic linear basis -> the
overcompleteness was wasted capacity, and the seed-instability you measured
was a symptom of fitting 8192 atoms to a ~few-hundred-rank space. That is the
result: the right decomposition is the principal subspace at the true rank.

HOW IT PLUGS IN
---------------
check_drop_csae_fixed.load_csae_module accepts any MultiChannelConvSAE
instance. PCAReconstructor SUBCLASSES MultiChannelConvSAE, so isinstance
passes and every attribute the eval touches (encoder.weight, in_channels,
hidden_dim, top_k) exists. We override forward() to do the PCA projection
instead of encode/top-k/decode, keeping the (reconstruction, z) contract.

The basis is built on the SAME normalized cached activations the CSAE trained
on (collect_activation_maps_chunked(normalize=True)), so it lives in the same
normalized space check_drop_csae_fixed normalizes test images into. The
comparison is therefore apples-to-apples in normalized space (read
avg_relerr_normalized -- the scale-independent model-quality number).

IMPORTANT (pickle import path)
------------------------------
The saved .pkl is an instance of THIS module's PCAReconstructor class. joblib
stores the qualified name csae_pca_baseline.PCAReconstructor, so at load time
check_drop_csae_fixed.py must be able to import csae_pca_baseline -- i.e. keep
THIS FILE in the same directory you run check_drop_csae_fixed.py from (it does
sys.path.append('.'), so the cwd works).

Usage
-----
  # 1) build the rank-D PCA reconstructor from the cache
  python csae_pca_baseline.py \
      --cache_dir cache_activations \
      --cache_key activations_resnet50_layer3_thresh0p95_samples50000_chunk100_gcmap1 \
      --D 200 --device cuda --save pca_baseline_resnet50_D200_model.pkl

  # 2) evaluate it with the SAME fixed eval used for CSAE
  python check_drop_csae_fixed.py --model resnet50 \\
      --csae_model pca_baseline_resnet50_D200_model.pkl \\
      --norm_mode per_image --debug_batches 3

  # 3) sweep D to find where accuracy saturates (the true usable rank)
  for D in 50 100 200 400; do
    python csae_pca_baseline.py --cache_dir cache_activations \\
        --cache_key <gcmap1 dir> --D $D --device cuda \\
        --save pca_baseline_resnet50_D$D_model.pkl
    python check_drop_csae_fixed.py --model resnet50 \\
        --csae_model pca_baseline_resnet50_D$D_model.pkl --norm_mode per_image
  done
"""

import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import joblib

import sys
sys.path.append('.')
from run_xcsae_full import MultiChannelConvSAE
# reuse the streamed covariance + device helpers (no transcription drift)
from misc.csae_svd_anchor import (
    build_percell_covariance, resolve_device, torch_dtype,
)


# ==========================================
# PCA reconstructor as a MultiChannelConvSAE subclass
# ==========================================

class PCAReconstructor(MultiChannelConvSAE):
    """Deterministic rank-D PCA reconstruction wearing the CSAE interface.

    forward(x) returns (x_hat, z) where
        z      = V^T (x - mu)         per-cell PCA coefficients [B, D, H, W]
        x_hat  = mu + V z             per-cell reconstruction   [B, C, H, W]

    NOT a sparse code -- z is a DENSE linear projection (coefficients can be
    negative). The 'active units' sparsity number the eval prints will sit
    near ~50% (sign of the coefficients), which is expected and meaningless
    for a linear baseline; ignore it. The number that matters is
    avg_relerr_normalized.
    """

    def __init__(self, in_channels: int, D: int,
                 mu: torch.Tensor, V: torch.Tensor):
        # D == 0 is the MEAN-ONLY control: reconstruct every cell as mu, with
        # NO projection. This is the critical baseline -- if it classifies
        # well, the "PCA works" result is really "the mean activation pattern
        # classifies" (an eval artifact). If it's near-chance, the projection
        # is doing the real work.
        #
        # The Conv2d skeleton needs >=1 hidden unit, so for D==0 we build it
        # with hidden_dim=1 (unused) and flag pca_rank=0.
        D = int(D)
        hd = max(D, 1)
        # top_k = hd so the eval's "expected active = top_k/hidden_dim" prints
        # 100% (dense); every kept component is used.
        super().__init__(in_channels=in_channels, hidden_dim=hd,
                         kernel_size=1, top_k=hd)
        self.pca_rank = D
        self.register_buffer('pca_mu', mu.view(in_channels).contiguous())

        if D > 0:
            self.register_buffer('pca_V', V.contiguous())        # [C, D]
            # Make encoder/decoder weights MEANINGFUL (unused by forward, but
            # the loader prints encoder row-norms as a sanity check): encoder
            # row d = eigenvector d (unit norm), decoder column d = same.
            with torch.no_grad():
                self.encoder.weight.copy_(V.T.view(D, in_channels, 1, 1))
                if self.encoder.bias is not None:
                    self.encoder.bias.zero_()
                self.decoder.weight.copy_(V.view(in_channels, D, 1, 1))
        else:
            # mean-only: no basis. Empty V buffer; zero the (unused) skeleton
            # weights so the loader's row-norm print reads ~0 -- a clear visual
            # signal that this checkpoint is the mean-only control.
            self.register_buffer('pca_V', torch.zeros(in_channels, 0))
            with torch.no_grad():
                self.encoder.weight.zero_()
                if self.encoder.bias is not None:
                    self.encoder.bias.zero_()
                self.decoder.weight.zero_()

    def forward(self, x: torch.Tensor, use_topk: bool = True):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        if self.pca_rank > 0:
            # project each spatial cell onto the top-D subspace
            xc = x.permute(0, 2, 3, 1).reshape(-1, C)        # [N, C]
            centered = xc - self.pca_mu                      # [N, C]
            coeff = centered @ self.pca_V                    # [N, D]
            recon = coeff @ self.pca_V.T + self.pca_mu       # [N, C]
            recon = recon.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z = coeff.view(B, H, W, self.pca_rank).permute(
                0, 3, 1, 2).contiguous()
        else:
            # mean-only: every cell -> mu (broadcast). z is a single dead unit
            # so the eval's (z>0) sparsity stat is well-defined (0% active).
            recon = self.pca_mu.view(1, C, 1, 1).expand(
                B, C, H, W).contiguous()
            z = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        return recon, z


# ==========================================
# Build the basis from the cache
# ==========================================

def build_pca_reconstructor(cache_path: Path, D: int,
                            device="auto", dtype="float64",
                            subsample_cells=200000, max_chunks=None,
                            seed=0):
    """Stream the cache, compute the per-cell mean + covariance, take the
    top-D eigenvectors, and return a PCAReconstructor (on CPU, float32).

    Also returns a diagnostics dict including the explained-variance ratio at
    D -- the fraction of (normalized) per-cell variance the rank-D subspace
    captures. If accuracy saturates at a D where this ratio is already high,
    the space really is low-rank and overcompleteness was unnecessary.
    """
    dev = resolve_device(device)
    dt = torch_dtype(dtype)

    mean, cov = build_percell_covariance(
        cache_path, dev, dt, max_chunks=max_chunks,
        subsample_cells=subsample_cells, seed=seed)
    C = cov.shape[0]
    D = min(D, C)

    cov_t = torch.as_tensor(cov, device=dev, dtype=dt)
    cov_t = 0.5 * (cov_t + cov_t.T)
    w, V = torch.linalg.eigh(cov_t)                  # ascending
    w = w.flip(0).clamp_min(0.0)                     # descending eigenvalues
    V = V.flip(1)                                    # [C, C], columns = eigvecs

    total_var = float(w.sum().item())
    Vk = V[:, :D]                                    # [C, D]  (empty if D==0)
    evr = (float(w[:D].sum().item()) / max(total_var, 1e-12)) if D > 0 else 0.0

    print(f"\n{'='*64}")
    if D == 0:
        print(f"  MEAN-ONLY baseline (D=0): every cell reconstructed as mu, "
              f"no projection.  C={C}")
        print(f"  CONTROL: if accuracy here is high, the 'PCA works' result "
              f"is really 'the mean pattern classifies' (eval artifact). "
              f"If near-chance, the projection does the work.")
    else:
        print(f"  PCA basis: C={C}, D={D}")
        print(f"  eigval range (top-D) [{w[D-1]:.4e}, {w[0]:.4e}]")
        print(f"  EXPLAINED VARIANCE at D={D}: {evr:.4f} "
              f"({100*evr:.1f}% of normalized per-cell variance)")
    # where does cumulative variance hit common thresholds?
    cum = torch.cumsum(w, dim=0) / max(total_var, 1e-12)
    for thr in (0.90, 0.95, 0.99):
        k = int(torch.searchsorted(
            cum, torch.tensor(thr, dtype=cum.dtype, device=cum.device)
        ).item()) + 1
        print(f"    {int(thr*100)}% variance reached at rank {k}")
    print(f"{'='*64}")

    mu32 = torch.as_tensor(mean, dtype=torch.float32)
    V32 = Vk.to(torch.float32).cpu()                 # [C, D]; [C, 0] if D==0
    module = PCAReconstructor(in_channels=C, D=D, mu=mu32, V=V32).cpu().eval()

    diag = {'C': C, 'D': D, 'explained_variance_ratio': evr,
            'eigvals_topD': w[:D].double().cpu().numpy(),
            'total_variance': total_var}
    return module, diag


def main():
    ap = argparse.ArgumentParser(
        description="Build a deterministic rank-D PCA reconstructor that "
                    "drops into check_drop_csae_fixed.py.")
    ap.add_argument("--cache_dir", type=str, default="cache_activations")
    ap.add_argument("--cache_key", type=str, required=True,
                    help="The gcmap1 activation cache dir name under "
                         "--cache_dir (same one used for the CSAE).")
    ap.add_argument("--D", type=int, default=200,
                    help="PCA rank. Sweep this; accuracy should saturate near "
                         "the true effective rank. D=0 is the MEAN-ONLY "
                         "control (reconstruct every cell as mu, no "
                         "projection) -- run it first to check the result "
                         "isn't just the mean pattern classifying.")
    ap.add_argument("--subsample_cells", type=int, default=200000)
    ap.add_argument("--max_chunks", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="float64",
                    choices=["float64", "float32"])
    ap.add_argument("--save", type=str, default="pca_baseline_model.pkl",
                    help="Output path. Pass this as --csae_model to "
                         "check_drop_csae_fixed.py. Keep csae_pca_baseline.py "
                         "in the same directory so the pickle can be loaded.")
    args = ap.parse_args()

    cache_path = Path(args.cache_dir) / args.cache_key
    if not cache_path.exists():
        raise SystemExit(f"Cache not found: {cache_path}")

    module, diag = build_pca_reconstructor(
        cache_path, args.D, device=args.device, dtype=args.dtype,
        subsample_cells=args.subsample_cells, max_chunks=args.max_chunks)

    joblib.dump(module, args.save)
    print(f"\nSaved PCA reconstructor to {args.save}")
    print(f"  C={diag['C']}  D={diag['D']}  "
          f"explained_var={diag['explained_variance_ratio']:.4f}")
    print(f"\nNow evaluate with the SAME fixed eval used for CSAE:")
    print(f"  python check_drop_csae_fixed.py --model resnet50 \\")
    print(f"      --csae_model {args.save} \\")
    print(f"      --norm_mode per_image --debug_batches 3")
    print(f"\nCompare acc_reconstructed and avg_relerr_normalized against the "
          f"8192-atom CSAE. If they match, overcompleteness was wasted "
          f"capacity -- that's the result.")


if __name__ == "__main__":
    main()