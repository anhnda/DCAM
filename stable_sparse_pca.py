"""
stable_sparse_pca.py
====================
Stable, k-SPARSE, ADAPTIVE-code decomposition of the per-cell channel
covariance Sigma produced by export_pca_basics.py.

WHAT PROBLEM THIS SOLVES
------------------------
Dense PCA (csae_pca_baseline / export_pca_basics) gives DETERMINISTIC,
seed-stable eigenvectors -- but every loading v_d mixes ~all C channels
(your Nv = 338..652), so:
  * components are not interpretable ("which channels IS this?"),
  * the top-variance components are class-agnostic (every image lands on the
    same v0/v2), and
  * near-degenerate eigenvalues (your v4 ~ v17 ~ v18, all beta ~ 0.02, 3%
    energy) make individual eigenvectors seed-FRAGILE even though the
    SUBSPACE is stable.

This module replaces the dense eigendecomposition with a decomposition whose
atoms are k_d-sparse (k_d in [k_min, k_max], median ~16), where the support of
each atom is chosen to be STABLE ACROSS RESAMPLES (stability selection), and
whose code is the ADAPTIVE oblique projection (because sparse atoms are not
orthogonal). It is a drop-in for export_pca_basics.PCAReconstructor: the saved
.pkl is a MultiChannelConvSAE subclass and loads unchanged in
check_drop_csae_fixed.py and hier_visualize_pca.py.

THE FOUR STAGES (all consume the SAME streamed (mu, Sigma))
-----------------------------------------------------------
  1. STABLE SUBSPACE.  The top-D eigenvectors define a subspace U whose
     rotation under resampling is bounded by Davis-Kahan via the gap
     gap(lambda_D, lambda_{D+1}) -- the gap BELOW the block, not the (tiny)
     within-block gaps. So we sparsify against the subspace, never trusting
     an individual degenerate eigenvector.

  2. SPARSE ATOMS (deterministic deflation).  For d = 1..D on the residual
     Sigma_d:
        w        = leading eigenvector of Sigma_d                  (det.)
        S_d      = smallest channel set capturing energy_target of ||w||^2,
                   clamped to [k_min, k_max]                        (support)
        v_d      = leading eigenvector of Sigma_d RESTRICTED to S_d x S_d
                   -- the OPTIMAL sparse direction within that support,
                   not a mere truncation                            (refine)
        Sigma_{d+1} = (I - v_d v_d^T) Sigma_d (I - v_d v_d^T)      (PROJECTION
                   deflation -- keeps the residual PSD; Hotelling would not,
                   since v_d is not an exact eigenvector of Sigma_d.)

  3. STABILITY SELECTION (the cross-seed guarantee).  Re-run stage 2 on B
     bootstrap covariances; match bootstrap atoms to the reference by maximal
     |cos| (Hungarian, NOT by index -- degenerate atoms permute freely).
     selection_prob Pi_{d,c} = fraction of bootstraps where channel c is in
     atom d's support. Keep channels with Pi >= pi_thresh; refit v_d on that
     pruned support. (Meinshausen-Buhlmann stability selection: finite-sample
     FDR control on the retained set.) Per-atom mean |cos| = atom_stability;
     atoms below stability_floor are flagged tied_block and should be REPORTED
     AS A SUBSPACE, not as individual interpretable directions.

  4. ADAPTIVE (OBLIQUE) CODE.  Sparse atoms are NOT orthogonal, so the naive
     z = V^T (x - mu) is wrong (it double-counts shared channels). The
     adaptive code is the oblique projection / least-squares solution
        z(x) = G V^T (x - mu),     G = (V^T V)^{-1}   [D x D, precomputed].
     z_d therefore DEPENDS on the other atoms through G: a channel shared
     between atoms has its contribution apportioned by the cross-atom Gram
     matrix instead of counted twice. With orthonormal V, G = I and this
     reduces EXACTLY to the dense PCAReconstructor -- a strict generalization.

STABILITY IS PER-ATOM, NOT GLOBAL
---------------------------------
The honest claim this method supports: stability tracks the eigengap. Well-
separated atoms reach |cos| ~ 1 across resamples; degenerate ones do not, and
those are FLAGGED rather than dressed up as stable. Report atom_stability and
selection_prob in the figure; that is the interpretability-vs-stability result,
per atom.

BOOTSTRAP MODES
---------------
  * GAUSSIAN SURROGATE (default; needs only mu, Sigma): resample cells from
    N(mu, Sigma), re-estimate Sigma. Fast; it is the null the FDR theory
    assumes. Good for fast iteration.
  * REAL RESAMPLING (recommended for NeurIPS/AIM submission): re-stream Sigma
    over bootstrap subsamples of the IMAGES, using your LayerActivationStreamer
    -- faithful to the true (non-Gaussian, ReLU'd, 0.99-clamped) activation
    distribution. Use fit_from_streamer(...) for this path.

USAGE
-----
  # 1) build Sigma exactly as before
  python export_pca_basics.py --model resnet50 --target_layer layer3 \
      --D 200 --save_cov cov_resnet50_layer3_D200.npz   # (no --save needed)

  # 2) build the stable sparse reconstructor FROM that covariance .npz
  python stable_sparse_pca.py \
      --cov_npz cov_resnet50_layer3_D200.npz \
      --D 64 --k_max 32 --k_min 4 --energy_target 0.90 \
      --n_boot 40 --pi_thresh 0.8 \
      --save ssparse_resnet50_D64_k32_model.pkl \
      --save_decomp ssparse_resnet50_D64_k32_decomp.npz

  # 3) evaluate / visualize with the SAME fixed tools (it loads unchanged)
  python check_drop_csae_fixed.py --model resnet50 \
      --csae_model ssparse_resnet50_D64_k32_model.pkl --norm_mode per_image
  python hier_visualize_pca.py --class_id 207 \
      --pca_model ssparse_resnet50_D64_k32_model.pkl

NOTE keep this file importable as `stable_sparse_pca` from the directory you
run the eval in, because joblib stores the qualified class name
stable_sparse_pca.SparsePCAReconstructor in the .pkl.
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
import joblib

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:                                  # pragma: no cover
    _HAVE_SCIPY = False

# Same base class so the saved .pkl is loadable by the fixed eval / visualizer
# (joblib stores the parent's module path too). Matches export_pca_basics.py.
sys.path.append('.')
from run_xcsae_full import MultiChannelConvSAE


# ==========================================================================
# Config
# ==========================================================================

@dataclass
class SSPCAConfig:
    D: int = 64                      # number of atoms to extract
    energy_target: float = 0.90      # per-atom loading-energy budget for support
    k_min: int = 4                   # floor on support size (no 1-2 ch overfits)
    k_max: int = 32                  # cap on support size (interpretability bound)
    n_boot: int = 40                 # bootstrap resamples for stability selection
    n_cells: int = 8000              # cells per Gaussian-surrogate resample
    pi_thresh: float = 0.80          # keep channels with selection prob >= this
    stability_floor: float = 0.70    # atoms below this mean|cos| -> tied_block
    ridge: float = 1e-9              # ridge on (V^T V) for the oblique G
    dtype: torch.dtype = torch.float64
    seed: int = 0


# ==========================================================================
# Stage 2 -- deterministic sparse deflation (pure function of Sigma)
# ==========================================================================

def _energy_support(w: torch.Tensor, energy_target: float,
                    k_min: int, k_max: int) -> torch.Tensor:
    """Smallest set of channels (most-significant first by |w_c|) capturing
    `energy_target` of ||w||^2, clamped to [k_min, k_max]. Returns SORTED ids."""
    order = torch.argsort(w.abs(), descending=True)
    e = torch.cumsum(w[order] ** 2, dim=0) / (w @ w).clamp_min(1e-12)
    kd = int(torch.searchsorted(
        e, torch.tensor(energy_target, dtype=e.dtype, device=e.device)).item()) + 1
    kd = max(k_min, min(kd, k_max))
    return order[:kd].sort().values


def sparse_deflate(cov: torch.Tensor, D: int, energy_target: float,
                   k_min: int, k_max: int
                   ) -> Tuple[torch.Tensor, List[set]]:
    """Deterministic k-sparse deflation. Returns V [C, D] (unit-norm sparse
    columns) and a list of support sets. No RNG -> bitwise deterministic in
    Sigma."""
    C = cov.shape[0]
    S = cov.clone()
    V = torch.zeros(C, D, dtype=cov.dtype, device=cov.device)
    sups: List[set] = []
    eye = torch.eye(C, dtype=cov.dtype, device=cov.device)
    for d in range(D):
        _, Q = torch.linalg.eigh(S)                       # ascending
        w = Q[:, -1]                                      # leading eigenvector
        sup = _energy_support(w, energy_target, k_min, k_max)
        sub = S.index_select(0, sup).index_select(1, sup)  # k_d x k_d
        _, Qs = torch.linalg.eigh(sub)
        v = torch.zeros(C, dtype=cov.dtype, device=cov.device)
        v[sup] = Qs[:, -1]                                # optimal within support
        v = v / v.norm().clamp_min(1e-12)
        V[:, d] = v
        sups.append(set(sup.tolist()))
        P = eye - torch.outer(v, v)                       # projection deflation
        S = P @ S @ P
        S = 0.5 * (S + S.T)                               # keep symmetric/PSD
    return V, sups


# ==========================================================================
# Bootstrap covariance samplers
# ==========================================================================

class GaussianSurrogateSampler:
    """Fast bootstrap: resample n_cells cells ~ N(mu, Sigma), re-estimate cov.
    This is the null assumed by stability-selection FDR theory. Needs only
    (mu, Sigma) -- works from the export_pca_basics .npz alone."""

    def __init__(self, mu: torch.Tensor, cov: torch.Tensor, n_cells: int,
                 seed: int = 0):
        self.mu = mu
        self.C = cov.shape[0]
        self.n_cells = n_cells
        jitter = 1e-6 * torch.eye(self.C, dtype=cov.dtype, device=cov.device)
        self.L = torch.linalg.cholesky(cov + jitter)
        self.dtype = cov.dtype
        self.device = cov.device
        self.gen = torch.Generator(device='cpu').manual_seed(seed)

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        Z = torch.randn(self.C, self.n_cells, dtype=self.dtype,
                        generator=self.gen)               # cpu gen
        Z = Z.to(self.device)
        X = (self.L @ Z).T + self.mu                      # [n_cells, C]
        Xc = X - X.mean(0, keepdim=True)
        cov_b = (Xc.T @ Xc) / self.n_cells
        return self.mu, 0.5 * (cov_b + cov_b.T)


class RestreamSampler:
    """Faithful bootstrap for submission: re-stream Sigma over a bootstrap
    subsample of the IMAGES, using your export_pca_basics machinery. Captures
    the true non-Gaussian activation distribution.

    Parameters
    ----------
    streamer        : export_pca_basics.LayerActivationStreamer
    dataset         : the full ImageNet1kSampledDataset
    scales          : per-channel 0.99 normalization scales (or None)
    make_loader     : fn(subset_indices) -> torch.utils.data.DataLoader
    frac            : fraction of images per bootstrap (default 1.0 = full
                      resample WITH replacement; <1.0 = subsample)
    accum_device    : where to accumulate the C x C cov
    """

    def __init__(self, streamer, dataset, scales, make_loader: Callable,
                 frac: float = 1.0, accum_device: str = 'cpu',
                 dtype: torch.dtype = torch.float64, seed: int = 0):
        # local imports so the math module doesn't hard-depend on the pipeline
        from export_pca_basics import (StreamingCovariance, normalize_batch)
        self._SC = StreamingCovariance
        self._normb = normalize_batch
        self.streamer = streamer
        self.dataset = dataset
        self.scales = scales
        self.make_loader = make_loader
        self.frac = frac
        self.accum_device = torch.device(accum_device)
        self.dtype = dtype
        self.N = len(dataset)
        self.gen = np.random.default_rng(seed)

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        m = int(round(self.frac * self.N))
        idx = self.gen.choice(self.N, size=m, replace=True)   # bootstrap images
        loader = self.make_loader(idx.tolist())
        C = self.streamer.num_channels
        acc = self._SC(C, device=self.accum_device, dtype=self.dtype)
        for acts in self.streamer.iter_batches(loader, desc="boot/cov"):
            a = acts.detach().to(self.accum_device)
            if self.scales is not None:
                a = self._normb(a, self.scales.to(self.accum_device))
            acc.update(a)
        mu, cov = acc.finalize()
        return mu.to(self.dtype), cov.to(self.dtype)


# ==========================================================================
# Stage 3 -- stability selection
# ==========================================================================

def _match_atoms(V_ref: torch.Tensor, V_b: torch.Tensor
                 ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Match bootstrap atoms to reference by maximal |cos| (Hungarian if scipy,
    else greedy). Returns (ref_idx, boot_idx, |cos| matrix)."""
    M = (V_ref.T @ V_b).abs()                              # [D, D]
    if _HAVE_SCIPY:
        ri, ci = linear_sum_assignment(-M.cpu().numpy())
        return ri, ci, M
    # greedy fallback
    D = M.shape[0]
    Mc = M.clone()
    ri, ci = [], []
    for _ in range(D):
        flat = torch.argmax(Mc).item()
        r, c = divmod(flat, D)
        ri.append(r); ci.append(c)
        Mc[r, :] = -1; Mc[:, c] = -1
    return np.array(ri), np.array(ci), M


def stability_select(cov: torch.Tensor, V_ref: torch.Tensor,
                     sampler: Callable, cfg: SSPCAConfig
                     ) -> Tuple[torch.Tensor, torch.Tensor, List[set]]:
    """Bootstrap stability selection. Returns
        Pi             [C, D]  selection probability per (channel, atom)
        atom_stability [D]     mean |cos| of each atom to its reference
        pruned         list    channels with Pi >= pi_thresh, per atom."""
    C, D = V_ref.shape
    Pi = torch.zeros(C, D, dtype=cov.dtype, device=cov.device)
    atom_cos = torch.zeros(D, dtype=cov.dtype, device=cov.device)
    for _ in tqdm(range(cfg.n_boot)):
        _, cov_b = sampler()
        cov_b = cov_b.to(device=cov.device, dtype=cov.dtype)
        V_b, sup_b = sparse_deflate(cov_b, D, cfg.energy_target,
                                    cfg.k_min, cfg.k_max)
        ri, ci, M = _match_atoms(V_ref, V_b)
        for r, c in zip(ri, ci):
            atom_cos[r] += M[r, c]
            for ch in sup_b[c]:
                Pi[ch, r] += 1.0
    Pi /= cfg.n_boot
    atom_cos /= cfg.n_boot
    pruned = [set(torch.nonzero(Pi[:, d] >= cfg.pi_thresh).flatten().tolist())
              for d in range(D)]
    return Pi, atom_cos, pruned


# ==========================================================================
# Top-level fit  (replaces export_pca_basics.eigendecompose)
# ==========================================================================

def fit_stable_sparse_pca(mu: torch.Tensor, cov: torch.Tensor,
                          cfg: SSPCAConfig,
                          sampler: Optional[Callable] = None) -> Dict:
    """Stable, k-sparse, adaptive decomposition of (mu, Sigma).

    sampler : a callable returning (mu_b, cov_b) per bootstrap. If None, a
              GaussianSurrogateSampler(mu, cov, cfg.n_cells) is used. Pass a
              RestreamSampler for the faithful (submission) bootstrap.

    Returns a dict consumed by build_reconstructor / save_decomp_npz."""
    mu = mu.to(cfg.dtype)
    cov = (0.5 * (cov + cov.T)).to(cfg.dtype)
    C = cov.shape[0]
    D = min(cfg.D, C)
    print("Starge 2: deterministic sparse deflation\n")
    # Stage 2: deterministic sparse atoms
    V, sups = sparse_deflate(cov, D, cfg.energy_target, cfg.k_min, cfg.k_max)
    print("\nStarge 3: stability selection\n")
    # Stage 3: stability selection
    if sampler is None:
        sampler = GaussianSurrogateSampler(mu, cov, cfg.n_cells, seed=cfg.seed)
    Pi, atom_stability, pruned = stability_select(cov, V, sampler, cfg)

    # refit each atom on its STABILITY-SELECTED support (fall back to the
    # deterministic support if pruning emptied it)
    for d in tqdm(range(D)):
        sup = sorted(pruned[d]) if len(pruned[d]) >= cfg.k_min else sorted(sups[d])
        sup_t = torch.tensor(sup, dtype=torch.long, device=cov.device)
        sub = cov.index_select(0, sup_t).index_select(1, sup_t)
        _, Qs = torch.linalg.eigh(sub)
        v = torch.zeros(C, dtype=cov.dtype, device=cov.device)
        v[sup_t] = Qs[:, -1]
        V[:, d] = v / v.norm().clamp_min(1e-12)
        pruned[d] = set(sup)

    k_per_atom = torch.tensor([len(s) for s in pruned], dtype=torch.long)

    # Stage 4: oblique adaptive-code matrix  G = (V^T V)^{-1}
    G = torch.linalg.inv(V.T @ V + cfg.ridge * torch.eye(D, dtype=cov.dtype,
                                                         device=cov.device))

    # variance bookkeeping (captured by the sparse atoms vs total)
    total_var = float(torch.diagonal(cov).sum().item())
    # sparse-reconstruction variance: tr(V G V^T Sigma V G V^T)... report the
    # simple captured-energy proxy tr(C_proj) where C_proj = P Sigma P,
    # P = V G V^T (oblique projector onto span V)
    P = V @ G @ V.T
    captured = float(torch.diagonal(P @ cov @ P.T).sum().item())
    evr_sparse = captured / max(total_var, 1e-12)

    return {
        'V': V.cpu(), 'mu': mu.cpu(), 'G': G.cpu(),
        'supports': [sorted(s) for s in pruned],
        'selection_prob': Pi.cpu(),
        'atom_stability': atom_stability.cpu(),
        'k_per_atom': k_per_atom.cpu(),
        'tied_block': (atom_stability.cpu() < cfg.stability_floor),
        'median_k': int(k_per_atom.float().median().item()),
        'C': C, 'D': D,
        'total_variance': total_var,
        'explained_variance_ratio_sparse': evr_sparse,
        'config': cfg,
    }


# ==========================================================================
# SparsePCAReconstructor -- drop-in for export_pca_basics.PCAReconstructor
# ==========================================================================

class SparsePCAReconstructor(MultiChannelConvSAE):
    """k-sparse, ADAPTIVE-code reconstructor wearing the CSAE interface.

    forward(x) returns (x_hat, z) where
        z      = G V^T (x - mu)        oblique adaptive code  [B, D, H, W]
        x_hat  = mu + V z              reconstruction         [B, C, H, W]
    With orthonormal V (G = I) this is identical to the dense PCAReconstructor;
    with sparse (non-orthogonal) V the oblique G apportions shared-channel
    contributions correctly. z is dense over D; the eval's "active units" stat
    is meaningless here -- read avg_relerr_normalized.

    Extra buffers (for hier_visualize_pca.py): pca_G, selection_prob,
    atom_stability, k_per_atom, tied_block, and the python-side `supports`."""

    def __init__(self, in_channels: int, D: int,
                 mu: torch.Tensor, V: torch.Tensor, G: torch.Tensor,
                 supports: Optional[List[List[int]]] = None,
                 selection_prob: Optional[torch.Tensor] = None,
                 atom_stability: Optional[torch.Tensor] = None,
                 k_per_atom: Optional[torch.Tensor] = None,
                 tied_block: Optional[torch.Tensor] = None):
        D = int(D)
        hd = max(D, 1)
        super().__init__(in_channels=in_channels, hidden_dim=hd,
                         kernel_size=1, top_k=hd)
        self.pca_rank = D
        self.register_buffer('pca_mu', mu.view(in_channels).contiguous())

        if D > 0:
            self.register_buffer('pca_V', V.contiguous())          # [C, D]
            self.register_buffer('pca_G', G.contiguous())          # [D, D]
            # wire encoder/decoder for any code path that inspects them; the
            # forward() below uses pca_V/pca_G directly. encoder = G V^T so the
            # conv also yields the oblique code if ever called.
            with torch.no_grad():
                enc = (G @ V.T).view(D, in_channels, 1, 1)
                self.encoder.weight.copy_(enc.to(self.encoder.weight.dtype))
                if self.encoder.bias is not None:
                    self.encoder.bias.zero_()
                self.decoder.weight.copy_(
                    V.view(in_channels, D, 1, 1).to(self.decoder.weight.dtype))
        else:
            self.register_buffer('pca_V', torch.zeros(in_channels, 0))
            self.register_buffer('pca_G', torch.zeros(0, 0))
            with torch.no_grad():
                self.encoder.weight.zero_()
                if self.encoder.bias is not None:
                    self.encoder.bias.zero_()
                self.decoder.weight.zero_()

        # interpretability / stability metadata (buffers so they survive .to())
        if selection_prob is not None:
            self.register_buffer('selection_prob', selection_prob.contiguous())
        if atom_stability is not None:
            self.register_buffer('atom_stability', atom_stability.contiguous())
        if k_per_atom is not None:
            self.register_buffer('k_per_atom', k_per_atom.contiguous())
        if tied_block is not None:
            self.register_buffer('tied_block', tied_block.contiguous())
        # python-side ragged supports (not a tensor)
        self.supports = supports if supports is not None else []

    def forward(self, x: torch.Tensor, use_topk: bool = True):
        B, C, H, W = x.shape
        if self.pca_rank > 0:
            xc = x.permute(0, 2, 3, 1).reshape(-1, C)              # [N, C]
            centered = xc - self.pca_mu                            # [N, C]
            # oblique adaptive code: z = (x-mu) V G   (row form, G symmetric)
            coeff = (centered @ self.pca_V) @ self.pca_G           # [N, D]
            recon = coeff @ self.pca_V.T + self.pca_mu             # [N, C]
            recon = recon.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z = coeff.view(B, H, W, self.pca_rank).permute(
                0, 3, 1, 2).contiguous()
        else:
            recon = self.pca_mu.view(1, C, 1, 1).expand(
                B, C, H, W).contiguous()
            z = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        return recon, z


def build_reconstructor(res: Dict) -> SparsePCAReconstructor:
    """Build a CPU float32 SparsePCAReconstructor from a fit_stable_sparse_pca
    result dict, ready for joblib.dump."""
    C, D = res['C'], res['D']
    V = res['V'].to(torch.float32)
    mu = res['mu'].to(torch.float32)
    G = res['G'].to(torch.float32)
    module = SparsePCAReconstructor(
        in_channels=C, D=D, mu=mu, V=V, G=G,
        supports=res['supports'],
        selection_prob=res['selection_prob'].to(torch.float32),
        atom_stability=res['atom_stability'].to(torch.float32),
        k_per_atom=res['k_per_atom'].to(torch.long),
        tied_block=res['tied_block'],
    )
    return module.cpu().eval()


# ==========================================================================
# Persistence
# ==========================================================================

def save_decomp_npz(path: str, res: Dict, meta: Dict):
    """Separate analysis file mirroring export_pca_basics' --save_cov, plus the
    sparse/stability fields. supports is ragged -> stored as an object array."""
    np.savez_compressed(
        path,
        V=res['V'].double().numpy(),
        mu=res['mu'].double().numpy(),
        G=res['G'].double().numpy(),
        selection_prob=res['selection_prob'].double().numpy(),
        atom_stability=res['atom_stability'].double().numpy(),
        k_per_atom=res['k_per_atom'].numpy(),
        tied_block=res['tied_block'].numpy(),
        supports=np.array([np.array(s, dtype=np.int64)
                           for s in res['supports']], dtype=object),
        meta=np.array([meta], dtype=object),
    )


def load_cov_npz(path: str, dtype: torch.dtype = torch.float64
                 ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Load (mu, Sigma) from an export_pca_basics.py --save_cov .npz."""
    z = np.load(path, allow_pickle=True)
    mu = torch.from_numpy(z['mean']).to(dtype)
    cov = torch.from_numpy(z['cov']).to(dtype)
    meta = z['meta'][0] if 'meta' in z.files else {}
    return mu, cov, meta


# ==========================================================================
# Convenience: faithful (re-streaming) fit straight from a streamer
# ==========================================================================

def fit_from_streamer(mu, cov, streamer, dataset, scales, make_loader,
                      cfg: SSPCAConfig, frac: float = 1.0,
                      accum_device: str = 'cpu') -> Dict:
    """Faithful-bootstrap fit: stability selection re-streams Sigma over
    bootstrap IMAGE subsamples (non-Gaussian, true distribution)."""
    sampler = RestreamSampler(streamer, dataset, scales, make_loader,
                              frac=frac, accum_device=accum_device,
                              dtype=cfg.dtype, seed=cfg.seed)
    return fit_stable_sparse_pca(mu, cov, cfg, sampler=sampler)


# ==========================================================================
# Reporting
# ==========================================================================

def report(res: Dict):
    D = res['D']; C = res['C']
    k = res['k_per_atom']
    st = res['atom_stability']
    tied = res['tied_block']
    print(f"\n{'='*70}")
    print(f"  STABLE SPARSE PCA  C={C}  D={D}")
    print(f"  support k_d: median={res['median_k']}  "
          f"min={int(k.min())}  max={int(k.max())}  "
          f"(budget [{res['config'].k_min},{res['config'].k_max}])")
    print(f"  atom stability |cos|: median={float(st.median()):.3f}  "
          f"min={float(st.min()):.3f}")
    print(f"  tied/unstable atoms (|cos|<{res['config'].stability_floor}): "
          f"{int(tied.sum())}/{D}  -> report these as a SUBSPACE")
    print(f"  oblique-reconstruction EVR: "
          f"{res['explained_variance_ratio_sparse']:.4f} "
          f"({100*res['explained_variance_ratio_sparse']:.1f}% of per-cell var)")
    print(f"{'='*70}")
    # per-atom table (top 12)
    print(f"  {'atom':>4} {'k':>3} {'stab':>6} {'tied':>5}   top channels")
    for d in range(min(D, 12)):
        sup = res['supports'][d]
        pis = res['selection_prob'][sup, d]
        topc = sorted(zip(sup, pis.tolist()), key=lambda t: -t[1])[:6]
        chs = ' '.join(f"ch{c}(p={p:.2f})" for c, p in topc)
        print(f"  v{d:<3d} {len(sup):>3} {float(st[d]):>6.3f} "
              f"{'YES' if bool(tied[d]) else '  -':>5}   {chs}")
    if D > 12:
        print(f"  ... ({D-12} more atoms)")


# ==========================================================================
# CLI
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Stable, k-sparse, adaptive-code decomposition of the "
                    "per-cell covariance from export_pca_basics.py.")
    ap.add_argument('--cov_npz', type=str, required=True,
                    help="export_pca_basics.py --save_cov output (.npz): "
                         "needs keys cov[C,C], mean[C].")
    ap.add_argument('--D', type=int, default=64,
                    help="Number of sparse atoms to extract.")
    ap.add_argument('--energy_target', type=float, default=0.90,
                    help="Per-atom loading-energy budget that sets k_d.")
    ap.add_argument('--k_min', type=int, default=4)
    ap.add_argument('--k_max', type=int, default=32)
    ap.add_argument('--n_boot', type=int, default=40,
                    help="Bootstrap resamples for stability selection.")
    ap.add_argument('--n_cells', type=int, default=8000,
                    help="Cells per Gaussian-surrogate bootstrap resample.")
    ap.add_argument('--pi_thresh', type=float, default=0.80,
                    help="Keep channels with selection prob >= this.")
    ap.add_argument('--stability_floor', type=float, default=0.70,
                    help="Atoms with mean|cos| below this are flagged tied.")
    ap.add_argument('--ridge', type=float, default=1e-9)
    ap.add_argument('--dtype', type=str, default='float64',
                    choices=['float64', 'float32'])
    ap.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save', type=str, default='ssparse_model.pkl',
                    help="SparsePCAReconstructor .pkl (drop-in for the eval).")
    ap.add_argument('--save_decomp', type=str, default='ssparse_decomp.npz',
                    help="Analysis .npz: V, G, supports, selection_prob, "
                         "atom_stability, k_per_atom, tied_block, meta.")
    args = ap.parse_args()

    if args.device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(args.device)
    dt = torch.float64 if args.dtype == 'float64' else torch.float32

    print("=" * 70)
    print("stable_sparse_pca: stable k-sparse adaptive decomposition")
    print(f"  cov source : {args.cov_npz}")
    print(f"  device={dev}  dtype={args.dtype}  D={args.D}  "
          f"k in [{args.k_min},{args.k_max}]  energy={args.energy_target}")
    print(f"  bootstrap  : Gaussian surrogate, n_boot={args.n_boot}, "
          f"n_cells={args.n_cells}, pi>={args.pi_thresh}")
    print("=" * 70)

    mu, cov, cov_meta = load_cov_npz(args.cov_npz, dtype=dt)
    mu = mu.to(dev); cov = cov.to(dev)
    print(f"  loaded Sigma {tuple(cov.shape)}  (mu norm={mu.norm().item():.4g})")

    cfg = SSPCAConfig(
        D=args.D, energy_target=args.energy_target,
        k_min=args.k_min, k_max=args.k_max, n_boot=args.n_boot,
        n_cells=args.n_cells, pi_thresh=args.pi_thresh,
        stability_floor=args.stability_floor, ridge=args.ridge,
        dtype=dt, seed=args.seed)

    res = fit_stable_sparse_pca(mu, cov, cfg)   # Gaussian-surrogate bootstrap
    report(res)

    meta = {
        'source_cov_npz': args.cov_npz,
        'source_cov_meta': cov_meta,
        'D': res['D'], 'C': res['C'],
        'energy_target': args.energy_target,
        'k_min': args.k_min, 'k_max': args.k_max, 'median_k': res['median_k'],
        'n_boot': args.n_boot, 'pi_thresh': args.pi_thresh,
        'stability_floor': args.stability_floor,
        'bootstrap': 'gaussian_surrogate',
        'explained_variance_ratio_sparse':
            res['explained_variance_ratio_sparse'],
        'n_tied_atoms': int(res['tied_block'].sum()),
    }
    save_decomp_npz(args.save_decomp, res, meta)
    print(f"\nSaved decomposition -> {args.save_decomp}")
    print(f"  keys: V[C,D], mu[C], G[D,D], selection_prob[C,D], "
          f"atom_stability[D], k_per_atom[D], tied_block[D], supports(obj), meta")

    module = build_reconstructor(res)
    joblib.dump(module, args.save)
    print(f"Saved SparsePCAReconstructor (D={res['D']}, "
          f"median k={res['median_k']}) -> {args.save}")

    print(f"\n{'='*70}")
    print("Evaluate / visualize with the SAME fixed tools (loads unchanged):")
    print(f"  python check_drop_csae_fixed.py --model {cov_meta.get('model','resnet50')} \\")
    print(f"      --csae_model {args.save} --norm_mode per_image")
    print(f"  python hier_visualize_pca.py --class_id 207 \\")
    print(f"      --pca_model {args.save}")
    print(f"\nFor the SUBMISSION bootstrap (faithful, non-Gaussian), call")
    print(f"  fit_from_streamer(...) with a RestreamSampler instead of the CLI.")
    print(f"Keep stable_sparse_pca.py importable so the .pkl loads.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()