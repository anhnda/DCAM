"""
fuse_cov_fft.py
===============
Fuse a prior PCA-covariance run (export_pca_basics.py --save_cov) with a prior
|FFT|-descriptor run (export_fft_basics.py --save_fft) into ONE basis, WITHOUT
re-streaming ImageNet. Sibling of recluster_fft.py: pure post-processing on the
two .npz files.

WHY THIS IS SAFE TO SKIP STREAMING
----------------------------------
Both heavy scripts already saved everything fusion needs:

  cov_*.npz  (export_pca_basics --save_cov):
      cov          [C,C]   per-cell channel covariance Sigma
      mean         [C]     per-cell channel mean (mu_cov)
      eigvals/eigvecs      (we recompute from cov; not required)
      channel_scale[C]

  fft_*.npz  (export_fft_basics --save_fft):
      fft_mean     [C,H,W] per-channel mean |FFT| magnitude
      descriptor   [C,F]   the descriptor that WAS clustered (may be reused)
      channel_mean [C]     per-channel activation mean (mu_fft)
      channel_scale[C]

Sigma is a C x C second-moment in ACTIVATION space.
The FFT side gives a C x C affinity in SPECTRUM-SHAPE space.
"Fusion" = put both on comparable [0,1] units, blend with --lam, then derive a
single [C,k] basis V and wrap it in the SAME FFTBasisReconstructor the
visualizer already loads.

THE TWO C x C AFFINITIES (made commensurable before blending)
-------------------------------------------------------------
  A_cov : Sigma -> correlation matrix (scale-free), clamped >=0.
              R_ij = Sigma_ij / sqrt(Sigma_ii Sigma_jj),  R in [-1,1] -> clamp.
          This removes per-channel scale so it is comparable to a cosine.
  A_fft : descriptor rows -> L2-normalized -> cosine Gram, clamped >=0.
              A_fft = clamp(Xn @ Xn^T, 0),  in [0,1].

  A = lam * A_cov + (1 - lam) * A_fft        (lam in [0,1])

  lam = 1.0  -> pure co-activation grouping (covariance only)
  lam = 0.0  -> pure spectral-shape grouping (FFT only)

HONESTY NOTE
------------
correlation and cosine-of-spectrum are DIFFERENT units; the lam blend is a
heuristic, not a principled joint objective. It is a good DIAGNOSTIC: sweep lam
and watch whether the grouping (or downstream viz) actually moves. If the two
endpoints agree, Sigma and the FFT descriptor carry redundant grouping info and
a fancier joint objective buys little. If an intermediate lam is clearly better
(commit to the criterion BEFORE looking), that justifies escalating to the
generalized-eigenproblem version (Sigma v = mu L v, L = Laplacian of A_fft),
left as a TODO stub at the bottom so it drops in without touching this code.

THREE BASIS MODES FROM THE FUSED MATRIX (different meanings -- read carefully)
-----------------------------------------------------------------------------
  --fuse_mode kmeans  (DEFAULT)
      Hard k-means on the ROWS of A (each channel described by its affinity to
      all channels). membership -> membership_to_V -> oblique [C,k] basis.
      Drop-in identical contract to the FFT path. Columns NOT orthonormal.

  --fuse_mode nmf
      NMF of A (>=0) -> overlapping soft membership -> oblique [C,k] basis.
      A channel can join several basics.

  --fuse_mode pca
      Symmetric eigendecomposition of A; top-k eigenvectors become ORTHONORMAL
      columns of V. Restores an orthogonal-projection basis -- BUT these are
      eigenvectors of the FUSED AFFINITY, not of Sigma. So a recon cosine here
      is "variance explained in fused-affinity space", NOT raw activation
      variance. Do not compare it to the PCA baseline's explained_variance.

mu USED FOR CENTERING
---------------------
The reconstructor needs pca_mu so the visualizer's "centered cells" match. cov
and fft both saved a per-channel mean; they should be ~identical if both runs
used the same normalization. We default to the FFT run's channel_mean (so the
viz centering is byte-identical to the FFT path) and assert agreement with the
cov mean within --mu_tol, warning loudly on mismatch (a mismatch means the two
runs used different norm settings and fusion is comparing apples to oranges).

OUTPUTS
-------
  --save       FFTBasisReconstructor .pkl  (drop-in for hier_visualize_pca.py)
  --save_npz   optional .npz: A_cov, A_fft, A_fused, descriptor, labels,
               membership, V, channel_mean, meta

USAGE
-----
  # default fusion: lam=0.5, kmeans on the fused affinity, k=128
  python fuse_cov_fft.py \
      --cov_npz cov_resnet50_layer3_D200.npz \
      --fft_npz fft_resnet50_layer3_k128.npz \
      --lam 0.5 --fuse_mode kmeans --k 128 \
      --save fused_resnet50_k128_lam05_model.pkl \
      --save_npz fused_resnet50_k128_lam05.npz

  # orthonormal PCA of the fused affinity, rank 64
  python fuse_cov_fft.py --cov_npz cov_*.npz --fft_npz fft_*.npz \
      --fuse_mode pca --k 64 --lam 0.5 \
      --save fused_pca_resnet50_D64_model.pkl

  # rebuild the FFT side with a DIFFERENT signal than was originally clustered
  # (reads fft_mean from the .npz and re-reduces it), e.g. drop the DC loudness:
  python fuse_cov_fft.py --cov_npz cov_*.npz --fft_npz fft_*.npz \
      --fft_signal log_nodc --descriptor full --lam 0.5 \
      --save fused_resnet50_k128_lognodc_model.pkl

  # lam sweep + seed stability, all reusing the SAME two .npz (seconds each):
  for L in 0.0 0.25 0.5 0.75 1.0; do
    python fuse_cov_fft.py --cov_npz cov_*.npz --fft_npz fft_*.npz \
      --lam $L --fuse_mode kmeans --k 128 --stability_seeds 3 \
      --save fused_resnet50_k128_lam${L}_model.pkl
  done

  # then visualize with the UNCHANGED PCA/FFT viewer
  python hier_visualize_pca.py --class_id 207 \
      --pca_model fused_resnet50_k128_lam05_model.pkl \
      --ring_components 16 --top_channels 4 --recon_D 64
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F
import joblib

# Reuse the (patched) descriptor / cluster / wrap machinery so this script and
# export_fft_basics.py never drift. Requires the desc_transform patch already
# applied to export_fft_basics.py (same requirement as recluster_fft.py).
sys.path.append('.')
from export_fft_basics import (            # noqa: E402
    build_descriptor, cluster_kmeans, cluster_nmf, membership_to_V,
    make_reconstructor, report_clusters, l2norm_rows,
)


# ==========================================
# Affinity construction
# ==========================================

def cov_to_correlation(cov: torch.Tensor, clamp_neg: bool = True
                       ) -> torch.Tensor:
    """Sigma [C,C] -> correlation matrix R [C,C], scale-free, optionally
    clamped to >=0 so it can be blended with a cosine affinity.

        R_ij = Sigma_ij / sqrt(Sigma_ii * Sigma_jj)

    Degenerate (zero-variance) channels get 0 off-diagonal, 1 on the diagonal.
    """
    d = torch.sqrt(torch.clamp(torch.diag(cov), min=0.0))         # [C]
    denom = torch.outer(d, d)                                     # [C,C]
    valid = denom > 1e-12
    R = torch.zeros_like(cov)
    R[valid] = cov[valid] / denom[valid]
    R = R.clamp(-1.0, 1.0)
    # restore a clean unit diagonal (numerical safety)
    idx = torch.arange(R.shape[0])
    R[idx, idx] = 1.0
    if clamp_neg:
        R = R.clamp_min(0.0)
    return R


def descriptor_to_cosine(descriptor: torch.Tensor) -> torch.Tensor:
    """[C,F] descriptor -> cosine Gram [C,C], L2-normalized rows, clamped >=0."""
    Xn = l2norm_rows(descriptor.to(torch.float32))
    A = (Xn @ Xn.T).clamp_min(0.0)
    idx = torch.arange(A.shape[0])
    A[idx, idx] = 1.0
    return A


def blend_affinities(A_cov: torch.Tensor, A_fft: torch.Tensor,
                     lam: float) -> torch.Tensor:
    """A = lam * A_cov + (1-lam) * A_fft. Both already in [0,1], same shape."""
    if A_cov.shape != A_fft.shape:
        raise ValueError(f"affinity shape mismatch: cov {tuple(A_cov.shape)} "
                         f"vs fft {tuple(A_fft.shape)} -- the two .npz files "
                         f"must come from the SAME model/layer (same C).")
    lam = float(lam)
    A = lam * A_cov + (1.0 - lam) * A_fft
    return 0.5 * (A + A.T)                                        # symmetrize


# ==========================================
# Basis from the fused affinity
# ==========================================

def basis_pca(A: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k symmetric eigenvectors of A -> ORTHONORMAL [C,k] basis (columns).

    NOTE: eigenvectors of the FUSED AFFINITY, not of Sigma. Recon cosine in the
    viz is variance in fused-affinity space, not raw activation variance.
    """
    C = A.shape[0]
    k = max(0, min(int(k), C))
    w, V = torch.linalg.eigh(A.to(torch.float64))    # ascending
    w = w.flip(0)                                    # descending
    V = V.flip(1)
    Vk = V[:, :k].to(torch.float32)                  # [C,k], orthonormal cols
    evr = (float(w[:k].clamp_min(0).sum()) /
           float(w.clamp_min(0).sum().clamp_min(1e-12))) if k > 0 else 0.0
    print(f"  [pca-of-A] top-{k} eigen-affinity ratio = {evr:.4f} "
          f"(NOT activation explained-variance)")
    return Vk


def basis_cluster(A: torch.Tensor, k: int, mode: str, seed: int):
    """Cluster the ROWS of A (each channel = its affinity profile) into k basics.
    Returns (V [C,k], membership [C,k], labels [C]).
    """
    rows = A.to(torch.float32)                        # [C,C] used as features
    if mode == 'kmeans':
        membership, centers, labels = cluster_kmeans(rows, k, seed=seed)
    elif mode == 'nmf':
        membership, centers, labels = cluster_nmf(rows, k, seed=seed)
    else:
        raise ValueError(f"unknown cluster mode: {mode}")
    report_clusters(labels, k)
    V = membership_to_V(membership)                  # [C,k] oblique
    return V, membership, labels


# ==========================================
# Stability (kmeans on fused rows, ARI across seeds) -- ported from recluster
# ==========================================

def _adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    n = a.shape[0]
    ca = {v: i for i, v in enumerate(np.unique(a))}
    cb = {v: i for i, v in enumerate(np.unique(b))}
    cont = np.zeros((len(ca), len(cb)), dtype=np.int64)
    for i in range(n):
        cont[ca[a[i]], cb[b[i]]] += 1
    sum_comb = lambda x: (x * (x - 1) // 2).sum()
    index = sum_comb(cont)
    sa = sum_comb(cont.sum(axis=1))
    sb = sum_comb(cont.sum(axis=0))
    expected = sa * sb / max(n * (n - 1) // 2, 1)
    maxidx = 0.5 * (sa + sb)
    denom = maxidx - expected
    return 1.0 if denom == 0 else float((index - expected) / denom)


def stability_check(A: torch.Tensor, k: int, seeds: int) -> None:
    label_sets = []
    for s in range(seeds):
        _, _, labels = cluster_kmeans(A.to(torch.float32), k, seed=s)
        label_sets.append(labels.cpu().numpy())
    aris = []
    for i in range(seeds):
        for j in range(i + 1, seeds):
            aris.append(_adjusted_rand_index(label_sets[i], label_sets[j]))
    if aris:
        print(f"\n  [stability] mean pairwise ARI over {seeds} seeds = "
              f"{np.mean(aris):.3f}  (min={min(aris):.3f}, max={max(aris):.3f})")
        print(f"  [stability] ~>0.8 = stable grouping; low = fitting noise. "
              f"Commit to a threshold BEFORE the lam sweep.")
    else:
        print("  [stability] need >=2 seeds for ARI.")


# ==========================================
# Loaders
# ==========================================

def load_cov(path: str):
    d = np.load(path, allow_pickle=True)
    cov = torch.from_numpy(d['cov']).to(torch.float64)            # [C,C]
    mu_cov = torch.from_numpy(d['mean']).to(torch.float64)        # [C]
    return cov, mu_cov


def load_fft(path: str, fft_signal: str, descriptor_mode: str,
             n_rbin: int, hp_radius: float):
    d = np.load(path, allow_pickle=True)
    mu_fft = torch.from_numpy(d['channel_mean']).to(torch.float64)  # [C]
    if fft_signal is None:
        # reuse the descriptor exactly as it was saved/clustered
        descriptor = torch.from_numpy(d['descriptor']).to(torch.float64)
        desc_info = "reused saved descriptor (no re-reduction)"
        fft_mean = (torch.from_numpy(d['fft_mean']).to(torch.float64)
                    if 'fft_mean' in d.files else None)
    else:
        # rebuild the FFT side from fft_mean with a (possibly different) signal
        if 'fft_mean' not in d.files:
            raise KeyError(f"{path} has no 'fft_mean'; cannot apply "
                          f"--fft_signal {fft_signal}. Omit --fft_signal to "
                          f"reuse the saved descriptor.")
        fft_mean = torch.from_numpy(d['fft_mean']).to(torch.float64)  # [C,H,W]
        descriptor, desc_info = build_descriptor(
            fft_mean, mode=descriptor_mode, n_rbin=n_rbin,
            desc_transform=fft_signal, hp_radius=hp_radius)
    return descriptor, mu_fft, desc_info, fft_mean


# ==========================================
# CLI
# ==========================================

def main():
    ap = argparse.ArgumentParser(
        description="Fuse a covariance .npz and an |FFT| .npz into one basis "
                    "(pca / kmeans / nmf of the blended C x C affinity), no "
                    "re-streaming.")
    ap.add_argument('--cov_npz', type=str, required=True,
                    help="export_pca_basics.py --save_cov output (cov, mean).")
    ap.add_argument('--fft_npz', type=str, required=True,
                    help="export_fft_basics.py --save_fft output "
                         "(fft_mean, descriptor, channel_mean).")

    ap.add_argument('--lam', type=float, default=0.5,
                    help="Blend weight: A = lam*A_cov + (1-lam)*A_fft. "
                         "1.0=covariance only, 0.0=FFT only. Default 0.5.")
    ap.add_argument('--fuse_mode', type=str, default='kmeans',
                    choices=['kmeans', 'nmf', 'pca'],
                    help="How to derive V from the fused affinity. "
                         "kmeans/nmf = cluster rows (oblique basis); "
                         "pca = top-k eigenvectors (orthonormal, but of the "
                         "AFFINITY not Sigma).")
    ap.add_argument('--k', type=int, default=128,
                    help="Number of basics / PCA rank from the fused affinity.")
    ap.add_argument('--cluster_seed', type=int, default=0,
                    help="Seed for kmeans++/NMF init (fuse_mode kmeans/nmf).")
    ap.add_argument('--stability_seeds', type=int, default=0,
                    help="If >=2 and fuse_mode=kmeans, recluster across seeds "
                         "and report mean pairwise ARI on the fused affinity.")

    # FFT-signal controls (let the fused FFT side differ from what was clustered)
    ap.add_argument('--fft_signal', type=str, default=None,
                    choices=['none', 'log', 'log_nodc', 'radial_ramp',
                             'highpass'],
                    help="Rebuild the FFT descriptor from fft_mean with THIS "
                         "transform before fusing. Omit to reuse the saved "
                         "descriptor verbatim.")
    ap.add_argument('--descriptor', type=str, default='full',
                    choices=['full', 'radial'],
                    help="Descriptor reduction when --fft_signal is given.")
    ap.add_argument('--n_rbin', type=int, default=8,
                    help="Radial bins when --descriptor radial.")
    ap.add_argument('--hp_radius', type=float, default=0.25,
                    help="Low-freq disk radius for --fft_signal highpass.")

    ap.add_argument('--keep_neg_corr', action='store_true',
                    help="Keep negative covariance correlations instead of "
                         "clamping to 0 (anti-correlated channels then push "
                         "apart). Default clamps >=0 to stay comparable to the "
                         "non-negative cosine affinity.")
    ap.add_argument('--mu_source', type=str, default='fft',
                    choices=['fft', 'cov', 'mean'],
                    help="Which per-channel mean to use for centering in the "
                         "reconstructor. 'fft' (default) keeps viz centering "
                         "byte-identical to the FFT path.")
    ap.add_argument('--mu_tol', type=float, default=1e-3,
                    help="Warn if cov-mean and fft-mean disagree by more than "
                         "this relative L2 (signals mismatched norm settings).")

    ap.add_argument('--save', type=str, default='fused_cov_fft_model.pkl',
                    help="FFTBasisReconstructor .pkl (drop-in for "
                         "hier_visualize_pca.py).")
    ap.add_argument('--save_npz', type=str, default=None,
                    help="Optional: save A_cov, A_fft, A_fused, V, labels, "
                         "membership, channel_mean, meta.")
    args = ap.parse_args()

    print("=" * 80)
    print("fuse_cov_fft: blend covariance + |FFT| affinity into one basis "
          "(no streaming)")
    print(f"  cov_npz={args.cov_npz}")
    print(f"  fft_npz={args.fft_npz}")
    print(f"  lam={args.lam}  fuse_mode={args.fuse_mode}  k={args.k}  "
          f"fft_signal={args.fft_signal or '(saved descriptor)'}")
    print("=" * 80)

    # ---- load ----
    cov, mu_cov = load_cov(args.cov_npz)
    descriptor, mu_fft, desc_info, fft_mean = load_fft(
        args.fft_npz, args.fft_signal, args.descriptor,
        args.n_rbin, args.hp_radius)

    C = cov.shape[0]
    if descriptor.shape[0] != C:
        raise ValueError(
            f"channel-count mismatch: cov C={C} but fft C={descriptor.shape[0]}."
            f" The two .npz must be the SAME model+layer.")
    print(f"  loaded: cov[{C},{C}], descriptor[{C},{descriptor.shape[1]}] "
          f"[{desc_info}]")

    # ---- mu agreement check ----
    rel = (mu_cov - mu_fft).norm() / mu_fft.norm().clamp_min(1e-12)
    if rel > args.mu_tol:
        print(f"  WARNING: cov-mean vs fft-mean relative L2 = {rel:.3g} "
              f"> mu_tol={args.mu_tol}. The two runs likely used DIFFERENT "
              f"normalization -- fusion is comparing apples to oranges. "
              f"Re-run one export to match before trusting results.")
    else:
        print(f"  mu agreement OK (relative L2 = {rel:.3g})")
    if args.mu_source == 'cov':
        mu = mu_cov
    elif args.mu_source == 'fft':
        mu = mu_fft
    else:
        mu = 0.5 * (mu_cov + mu_fft)

    # ---- build the two affinities, blend ----
    A_cov = cov_to_correlation(cov, clamp_neg=not args.keep_neg_corr)
    A_fft = descriptor_to_cosine(descriptor)
    A = blend_affinities(A_cov, A_fft, args.lam)
    print(f"\n  A_cov: mean off-diag = "
          f"{A_cov[~torch.eye(C, dtype=torch.bool)].mean().item():.4f}")
    print(f"  A_fft: mean off-diag = "
          f"{A_fft[~torch.eye(C, dtype=torch.bool)].mean().item():.4f}")
    print(f"  A_fused (lam={args.lam}): mean off-diag = "
          f"{A[~torch.eye(C, dtype=torch.bool)].mean().item():.4f}")

    k = max(0, min(int(args.k), C))
    if k != args.k:
        print(f"  NOTE: clamped k from {args.k} to {k} (must be <= C={C}).")

    # ---- optional stability sweep (kmeans on fused affinity) ----
    if args.stability_seeds and args.stability_seeds >= 2:
        if args.fuse_mode != 'kmeans':
            print("  [stability] ARI check is kmeans-only; skipping.")
        else:
            stability_check(A, k, args.stability_seeds)

    # ---- derive basis ----
    membership = None
    labels = None
    if args.fuse_mode == 'pca':
        V = basis_pca(A, k)
        print(f"\n  basis: orthonormal PCA-of-affinity, V[{C},{k}]")
    else:
        print(f"\nClustering fused-affinity rows into {k} basics "
              f"({args.fuse_mode}, seed={args.cluster_seed})...")
        V, membership, labels = basis_cluster(
            A, k, mode=args.fuse_mode, seed=args.cluster_seed)
        print(f"  basis: oblique cluster-membership, V[{C},{k}]")

    # ---- wrap + save ----
    module = make_reconstructor(C, k, mu, V)
    joblib.dump(module, args.save)
    print(f"\nSaved fused FFTBasisReconstructor (k={k}, lam={args.lam}, "
          f"mode={args.fuse_mode}) -> {args.save}")

    if args.save_npz:
        meta = {
            'cov_npz': args.cov_npz,
            'fft_npz': args.fft_npz,
            'lam': args.lam,
            'fuse_mode': args.fuse_mode,
            'k': k,
            'cluster_seed': args.cluster_seed,
            'fft_signal': args.fft_signal,
            'descriptor_mode': args.descriptor,
            'desc_info': desc_info,
            'keep_neg_corr': args.keep_neg_corr,
            'mu_source': args.mu_source,
            'C': C,
        }
        np.savez_compressed(
            args.save_npz,
            A_cov=A_cov.double().cpu().numpy(),
            A_fft=A_fft.double().cpu().numpy(),
            A_fused=A.double().cpu().numpy(),
            descriptor=descriptor.double().cpu().numpy(),
            V=V.double().cpu().numpy(),
            labels=(labels.long().cpu().numpy() if labels is not None
                    else np.array([])),
            membership=(membership.double().cpu().numpy()
                        if membership is not None else np.array([])),
            channel_mean=mu.double().cpu().numpy(),
            meta=np.array([meta], dtype=object),
        )
        print(f"Saved fused arrays -> {args.save_npz}")

    print(f"\n{'='*80}")
    print(f"Done. Visualize with the UNCHANGED viewer:")
    print(f"  python hier_visualize_pca.py --class_id 207 \\")
    print(f"      --pca_model {args.save} \\")
    print(f"      --ring_components 16 --top_channels 4 --recon_D {min(64, k)}")
    if args.fuse_mode != 'pca':
        print(f"\nEach 'component' is a CHANNEL-GROUP from the FUSED affinity "
              f"(co-activation + spectral shape), not a PCA direction.")
    else:
        print(f"\nColumns are eigenvectors of the FUSED AFFINITY, not of Sigma; "
              f"do NOT read recon cosine as activation explained-variance.")
    print(f"{'='*80}")


# ============================================================================
# TODO (escalation, only if the lam sweep says fusion genuinely helps):
#   Principled joint objective instead of the heuristic lam blend.
#   Generalized eigenproblem: maximize activation variance (Sigma) subject to
#   smoothness on the FFT-affinity graph:
#
#       Sigma v = mu * L v ,   L = D - A_fft  (graph Laplacian of A_fft)
#
#   i.e. directions of high activation variance whose loadings vary little
#   between channels with similar spectra. Degenerates to plain PCA of Sigma
#   when A_fft is uniform. Solve via scipy.linalg.eigh(Sigma, L) (or a
#   regularized L + eps*I to keep it PD). Wire as --fuse_mode generalized.
#   Deliberately NOT implemented until the blend diagnostic justifies it.
# ============================================================================


if __name__ == "__main__":
    main()