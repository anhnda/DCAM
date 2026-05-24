"""
csae_svd_anchor.py  (GPU-accelerated)
=====================================
Build a seed-free, SIGNED anchor for CSAE's encoder from the per-cell
activation statistics, via SVD only (no ICA rotation).

Relationship to csae_ica_anchor.py
----------------------------------
Same streamed per-cell covariance, same eigendecomposition. The ICA module
adds an independence rotation on top of the whitened SVD subspace; this
module stops at SVD and uses the signed top-D eigenvectors directly as the
anchor.

When to use this instead of the ICA anchor
------------------------------------------
Use this when ICA does not converge cleanly at the D you want (final
fixed-point diff stays large after max_iter, or cross-seed distance > 0.05).
That happens when the whitened tail eigenvalues span enough range that the
non-Gaussian signal in the lower-variance directions is too weak for FastICA
to rotate stably -- the algorithm wanders for the full iteration budget and
each seed lands somewhere different. ResNet50 layer3 at D=200 with the
eigval range [2.24e-02, 4.47e-01] hits this regime.

What you give up: the "independence rotation aligns with distinct generative
factors" semantic argument. PCA atoms are holistic signed mixtures of
channels -- mathematically orthogonal but not concept-aligned in any strong
sense. What you keep: a SIGNED, unit-norm, fully deterministic anchor that
matches CSAE's encoder geometry (no NMF-style sign constraint) and that the
verifier will agree on to float-noise across seeds.

Determinism
-----------
The signed top-D PCA basis is a fixed function of the covariance matrix,
which is a fixed function of the cells in the cache. The only seed input is
the covariance subsample draw (and we fix that too in the verifier). So the
cross-seed verifier should report numbers at the GPU float-noise floor; if
it does not, that is a real bug to chase before training.

A subtle point: eigenvectors are defined up to sign, and eigh's output sign
is implementation-dependent. We pin signs deterministically: largest-
magnitude entry of each row is forced positive. With that fix, the anchor is
identical across runs (modulo floating-point reductions).

What this module produces
-------------------------
build_svd_anchor() -> W0  in  R^{D x C}
  D signed unit-norm rows over the C backbone channels.

The saved .npy is drop-in compatible with csae_stable.py's --ica_anchor flag
(the training script only loads a [D, C] array; it does not care how it was
built).

Usage
-----
    python csae_svd_anchor.py \\
        --cache_dir cache_activations \\
        --cache_key resnet50_layer3_thresh0p85_samples50000_chunk100_gcmap1 \\
        --D 200 --verify_seeds 0 1 2 --device cuda
"""

import argparse
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError:
    torch = None
try:
    import joblib
except ImportError:
    joblib = None


# ----------------------------------------------------------------------
# Device / dtype helpers (verbatim from csae_ica_anchor.py for parity)
# ----------------------------------------------------------------------

def resolve_device(device: str):
    """Resolve 'auto'/'cuda'/'cpu' to a torch.device, with a CPU fallback."""
    if torch is None:
        return None
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[svd anchor] CUDA requested but not available; using CPU.")
        device = "cpu"
    return torch.device(device)


def torch_dtype(dtype: str):
    return torch.float64 if dtype == "float64" else torch.float32


def _chunk_to_cells(part):
    """Turn a cached chunk into a [n_cells, C] float array (still on CPU)."""
    act = part["activation"]
    if torch is not None and isinstance(act, torch.Tensor):
        act = act.detach().cpu().numpy()
    else:
        act = np.asarray(act)
    if act.ndim == 4:
        n, C, H, W = act.shape
        cells = act.transpose(0, 2, 3, 1).reshape(-1, C)
    elif act.ndim == 2:
        cells = act
    else:
        raise ValueError(f"unexpected ndim {act.ndim}")
    return cells


# ----------------------------------------------------------------------
# Per-cell covariance (streamed, GPU-accumulated) -- same as ICA module
# ----------------------------------------------------------------------

def build_percell_covariance(cache_path: Path, device, dtype,
                             max_chunks=None, subsample_cells=200000, seed=0):
    """Per-cell channel covariance + mean, streamed one chunk at a time.

    Identical to csae_ica_anchor.build_percell_covariance -- duplicated here
    so this module stands alone and can be regenerated/edited without
    touching the ICA file. Returns (mean [C], cov [C, C]) as float64 numpy.
    """
    parts = sorted(cache_path.glob("part_*.pkl"))
    if not parts:
        raise FileNotFoundError(f"No part_*.pkl in {cache_path}")
    if max_chunks is not None:
        parts = parts[:max_chunks]
    rng = np.random.default_rng(seed)

    n_cells = 0
    sum_x = None        # [C] on device
    sum_xxT = None      # [C, C] on device
    print(f"Streaming {len(parts)} chunk(s) for per-cell covariance "
          f"on {device} ({dtype})...")
    for i, p in enumerate(parts):
        part = joblib.load(p)
        cells = _chunk_to_cells(part)
        if subsample_cells and cells.shape[0] > subsample_cells:
            keep = rng.choice(cells.shape[0], subsample_cells, replace=False)
            cells = cells[keep]
        xt = torch.as_tensor(np.ascontiguousarray(cells)).to(device=device,
                                                             dtype=dtype)
        if sum_x is None:
            C = xt.shape[1]
            sum_x = torch.zeros(C, device=device, dtype=dtype)
            sum_xxT = torch.zeros((C, C), device=device, dtype=dtype)
        sum_x += xt.sum(dim=0)
        sum_xxT += xt.T @ xt
        n_cells += xt.shape[0]
        del part, cells, xt
        if (i + 1) % 20 == 0 or (i + 1) == len(parts):
            print(f"  ...{i+1}/{len(parts)} chunks, {n_cells} cells")

    inv_n = 1.0 / max(n_cells, 1)
    mean_t = sum_x * inv_n
    cov_t = sum_xxT * inv_n - torch.outer(mean_t, mean_t)
    cov_t = 0.5 * (cov_t + cov_t.T)
    mean = mean_t.double().cpu().numpy()
    cov = cov_t.double().cpu().numpy()
    print(f"Per-cell covariance built: C={cov.shape[0]}, {n_cells} cells.")
    return mean, cov


# ----------------------------------------------------------------------
# Signed top-D PCA basis (the whole anchor, no ICA)
# ----------------------------------------------------------------------

def signed_pca_basis(cov, D, device, dtype):
    """Top-D signed orthonormal eigenvectors of the covariance.

    Returns:
        Vk [C, D]      top-D eigenvectors, signed, orthonormal (numpy f64)
        eigvals [D]    descending eigenvalues (numpy f64)
    """
    cov_t = torch.as_tensor(cov, device=device, dtype=dtype)
    cov_t = 0.5 * (cov_t + cov_t.T)
    w, V = torch.linalg.eigh(cov_t)             # ascending
    w = w.flip(0)                                # descending
    V = V.flip(1)
    eigvals = w[:D].clamp_min(1e-12)
    Vk = V[:, :D]                                # [C, D]
    return Vk.double().cpu().numpy(), eigvals.double().cpu().numpy()


def build_svd_anchor(cache_path: Path, D=200, seed=0, max_chunks=None,
                     subsample_cells=200000,
                     device="auto", dtype="float64",
                     return_diagnostics=False, _reuse_cov=None):
    """Build the signed SVD anchor W0 in R^{D x C}.

    Pipeline: per-cell covariance -> top-D signed eigenvectors -> transpose
    to [D, C] -> unit-normalize each row -> deterministic sign fix.

    Args:
        D: number of anchor atoms.
        seed: only affects the covariance subsample draw. The SVD itself is
            deterministic given the covariance.
        device: 'auto' | 'cuda' | 'cpu'.
        dtype: 'float64' (recommended for reproducibility) | 'float32'.
        _reuse_cov: optional (mean, cov) tuple to skip the covariance pass
            (used by the verifier so the cache is streamed only once).
    Returns:
        W0 [D, C] signed, unit-norm rows. (Plus a diagnostics dict if asked.)
    """
    dev = resolve_device(device)
    dt = torch_dtype(dtype)

    if _reuse_cov is not None:
        mean, cov = _reuse_cov
    else:
        mean, cov = build_percell_covariance(
            cache_path, dev, dt, max_chunks=max_chunks,
            subsample_cells=subsample_cells, seed=seed)
    C = cov.shape[0]
    D = min(D, C)

    Vk, eigvals = signed_pca_basis(cov, D, dev, dt)
    print(f"[svd anchor] kept D={D}, "
          f"eigval range [{eigvals.min():.4e}, {eigvals.max():.4e}], "
          f"condition {eigvals.max()/max(eigvals.min(),1e-12):.2e}")

    # rows = atoms over channels
    W0 = Vk.T                                    # [D, C]
    norms = np.linalg.norm(W0, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    W0 = W0 / norms

    # deterministic sign fix: largest-magnitude entry of each row positive.
    # eigh's sign is implementation-defined; this pins it so the cross-seed
    # verifier sees identical anchors to float-noise.
    for d in range(W0.shape[0]):
        j = np.argmax(np.abs(W0[d]))
        if W0[d, j] < 0:
            W0[d] = -W0[d]

    print(f"[svd anchor] W0 ready: shape {W0.shape}, signed, unit rows.")

    if return_diagnostics:
        diag = {'eigvals': eigvals, 'mean': mean,
                'n_cells_cov': subsample_cells, 'device': str(dev),
                'dtype': dtype}
        return W0, diag
    return W0


# ----------------------------------------------------------------------
# Seed-stability verification
# ----------------------------------------------------------------------

def matched_component_distance(A, B):
    """Sign- and permutation-invariant distance between two [D, C] signed
    bases. Hungarian matching on (1 - |cosine|); returns mean matched value.
    For an SVD anchor with the sign fix, this should be at float-noise.
    """
    from scipy.optimize import linear_sum_assignment
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    D = min(A.shape[0], B.shape[0])
    A, B = A[:D], B[:D]
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    cos = An @ Bn.T
    cost = 1.0 - np.abs(cos)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def verify_anchor_determinism(cache_path, D, seeds, device="auto",
                              dtype="float64", **kw):
    """Build the SVD anchor at several seeds and report cross-seed agreement.

    The SVD itself is deterministic given the covariance, and the covariance
    is built once and reused -- so all three seeds should produce literally
    the same numbers (up to GPU reduction nondeterminism). If you see >> 1e-6
    here, something is wrong (most likely the sign-fix tie-breaking on a row
    whose largest entry is genuinely ambiguous).
    """
    print(f"\n{'='*60}\nSVD ANCHOR DETERMINISM CHECK (seeds={seeds})"
          f"\n{'='*60}")
    dev = resolve_device(device)
    dt = torch_dtype(dtype)
    cov_seed = seeds[0]
    mean, cov = build_percell_covariance(
        cache_path, dev, dt, max_chunks=kw.get("max_chunks"),
        subsample_cells=kw.get("subsample_cells", 200000), seed=cov_seed)
    anchors = []
    for sd in seeds:
        print(f"\n--- seed {sd} ---")
        W0 = build_svd_anchor(cache_path, D=D, seed=sd, device=device,
                              dtype=dtype, _reuse_cov=(mean, cov), **kw)
        anchors.append(W0)
    dists = []
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            d = matched_component_distance(anchors[i], anchors[j])
            dists.append(d)
            print(f"  seeds {seeds[i]}<->{seeds[j]}: "
                  f"matched (1-|cos|) = {d:.4e}")
    mean_d = float(np.mean(dists)) if dists else float('nan')
    print(f"\n  mean cross-seed component distance = {mean_d:.4e}")
    if mean_d < 1e-4:
        print("  --> SVD anchor is DETERMINISTIC (as expected). Safe to use.")
    elif mean_d < 0.05:
        print("  --> SVD anchor is seed-stable but not bit-exact across runs "
              "(GPU reduction noise or sign-fix ties on borderline rows). "
              "Acceptable; fix one seed and move on.")
    else:
        print("  --> Unexpected: SVD should be deterministic. Check the "
              "sign-fix code path and that the covariance is actually "
              "being reused across the verifier's seed loop.")
    return mean_d, anchors


def main():
    ap = argparse.ArgumentParser(
        description="Build / verify the signed SVD anchor for CSAE.")
    ap.add_argument("--cache_dir", type=str, default="cache_activations")
    ap.add_argument("--cache_key", type=str, required=True,
                    help="CSAE activation cache dir (the gcmap1 one).")
    ap.add_argument("--D", type=int, default=200,
                    help="Anchor atom count (~200 for ResNet50 layer3). "
                         "PCA condition number is in the verbose output; "
                         "trim D if the tail eigenvalues are noise.")
    ap.add_argument("--verify_seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Seeds for the determinism check. SVD should agree "
                         "to float-noise across seeds.")
    ap.add_argument("--max_chunks", type=int, default=None)
    ap.add_argument("--subsample_cells", type=int, default=200000)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="float64",
                    choices=["float64", "float32"])
    ap.add_argument("--save", type=str, default="csae_svd_anchor_W0.npy",
                    help="Where to save the anchor. Drop-in for "
                         "csae_stable.py's --ica_anchor flag.")
    args = ap.parse_args()

    if joblib is None or torch is None:
        raise SystemExit("Needs joblib and torch installed.")

    cache_path = Path(args.cache_dir) / args.cache_key
    if not cache_path.exists():
        raise SystemExit(f"Cache not found: {cache_path}")

    mean_d, anchors = verify_anchor_determinism(
        cache_path, args.D, args.verify_seeds,
        device=args.device, dtype=args.dtype,
        max_chunks=args.max_chunks, subsample_cells=args.subsample_cells)

    W0 = anchors[0]
    np.save(args.save, W0)
    print(f"\nSaved SVD anchor to {args.save}  (shape {W0.shape})")
    print(f"Use this file as --ica_anchor in csae_stable.py "
          f"(the flag name is historical; the training script accepts any "
          f"[D, C] anchor).")


if __name__ == "__main__":
    main()