"""
csae_ica_anchor.py
==================
Build a seed-free, SIGNED anchor for CSAE's encoder from the per-cell
activation statistics, via SVD-whitening + ICA.

Why ICA, not NMF
----------------
CSAE's encoder is a free nn.Conv2d with SIGNED weights (kaiming init, no
non-negativity). NMF produces a NON-NEGATIVE factor; anchoring a signed
encoder to a non-negative basis reintroduces exactly the constraint that made
DCAM fail (DCAM's simplex encoder gave rel_err > 3 on ResNet50 layer3).
So the anchor for CSAE must be SIGNED.

SVD gives a signed orthogonal basis but its components are holistic mixtures
(orthogonality is a mathematical convenience, not semantic). ICA rotates the
whitened SVD subspace toward statistical independence, which tends to align
with distinct generative factors -- a better "concept" target. Standard ICA
IS svd-whiten -> independence-rotation, so this module does both.

Determinism
-----------
The whole pipeline is run with a FIXED seed, so the anchor is seed-free by
construction. We VERIFY this by building it at several seeds and checking the
components match (Hungarian-matched, sign/permutation aware) -- the same
discipline that exposed the NMF convergence bug earlier. ICA has an inherent
sign+permutation ambiguity; verification accounts for it.

What this module produces
-------------------------
build_ica_anchor() -> W0  in  R^{D x C}
  D signed unit-norm rows over the C backbone channels. This is the anchor
  target for CSAE's encoder weights (a 1x1 conv weight is [hidden, C, 1, 1];
  the [hidden, C] slice is what gets anchored).

Memory
------
The per-cell covariance is C x C and is streamed one cache chunk at a time;
activations are never fully loaded. ICA runs on the C x C scale, never on the
full activation set.

Usage (standalone verification -- DO THIS BEFORE training CSAE)
---------------------------------------------------------------
    python csae_ica_anchor.py \\
        --cache_dir cache_activations \\
        --cache_key resnet50_layer3_thresh0p85_samples50000_chunk100_gcmap1 \\
        --D 200 --verify_seeds 0 1 2
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
# Per-cell covariance (streamed)
# ----------------------------------------------------------------------

def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def build_percell_covariance(cache_path: Path, max_chunks=None,
                             subsample_cells=200000, seed=0):
    """Per-cell channel covariance + mean, streamed one chunk at a time.

    Returns (mean [C], cov [C, C]) over all spatial cells. ICA needs the
    mean to center and the covariance to whiten. Memory: one chunk resident
    at a time.
    """
    parts = sorted(cache_path.glob("part_*.pkl"))
    if not parts:
        raise FileNotFoundError(f"No part_*.pkl in {cache_path}")
    if max_chunks is not None:
        parts = parts[:max_chunks]
    rng = np.random.default_rng(seed)

    n_cells = 0
    sum_x = None        # [C]
    sum_xxT = None      # [C, C]
    print(f"Streaming {len(parts)} chunk(s) for per-cell covariance...")
    for i, p in enumerate(parts):
        part = joblib.load(p)
        act = _to_numpy(part["activation"])             # [n, C, H, W]
        if act.ndim == 4:
            n, C, H, W = act.shape
            cells = act.transpose(0, 2, 3, 1).reshape(-1, C)
        elif act.ndim == 2:
            cells = act
        else:
            raise ValueError(f"unexpected ndim {act.ndim}")
        if subsample_cells and cells.shape[0] > subsample_cells:
            keep = rng.choice(cells.shape[0], subsample_cells, replace=False)
            cells = cells[keep]
        if sum_x is None:
            C = cells.shape[1]
            sum_x = np.zeros(C, dtype=np.float64)
            sum_xxT = np.zeros((C, C), dtype=np.float64)
        sum_x += cells.sum(axis=0)
        sum_xxT += cells.T @ cells
        n_cells += cells.shape[0]
        del part, act, cells
        if (i + 1) % 20 == 0 or (i + 1) == len(parts):
            print(f"  ...{i+1}/{len(parts)} chunks, {n_cells} cells")

    mean = sum_x / max(n_cells, 1)
    cov = sum_xxT / max(n_cells, 1) - np.outer(mean, mean)
    cov = 0.5 * (cov + cov.T)
    print(f"Per-cell covariance built: C={cov.shape[0]}, {n_cells} cells.")
    return mean, cov


# ----------------------------------------------------------------------
# SVD-whitening + FastICA
# ----------------------------------------------------------------------

def svd_whiten(cov, D):
    """Top-D PCA whitening transform from the covariance.

    Returns:
        V  [C, D]   top-D eigenvectors (signed, orthonormal)
        whiten [D, C]  the whitening map  diag(1/sqrt(lambda)) V^T
        eigvals [D]    the kept eigenvalues
    """
    w, V = np.linalg.eigh(cov)              # ascending
    order = np.argsort(w)[::-1]
    keep = order[:D]
    eigvals = np.clip(w[keep], 1e-12, None)
    Vk = V[:, keep]                         # [C, D]
    whiten = (Vk / np.sqrt(eigvals)).T      # [D, C]
    return Vk, whiten, eigvals


def fastica(X_white, n_components, seed=0, max_iter=500, tol=1e-5):
    """Symmetric (parallel) FastICA with the logcosh nonlinearity.

    Args:
        X_white: [N, D] whitened, zero-mean samples (rows = samples).
        n_components: number of independent components (<= D).
        seed: FIXED rng seed for the W init -- determinism comes from here.
    Returns:
        W_ica [n_components, D]  unmixing matrix (rows = components in the
        whitened space). Each row is unit-norm.
    """
    rng = np.random.default_rng(seed)
    N, D = X_white.shape
    n = min(n_components, D)

    # deterministic orthonormal init
    W = rng.standard_normal((n, D))
    # symmetric orthonormalization:  W <- (W W^T)^-1/2 W
    def sym_orth(M):
        u, s, vt = np.linalg.svd(M, full_matrices=False)
        return u @ vt
    W = sym_orth(W)

    for it in range(max_iter):
        WX = W @ X_white.T                  # [n, N]
        g = np.tanh(WX)                     # logcosh derivative
        g_prime = 1.0 - g ** 2              # [n, N]
        # FastICA fixed-point update
        W_new = (g @ X_white) / N - (g_prime.mean(axis=1)[:, None] * W)
        W_new = sym_orth(W_new)
        # convergence: max abs change in component directions (sign-agnostic)
        diff = np.max(np.abs(np.abs((W_new * W).sum(axis=1)) - 1.0))
        W = W_new
        if diff < tol:
            break
    return W


def build_ica_anchor(cache_path: Path, D=200, seed=0, max_chunks=None,
                     subsample_cells=200000, ica_sample_cells=100000,
                     return_diagnostics=False):
    """Build the signed ICA anchor W0 in R^{D x C}.

    Pipeline: per-cell covariance -> top-D SVD whitening -> FastICA in the
    whitened space -> map components back to the C-channel space -> unit-
    normalize each row.

    Args:
        D: number of anchor atoms. Set to the per-cell effective rank
           (~200 for ResNet50 layer3) -- NOT an arbitrary number.
        seed: FIXED seed (whitening is deterministic; this seeds the ICA
              init). Determinism of the anchor depends on keeping it fixed.
        ica_sample_cells: how many whitened cells to feed FastICA. ICA needs
              samples, not just the covariance; we re-stream a subsample.
    Returns:
        W0 [D, C] signed, unit-norm rows. (Plus a diagnostics dict if asked.)
    """
    mean, cov = build_percell_covariance(
        cache_path, max_chunks=max_chunks, subsample_cells=subsample_cells,
        seed=seed)
    C = cov.shape[0]
    D = min(D, C)

    Vk, whiten, eigvals = svd_whiten(cov, D)
    print(f"[ica anchor] SVD whitening: kept D={D}, "
          f"eigval range [{eigvals.min():.4e}, {eigvals.max():.4e}]")

    # collect a whitened sample set for ICA (re-stream a subsample of cells)
    parts = sorted(cache_path.glob("part_*.pkl"))
    if max_chunks is not None:
        parts = parts[:max_chunks]
    rng = np.random.default_rng(seed)
    want_per_chunk = max(1, ica_sample_cells // len(parts))
    whitened = []
    for p in parts:
        part = joblib.load(p)
        act = _to_numpy(part["activation"])
        if act.ndim == 4:
            n, _, H, W = act.shape
            cells = act.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            cells = act
        if cells.shape[0] > want_per_chunk:
            keep = rng.choice(cells.shape[0], want_per_chunk, replace=False)
            cells = cells[keep]
        # center then whiten:  x_white = whiten @ (x - mean)
        xw = (cells - mean) @ whiten.T          # [n', D]
        whitened.append(xw)
        del part, act, cells
    X_white = np.concatenate(whitened, axis=0)
    print(f"[ica anchor] FastICA on {X_white.shape[0]} whitened cells "
          f"(D={D})")

    W_ica = fastica(X_white, D, seed=seed)      # [D, D] in whitened space

    # map ICA components back to the C-channel space:
    #   a whitened-space component  w  corresponds to channel-space
    #   direction  w @ whiten  (since x_white = whiten @ x_centered).
    W0 = W_ica @ whiten                         # [D, C]
    # unit-normalize each row
    norms = np.linalg.norm(W0, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    W0 = W0 / norms

    # fix sign ambiguity deterministically: make the largest-magnitude
    # entry of each row positive (a fixed, seed-independent convention)
    for d in range(W0.shape[0]):
        j = np.argmax(np.abs(W0[d]))
        if W0[d, j] < 0:
            W0[d] = -W0[d]

    print(f"[ica anchor] W0 ready: shape {W0.shape}, signed, unit rows.")

    if return_diagnostics:
        diag = {'eigvals': eigvals, 'mean': mean,
                'n_cells_cov': subsample_cells}
        return W0, diag
    return W0


# ----------------------------------------------------------------------
# Seed-stability verification (run BEFORE using the anchor)
# ----------------------------------------------------------------------

def matched_component_distance(A, B):
    """Sign- and permutation-invariant distance between two [D, C] signed
    bases. Hungarian matching on (1 - |cosine|); returns mean matched
    (1 - |cos|) -- 0 means identical up to sign+permutation.
    """
    from scipy.optimize import linear_sum_assignment
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    D = min(A.shape[0], B.shape[0])
    A, B = A[:D], B[:D]
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    cos = An @ Bn.T                              # [D, D]
    cost = 1.0 - np.abs(cos)                     # sign-invariant
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def verify_anchor_determinism(cache_path, D, seeds, **kw):
    """Build the ICA anchor at several seeds and report cross-seed agreement.

    ICA's fixed-point iteration is seeded; if the anchor is to be a
    deterministic, reportable object the cross-seed distance must be ~0.
    A non-zero value means ICA is finding different independent bases on
    different seeds -- in which case the anchor must fix ONE seed and the
    paper must say so explicitly (as with the NNDSVD anchor).
    """
    print(f"\n{'='*60}\nANCHOR DETERMINISM CHECK (seeds={seeds})\n{'='*60}")
    anchors = []
    for sd in seeds:
        print(f"\n--- seed {sd} ---")
        W0 = build_ica_anchor(cache_path, D=D, seed=sd, **kw)
        anchors.append(W0)
    dists = []
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            d = matched_component_distance(anchors[i], anchors[j])
            dists.append(d)
            print(f"  seeds {seeds[i]}<->{seeds[j]}: "
                  f"matched (1-|cos|) = {d:.4f}")
    mean_d = float(np.mean(dists)) if dists else float('nan')
    print(f"\n  mean cross-seed component distance = {mean_d:.4f}")
    if mean_d < 0.05:
        print("  --> ICA anchor is SEED-STABLE. Safe to use any fixed seed; "
              "the anchor is a well-defined object.")
    else:
        print("  --> ICA anchor VARIES across seeds. The anchor must FIX one "
              "seed, and the paper must report it as 'the seed-0 ICA "
              "anchor', not as intrinsically unique. (Same situation as the "
              "NNDSVD anchor: determinism by fixed seed, not by uniqueness.)")
    return mean_d, anchors


def main():
    ap = argparse.ArgumentParser(
        description="Build / verify the signed ICA anchor for CSAE.")
    ap.add_argument("--cache_dir", type=str, default="cache_activations")
    ap.add_argument("--cache_key", type=str, required=True,
                    help="CSAE activation cache dir (the gcmap1 one).")
    ap.add_argument("--D", type=int, default=200,
                    help="Anchor atom count. Set to the per-cell effective "
                         "rank (~200 for ResNet50 layer3).")
    ap.add_argument("--verify_seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Seeds for the determinism check.")
    ap.add_argument("--max_chunks", type=int, default=None)
    ap.add_argument("--subsample_cells", type=int, default=200000)
    ap.add_argument("--ica_sample_cells", type=int, default=100000)
    ap.add_argument("--save", type=str, default="csae_ica_anchor_W0.npy",
                    help="Where to save the seed-0 anchor.")
    args = ap.parse_args()

    if joblib is None or torch is None:
        raise SystemExit("Needs joblib and torch installed.")

    cache_path = Path(args.cache_dir) / args.cache_key
    if not cache_path.exists():
        raise SystemExit(f"Cache not found: {cache_path}")

    # 1. verify determinism across seeds
    mean_d, anchors = verify_anchor_determinism(
        cache_path, args.D, args.verify_seeds,
        max_chunks=args.max_chunks, subsample_cells=args.subsample_cells,
        ica_sample_cells=args.ica_sample_cells)

    # 2. save the seed-0 anchor for the training script to load
    W0 = anchors[0]
    np.save(args.save, W0)
    print(f"\nSaved seed-{args.verify_seeds[0]} ICA anchor to {args.save}  "
          f"(shape {W0.shape})")
    print("Use this file as --ica_anchor in csae_stable.py.")
    if mean_d >= 0.05:
        print("NOTE: cross-seed distance was not ~0 -- treat the saved file "
              "as 'the seed-0 anchor' in the paper, not a unique object.")


if __name__ == "__main__":
    main()