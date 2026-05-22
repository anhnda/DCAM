"""
csae_ica_anchor.py  (GPU-accelerated)
=====================================
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

GPU acceleration
----------------
The heavy linear algebra now runs on GPU (torch.cuda) when available:
  * per-cell covariance accumulation (X^T X) per chunk,
  * eigh-based SVD whitening (C x C),
  * FastICA fixed-point loop (the hot path -- g @ X_white every iter),
  * batched whitening of the ICA sample set.
Cache I/O (joblib) stays on CPU; each chunk is moved to GPU once and freed.
Everything still runs in float64 by default for numerical parity with the
old CPU path; pass --dtype float32 for a large extra speedup on consumer GPUs.

Determinism
-----------
The whole pipeline is run with a FIXED seed, so the anchor is seed-free by
construction. We VERIFY this by building it at several seeds and checking the
components match (Hungarian-matched, sign/permutation aware). ICA has an
inherent sign+permutation ambiguity; verification accounts for it.

NOTE on GPU determinism: torch CUDA reductions/SVD are not always bitwise
reproducible, but the cross-seed determinism check tolerates that (it
measures component agreement up to sign+perm, well above floating-point
noise). For an exact-reproducible saved anchor, build the final seed-0
anchor on CPU (--device cpu) or accept the (tiny) CUDA nondeterminism.

What this module produces
-------------------------
build_ica_anchor() -> W0  in  R^{D x C}
  D signed unit-norm rows over the C backbone channels.

Usage (standalone verification -- DO THIS BEFORE training CSAE)
---------------------------------------------------------------
    python csae_ica_anchor.py \\
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
# Device / dtype helpers
# ----------------------------------------------------------------------

def resolve_device(device: str):
    """Resolve 'auto'/'cuda'/'cpu' to a torch.device, with a CPU fallback."""
    if torch is None:
        return None
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[ica anchor] CUDA requested but not available; using CPU.")
        device = "cpu"
    return torch.device(device)


def torch_dtype(dtype: str):
    return torch.float64 if dtype == "float64" else torch.float32


def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _chunk_to_cells(part, C_hint=None):
    """Turn a cached chunk into a [n_cells, C] float array (still on CPU).

    Kept in the cache's native dtype to avoid an extra float64 upcast on CPU;
    the upcast happens once on the GPU side.
    """
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
# Per-cell covariance (streamed, GPU-accumulated)
# ----------------------------------------------------------------------

def build_percell_covariance(cache_path: Path, device, dtype,
                             max_chunks=None, subsample_cells=200000, seed=0):
    """Per-cell channel covariance + mean, streamed one chunk at a time.

    The X^T X accumulation runs on `device`. Each chunk is moved to the GPU
    once, reduced, and freed -- activations are never all resident at once.

    Returns (mean [C], cov [C, C]) as float64 numpy arrays.
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
        cells = _chunk_to_cells(part)                   # [n, C] cpu
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
# SVD-whitening + FastICA (GPU)
# ----------------------------------------------------------------------

def svd_whiten(cov, D, device, dtype):
    """Top-D PCA whitening transform from the covariance, via torch.eigh.

    Returns:
        Vk  [C, D]      top-D eigenvectors (signed, orthonormal), numpy f64
        whiten [D, C]   the whitening map diag(1/sqrt(lambda)) V^T, numpy f64
        eigvals [D]     the kept eigenvalues, numpy f64
    """
    cov_t = torch.as_tensor(cov, device=device, dtype=dtype)
    cov_t = 0.5 * (cov_t + cov_t.T)
    w, V = torch.linalg.eigh(cov_t)             # ascending
    w = w.flip(0)
    V = V.flip(1)
    keep = slice(0, D)
    eigvals = w[keep].clamp_min(1e-12)
    Vk = V[:, keep]                             # [C, D]
    whiten = (Vk / torch.sqrt(eigvals)).T       # [D, C]
    return (Vk.double().cpu().numpy(),
            whiten.double().cpu().numpy(),
            eigvals.double().cpu().numpy())


def fastica(X_white, n_components, device, dtype, seed=0,
            max_iter=500, tol=1e-5, verbose=True):
    Xw = torch.as_tensor(np.asarray(X_white)) if not isinstance(
        X_white, torch.Tensor) else X_white
    Xw = Xw.to(device=device, dtype=dtype)
    N, D = Xw.shape
    n = min(n_components, D)
    g_cpu = torch.Generator(device="cpu").manual_seed(int(seed))
    W = torch.randn(n, D, generator=g_cpu, dtype=torch.float64).to(device=device, dtype=dtype)

    def sym_orth(M):
        u, s, vt = torch.linalg.svd(M, full_matrices=False)
        return u @ vt

    W = sym_orth(W)
    XwT = Xw.T.contiguous()
    converged, last_diff = False, float('nan')
    for it in range(max_iter):
        WX = W @ XwT
        g = torch.tanh(WX)
        g_prime = 1.0 - g * g
        W_new = (g @ Xw) / N - g_prime.mean(dim=1, keepdim=True) * W
        W_new = sym_orth(W_new)
        last_diff = (((W_new * W).sum(dim=1).abs()) - 1.0).abs().max().item()
        W = W_new
        if last_diff < tol:
            converged = True
            break
    if verbose:
        status = "converged" if converged else f"HIT max_iter={max_iter}"
        print(f"[fastica] seed={seed}: {status} at it={it+1}, "
              f"final diff={last_diff:.2e}")
    return W.double().cpu().numpy()


def build_ica_anchor(cache_path: Path, D=200, seed=0, max_chunks=None,
                     subsample_cells=200000, ica_sample_cells=100000,
                     device="auto", dtype="float64",
                     return_diagnostics=False, _reuse_cov=None):
    """Build the signed ICA anchor W0 in R^{D x C} (GPU-accelerated).

    Pipeline: per-cell covariance -> top-D SVD whitening -> FastICA in the
    whitened space -> map components back to the C-channel space -> unit-
    normalize each row.

    Args:
        D: number of anchor atoms (~200 for ResNet50 layer3).
        seed: FIXED seed (whitening is deterministic; this seeds ICA init).
        device: 'auto' | 'cuda' | 'cpu'.
        dtype: 'float64' (parity with CPU path) | 'float32' (faster on GPU).
        _reuse_cov: optional (mean, cov) tuple to skip the covariance pass
            (used by the verifier to stream the cache only once).
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

    Vk, whiten, eigvals = svd_whiten(cov, D, dev, dt)
    print(f"[ica anchor] SVD whitening: kept D={D}, "
          f"eigval range [{eigvals.min():.4e}, {eigvals.max():.4e}]")

    # collect a whitened sample set for ICA (re-stream a subsample of cells);
    # the per-chunk whitening (cells - mean) @ whiten.T runs on GPU.
    parts = sorted(cache_path.glob("part_*.pkl"))
    if max_chunks is not None:
        parts = parts[:max_chunks]
    rng = np.random.default_rng(seed)
    want_per_chunk = max(1, ica_sample_cells // len(parts))
    mean_t = torch.as_tensor(mean, device=dev, dtype=dt)
    whiten_t = torch.as_tensor(whiten, device=dev, dtype=dt)   # [D, C]
    whitened = []
    for p in parts:
        part = joblib.load(p)
        cells = _chunk_to_cells(part)
        if cells.shape[0] > want_per_chunk:
            keep = rng.choice(cells.shape[0], want_per_chunk, replace=False)
            cells = cells[keep]
        xt = torch.as_tensor(np.ascontiguousarray(cells)).to(
            device=dev, dtype=dt)
        xw = (xt - mean_t) @ whiten_t.T              # [n', D]
        whitened.append(xw.cpu())
        del part, cells, xt, xw
    X_white = torch.cat(whitened, dim=0)             # cpu tensor
    print(f"[ica anchor] FastICA on {X_white.shape[0]} whitened cells "
          f"(D={D}) on {dev}")

    W_ica = fastica(X_white, D, dev, dt, seed=seed)  # [D, D] numpy f64

    # map ICA components back to channel space: w @ whiten
    W0 = W_ica @ whiten                              # [D, C]
    norms = np.linalg.norm(W0, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    W0 = W0 / norms

    # deterministic sign fix: largest-magnitude entry of each row positive
    for d in range(W0.shape[0]):
        j = np.argmax(np.abs(W0[d]))
        if W0[d, j] < 0:
            W0[d] = -W0[d]

    print(f"[ica anchor] W0 ready: shape {W0.shape}, signed, unit rows.")

    if return_diagnostics:
        diag = {'eigvals': eigvals, 'mean': mean,
                'n_cells_cov': subsample_cells, 'device': str(dev),
                'dtype': dtype}
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


def verify_anchor_determinism(cache_path, D, seeds, device="auto",
                              dtype="float64", **kw):
    """Build the ICA anchor at several seeds and report cross-seed agreement.

    The per-cell covariance does NOT depend on the ICA seed (only on the
    covariance-subsample rng), so it is built ONCE and reused across seeds --
    this halves the cache streaming compared to the old per-seed rebuild.
    """
    print(f"\n{'='*60}\nANCHOR DETERMINISM CHECK (seeds={seeds})\n{'='*60}")
    dev = resolve_device(device)
    dt = torch_dtype(dtype)
    # build covariance once (seed only affects the subsample draw; fix it)
    cov_seed = seeds[0]
    mean, cov = build_percell_covariance(
        cache_path, dev, dt, max_chunks=kw.get("max_chunks"),
        subsample_cells=kw.get("subsample_cells", 200000), seed=cov_seed)
    anchors = []
    for sd in seeds:
        print(f"\n--- seed {sd} ---")
        W0 = build_ica_anchor(cache_path, D=D, seed=sd, device=device,
                              dtype=dtype, _reuse_cov=(mean, cov), **kw)
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
        description="Build / verify the signed ICA anchor for CSAE (GPU).")
    ap.add_argument("--cache_dir", type=str, default="cache_activations")
    ap.add_argument("--cache_key", type=str, required=True,
                    help="CSAE activation cache dir (the gcmap1 one).")
    ap.add_argument("--D", type=int, default=200,
                    help="Anchor atom count (~200 for ResNet50 layer3).")
    ap.add_argument("--verify_seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Seeds for the determinism check.")
    ap.add_argument("--max_chunks", type=int, default=None)
    ap.add_argument("--subsample_cells", type=int, default=200000)
    ap.add_argument("--ica_sample_cells", type=int, default=100000)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "cpu"],
                    help="Where to run the linear algebra.")
    ap.add_argument("--dtype", type=str, default="float64",
                    choices=["float64", "float32"],
                    help="float64 = CPU-parity; float32 = faster on GPU.")
    ap.add_argument("--save", type=str, default="csae_ica_anchor_W0.npy",
                    help="Where to save the seed-0 anchor.")
    args = ap.parse_args()

    if joblib is None or torch is None:
        raise SystemExit("Needs joblib and torch installed.")

    cache_path = Path(args.cache_dir) / args.cache_key
    if not cache_path.exists():
        raise SystemExit(f"Cache not found: {cache_path}")

    # 1. verify determinism across seeds (covariance built once, reused)
    mean_d, anchors = verify_anchor_determinism(
        cache_path, args.D, args.verify_seeds,
        device=args.device, dtype=args.dtype,
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