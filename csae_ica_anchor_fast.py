"""
csae_ica_anchor_fast.py
=======================
Drop-in replacements for the two slow functions in csae_ica_anchor.py.

The real cost in the original is streaming the cache TWICE (once for the
covariance, once to collect whitened cells for ICA). This module streams
ONCE: it accumulates the covariance AND stashes a raw-cell subsample in the
same pass, then whitens that subsample in memory afterwards.

A GPU path for fastica() is included but OPTIONAL and guarded -- it only
helps if FastICA is genuinely your bottleneck, which at D=200 it usually is
not. Covariance streaming is I/O-bound; no GPU helps that.
"""

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
try:
    import cupy as cp
except ImportError:
    cp = None


def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


# ----------------------------------------------------------------------
# SINGLE-PASS: covariance + raw-cell subsample in one stream
# ----------------------------------------------------------------------

def stream_cov_and_sample(cache_path: Path, max_chunks=None,
                          subsample_cells=200000, ica_sample_cells=100000,
                          seed=0):
    """One pass over the cache. Returns (mean[C], cov[C,C], raw_sample[M,C]).

    raw_sample is uncentered/unwhitened cells -- centering+whitening is done
    by the caller once the covariance (hence the whitening map) exists.
    """
    parts = sorted(cache_path.glob("part_*.pkl"))
    if not parts:
        raise FileNotFoundError(f"No part_*.pkl in {cache_path}")
    if max_chunks is not None:
        parts = parts[:max_chunks]
    rng = np.random.default_rng(seed)
    want_per_chunk = max(1, ica_sample_cells // len(parts))

    n_cells = 0
    sum_x = None
    sum_xxT = None
    sample = []
    print(f"Single-pass stream over {len(parts)} chunk(s)...")
    for i, p in enumerate(parts):
        part = joblib.load(p)
        act = _to_numpy(part["activation"])
        if act.ndim == 4:
            n, C, H, W = act.shape
            cells = act.transpose(0, 2, 3, 1).reshape(-1, C)
        elif act.ndim == 2:
            cells = act
        else:
            raise ValueError(f"unexpected ndim {act.ndim}")

        # covariance subsample
        cov_cells = cells
        if subsample_cells and cells.shape[0] > subsample_cells:
            keep = rng.choice(cells.shape[0], subsample_cells, replace=False)
            cov_cells = cells[keep]
        if sum_x is None:
            C = cov_cells.shape[1]
            sum_x = np.zeros(C, dtype=np.float64)
            sum_xxT = np.zeros((C, C), dtype=np.float64)
        sum_x += cov_cells.sum(axis=0)
        sum_xxT += cov_cells.T @ cov_cells
        n_cells += cov_cells.shape[0]

        # ICA subsample -- stashed RAW, whitened later
        if cells.shape[0] > want_per_chunk:
            keep = rng.choice(cells.shape[0], want_per_chunk, replace=False)
            sample.append(cells[keep].copy())
        else:
            sample.append(cells.copy())

        del part, act, cells
        if (i + 1) % 20 == 0 or (i + 1) == len(parts):
            print(f"  ...{i+1}/{len(parts)} chunks, {n_cells} cov cells")

    mean = sum_x / max(n_cells, 1)
    cov = sum_xxT / max(n_cells, 1) - np.outer(mean, mean)
    cov = 0.5 * (cov + cov.T)
    raw_sample = np.concatenate(sample, axis=0)
    print(f"Done: C={cov.shape[0]}, {n_cells} cov cells, "
          f"{raw_sample.shape[0]} ICA cells -- ONE pass.")
    return mean, cov, raw_sample


def svd_whiten(cov, D):
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    keep = order[:D]
    eigvals = np.clip(w[keep], 1e-12, None)
    Vk = V[:, keep]
    whiten = (Vk / np.sqrt(eigvals)).T
    return Vk, whiten, eigvals


# ----------------------------------------------------------------------
# FastICA -- CPU, with an OPTIONAL GPU path
# ----------------------------------------------------------------------

def fastica(X_white, n_components, seed=0, max_iter=500, tol=1e-5,
            use_gpu=False):
    """Symmetric FastICA, logcosh. Identical math to the original.

    use_gpu=True moves the fixed-point loop to cupy IF cupy is importable
    AND the problem is large enough that the transfer pays off. Otherwise it
    silently runs on CPU -- there is no behavioural difference, only speed.
    """
    xp = np
    on_gpu = False
    # only bother with GPU for genuinely large sample counts; below this the
    # host<->device copy costs more than it saves.
    if use_gpu and cp is not None and X_white.shape[0] >= 50000:
        xp = cp
        on_gpu = True
        X_white = cp.asarray(X_white)
        print(f"[fastica] running on GPU (cupy), N={X_white.shape[0]}")
    elif use_gpu and cp is None:
        print("[fastica] use_gpu set but cupy not installed -- CPU fallback.")
    elif use_gpu:
        print(f"[fastica] N={X_white.shape[0]} too small for GPU -- CPU.")

    rng = np.random.default_rng(seed)
    N, D = X_white.shape
    n = min(n_components, D)

    # init on CPU for seed-identical behaviour, then move
    W = rng.standard_normal((n, D))
    if on_gpu:
        W = cp.asarray(W)

    def sym_orth(M):
        u, s, vt = xp.linalg.svd(M, full_matrices=False)
        return u @ vt

    W = sym_orth(W)
    Xt = X_white.T
    for it in range(max_iter):
        WX = W @ Xt
        g = xp.tanh(WX)
        g_prime = 1.0 - g ** 2
        W_new = (g @ X_white) / N - (g_prime.mean(axis=1)[:, None] * W)
        W_new = sym_orth(W_new)
        diff = xp.max(xp.abs(xp.abs((W_new * W).sum(axis=1)) - 1.0))
        W = W_new
        if float(diff) < tol:
            break

    if on_gpu:
        W = cp.asnumpy(W)
    return W


# ----------------------------------------------------------------------
# Single-pass anchor build
# ----------------------------------------------------------------------

def build_ica_anchor(cache_path: Path, D=200, seed=0, max_chunks=None,
                     subsample_cells=200000, ica_sample_cells=100000,
                     use_gpu=False, return_diagnostics=False):
    """Same output as the original build_ica_anchor, ONE cache pass.

    use_gpu only affects fastica(); covariance streaming is I/O-bound and
    runs on CPU regardless.
    """
    mean, cov, raw_sample = stream_cov_and_sample(
        cache_path, max_chunks=max_chunks, subsample_cells=subsample_cells,
        ica_sample_cells=ica_sample_cells, seed=seed)
    C = cov.shape[0]
    D = min(D, C)

    Vk, whiten, eigvals = svd_whiten(cov, D)
    print(f"[ica anchor] SVD whitening: kept D={D}, "
          f"eigval range [{eigvals.min():.4e}, {eigvals.max():.4e}]")

    # whiten the stashed raw sample in memory -- no second cache stream
    X_white = (raw_sample - mean) @ whiten.T
    print(f"[ica anchor] FastICA on {X_white.shape[0]} whitened cells (D={D})")

    W_ica = fastica(X_white, D, seed=seed, use_gpu=use_gpu)

    W0 = W_ica @ whiten
    norms = np.linalg.norm(W0, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    W0 = W0 / norms

    for d in range(W0.shape[0]):
        j = np.argmax(np.abs(W0[d]))
        if W0[d, j] < 0:
            W0[d] = -W0[d]

    print(f"[ica anchor] W0 ready: shape {W0.shape}, signed, unit rows.")
    if return_diagnostics:
        return W0, {'eigvals': eigvals, 'mean': mean}
    return W0