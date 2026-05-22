"""
check_anchor_nonnegativity.py
=============================
Diagnose the DCAM anchor: rank, non-negativity, and atom-stability of the
channel co-activation matrix S.

Four analyses
-------------
1. NON-NEGATIVITY GAP. Factorize S at rank D three ways and compare relative
   error ||S - S_hat||_F / ||S||_F:
     - signed eig: best symmetric rank-D fit (any sign). The floor.
     - signed GG^T: PSD signed fit, the fair analogue of NMF's W W^T.
     - NMF W W^T (W >= 0): the anchor's factorization, run to convergence.
   gap = err_NMF - err_signed. gap ~ 0 => non-negativity is cheap (any anchor
   failure is a SOLVER problem). gap large => S has no good non-negative
   factorization (the premise is broken).

2. EFFECTIVE RANK. Participation ratio (sum l)^2/sum(l^2), entropy rank, and
   eigenvalue counts. A steep spectrum means only ~R directions carry signal;
   asking for D >> R atoms factorizes a noise floor.

3. ATOM-STABILITY SWEEP. For each D, run NMF from several seeds and measure
   how much the atom sets disagree (Hungarian-matched distance). Turns
   "atoms split and absorb" into a number. Prediction: stable at D ~ R,
   unstable once D >> R.

4. PER-CELL RANK CHECK (--percell_rank). S is built from spatially-AVERAGED
   activations; averaging can crush rank. This streams the per-cell channel
   covariance and compares its effective rank to S's. Decides whether a
   near-low-rank spectrum is a property of the LAYER or only of the anchor's
   averaged construction.

Memory
------
Activations are NEVER fully loaded. Cache chunks are streamed one at a time
and reduced to a [C, C] matrix, then freed. Peak extra RAM is one chunk.
--max_chunks / --subsample_rows / --percell_subsample lighten it further.

Usage
-----
    # full sweep over the small-D regime where the rank elbow lives
    python check_anchor_nonnegativity.py \\
        --cache_dir cache_activations \\
        --cache_key <activations_... dir> \\
        --D 10 25 50 100 200 512 \\
        --percell_rank --device cuda

    # reuse a precomputed S (skips streaming; per-cell check still needs cache)
    python check_anchor_nonnegativity.py --S_path anchor_nonnegativity_check_S.npy

If you don't know the cache key, run with --list.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import joblib
except ImportError:
    joblib = None
try:
    import torch
except ImportError:
    torch = None


# ----------------------------------------------------------------------
# Streaming construction of S from cached activation chunks
# ----------------------------------------------------------------------

def list_caches(cache_dir: Path):
    """Print candidate cache directories (those containing metadata.pkl)."""
    print(f"Scanning {cache_dir} for caches...")
    found = []
    if not cache_dir.exists():
        print(f"  {cache_dir} does not exist.")
        return found
    for d in sorted(cache_dir.iterdir()):
        if d.is_dir() and (d / "metadata.pkl").exists():
            parts = sorted(d.glob("part_*.pkl"))
            print(f"  {d.name}   ({len(parts)} parts)")
            found.append(d)
    if not found:
        print("  No chunked caches found.")
    return found


def _to_numpy(x):
    """Accept a torch tensor or ndarray, return float64 ndarray."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def build_S_streaming(cache_path: Path, max_chunks=None, subsample_rows=None):
    """Build S = E[ Abar Abar^T ] by streaming cache parts one at a time.

    Abar = spatial mean of the activation tensor -> one C-vector per image.
    S accumulates Abar^T Abar over all images, divided by the image count.

    Memory: only ONE chunk is resident at a time. S itself is C x C.

    Args:
        cache_path: directory containing part_XXXX.pkl files.
        max_chunks: if set, only stream this many chunks (subsampling).
        subsample_rows: if set, randomly keep this many images per chunk.
    Returns:
        S [C, C] float64 ndarray, and the total image count used.
    """
    parts = sorted(cache_path.glob("part_*.pkl"))
    if not parts:
        raise FileNotFoundError(f"No part_*.pkl in {cache_path}")
    if max_chunks is not None:
        parts = parts[:max_chunks]

    print(f"Streaming {len(parts)} cache part(s) from {cache_path.name} ...")
    S = None
    n_total = 0
    rng = np.random.default_rng(0)

    for i, p in enumerate(parts):
        part = joblib.load(p)                       # one chunk in RAM
        act = part["activation"]                    # [n, C, H, W]
        act = _to_numpy(act)
        # spatial mean -> [n, C]
        if act.ndim == 4:
            abar = act.mean(axis=(2, 3))
        elif act.ndim == 2:
            abar = act
        else:
            raise ValueError(f"unexpected activation ndim {act.ndim} in {p}")

        if subsample_rows is not None and abar.shape[0] > subsample_rows:
            keep = rng.choice(abar.shape[0], subsample_rows, replace=False)
            abar = abar[keep]

        if S is None:
            C = abar.shape[1]
            S = np.zeros((C, C), dtype=np.float64)
        S += abar.T @ abar
        n_total += abar.shape[0]

        del part, act, abar                         # free the chunk
        if (i + 1) % 10 == 0 or (i + 1) == len(parts):
            print(f"  ...{i + 1}/{len(parts)} chunks, {n_total} images so far")

    S /= max(n_total, 1)
    print(f"Built S: shape {S.shape}, {n_total} images.")
    return S, n_total


def build_S_percell_streaming(cache_path: Path, max_chunks=None,
                              subsample_cells=None):
    """Build the PER-CELL channel covariance  S_cell = E[ a a^T ]  where a is
    the C-vector of activations at ONE spatial cell (not spatially averaged).

    Why this exists
    ---------------
    The main S in build_S_streaming uses Abar = spatial MEAN over H,W. Spatial
    averaging can crush rank: it is plausible the near-low-rank spectrum is an
    artifact of averaging, and the per-cell activations the encoder actually
    sees are much higher rank. If S_cell is ALSO near-low-rank, the
    "layer supports only ~R concepts" conclusion is about the LAYER and is
    safe to put in the paper. If S_cell is high rank, the conclusion is only
    about the anchor's averaged construction -- a finding about build_anchor,
    not the layer.

    Memory: identical streaming discipline -- one chunk resident at a time.
    Each [n, C, H, W] chunk is reshaped to [n*H*W, C] cells; cells are
    optionally subsampled (there are n*H*W of them, e.g. 100*196 per chunk),
    then accumulated into the C x C matrix.

    Returns:
        S_cell [C, C] float64 ndarray, total cell count used.
    """
    parts = sorted(cache_path.glob("part_*.pkl"))
    if not parts:
        raise FileNotFoundError(f"No part_*.pkl in {cache_path}")
    if max_chunks is not None:
        parts = parts[:max_chunks]

    print(f"Streaming {len(parts)} part(s) for PER-CELL covariance "
          f"from {cache_path.name} ...")
    S = None
    n_cells = 0
    rng = np.random.default_rng(0)

    for i, p in enumerate(parts):
        part = joblib.load(p)
        act = _to_numpy(part["activation"])         # [n, C, H, W]
        if act.ndim == 4:
            n, C, H, W = act.shape
            # [n, C, H, W] -> [n*H*W, C]
            cells = act.transpose(0, 2, 3, 1).reshape(-1, C)
        elif act.ndim == 2:
            cells = act
        else:
            raise ValueError(f"unexpected activation ndim {act.ndim} in {p}")

        if subsample_cells is not None and cells.shape[0] > subsample_cells:
            keep = rng.choice(cells.shape[0], subsample_cells, replace=False)
            cells = cells[keep]

        if S is None:
            S = np.zeros((cells.shape[1], cells.shape[1]), dtype=np.float64)
        S += cells.T @ cells
        n_cells += cells.shape[0]

        del part, act, cells
        if (i + 1) % 10 == 0 or (i + 1) == len(parts):
            print(f"  ...{i + 1}/{len(parts)} chunks, "
                  f"{n_cells} cells so far")

    S /= max(n_cells, 1)
    print(f"Built S_cell: shape {S.shape}, {n_cells} spatial cells.")
    return S, n_cells


def effective_rank_metrics(evals):
    """Summarize how concentrated a PSD spectrum is -- i.e. how many
    directions actually carry signal.

    Args:
        evals: 1-D array of eigenvalues, DESCENDING order.
    Returns:
        dict with:
          participation_ratio : (sum l)^2 / sum(l^2). Equals the true rank
                                 for a flat spectrum, ~1 for a rank-1 spike.
                                 The single most honest "effective rank".
          entropy_rank        : exp(Shannon entropy of the normalized
                                 spectrum) -- another standard effective rank.
          n_above_1pct        : count of eigenvalues > 1% of the largest.
          n_above_0p1pct      : count of eigenvalues > 0.1% of the largest.
          n_90 / n_95 / n_99  : how many eigenvalues to reach that fraction
                                 of total positive energy.
    """
    ev = np.asarray(evals, dtype=np.float64)
    pos = ev[ev > 0]
    if len(pos) == 0:
        return {}
    s1 = pos.sum()
    s2 = (pos ** 2).sum()
    participation = float(s1 * s1 / (s2 + 1e-30))
    p = pos / s1
    entropy = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    cume = np.cumsum(pos) / s1

    def n_for(frac):
        idx = np.searchsorted(cume, frac)
        return int(min(idx + 1, len(pos)))

    return {
        "participation_ratio": participation,
        "entropy_rank": entropy,
        "n_above_1pct": int((pos > 0.01 * pos[0]).sum()),
        "n_above_0p1pct": int((pos > 0.001 * pos[0]).sum()),
        "n_90": n_for(0.90),
        "n_95": n_for(0.95),
        "n_99": n_for(0.99),
    }


def spectrum_of(S, device="cpu"):
    """Descending eigenvalues of a symmetric matrix S (GPU eigh if available)."""
    if torch is not None:
        evt = torch.linalg.eigvalsh(_as_tensor(S, device))
        return evt.flip(0).cpu().numpy()
    return np.linalg.eigvalsh(S)[::-1]


# ----------------------------------------------------------------------
# Factorizations
# ----------------------------------------------------------------------

# All factorization math runs on torch tensors so it can use the GPU.
# S is only C x C (~1024^2), so the matrices are tiny; the GPU mainly helps
# the NMF multiplicative-update loop at large D. eigh/IO stay cheap either way.
#
# resolve_device() picks cuda if asked-for and available, else falls back to
# cpu with a printed warning -- the script must never hard-fail on a missing
# GPU. All tensors are float64 for numerical headroom (NMF ratios, eigh).

def resolve_device(requested: str) -> str:
    """Return a usable torch device string, falling back to cpu if needed."""
    if torch is None:
        print("  [device] torch not importable -> using numpy/cpu paths.")
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested in ("cuda", "gpu") or requested.startswith("cuda:"):
        if torch.cuda.is_available():
            dev = "cuda" if requested in ("cuda", "gpu") else requested
            name = torch.cuda.get_device_name(
                0 if dev == "cuda" else int(dev.split(":")[1]))
            print(f"  [device] using {dev}  ({name})")
            return dev
        print("  [device] cuda requested but not available -> cpu fallback.")
        return "cpu"
    return "cpu"


def _as_tensor(S, device):
    """numpy array -> float64 torch tensor on the given device."""
    if torch is None:
        return S
    if isinstance(S, torch.Tensor):
        return S.to(device=device, dtype=torch.float64)
    return torch.as_tensor(S, dtype=torch.float64, device=device)


def rel_err(S, S_hat):
    """Relative Frobenius error. Accepts numpy arrays or torch tensors."""
    if torch is not None and isinstance(S, torch.Tensor):
        num = torch.linalg.norm(S - S_hat)
        den = torch.linalg.norm(S) + 1e-12
        return float((num / den).item())
    return float(np.linalg.norm(S - S_hat) / (np.linalg.norm(S) + 1e-12))


def signed_eig_lowrank(S, D, device="cpu"):
    """Best symmetric rank-D fit: keep the D eigenpairs of largest |lambda|.

    Runs torch.linalg.eigh on `device`. For D >= rank(S) the error is ~0.
    Returns (S_hat as numpy, relative error).
    """
    St = _as_tensor(S, device)
    w, V = torch.linalg.eigh(St)                    # ascending eigenvalues
    order = torch.argsort(w.abs(), descending=True)
    keep = order[:min(D, w.shape[0])]
    Vk = V[:, keep]
    S_hat = (Vk * w[keep]) @ Vk.t()
    e = rel_err(St, S_hat)
    return S_hat.cpu().numpy(), e


def signed_GGt_psd(S, D, device="cpu"):
    """Signed G G^T fit, G in R^{C x D}. For a PSD S this equals the
    truncated eigendecomposition restricted to POSITIVE eigenvalues
    (G G^T is itself PSD, so it cannot use negative-eigenvalue directions).

    This is the fair signed analogue of NMF's W W^T (both yield PSD S_hat).
    Returns (S_hat as numpy, relative error, number of positive dirs used).
    """
    St = _as_tensor(S, device)
    w, V = torch.linalg.eigh(St)
    pos = w > 0
    w_pos, V_pos = w[pos], V[:, pos]
    order = torch.argsort(w_pos, descending=True)
    keep = order[:min(D, w_pos.shape[0])]
    g = V_pos[:, keep] * torch.sqrt(w_pos[keep])    # [C, D'] -> G
    S_hat = g @ g.t()
    e = rel_err(St, S_hat)
    return S_hat.cpu().numpy(), e, int(g.shape[1])


def nndsvd_init_symmetric(S_work, D, device="cpu"):
    """Deterministic NNDSVD init for symmetric NMF  S ~ W W^T.

    NNDSVD (Boutsidis & Gallopoulos, 2008) seeds NMF from the truncated SVD,
    so the init is FULLY DETERMINISTIC -- no seed. For a symmetric PSD S we
    use the eigendecomposition (eigenvectors = singular vectors, eigenvalues
    >= 0 = singular values).

    For each of the top-D eigenpairs (lambda_j, v_j), v_j has mixed sign but
    sqrt(lambda_j) v_j is a valid signed factor column. NNDSVD turns it into
    a non-negative column by taking whichever of the positive part v_j+ or
    the negative part v_j- carries more energy, scaled by sqrt(lambda_j) and
    the relevant norm. This gives W >= 0 with W W^T already close to the
    leading structure of S.

    Why this matters here: the random-init NMF gave cross-seed atom distance
    ~0.12 flat in D. If that is just different random basins, a deterministic
    NNDSVD init should give cross-seed distance ~0 (identical start ->
    identical descent). If NNDSVD-init NMF is STILL seed-varying, something
    is non-deterministic in the solver itself; if it is stable, the anchor
    can be made deterministic via this init.

    Returns W [C, D] non-negative float64 tensor on `device`.
    """
    St = _as_tensor(S_work, device)
    w, V = torch.linalg.eigh(St)                    # ascending
    order = torch.argsort(w, descending=True)
    keep = order[:min(D, w.shape[0])]
    w = torch.clamp(w[keep], min=0.0)               # PSD: tiny negs -> 0
    V = V[:, keep]                                  # [C, D']

    C = St.shape[0]
    Dk = V.shape[1]
    W = torch.zeros((C, Dk), dtype=torch.float64, device=device)
    for j in range(Dk):
        v = V[:, j]
        s = torch.sqrt(w[j])
        vp = torch.clamp(v, min=0.0)
        vn = torch.clamp(-v, min=0.0)
        np_ = torch.linalg.norm(vp)
        nn_ = torch.linalg.norm(vn)
        if np_ >= nn_:
            col = vp / (np_ + 1e-12)
        else:
            col = vn / (nn_ + 1e-12)
        W[:, j] = s * col
    # pad with tiny positive values if D > rank kept
    if Dk < D:
        pad = torch.full((C, D - Dk), 1e-4, dtype=torch.float64,
                         device=device)
        W = torch.cat([W, pad], dim=1)
    W = torch.clamp(W, min=1e-9)
    return W


def nmf_symmetric(S, D, iters=4000, restarts=3, normalize=True, seed=0,
                  tol=1e-7, verbose=True, device="cpu", init="random"):
    """Symmetric NMF  S ~ W W^T,  W >= 0,  via multiplicative updates.

    GPU-accelerated: every per-iteration op (the [C,D] matmuls SW, W W^T W)
    runs on `device`. This is the part that actually benefits from a GPU at
    large D -- eigh and IO do not.

    A stronger version of build_anchor's NMF: more iters, optional S
    normalization to a correlation matrix, multiple random restarts, and a
    real convergence trace (build_anchor logs only the final iter).

    Args:
        init: 'random'  -- W = rescaled rand(C,D), seeded by `seed`. Each seed
                           gives a different basin (this is the seed-varying
                           path; restarts > 1 keeps the best).
              'nndsvd'  -- W = deterministic NNDSVD(S). The init is identical
                           for every seed, so the result is seed-FREE.
                           restarts is forced to 1 (a deterministic init has
                           nothing to restart). Use this to test whether the
                           per-cell anchor's seed-sensitivity is just random
                           init or something intrinsic.

    Returns the best (lowest-error) W as numpy, its relative error against
    the ORIGINAL S, and the per-iteration error trace of the best restart.
    """
    St = _as_tensor(S, device)
    S_work = torch.clamp(St, min=0.0).clone()       # S is PSD; kill tiny negs

    scale_back = None
    if normalize:
        d = torch.sqrt(torch.clamp(torch.diag(S_work), min=1e-12))
        Dinv = 1.0 / d
        S_work = S_work * torch.outer(Dinv, Dinv)   # correlation matrix
        scale_back = d                              # to undo for reporting

    C = S_work.shape[0]
    norm_Swork = torch.linalg.norm(S_work)
    best_W, best_err, best_trace = None, float("inf"), None

    if init == "nndsvd":
        # deterministic init: one restart only, seed is irrelevant
        restarts = 1

    for r in range(restarts):
        if init == "nndsvd":
            W = nndsvd_init_symmetric(S_work, D, device=device)
            # rescale to S_work's magnitude, same as the random path
            W *= torch.sqrt(norm_Swork
                            / (torch.linalg.norm(W @ W.t()) + 1e-12))
        else:
            g = torch.Generator(device="cpu").manual_seed(seed + r)
            # generate on cpu for cross-device-deterministic init, then move
            W = torch.rand((C, D), generator=g, dtype=torch.float64)
            W = W.to(device)
            # scale W so W W^T starts near S_work's magnitude
            W *= torch.sqrt(norm_Swork
                            / (torch.linalg.norm(W @ W.t()) + 1e-12))
        trace = []
        prev = float("inf")
        for it in range(iters):
            SW = S_work @ W                         # [C, D]
            WWtW = W @ (W.t() @ W)                  # [C, D]
            W *= SW / (WWtW + 1e-9)
            W = torch.clamp(W, min=1e-9)
            if (it + 1) % 50 == 0 or it == iters - 1:
                e = rel_err(S_work, W @ W.t())
                trace.append((it + 1, e))
                if verbose and ((it + 1) % 500 == 0 or it == iters - 1):
                    print(f"    [NMF D={D} restart {r}] iter {it+1}: "
                          f"rel_err(S_norm)={e:.4f}")
                if abs(prev - e) < tol:
                    if verbose:
                        print(f"    [NMF D={D} restart {r}] converged "
                              f"at iter {it+1}")
                    break
                prev = e

        # error against the (normalized) working S, then map W back to
        # original-S scale for the reported number.
        e_norm = rel_err(S_work, W @ W.t())
        if scale_back is not None:
            W_orig = W * scale_back[:, None]        # undo the correlation norm
            e_orig = rel_err(St, W_orig @ W_orig.t())
        else:
            W_orig, e_orig = W, e_norm
        if e_orig < best_err:
            best_W = W_orig.cpu().numpy()
            best_err, best_trace = e_orig, trace

    return best_W, best_err, best_trace


# ----------------------------------------------------------------------
# Atom-stability across seeds
# ----------------------------------------------------------------------

def _l1_normalize_rows(M):
    """Row-L1-normalize a [D, C] matrix (atoms as rows on the L1 simplex).
    Mirrors what build_anchor does before comparing atoms."""
    s = np.abs(M).sum(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0
    return M / s


def matched_atom_distance(W_a, W_b):
    """Permutation-invariant distance between two atom sets.

    W_a, W_b are NMF factors [C, D]. Atoms are the COLUMNS; we compare them
    as rows-over-channels (transpose), L1-normalized, via a Hungarian
    matching on pairwise L2 distance -- the same construction as
    run_dcam_full.atom_distance, so the numbers are comparable to DCAM's.

    Returns:
        mean_matched_l2 : average L2 distance between matched atom pairs.
                          Scale-comparable across different D (it is a mean,
                          not a sum).
        frac_unmatched  : fraction of atoms whose matched distance exceeds a
                          'basically a different atom' threshold (0.5 of the
                          max possible L1-simplex distance, ~sqrt(2)).
    """
    from scipy.optimize import linear_sum_assignment
    A = _l1_normalize_rows(np.asarray(W_a, dtype=np.float64).T)  # [D, C]
    B = _l1_normalize_rows(np.asarray(W_b, dtype=np.float64).T)  # [D, C]
    D = min(A.shape[0], B.shape[0])
    A, B = A[:D], B[:D]
    # pairwise L2 cost matrix
    cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)  # [D, D]
    r, c = linear_sum_assignment(cost)
    matched = cost[r, c]
    mean_matched = float(matched.mean())
    # two L1-simplex points are maximally ~sqrt(2) apart; >0.5*sqrt(2)
    # means the matched partner is essentially a different atom.
    frac_unmatched = float((matched > 0.5 * np.sqrt(2.0)).mean())
    return mean_matched, frac_unmatched


def atom_stability_sweep(S, D_list, seeds, iters, restarts, normalize,
                         device="cpu", verbose=False):
    """For each D, run NMF from several DIFFERENT seeds and measure how much
    the resulting atom sets disagree (mean pairwise matched-atom distance).

    This is the experiment that turns "atoms split and absorb" from an
    observation into a number. The prediction: stable (small distance) at D
    near the effective rank, degrading sharply once D far exceeds it, because
    beyond the effective rank the extra atoms fit a noise floor and are
    determined by the init rather than by S.

    Args:
        S: [C, C] co-activation matrix.
        D_list: ranks to test.
        seeds: list of NMF seeds (>= 2). Each gives one atom set per D.
        iters, restarts, normalize: passed to nmf_symmetric. NOTE restarts is
            forced to 1 here -- we want each SEED to give one deterministic
            solution, not a best-of-restarts, so the comparison is honest.
    Returns:
        list of dicts: {D, mean_dist, std_dist, mean_frac_unmatched}.
    """
    out = []
    for D in D_list:
        Ws = []
        for sd in seeds:
            W, e, _ = nmf_symmetric(
                S, D, iters=iters, restarts=1, normalize=normalize,
                seed=sd, verbose=False, device=device)
            Ws.append(W)
        # all unordered seed pairs
        dists, fracs = [], []
        for i in range(len(Ws)):
            for j in range(i + 1, len(Ws)):
                md, fu = matched_atom_distance(Ws[i], Ws[j])
                dists.append(md)
                fracs.append(fu)
        mean_d = float(np.mean(dists)) if dists else float("nan")
        std_d = float(np.std(dists)) if dists else float("nan")
        mean_f = float(np.mean(fracs)) if fracs else float("nan")
        print(f"  [stability D={D}] mean matched-atom dist = {mean_d:.4f} "
              f"+/- {std_d:.4f}   frac 'different atom' = {mean_f:.3f}  "
              f"({len(dists)} seed-pairs)")
        out.append(dict(D=D, mean_dist=mean_d, std_dist=std_d,
                        mean_frac_unmatched=mean_f))
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Signed-SVD vs NMF comparison on the co-activation "
                    "matrix S (memory-streaming).")
    ap.add_argument("--cache_dir", type=str, default="cache_activations")
    ap.add_argument("--cache_key", type=str, default=None,
                    help="Name of the cache subdirectory "
                         "(the activations_... folder).")
    ap.add_argument("--list", action="store_true",
                    help="List available caches and exit.")
    ap.add_argument("--S_path", type=str, default=None,
                    help="Optional: load a precomputed S (.npy/.pt) and skip "
                         "streaming entirely.")
    ap.add_argument("--D", type=int, nargs="+",
                    default=[10, 25, 50, 100, 200, 512],
                    help="Ranks D to test. Default spans the small-D regime "
                         "where the effective-rank elbow lives.")
    ap.add_argument("--max_chunks", type=int, default=None,
                    help="Stream at most this many chunks (subsample).")
    ap.add_argument("--subsample_rows", type=int, default=None,
                    help="Keep at most this many images per chunk.")
    ap.add_argument("--nmf_iters", type=int, default=4000)
    ap.add_argument("--nmf_restarts", type=int, default=3)
    ap.add_argument("--no_normalize", action="store_true",
                    help="Disable correlation-matrix normalization of S "
                         "before NMF.")
    ap.add_argument("--stability_seeds", type=int, nargs="+",
                    default=[0, 1, 2],
                    help="NMF seeds for the atom-stability sweep. Each seed "
                         "yields one atom set per D; pairwise matched-atom "
                         "distance measures seed-stability. Pass an empty "
                         "list / use --no_stability to skip.")
    ap.add_argument("--no_stability", action="store_true",
                    help="Skip the multi-seed atom-stability sweep.")
    ap.add_argument("--stability_iters", type=int, default=2000,
                    help="NMF iters for the stability sweep (each seed run "
                         "uses restarts=1). Lower than --nmf_iters since many "
                         "runs are needed; raise if traces look undertrained.")
    ap.add_argument("--percell_rank", action="store_true",
                    help="Also stream the PER-CELL channel covariance and "
                         "report its effective rank. Decides whether the "
                         "near-low-rank spectrum is a property of the LAYER "
                         "or only of the spatially-averaged anchor S.")
    ap.add_argument("--percell_subsample", type=int, default=20000,
                    help="Max spatial cells kept per chunk for the per-cell "
                         "covariance (memory control).")
    ap.add_argument("--out_prefix", type=str,
                    default="anchor_nonnegativity_check")
    ap.add_argument("--device", type=str, default="cuda",
                    help="Device for the factorization math: 'cuda', "
                         "'cuda:N', or 'cpu'. Falls back to cpu if cuda is "
                         "unavailable. (Streaming/IO always run on cpu.)")
    args = ap.parse_args()

    device = resolve_device(args.device)

    cache_dir = Path(args.cache_dir)

    if args.list:
        list_caches(cache_dir)
        return

    # ---- obtain S ------------------------------------------------------
    if args.S_path is not None:
        sp = Path(args.S_path)
        if sp.suffix == ".npy":
            S = np.load(sp).astype(np.float64)
        elif sp.suffix in (".pt", ".pth"):
            assert torch is not None, "torch needed to load a .pt S"
            S = torch.load(sp).detach().cpu().numpy().astype(np.float64)
        else:
            raise ValueError("S_path must be .npy or .pt")
        print(f"Loaded precomputed S from {sp}: shape {S.shape}")
        n_total = -1
    else:
        if joblib is None:
            sys.exit("joblib is required to read the cache; "
                     "install it or pass --S_path.")
        if args.cache_key is None:
            print("No --cache_key given. Available caches:")
            list_caches(cache_dir)
            sys.exit("\nRe-run with --cache_key <dirname>.")
        cache_path = cache_dir / args.cache_key
        S, n_total = build_S_streaming(
            cache_path, max_chunks=args.max_chunks,
            subsample_rows=args.subsample_rows)
        np.save(f"{args.out_prefix}_S.npy", S)
        print(f"Saved S to {args.out_prefix}_S.npy "
              f"(reuse with --S_path to skip streaming).")

    C = S.shape[0]
    # symmetrize defensively
    S = 0.5 * (S + S.T)

    # ---- spectrum summary + effective rank ----------------------------
    evals = spectrum_of(S, device=device)
    pos = evals[evals > 0]
    erm = effective_rank_metrics(evals)
    print(f"\nS spectrum (spatially-averaged co-activation): C={C}")
    print(f"  rank(S) (eigs > 1e-9 * max): "
          f"{int((evals > 1e-9 * evals[0]).sum())}")
    print(f"  top 5 eigenvalues:    {np.round(evals[:5], 5)}")
    print(f"  # negative eigenvalues: {int((evals < 0).sum())} "
          f"(should be ~0 for a PSD S)")
    print(f"  EFFECTIVE RANK:")
    print(f"    participation ratio (sum l)^2/sum l^2 : "
          f"{erm['participation_ratio']:.2f}")
    print(f"    entropy rank exp(H)                   : "
          f"{erm['entropy_rank']:.2f}")
    print(f"    # eigenvalues > 1%  of max            : {erm['n_above_1pct']}")
    print(f"    # eigenvalues > 0.1% of max           : "
          f"{erm['n_above_0p1pct']}")
    print(f"    eigenvalues to reach 90/95/99% energy : "
          f"{erm['n_90']} / {erm['n_95']} / {erm['n_99']}")
    if len(pos) > 0:
        cume = np.cumsum(pos) / pos.sum()
        for k in (10, 50, 100, 256, 512, 1024):
            if k <= len(pos):
                print(f"  top-{k} eigenvalues hold "
                      f"{100 * cume[k - 1]:.1f}% of positive energy")

    # ---- per-cell rank check (optional) -------------------------------
    # Decides whether the near-low-rank spectrum is a property of the LAYER
    # or only an artifact of spatial averaging in the anchor's S.
    percell_evals = None
    percell_erm = None
    if args.percell_rank:
        if args.S_path is not None and args.cache_key is None:
            print("\n[per-cell] WARNING: --percell_rank needs the activation "
                  "cache (--cache_key), not just --S_path. Skipping.")
        elif joblib is None:
            print("\n[per-cell] joblib unavailable; skipping per-cell check.")
        else:
            print(f"\n{'=' * 60}\nPER-CELL RANK CHECK\n{'=' * 60}")
            cache_path = cache_dir / args.cache_key
            S_cell, n_cells = build_S_percell_streaming(
                cache_path, max_chunks=args.max_chunks,
                subsample_cells=args.percell_subsample)
            S_cell = 0.5 * (S_cell + S_cell.T)
            np.save(f"{args.out_prefix}_S_percell.npy", S_cell)
            percell_evals = spectrum_of(S_cell, device=device)
            percell_erm = effective_rank_metrics(percell_evals)
            print(f"  PER-CELL effective rank:")
            print(f"    participation ratio : "
                  f"{percell_erm['participation_ratio']:.2f}")
            print(f"    entropy rank        : "
                  f"{percell_erm['entropy_rank']:.2f}")
            print(f"    # eigs > 1% of max  : {percell_erm['n_above_1pct']}")
            print(f"    eigs to 90/95/99%   : {percell_erm['n_90']} / "
                  f"{percell_erm['n_95']} / {percell_erm['n_99']}")
            pr_avg = erm['participation_ratio']
            pr_cell = percell_erm['participation_ratio']
            if pr_cell > 3 * pr_avg:
                print(f"  --> PER-CELL rank ({pr_cell:.0f}) is much higher "
                      f"than averaged-S rank ({pr_avg:.0f}). The low rank is "
                      f"an ARTIFACT of spatial averaging in the anchor's S, "
                      f"not a property of the layer. The 'layer supports "
                      f"only ~R concepts' claim is NOT supported -- this is a "
                      f"finding about build_anchor's construction instead.")
            else:
                print(f"  --> PER-CELL rank ({pr_cell:.0f}) is comparable to "
                      f"averaged-S rank ({pr_avg:.0f}). The near-low-rank "
                      f"structure is a property of the LAYER itself; the "
                      f"'limited number of stable concepts' conclusion is "
                      f"safe to state for the layer.")

    # ---- factorize at each D ------------------------------------------
    results = []
    for D in args.D:
        print(f"\n{'=' * 60}\nD = {D}\n{'=' * 60}")

        S_eig, e_eig = signed_eig_lowrank(S, D, device=device)
        print(f"  signed eig  (best rank-D, ANY sign): rel_err = {e_eig:.5f}")

        S_ggt, e_ggt, dprime = signed_GGt_psd(S, D, device=device)
        print(f"  signed GG^T (PSD, D'={dprime} pos-eig dirs used): "
              f"rel_err = {e_ggt:.5f}")

        _, e_nmf, trace = nmf_symmetric(
            S, D, iters=args.nmf_iters, restarts=args.nmf_restarts,
            normalize=not args.no_normalize, device=device)
        print(f"  NMF  W W^T  (W >= 0): rel_err = {e_nmf:.5f}")

        gap = e_nmf - e_ggt
        print(f"  --> non-negativity gap (NMF - signed GG^T) = {gap:.5f}")
        results.append(dict(D=D, e_eig=e_eig, e_ggt=e_ggt, e_nmf=e_nmf,
                            gap=gap, trace=trace))

    # ---- verdict -------------------------------------------------------
    print(f"\n{'=' * 60}\nVERDICT\n{'=' * 60}")
    for r in results:
        D = r["D"]
        if r["e_ggt"] < 0.05 and r["gap"] > 0.20:
            msg = ("LARGE non-negativity gap. Signed fit is ~exact but NMF "
                   "is not -> S has no good non-negative factorization. "
                   "The non-negative-anchor / soft-membership premise does "
                   "NOT hold here. This is a real negative result, not a "
                   "solver bug.")
        elif r["gap"] < 0.05:
            msg = ("Gap ~ 0. Non-negativity costs little. If build_anchor "
                   "still fails, it is a SOLVER problem -> more iters / "
                   "normalize S / better init.")
        else:
            msg = ("Intermediate gap. Non-negativity has a real but partial "
                   "cost. Inspect the NMF convergence trace before "
                   "concluding.")
        print(f"  D={D}: signed GG^T err={r['e_ggt']:.4f}, "
              f"NMF err={r['e_nmf']:.4f}, gap={r['gap']:.4f}\n      {msg}")

    # ---- atom-stability sweep -----------------------------------------
    stability = []
    if not args.no_stability and len(args.stability_seeds) >= 2:
        print(f"\n{'=' * 60}\nATOM-STABILITY SWEEP "
              f"(seeds={args.stability_seeds})\n{'=' * 60}")
        print("Running NMF from each seed at each D; comparing atom sets "
              "by Hungarian-matched distance.")
        stability = atom_stability_sweep(
            S, args.D, seeds=args.stability_seeds,
            iters=args.stability_iters, restarts=1,
            normalize=not args.no_normalize, device=device)
        # interpret: where does it stop being stable?
        stable_Ds = [s["D"] for s in stability if s["mean_dist"] < 0.10]
        if stable_Ds:
            print(f"\n  Atoms are seed-stable (mean dist < 0.10) up to "
                  f"D <= {max(stable_Ds)}.")
        else:
            print(f"\n  No tested D is seed-stable below the 0.10 threshold.")
        unstable_Ds = [s["D"] for s in stability if s["mean_dist"] >= 0.10]
        if stable_Ds and unstable_Ds:
            print(f"  Instability sets in by D >= {min(unstable_Ds)}. "
                  f"If this elbow matches the effective rank "
                  f"(~{erm['n_above_1pct']}), the instability is explained: "
                  f"atoms beyond the effective rank fit a noise floor and "
                  f"are determined by the init, not by S.")

    # ---- plot ----------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Anchor analysis: rank, non-negativity, and atom-stability",
                 fontsize=13, fontweight="bold")

    # (0,0) error vs D, grouped bars
    Ds = [r["D"] for r in results]
    x = np.arange(len(Ds))
    w = 0.27
    axs[0, 0].bar(x - w, [r["e_eig"] for r in results], w,
                  label="signed eig (any sign)", color="seagreen")
    axs[0, 0].bar(x, [r["e_ggt"] for r in results], w,
                  label="signed GG^T (PSD)", color="steelblue")
    axs[0, 0].bar(x + w, [r["e_nmf"] for r in results], w,
                  label="NMF W W^T (W>=0)", color="crimson")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels([f"{d}" for d in Ds])
    axs[0, 0].set_xlabel("D")
    axs[0, 0].set_ylabel("rel. reconstruction error")
    axs[0, 0].set_title("Reconstruction error by method and D")
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(True, axis="y", alpha=0.3)

    # (0,1) non-negativity gap
    axs[0, 1].bar(x, [r["gap"] for r in results], 0.5, color="darkorange")
    axs[0, 1].axhline(0.20, color="red", ls="--", lw=0.8,
                      label="0.20: premise-broken")
    axs[0, 1].axhline(0.05, color="green", ls="--", lw=0.8,
                      label="0.05: solver-problem")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels([f"{d}" for d in Ds])
    axs[0, 1].set_xlabel("D")
    axs[0, 1].set_title("Non-negativity gap (NMF - signed GG^T)")
    axs[0, 1].set_ylabel("gap")
    axs[0, 1].legend(fontsize=8)
    axs[0, 1].grid(True, axis="y", alpha=0.3)

    # (0,2) NMF convergence traces
    for r in results:
        if r["trace"]:
            its = [t[0] for t in r["trace"]]
            es = [t[1] for t in r["trace"]]
            axs[0, 2].plot(its, es, lw=1.5, label=f"D={r['D']}")
    axs[0, 2].set_xlabel("NMF iteration")
    axs[0, 2].set_ylabel("rel_err (normalized S)")
    axs[0, 2].set_title("NMF convergence trace")
    axs[0, 2].legend(fontsize=8)
    axs[0, 2].grid(True, alpha=0.3)

    # (1,0) SCREE PLOT -- log eigenvalue spectrum, with effective rank marked
    rank_idx = np.arange(1, len(evals) + 1)
    axs[1, 0].semilogy(rank_idx, np.clip(evals, 1e-12, None),
                       color="navy", lw=1.5, label="S (averaged)")
    pr = erm["participation_ratio"]
    axs[1, 0].axvline(pr, color="crimson", ls="--", lw=1.0,
                      label=f"participation rank ~{pr:.0f}")
    axs[1, 0].axvline(erm["n_above_1pct"], color="darkorange", ls=":",
                      lw=1.0, label=f"# eigs>1% = {erm['n_above_1pct']}")
    if percell_evals is not None:
        pc_idx = np.arange(1, len(percell_evals) + 1)
        axs[1, 0].semilogy(pc_idx, np.clip(percell_evals, 1e-12, None),
                           color="teal", lw=1.5, alpha=0.8,
                           label="S (per-cell)")
    axs[1, 0].set_xlabel("eigenvalue index")
    axs[1, 0].set_ylabel("eigenvalue (log)")
    axs[1, 0].set_title("Scree plot -- spectrum & effective rank")
    axs[1, 0].legend(fontsize=8)
    axs[1, 0].grid(True, which="both", alpha=0.3)

    # (1,1) cumulative energy
    if len(pos) > 0:
        cume = np.cumsum(pos) / pos.sum()
        axs[1, 1].plot(np.arange(1, len(cume) + 1), cume,
                       color="navy", lw=1.5)
        for frac, c in ((0.90, "green"), (0.95, "orange"), (0.99, "red")):
            axs[1, 1].axhline(frac, color=c, ls="--", lw=0.7)
        axs[1, 1].set_xlabel("# eigenvalues")
        axs[1, 1].set_ylabel("cumulative energy fraction")
        axs[1, 1].set_title("Cumulative spectral energy")
        axs[1, 1].set_xscale("log")
        axs[1, 1].grid(True, which="both", alpha=0.3)

    # (1,2) ATOM-STABILITY vs D
    if stability:
        sD = [s["D"] for s in stability]
        sM = [s["mean_dist"] for s in stability]
        sS = [s["std_dist"] for s in stability]
        axs[1, 2].errorbar(sD, sM, yerr=sS, marker="o", lw=1.5,
                           color="darkviolet", capsize=3,
                           label="mean matched-atom dist")
        axs[1, 2].axhline(0.10, color="green", ls="--", lw=0.8,
                          label="0.10: 'stable' threshold")
        axs[1, 2].axvline(pr, color="crimson", ls=":", lw=1.0,
                          label=f"effective rank ~{pr:.0f}")
        axs[1, 2].set_xlabel("D")
        axs[1, 2].set_ylabel("mean matched-atom distance (across seeds)")
        axs[1, 2].set_title("Atom seed-stability vs D")
        axs[1, 2].set_xscale("log")
        axs[1, 2].legend(fontsize=8)
        axs[1, 2].grid(True, which="both", alpha=0.3)
    else:
        axs[1, 2].text(0.5, 0.5, "atom-stability sweep skipped\n"
                       "(--no_stability or <2 seeds)",
                       ha="center", va="center", fontsize=10)
        axs[1, 2].set_title("Atom seed-stability vs D")

    plt.tight_layout()
    out_png = f"{args.out_prefix}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_png}")

    # ---- numeric tables -----------------------------------------------
    out_txt = f"{args.out_prefix}.txt"
    with open(out_txt, "w") as f:
        f.write("# effective rank of S (spatially-averaged co-activation)\n")
        for k, v in erm.items():
            f.write(f"#   {k}\t{v}\n")
        if percell_erm is not None:
            f.write("# effective rank of S_percell\n")
            for k, v in percell_erm.items():
                f.write(f"#   {k}\t{v}\n")
        f.write("\nD\tsigned_eig\tsigned_GGt\tNMF\tgap\n")
        for r in results:
            f.write(f"{r['D']}\t{r['e_eig']:.6f}\t{r['e_ggt']:.6f}\t"
                    f"{r['e_nmf']:.6f}\t{r['gap']:.6f}\n")
        if stability:
            f.write("\nD\tmean_atom_dist\tstd_atom_dist\tfrac_diff_atom\n")
            for s in stability:
                f.write(f"{s['D']}\t{s['mean_dist']:.6f}\t"
                        f"{s['std_dist']:.6f}\t"
                        f"{s['mean_frac_unmatched']:.6f}\n")
    print(f"Numeric results saved to {out_txt}")


if __name__ == "__main__":
    main()