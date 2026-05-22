"""
check_anchor_nonnegativity.py
=============================
Decide whether "NMF anchor is bad" means (a) the solver is bad, or
(b) non-negativity is the wrong prior for this layer.

The test
--------
S is the C x C channel co-activation matrix (PSD, rank <= C). We factorize it
at rank D in three ways and compare relative reconstruction error
||S - S_hat||_F / ||S||_F:

  1. SIGNED truncated eigendecomposition  S ~ U L U^T  (rank D).
     This is the BEST possible symmetric rank-D fit. For D >= C it is
     ESSENTIALLY ZERO by construction. It is the floor.

  2. SIGNED low-rank  S ~ G G^T  with G in R^{C x D}, G unconstrained.
     Same error as (1) -- included only as a sanity check / explicit
     "G G^T" form matching the anchor's W W^T shape.

  3. NMF  S ~ W W^T  with W >= 0  (the anchor: build_anchor's factorization).
     Run to convergence (many iters, normalized S, multi-restart).

Interpretation
--------------
  gap := err_NMF - err_signed   at a given D.
  * gap ~ 0            -> non-negativity costs nothing; if your build_anchor
                          still fails, it's a SOLVER problem (iters/init/norm).
  * gap large (>> 0)   -> S has no good NON-NEGATIVE factorization. Since
                          err_signed is ~0 at D >= C, a large NMF error is
                          ENTIRELY the cost of the W >= 0 constraint. The
                          non-negative-anchor / soft-membership premise does
                          not hold for this layer.

Memory
------
Activations are NEVER fully loaded. Cache chunks are streamed one at a time:
each [n, C, H, W] chunk is spatially averaged to [n, C] and accumulated into
a [C, C] matrix, then freed. Peak extra RAM is one chunk (~tens of MB).
Optionally subsample chunks with --max_chunks for an even lighter pass.

Usage
-----
    python check_anchor_nonnegativity.py \\
        --cache_dir cache_activations \\
        --cache_key <the activations_... dir name> \\
        --D 1000 4000

If you don't know the cache key, run with --list to see available caches.
You can also point --S_path at a precomputed S .pt/.npy to skip streaming.
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


# ----------------------------------------------------------------------
# Factorizations
# ----------------------------------------------------------------------

def rel_err(S, S_hat):
    return float(np.linalg.norm(S - S_hat) / (np.linalg.norm(S) + 1e-12))


def signed_eig_lowrank(S, D):
    """Best symmetric rank-D fit: keep the D eigenpairs of largest |lambda|.

    Returns S_hat and relative error. For D >= rank(S) this is ~0.
    """
    # S is symmetric; eigh gives ascending eigenvalues.
    w, V = np.linalg.eigh(S)
    order = np.argsort(np.abs(w))[::-1]             # by magnitude, descending
    keep = order[:min(D, len(w))]
    S_hat = (V[:, keep] * w[keep]) @ V[:, keep].T
    return S_hat, rel_err(S, S_hat)


def signed_GGt_psd(S, D):
    """Signed G G^T fit, G in R^{C x D}. For a PSD S this equals the
    truncated eigendecomposition restricted to POSITIVE eigenvalues
    (G G^T is itself PSD, so it cannot use negative-eigenvalue directions).

    This is the fair signed analogue of NMF's W W^T (both yield PSD S_hat).
    For a genuinely PSD S with rank <= C this is still ~0 at D >= rank.
    """
    w, V = np.linalg.eigh(S)
    pos = w > 0
    w_pos, V_pos = w[pos], V[:, pos]
    order = np.argsort(w_pos)[::-1]
    keep = order[:min(D, len(w_pos))]
    g = V_pos[:, keep] * np.sqrt(w_pos[keep])       # [C, D']  -> G
    S_hat = g @ g.T
    return S_hat, rel_err(S, S_hat), g.shape[1]


def nmf_symmetric(S, D, iters=4000, restarts=3, normalize=True, seed=0,
                  tol=1e-7, verbose=True):
    """Symmetric NMF  S ~ W W^T,  W >= 0,  via multiplicative updates.

    A stronger version of build_anchor's NMF: more iters, optional S
    normalization to a correlation matrix, multiple random restarts, and
    an actual convergence trace (build_anchor logs only the final iter).

    Returns the best (lowest-error) W, its relative error against the
    ORIGINAL S, and the per-iteration error trace of the best restart.
    """
    S_work = S.copy()
    S_work[S_work < 0] = 0.0                        # S is PSD; kill tiny negs

    scale_back = None
    if normalize:
        d = np.sqrt(np.clip(np.diag(S_work), 1e-12, None))
        Dinv = 1.0 / d
        S_work = S_work * np.outer(Dinv, Dinv)      # correlation matrix
        scale_back = d                              # to undo for reporting

    C = S_work.shape[0]
    best_W, best_err, best_trace = None, np.inf, None

    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        W = rng.random((C, D)).astype(np.float64)
        # scale W so W W^T starts near S_work's magnitude
        W *= np.sqrt(np.linalg.norm(S_work) / (np.linalg.norm(W @ W.T) + 1e-12))
        trace = []
        prev = np.inf
        for it in range(iters):
            SW = S_work @ W                         # [C, D]
            WWtW = W @ (W.T @ W)                    # [C, D]
            W *= SW / (WWtW + 1e-9)
            W = np.clip(W, 1e-9, None)
            if (it + 1) % 50 == 0 or it == iters - 1:
                e = rel_err(S_work, W @ W.T)
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
        e_norm = rel_err(S_work, W @ W.T)
        if scale_back is not None:
            W_orig = W * scale_back[:, None]        # undo the correlation norm
            e_orig = rel_err(S, W_orig @ W_orig.T)
        else:
            W_orig, e_orig = W, e_norm
        if e_orig < best_err:
            best_W, best_err, best_trace = W_orig, e_orig, trace

    return best_W, best_err, best_trace


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
    ap.add_argument("--D", type=int, nargs="+", default=[1000, 4000],
                    help="Ranks D to test.")
    ap.add_argument("--max_chunks", type=int, default=None,
                    help="Stream at most this many chunks (subsample).")
    ap.add_argument("--subsample_rows", type=int, default=None,
                    help="Keep at most this many images per chunk.")
    ap.add_argument("--nmf_iters", type=int, default=4000)
    ap.add_argument("--nmf_restarts", type=int, default=3)
    ap.add_argument("--no_normalize", action="store_true",
                    help="Disable correlation-matrix normalization of S "
                         "before NMF.")
    ap.add_argument("--out_prefix", type=str,
                    default="anchor_nonnegativity_check")
    args = ap.parse_args()

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

    # ---- spectrum summary ---------------------------------------------
    evals = np.linalg.eigvalsh(S)[::-1]             # descending
    pos = evals[evals > 0]
    print(f"\nS spectrum: C={C}")
    print(f"  rank(S) (eigs > 1e-9 * max): "
          f"{int((evals > 1e-9 * evals[0]).sum())}")
    print(f"  top 5 eigenvalues:    {np.round(evals[:5], 5)}")
    print(f"  # negative eigenvalues: {int((evals < 0).sum())} "
          f"(should be ~0 for a PSD S)")
    if len(pos) > 0:
        # how much energy in the top-k
        cume = np.cumsum(pos) / pos.sum()
        for k in (10, 50, 100, 256, 512, 1024):
            if k <= len(pos):
                print(f"  top-{k} eigenvalues hold "
                      f"{100 * cume[k - 1]:.1f}% of positive energy")

    # ---- factorize at each D ------------------------------------------
    results = []
    for D in args.D:
        print(f"\n{'=' * 60}\nD = {D}\n{'=' * 60}")

        S_eig, e_eig = signed_eig_lowrank(S, D)
        print(f"  signed eig  (best rank-D, ANY sign): rel_err = {e_eig:.5f}")

        S_ggt, e_ggt, dprime = signed_GGt_psd(S, D)
        print(f"  signed GG^T (PSD, D'={dprime} pos-eig dirs used): "
              f"rel_err = {e_ggt:.5f}")

        _, e_nmf, trace = nmf_symmetric(
            S, D, iters=args.nmf_iters, restarts=args.nmf_restarts,
            normalize=not args.no_normalize)
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

    # ---- plot ----------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Anchor non-negativity check: signed low-rank vs NMF on S",
                 fontsize=13, fontweight="bold")

    # (a) error vs D, bar chart
    Ds = [r["D"] for r in results]
    x = np.arange(len(Ds))
    w = 0.27
    axs[0].bar(x - w, [r["e_eig"] for r in results], w,
               label="signed eig (any sign)", color="seagreen")
    axs[0].bar(x, [r["e_ggt"] for r in results], w,
               label="signed GG^T (PSD)", color="steelblue")
    axs[0].bar(x + w, [r["e_nmf"] for r in results], w,
               label="NMF W W^T (W>=0)", color="crimson")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([f"D={d}" for d in Ds])
    axs[0].set_ylabel("relative reconstruction error  ||S - S_hat||_F / ||S||_F")
    axs[0].set_title("Reconstruction error by method and D")
    axs[0].legend(fontsize=8)
    axs[0].grid(True, axis="y", alpha=0.3)

    # (b) the non-negativity gap
    axs[1].bar(x, [r["gap"] for r in results], 0.5, color="darkorange")
    axs[1].axhline(0.20, color="red", ls="--", lw=0.8,
                   label="0.20: 'premise broken' threshold")
    axs[1].axhline(0.05, color="green", ls="--", lw=0.8,
                   label="0.05: 'solver problem' threshold")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([f"D={d}" for d in Ds])
    axs[1].set_title("Non-negativity gap  (NMF err - signed GG^T err)")
    axs[1].set_ylabel("gap")
    axs[1].legend(fontsize=8)
    axs[1].grid(True, axis="y", alpha=0.3)

    # (c) NMF convergence traces
    for r in results:
        if r["trace"]:
            its = [t[0] for t in r["trace"]]
            es = [t[1] for t in r["trace"]]
            axs[2].plot(its, es, lw=1.5, label=f"D={r['D']}")
    axs[2].set_xlabel("NMF iteration")
    axs[2].set_ylabel("rel_err (normalized S)")
    axs[2].set_title("NMF convergence trace (is it still descending?)")
    axs[2].legend(fontsize=8)
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = f"{args.out_prefix}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_png}")

    # ---- numeric table -------------------------------------------------
    out_txt = f"{args.out_prefix}.txt"
    with open(out_txt, "w") as f:
        f.write("D\tsigned_eig\tsigned_GGt\tNMF\tgap\n")
        for r in results:
            f.write(f"{r['D']}\t{r['e_eig']:.6f}\t{r['e_ggt']:.6f}\t"
                    f"{r['e_nmf']:.6f}\t{r['gap']:.6f}\n")
    print(f"Numeric results saved to {out_txt}")


if __name__ == "__main__":
    main()