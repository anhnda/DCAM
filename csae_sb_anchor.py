"""
csae_sb_anchor.py  (GPU-accelerated)
====================================
Build a seed-free, SIGNED anchor for CSAE's encoder from the BETWEEN-CLASS
scatter S_B of the cached layer activations, via eigendecomposition only.

Why this exists (the theory in one paragraph)
---------------------------------------------
An overcomplete CSAE over rank-r data has the symmetry W -> W A^-1, Z -> A Z,
so the per-cell code Z is NON-identifiable: seeds land at different points on
the solution manifold (feature absorption/splitting -> seed-unstable atoms).
A penalty on the *code* cannot fix this -- it lives inside the symmetry orbit.
A FIXED, label-derived, channel-space anchor on the *encoder weights* DOES
break the symmetry (the term ||W_enc - W0||/cos is not invariant under A).
csae_svd_anchor.py used the TOTAL covariance (PCA) for W0 -- label-blind, and
(per its own diagnostics) a flat continuum with no part-like directions. This
module swaps that single matrix: W0 = top-D signed eigenbasis of S_B, the
between-class scatter. Same machinery, same determinism guarantees, same
[D, C] drop-in for csae_stable.py's --ica_anchor flag. The only new thing is
per-class accumulation in the streaming loop.

What S_B buys you
-----------------
  * Symmetry break toward LABEL-ALIGNED directions (stable like LDA
    directions are stable, because labels are a fixed external signal).
  * Shared vs class-specific split FOR FREE from the eigenvalue ordering:
    large-lambda rows separate many classes (shared sub-patterns), tail rows
    separate one-vs-rest (class-specific).
  * rank(S_B) <= (#classes - 1), so anchoring at most ~999 atoms is principled
    -- you only claim stability for the anchored sub-dictionary, not all D.

What S_B does NOT buy you (state this in the paper)
---------------------------------------------------
  * The atoms are DISCRIMINATIVE ("directions that separate classes"), not
    proven GENERATIVE part detectors. An S_B atom peaking on elephants is
    "the elephant-vs-rest direction", which MAY or may not be a leg detector.
  * If the gate ratio tr(S_B)/tr(S_T) is tiny, the anchored atoms are stable
    but explain little of what CSAE reconstructs (texture/pose dominate at
    mid-level conv layers). Read the gate BEFORE training.

THE GATE (read this number first)
---------------------------------
This script prints  tr(S_B)/tr(S_T)  = fraction of activation variance that
is between-class.
  >~ 0.20  : label-atoms are stable AND substantive -> proceed.
  ~ 0.03   : stable but thin -> you'll get a clean negative/footnote, not a
             constructive parts result. Decide accordingly.

Granularity (--pool)
--------------------
  --pool none   (DEFAULT): cell-level. Every (h,w) cell is a sample carrying
        its image's label. S_T here is EXACTLY csae_svd_anchor.py's `cov`, so
        the gate compares like-with-like against what CSAE actually decodes.
        Conflates "class uses these channels" with "class has more foreground
        cells", but matches CSAE granularity. The honest default.
  --pool image : mean-pool each image to one R^C vector first, then S_B over
        image-means. Cleaner class separation (drops within-image cell
        variance) but the gate is then an image-level ratio, not directly
        comparable to the per-cell S_T CSAE sees. Try both; report --pool none
        as primary.

Determinism
-----------
S_B is a fixed function of (cache cells, labels, subsample draw). The
eigendecomposition is deterministic given S_B; signs are pinned (largest-
magnitude row entry forced positive). The cross-seed verifier should report
at the GPU float-noise floor.

Usage
-----
    python csae_sb_anchor.py \\
        --cache_dir cache_activations \\
        --cache_key resnet50_layer3_thresh0p95_samples50000_chunk100_gcmap1 \\
        --num_classes 1000 --D 256 --pool none \\
        --verify_seeds 0 1 2 --device cuda \\
        --save csae_sb_anchor_W0.npy

The --cache_key is the directory name under --cache_dir (the gcmap1 one), the
same string you pass to csae_svd_anchor.py.
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
# Device / dtype helpers (verbatim from csae_svd_anchor.py for parity)
# ----------------------------------------------------------------------

def resolve_device(device: str):
    if torch is None:
        return None
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[sb anchor] CUDA requested but not available; using CPU.")
        device = "cpu"
    return torch.device(device)


def torch_dtype(dtype: str):
    return torch.float64 if dtype == "float64" else torch.float32


def _chunk_to_cells_and_labels(part, pool="none"):
    """Turn a cached chunk into (cells [N, C], labels [N]) numpy arrays.

    pool='none'  -> per-cell: N = n_images * H * W, label repeated H*W times.
    pool='image' -> per-image mean: N = n_images, label as-is.
    """
    act = part["activation"]
    lbl = part["label"]
    if torch is not None and isinstance(act, torch.Tensor):
        act = act.detach().cpu().numpy()
    else:
        act = np.asarray(act)
    if torch is not None and isinstance(lbl, torch.Tensor):
        lbl = lbl.detach().cpu().numpy()
    else:
        lbl = np.asarray(lbl)
    lbl = lbl.astype(np.int64).reshape(-1)               # [n_images]

    if act.ndim == 4:
        n, C, H, W = act.shape
        if pool == "image":
            cells = act.mean(axis=(2, 3))                # [n, C]
            labels = lbl                                 # [n]
        else:
            cells = act.transpose(0, 2, 3, 1).reshape(-1, C)   # [n*H*W, C]
            labels = np.repeat(lbl, H * W)               # [n*H*W]
    elif act.ndim == 2:
        cells = act
        labels = lbl
    else:
        raise ValueError(f"unexpected ndim {act.ndim}")
    return cells, labels


# ----------------------------------------------------------------------
# Streamed class statistics: global cov + per-class means in ONE pass
# ----------------------------------------------------------------------

def build_class_scatter(cache_path: Path, device, dtype, num_classes=1000,
                        pool="none", max_chunks=None,
                        subsample_cells=200000, seed=0):
    """Stream the cache once; return everything needed for S_B and the gate.

    Returns a dict with (all numpy float64):
        mean   [C]        global mean
        S_T    [C, C]     total scatter (== csae_svd_anchor's `cov` for pool=none)
        S_B    [C, C]     between-class scatter  Sum_c pi_c (mu_c-mu)(mu_c-mu)^T
        tr_ST, tr_SB      scalars
        gate              tr_SB / tr_ST  <-- THE NUMBER
        n_samples         total cells (or images if pool=image)
        present_classes   how many classes had >0 samples
    """
    parts = sorted(cache_path.glob("part_*.pkl"))
    if not parts:
        raise FileNotFoundError(f"No part_*.pkl in {cache_path}")
    if max_chunks is not None:
        parts = parts[:max_chunks]
    rng = np.random.default_rng(seed)

    n = 0
    sum_x = None          # [C]
    sum_xxT = None        # [C, C]
    sum_sq = torch.zeros((), device=device, dtype=dtype)   # Sum ||x||^2
    cls_sum = None        # [K, C]
    cls_cnt = None        # [K]
    K = num_classes

    print(f"Streaming {len(parts)} chunk(s) for class scatter on {device} "
          f"({dtype}), pool={pool}...")
    for i, p in enumerate(parts):
        part = joblib.load(p)
        cells, labels = _chunk_to_cells_and_labels(part, pool=pool)
        if pool == "none" and subsample_cells and cells.shape[0] > subsample_cells:
            keep = rng.choice(cells.shape[0], subsample_cells, replace=False)
            cells = cells[keep]
            labels = labels[keep]

        xt = torch.as_tensor(np.ascontiguousarray(cells)).to(device=device,
                                                             dtype=dtype)
        lt = torch.as_tensor(np.ascontiguousarray(labels)).to(device=device).long()

        if sum_x is None:
            C = xt.shape[1]
            sum_x = torch.zeros(C, device=device, dtype=dtype)
            sum_xxT = torch.zeros((C, C), device=device, dtype=dtype)
            cls_sum = torch.zeros((K, C), device=device, dtype=dtype)
            cls_cnt = torch.zeros(K, device=device, dtype=dtype)

        if lt.max().item() >= K:
            raise ValueError(f"label {lt.max().item()} >= num_classes {K}; "
                             f"raise --num_classes.")

        sum_x += xt.sum(dim=0)
        sum_xxT += xt.T @ xt
        sum_sq += (xt * xt).sum()
        cls_sum.index_add_(0, lt, xt)
        cls_cnt.index_add_(0, lt, torch.ones(xt.shape[0], device=device,
                                             dtype=dtype))
        n += xt.shape[0]
        del part, cells, labels, xt, lt
        if (i + 1) % 20 == 0 or (i + 1) == len(parts):
            print(f"  ...{i+1}/{len(parts)} chunks, {n} samples")

    inv_n = 1.0 / max(n, 1)
    mean_t = sum_x * inv_n
    S_T = sum_xxT * inv_n - torch.outer(mean_t, mean_t)
    S_T = 0.5 * (S_T + S_T.T)
    tr_ST = (sum_sq * inv_n - mean_t @ mean_t)

    present = cls_cnt > 0
    n_present = int(present.sum().item())
    mu_c = cls_sum[present] / cls_cnt[present].unsqueeze(1)     # [Kp, C]
    pi_c = cls_cnt[present] * inv_n                             # [Kp]
    diff = mu_c - mean_t.unsqueeze(0)                           # [Kp, C]
    S_B = (pi_c.unsqueeze(1) * diff).T @ diff                   # [C, C]
    S_B = 0.5 * (S_B + S_B.T)
    tr_SB = (pi_c * (diff * diff).sum(dim=1)).sum()

    gate = (tr_SB / tr_ST.clamp_min(1e-12)).item()

    out = {
        'mean': mean_t.double().cpu().numpy(),
        'S_T': S_T.double().cpu().numpy(),
        'S_B': S_B.double().cpu().numpy(),
        'tr_ST': float(tr_ST.item()),
        'tr_SB': float(tr_SB.item()),
        'gate': gate,
        'n_samples': n,
        'present_classes': n_present,
        'pool': pool,
    }

    print(f"\nClass scatter built: C={S_T.shape[0]}, {n} samples, "
          f"{n_present} classes present.")
    print(f"{'='*60}")
    print(f"  GATE  tr(S_B)/tr(S_T) = {gate:.4f}   "
          f"({100*gate:.1f}% of variance is between-class)")
    if gate >= 0.20:
        print("  --> SUBSTANTIVE: label-atoms are stable AND carry real "
              "signal. Proceed.")
    elif gate >= 0.08:
        print("  --> MODERATE: usable, but expect the anchored atoms to "
              "explain a minority of reconstruction. Report honestly.")
    else:
        print("  --> THIN: between-class variance is small. Anchored atoms "
              "will be stable but explain little of what CSAE reconstructs "
              "(this layer is mostly texture/pose). This points to a clean "
              "NEGATIVE result, not a constructive parts paper.")
    print(f"{'='*60}")
    return out


# ----------------------------------------------------------------------
# Signed top-D eigenbasis of S_B (the anchor)
# ----------------------------------------------------------------------

def signed_eigenbasis(M, D, device, dtype):
    """Top-D signed orthonormal eigenvectors of symmetric M.
    Returns Vk [C, D] (numpy f64) and eigvals [D] (descending, numpy f64).
    """
    M_t = torch.as_tensor(M, device=device, dtype=dtype)
    M_t = 0.5 * (M_t + M_t.T)
    w, V = torch.linalg.eigh(M_t)                # ascending
    w = w.flip(0)                                 # descending
    V = V.flip(1)
    eigvals = w[:D].clamp_min(0.0)
    Vk = V[:, :D]
    return Vk.double().cpu().numpy(), eigvals.double().cpu().numpy()


def build_sb_anchor(cache_path: Path, D=256, seed=0, num_classes=1000,
                    pool="none", max_chunks=None, subsample_cells=200000,
                    device="auto", dtype="float64",
                    return_diagnostics=False, _reuse_stats=None):
    """Build the signed S_B anchor W0 in R^{D x C}.

    Pipeline: between-class scatter S_B -> top-D signed eigenvectors ->
    transpose to [D, C] -> unit-normalize rows -> deterministic sign fix.

    D is capped at the usable rank of S_B (#eigenvalues above 1% of the max,
    and at most present_classes-1). A warning fires if you asked for more.
    """
    dev = resolve_device(device)
    dt = torch_dtype(dtype)

    if _reuse_stats is not None:
        stats = _reuse_stats
    else:
        stats = build_class_scatter(
            cache_path, dev, dt, num_classes=num_classes, pool=pool,
            max_chunks=max_chunks, subsample_cells=subsample_cells, seed=seed)

    S_B = stats['S_B']
    C = S_B.shape[0]

    # usable rank guidance
    _, all_eig = signed_eigenbasis(S_B, C, dev, dt)
    emax = max(all_eig.max(), 1e-12)
    usable = int((all_eig > 0.01 * emax).sum())
    rank_cap = min(stats['present_classes'] - 1, C)
    D_req = D
    D = min(D, rank_cap)
    if D_req > usable:
        print(f"[sb anchor] WARNING: requested D={D_req} but only ~{usable} "
              f"eigenvalues exceed 1% of the max (S_B rank cap {rank_cap}). "
              f"Anchoring to near-zero eigenvectors reintroduces instability. "
              f"Consider --D {usable}.")
    print(f"[sb anchor] usable S_B rank ~{usable} (>1% of max), "
          f"rank cap {rank_cap}; using D={D}.")

    Vk, eigvals = signed_eigenbasis(S_B, D, dev, dt)
    print(f"[sb anchor] kept D={D}, "
          f"eigval range [{eigvals.min():.4e}, {eigvals.max():.4e}], "
          f"condition {eigvals.max()/max(eigvals.min(),1e-12):.2e}")
    # shared-vs-specific guide: where does cumulative class-separation hit 50%?
    cum = np.cumsum(eigvals) / max(eigvals.sum(), 1e-12)
    half = int(np.searchsorted(cum, 0.5)) + 1
    print(f"[sb anchor] first {half}/{D} atoms carry 50% of between-class "
          f"separation (shared structure); the tail is class-specific.")

    W0 = Vk.T                                     # [D, C]
    norms = np.linalg.norm(W0, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    W0 = W0 / norms

    # deterministic sign fix
    for d in range(W0.shape[0]):
        j = np.argmax(np.abs(W0[d]))
        if W0[d, j] < 0:
            W0[d] = -W0[d]

    print(f"[sb anchor] W0 ready: shape {W0.shape}, signed, unit rows.")

    if return_diagnostics:
        diag = {'eigvals': eigvals, 'gate': stats['gate'],
                'tr_SB': stats['tr_SB'], 'tr_ST': stats['tr_ST'],
                'usable_rank': usable, 'present_classes': stats['present_classes'],
                'pool': pool, 'device': str(dev), 'dtype': dtype}
        return W0, diag
    return W0


# ----------------------------------------------------------------------
# Seed-stability verification (same contract as csae_svd_anchor.py)
# ----------------------------------------------------------------------

def matched_component_distance(A, B):
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


def verify_anchor_determinism(cache_path, D, seeds, num_classes=1000,
                              pool="none", device="auto", dtype="float64",
                              **kw):
    """Build the S_B anchor at several seeds (reusing one streamed S_B) and
    report cross-seed agreement. Should be at float-noise; S_B is a fixed
    function of cache+labels."""
    print(f"\n{'='*60}\nS_B ANCHOR DETERMINISM CHECK (seeds={seeds})"
          f"\n{'='*60}")
    dev = resolve_device(device)
    dt = torch_dtype(dtype)
    stats = build_class_scatter(
        cache_path, dev, dt, num_classes=num_classes, pool=pool,
        max_chunks=kw.get("max_chunks"),
        subsample_cells=kw.get("subsample_cells", 200000), seed=seeds[0])
    anchors = []
    for sd in seeds:
        print(f"\n--- seed {sd} ---")
        W0 = build_sb_anchor(cache_path, D=D, seed=sd, num_classes=num_classes,
                             pool=pool, device=device, dtype=dtype,
                             _reuse_stats=stats)
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
        print("  --> DETERMINISTIC (as expected). Safe to use.")
    elif mean_d < 0.05:
        print("  --> seed-stable, not bit-exact (GPU reductions / sign-fix "
              "ties on borderline rows). Acceptable; fix one seed.")
    else:
        print("  --> Unexpected for an eigendecomposition. Check the sign-fix "
              "path and that S_B is reused across the seed loop.")
    return mean_d, anchors, stats


def main():
    ap = argparse.ArgumentParser(
        description="Build / verify the signed S_B (between-class) anchor.")
    ap.add_argument("--cache_dir", type=str, default="cache_activations")
    ap.add_argument("--cache_key", type=str, required=True,
                    help="CSAE activation cache dir (the gcmap1 one).")
    ap.add_argument("--num_classes", type=int, default=1000)
    ap.add_argument("--D", type=int, default=256,
                    help="Anchor atom count. Capped at usable S_B rank "
                         "(<= #classes-1). Watch the eigenvalue cliff in the "
                         "verbose output and trim D to it.")
    ap.add_argument("--pool", type=str, default="none",
                    choices=["none", "image"],
                    help="'none'=cell-level (matches CSAE granularity, gate "
                         "comparable to S_T CSAE sees -- DEFAULT). "
                         "'image'=mean-pool per image first (cleaner class "
                         "separation, gate not directly comparable).")
    ap.add_argument("--verify_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max_chunks", type=int, default=None)
    ap.add_argument("--subsample_cells", type=int, default=200000)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="float64",
                    choices=["float64", "float32"])
    ap.add_argument("--save", type=str, default="csae_sb_anchor_W0.npy",
                    help="Where to save the anchor. Drop-in for "
                         "csae_stable.py's --ica_anchor flag.")
    args = ap.parse_args()

    if joblib is None or torch is None:
        raise SystemExit("Needs joblib and torch installed.")

    cache_path = Path(args.cache_dir) / args.cache_key
    if not cache_path.exists():
        raise SystemExit(f"Cache not found: {cache_path}")

    mean_d, anchors, stats = verify_anchor_determinism(
        cache_path, args.D, args.verify_seeds, num_classes=args.num_classes,
        pool=args.pool, device=args.device, dtype=args.dtype,
        max_chunks=args.max_chunks, subsample_cells=args.subsample_cells)

    W0 = anchors[0]
    np.save(args.save, W0)
    print(f"\nSaved S_B anchor to {args.save}  (shape {W0.shape})")
    print(f"GATE tr(S_B)/tr(S_T) = {stats['gate']:.4f}  "
          f"(decide go/no-go from this BEFORE training).")
    print(f"Use this file as --ica_anchor in csae_stable.py.")


if __name__ == "__main__":
    main()