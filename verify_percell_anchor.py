"""
verify_percell_anchor.py
========================
Before retraining DCAM, answer two questions about the per-cell anchor fix:

  Q1. Is the per-cell anchor still SEED-STABLE?
      The averaged-S anchor was stable (NMF of a near-rank-1 matrix is
      trivially reproducible). The per-cell S has effective rank ~200, so
      its NMF is NOT trivially reproducible. If per-cell NMF disagrees
      wildly across seeds, the anchor mechanism has a deeper problem and
      the honest paper is a negative result. If it is still reasonably
      stable, the fix is sound.

  Q2. Are the per-cell atoms NON-DEGENERATE?
      Stable-but-identical atoms are useless. We report nu (min pairwise
      atom distance). nu near 0 means the atoms collapse together.

This script does NOT retrain DCAM. It only checks the anchor. If both
answers are good, proceed to retrain with run_dcam_full.py (per-cell patch).

Usage
-----
    python verify_percell_anchor.py \\
        --cache_dir cache_activations \\
        --cache_key resnet50_layer3_thresh0p85_samples50000_chunk100_dcam_pi_v1 \\
        --D 512 --seeds 0 1 2
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import joblib
from scipy.optimize import linear_sum_assignment

from run_dcam_full import build_anchor
from dcam_percell_anchor import compute_coactivation_matrix_percell


def load_chunks(cache_dir, cache_key):
    """Load activation chunks from a DCAM cache directory."""
    d = Path(cache_dir) / cache_key
    md = joblib.load(d / "metadata.pkl")
    chunks = []
    for p in range(md["num_chunks"]):
        part = joblib.load(d / f"part_{p:04d}.pkl")
        chunks.append(part["activation"])
    print(f"Loaded {len(chunks)} chunks "
          f"({sum(c.shape[0] for c in chunks)} images)")
    return chunks


def matched_atom_distance(Pi_a, Pi_b):
    """Hungarian-matched mean L2 distance between two [D,C] atom sets.
    Same construction as run_dcam_full.atom_distance, but returns the MEAN
    (scale-comparable across D) rather than the Frobenius sum."""
    A = Pi_a.cpu().numpy().astype(np.float64)
    B = Pi_b.cpu().numpy().astype(np.float64)
    D = min(A.shape[0], B.shape[0])
    A, B = A[:D], B[:D]
    cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default="cache_activations")
    ap.add_argument("--cache_key", type=str, required=True)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--nmf_iters", type=int, default=4000)
    ap.add_argument("--subsample_cells", type=int, default=200000)
    args = ap.parse_args()

    chunks = load_chunks(args.cache_dir, args.cache_key)

    # build the per-cell S once (it does not depend on the NMF seed)
    S_cell = compute_coactivation_matrix_percell(
        chunks, subsample_cells=args.subsample_cells, seed=0)

    print(f"\n{'='*60}\nPer-cell anchor: seed-stability sweep\n{'='*60}")
    anchors = []
    nus = []
    for sd in args.seeds:
        Pi0, nu, D_eff = build_anchor(
            S_cell, args.D, nmf_iters=args.nmf_iters, seed=sd)
        anchors.append(Pi0)
        nus.append(nu)
        print(f"  seed {sd}: D_eff={D_eff}, nu={nu:.4e}")

    # pairwise matched distance across seeds
    dists = []
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            d = matched_atom_distance(anchors[i], anchors[j])
            dists.append(d)
            print(f"  seeds {args.seeds[i]}<->{args.seeds[j]}: "
                  f"matched-atom dist = {d:.4f}")

    mean_d = float(np.mean(dists)) if dists else float("nan")
    mean_nu = float(np.mean(nus))

    print(f"\n{'='*60}\nVERDICT\n{'='*60}")
    print(f"  mean cross-seed matched-atom distance : {mean_d:.4f}")
    print(f"  mean non-degeneracy nu                : {mean_nu:.4e}")
    stable = mean_d < 0.10
    nondegen = mean_nu > 1e-2
    if stable and nondegen:
        print("  --> Per-cell anchor is STABLE and NON-DEGENERATE. "
              "The fix is sound; proceed to retrain DCAM.")
    elif not stable:
        print("  --> Per-cell anchor is UNSTABLE across seeds. The anchor "
              "mechanism has a deeper problem than the averaging bug; "
              "the honest result is a negative one. Do NOT claim "
              "seed-stability in the paper.")
    else:
        print("  --> Per-cell anchor atoms are DEGENERATE (nu ~ 0): stable "
              "but collapsed onto each other. Reduce D or revisit the "
              "non-degeneracy merge.")


if __name__ == "__main__":
    main()