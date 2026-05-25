"""
verify_percell_anchor.py
========================
Before retraining DCAM, answer two questions about the per-cell anchor fix:

  Q1. Did the per-cell NMF actually CONVERGE?
      build_anchor's NMF stalled on the per-cell S at rel_err ~285 (a
      relative error must be in [0,~1]). A stability verdict computed from an
      unconverged NMF is meaningless -- three unconverged runs disagree just
      because they stopped at three different points on a slow descent. This
      script uses the converging solver (nmf_symmetric: rescaled init +
      correlation normalization) and REFUSES to issue a stability verdict
      unless rel_err < 1 for every seed.

  Q2. Given convergence, is the per-cell anchor SEED-STABLE and
      NON-DEGENERATE? Stable + nu well above 0 => the fix is sound. Unstable
      => the anchor mechanism has a real problem (a rank-~200 matrix factored
      into D=512 atoms is genuinely underdetermined) and the honest paper is
      a negative result.

This script does NOT retrain DCAM. It only checks the anchor.

Usage
-----
    python verify_percell_anchor.py \\
        --cache_dir cache_activations \\
        --cache_key resnet50_layer3_thresh0p85_samples50000_chunk100_dcam_pi_v1 \\
        --D 512 --seeds 0 1 2 --device cuda
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import joblib
from scipy.optimize import linear_sum_assignment

from dcam_percell_anchor import build_anchor_percell


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
    """Hungarian-matched MEAN L2 distance between two [D,C] atom sets
    (scale-comparable across D)."""
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
    ap.add_argument("--nmf_iters", type=int, default=8000,
                    help="Per-cell S is large-magnitude / high-rank; give the "
                         "NMF room. nmf_symmetric early-stops on plateau.")
    ap.add_argument("--subsample_cells", type=int, default=200000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--init", type=str, default="random",
                    choices=["random", "nndsvd"],
                    help="NMF init. 'random': seed-dependent. 'nndsvd': "
                         "deterministic SVD-based init -- the anchor is then "
                         "seed-free BY CONSTRUCTION, so cross-seed distance "
                         "should be ~0. If it is NOT ~0 with nndsvd, "
                         "something else in the solver is non-deterministic.")
    args = ap.parse_args()

    chunks = load_chunks(args.cache_dir, args.cache_key)

    if args.init == "nndsvd":
        print("\n[NOTE] --init nndsvd: the NMF init is deterministic, so all "
              "seeds share an identical starting point. Cross-seed distance "
              "is expected to be ~0; a non-zero value would indicate "
              "residual non-determinism (e.g. non-deterministic GPU "
              "reductions). The real question this answers: IS a "
              "deterministic per-cell anchor achievable at all.")

    print(f"\n{'='*60}\nPer-cell anchor: seed-stability sweep "
          f"(init={args.init})\n{'='*60}")
    anchors, nus, errs, traces = [], [], [], []
    for sd in args.seeds:
        Pi0, nu, D_eff, trace, err = build_anchor_percell(
            chunks, args.D, nmf_iters=args.nmf_iters, seed=sd,
            subsample_cells=args.subsample_cells, device=args.device,
            return_trace=True, init=args.init)
        anchors.append(Pi0)
        nus.append(nu)
        errs.append(err)
        traces.append(trace)
        print(f"  seed {sd}: D_eff={D_eff}, nu={nu:.4e}, "
              f"NMF rel_err={err:.4f}")

    # ---- convergence gate --------------------------------------------
    print(f"\n{'='*60}\nCONVERGENCE CHECK\n{'='*60}")
    all_converged = all(e < 1.0 for e in errs)
    for sd, e, tr in zip(args.seeds, errs, traces):
        tail = ""
        if tr and len(tr) >= 2:
            tail = (f"  (last step {tr[-2][1]:.4f} -> {tr[-1][1]:.4f})")
        flag = "OK" if e < 1.0 else "NOT CONVERGED"
        print(f"  seed {sd}: rel_err={e:.4f}  [{flag}]{tail}")

    if not all_converged:
        print(f"\n{'='*60}\nVERDICT: INCONCLUSIVE\n{'='*60}")
        print("  At least one NMF did not converge (rel_err >= 1). The "
              "stability numbers below would be meaningless -- NOT reporting "
              "a stability verdict. Raise --nmf_iters and re-run.")
        return

    # ---- stability (only if converged) -------------------------------
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
    print(f"  init mode                             : {args.init}")
    print(f"  all NMF runs converged                : yes "
          f"(max rel_err {max(errs):.4f})")
    print(f"  mean cross-seed matched-atom distance : {mean_d:.4f}")
    print(f"  mean non-degeneracy nu                : {mean_nu:.4e}")
    stable = mean_d < 0.10
    nondegen = mean_nu > 1e-2

    if args.init == "nndsvd":
        # deterministic init: stability is expected, not a finding.
        # the informative checks are (a) is it actually ~0 -- i.e. is the
        # pipeline truly deterministic -- and (b) is the result usable.
        if mean_d < 0.02 and nondegen:
            print("  --> NNDSVD init gives a DETERMINISTIC, non-degenerate "
                  "per-cell anchor (cross-seed distance ~0 as expected). "
                  "A seed-free per-cell anchor IS achievable: the random-init "
                  "~0.12 sensitivity was NMF basin-hopping, not an intrinsic "
                  "flaw. The paper can keep a seed-stability claim IF it "
                  "specifies the NNDSVD (deterministic) init.")
        elif mean_d >= 0.02:
            print(f"  --> NNDSVD init still gives cross-seed distance "
                  f"{mean_d:.4f}, NOT ~0. The init is deterministic, so this "
                  f"residual variation comes from elsewhere -- likely "
                  f"non-deterministic GPU reductions in the NMF loop. Re-run "
                  f"on --device cpu to confirm; if cpu gives ~0, the anchor "
                  f"IS deterministic and the gpu number was a red herring.")
        else:
            print("  --> NNDSVD init is deterministic but the atoms are "
                  "DEGENERATE (nu ~ 0). A seed-free but collapsed anchor is "
                  "not useful; reduce D.")
        # also compare to the random-init baseline for the writeup
        print(f"\n  For the paper: random-init per-cell anchor was ~0.12 "
              f"(flat in D); NNDSVD-init is {mean_d:.4f}. The contrast IS "
              f"the result -- determinism must be designed in, it is not a "
              f"property the method had.")
    else:
        if stable and nondegen:
            print("  --> Per-cell anchor is STABLE and NON-DEGENERATE on a "
                  "CONVERGED NMF. The fix is sound; proceed to retrain DCAM.")
        elif not stable:
            print("  --> Per-cell anchor is UNSTABLE across seeds even with "
                  "a converged NMF. Random NMF init lands in different "
                  "basins. Try --init nndsvd: if a deterministic init gives "
                  "cross-seed ~0, a seed-free anchor is still achievable.")
        else:
            print("  --> Per-cell anchor atoms are DEGENERATE (nu ~ 0): "
                  "stable but collapsed. Reduce D or revisit the merge.")


if __name__ == "__main__":
    main()