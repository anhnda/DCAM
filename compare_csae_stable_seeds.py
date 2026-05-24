"""
compare_csae_stable_seeds.py
============================
Cross-seed atom-distance comparison for csae_stable.py runs, split into
anchored slice (first D rows) and free slice (the rest).

Why split
---------
csae_stable.py's compare_seeds() compares the full encoder_weight_slice
across seeds. For --anchor_mode subspace that conflates two different
questions:

  1. ANCHORED SLICE [:D, :]  -- do the first D atoms agree across seeds?
     If we initialised them at W0 and held them there with a loss, of
     course they agree. The interesting number here is HOW MUCH they agree
     (close to 0 means strongly held, larger means λ_anchor let them drift).
     This bounds the anchor's pinning strength but is not itself the
     stability result.

  2. FREE SLICE [D:, :]  -- do the OTHER (hidden_dim - D) atoms agree?
     This is the open question from the csae_stable docstring: does
     anchoring 200/8192 units discipline the remaining 7992? Plain CSAE's
     atom distance on the free hidden_dim is the baseline; if csae_stable's
     free slice comes out meaningfully lower, the spine transfers stability
     to the unanchored majority -- the constructive result.

If both are small, the paper has a real claim. If only (1) is small, the
spine is stable but inert (the same problem we saw at λ=1.0, just at a
different scale).

Usage
-----
    python compare_csae_stable_seeds.py \\
        imagenet1k_csae_stable_resnet50_subspace_la0.01_seed0-0_result.pkl \\
        imagenet1k_csae_stable_resnet50_subspace_la0.01_seed0-1_result.pkl \\
        imagenet1k_csae_stable_resnet50_subspace_la0.01_seed0-2_result.pkl \\
        --D 200

Plain CSAE baseline (for context)
---------------------------------
Run the same comparison on plain CSAE result.pkls (where there is no
anchored/free split, so just pass --D 0 to compare the whole encoder), and
read off the number to put next to csae_stable's free-slice distance.
"""

import argparse
import numpy as np
import joblib
import torch
from scipy.optimize import linear_sum_assignment


def atom_distance(W_a, W_b):
    """Sign/permutation-invariant matched distance between two [H, C]
    encoder slices. Hungarian matching on (1 - |cosine|); returns mean
    matched value. Same metric as csae_stable.atom_distance and as
    csae_svd_anchor.matched_component_distance, so all three are directly
    comparable on the printed table.
    """
    A = np.asarray(W_a, np.float64)
    B = np.asarray(W_b, np.float64)
    H = min(A.shape[0], B.shape[0])
    A, B = A[:H], B[:H]
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    cos = An @ Bn.T
    cost = 1.0 - np.abs(cos)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def load_slice(path):
    """Pull the encoder weight slice [hidden_dim, C] out of a csae_stable
    checkpoint. Accepts either:
      * a *_result.pkl (reads 'encoder_weight_slice' from the saved dict),
      * a *_model.pth (loads the state dict, slices the encoder weight,
        and reads config from the matching *_result.pkl alongside).

    Returns (W [hidden_dim, C] float64 numpy, config dict).
    """
    from pathlib import Path
    p = Path(path)

    if p.suffix == ".pth" or p.name.endswith("_model.pth"):
        # state-dict path: read weights from .pth, config from sibling .pkl
        result_path = Path(str(p).replace("_model.pth", "_result.pkl"))
        if not result_path.exists():
            raise FileNotFoundError(
                f"Expected metadata {result_path} alongside {p} for the "
                f"config dict (D_anchor, hidden_dim, etc).")
        cfg = joblib.load(result_path).get('config', {})
        state = torch.load(str(p), map_location='cpu')
        # encoder.weight is [H, C, kH, kW]; for 1x1 kernel squeeze to [H, C]
        w = state['encoder.weight']
        if w.dim() == 4 and w.shape[2] == 1 and w.shape[3] == 1:
            w = w[:, :, 0, 0]
        elif w.dim() == 4:
            w = w.mean(dim=(2, 3))                       # fallback
        return np.asarray(w.detach().cpu().numpy(), np.float64), cfg

    # default: joblib result pickle
    blob = joblib.load(path)
    w = blob['encoder_weight_slice']
    if isinstance(w, torch.Tensor):
        w = w.detach().cpu().numpy()
    return np.asarray(w, np.float64), blob.get('config', {})


def pairwise(label, slices, indent="  "):
    """Print pairwise matched distances for a list of slices.
    Returns the mean distance (or NaN if <2 slices)."""
    if len(slices) < 2:
        print(f"{indent}{label}: need >= 2 runs.")
        return float('nan')
    dists = []
    for i in range(len(slices)):
        for j in range(i + 1, len(slices)):
            d = atom_distance(slices[i], slices[j])
            dists.append(d)
            print(f"{indent}{label} run{i} <-> run{j}: {d:.4f}")
    m = float(np.mean(dists))
    print(f"{indent}{label} mean = {m:.4f}")
    return m


def main():
    ap = argparse.ArgumentParser(
        description="Cross-seed atom-distance for csae_stable runs, "
                    "anchored vs free slices.")
    ap.add_argument('paths', nargs='+',
                    help='>=2 csae_stable checkpoint paths. Each can be '
                         'either a *_result.pkl (reads the saved '
                         'encoder_weight_slice) or a *_model.pth (loads '
                         'the state dict and auto-finds the matching '
                         '*_result.pkl for config). Mix and match freely.')
    ap.add_argument('--D', type=int, default=None,
                    help='Anchor atom count. If omitted, read from the '
                         "first checkpoint's config (D_anchor). Use --D 0 "
                         'to skip the split and just compare the whole '
                         'encoder (for plain CSAE baselines).')
    args = ap.parse_args()

    if len(args.paths) < 2:
        raise SystemExit("Need at least 2 checkpoints to compare.")

    print(f"Loading {len(args.paths)} runs:")
    slices, configs = [], []
    for p in args.paths:
        w, cfg = load_slice(p)
        slices.append(w)
        configs.append(cfg)
        print(f"  {p}: shape {w.shape}, "
              f"seeds data={cfg.get('data_seed','?')} "
              f"model={cfg.get('model_seed','?')}, "
              f"lambda_anchor={cfg.get('lambda_anchor','?')}")

    # decide D
    if args.D is None:
        D = configs[0].get('D_anchor')
        if D is None:
            raise SystemExit("--D not given and not in pkl config.")
    else:
        D = args.D

    # sanity: all encoders same shape
    shape0 = slices[0].shape
    for w in slices[1:]:
        if w.shape != shape0:
            raise SystemExit(f"Encoder shapes differ: {shape0} vs {w.shape}. "
                             "Are these from the same anchor_mode / "
                             "hidden_dim configuration?")
    H, C = shape0

    print(f"\nEncoder hidden_dim = {H}, channels = {C}, anchor D = {D}")
    print("=" * 60)
    print("Cross-seed atom distance (lower = more stable)")
    print("=" * 60)

    if D == 0 or D >= H:
        # plain CSAE baseline OR full anchor mode -- one number only
        print("\nWhole encoder:")
        pairwise("all atoms", slices)
    else:
        print(f"\nAnchored slice [:{D}, :]  (initialised at W0, held by "
              f"loss):")
        d_anc = pairwise("anchored", [s[:D] for s in slices])
        print(f"\nFree slice [{D}:, :]  (unconstrained -- the open "
              f"question):")
        d_free = pairwise("free    ", [s[D:] for s in slices])
        print(f"\nWhole encoder (for comparison with plain-CSAE baseline):")
        d_all = pairwise("all     ", slices)

        print("\n" + "=" * 60)
        print("Reading the result:")
        print(f"  anchored slice mean dist = {d_anc:.4f}")
        print(f"  free slice mean dist     = {d_free:.4f}")
        print(f"  whole encoder mean dist  = {d_all:.4f}")
        print("\nCompare 'free slice' to plain-CSAE's whole-encoder distance "
              "on the SAME (hidden_dim - D) atoms. If csae_stable's free "
              "slice is meaningfully lower, the spine transfers stability "
              "to the unanchored majority -- constructive result.")


if __name__ == "__main__":
    main()