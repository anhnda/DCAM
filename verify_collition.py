"""
verify_coalition_stability.py
=============================

GATING EXPERIMENT for the coalition-anchored-stability hypothesis.

Hypothesis under test
---------------------
SAE atoms are *coalitions of input filter channels*. Across training seeds the
individual atoms reshuffle (split / absorb), but:
  (H1) the backbone organizes channels into a FINITE number r* of reusable
       coalitions  -> NMF of the channel x class usage matrix U has an error
       plateau;
  (H2) that coalition basis is itself deterministic / data-stable
       -> NMF on two disjoint halves of the data gives matching coalitions;
  (H3) the disease is split/absorb, not subspace divergence
       -> two existing trained seeds have CLOSE recipe SUBSPACES (small
          principal angles) even though per-atom matching is poor.

If H1 fails (no plateau) the anchor approach is DEAD - stop.
If H1 holds but H2 fails, NMF must be replaced by a more stable factorization.
If H1+H2 hold and H3 shows small subspace angles + poor per-atom match, the
anchor is justified and split/absorb is confirmed as the mechanism.

This script touches NO training. It reads:
  - the cached activations + Grad-CAM maps (from run_xcsae_full.py), and
  - (optionally) two trained ConvSAE .pkl files for two seeds.

Usage
-----
  # Checks 1 & 2 only (need just the activation cache):
  python verify_coalition_stability.py --model resnet50

  # All three checks (also pass two trained seeds):
  python verify_coalition_stability.py --model resnet50 \
        --sae_seed_a imagenet1k_csae_resnet50_gcsum_seed42-42_model.pkl \
        --sae_seed_b imagenet1k_csae_resnet50_gcsum_seed42-43_model.pkl

Outputs
-------
  coalition_verify_<model>.png      diagnostic figure (3 panels)
  coalition_verify_<model>.json     machine-readable numbers + verdicts
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


# ----------------------------------------------------------------------
# Cache loading: rebuild the (activation, gradcam, label) tuples on disk
# ----------------------------------------------------------------------

def find_cache_dir(cache_root: Path, model: str, target_layer: str,
                   thresh: float) -> Path:
    """Locate the chunked cache dir written by MultiModelActivationExtractor.

    The cache key looks like:
        <model>_<layer>_thresh<th>_samples<N>_chunk<C>_gcmap1
    with '[' -> '_', ']' removed, '.' -> 'p'. We glob on the stable prefix
    and take the first match (there is normally exactly one per config).
    """
    layer_key = target_layer.replace('[', '_').replace(']', '')
    thresh_key = f"thresh{thresh}".replace('.', 'p')
    pattern = f"{model}_{layer_key}_{thresh_key}_*gcmap1"
    matches = sorted(cache_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No cache dir matching '{pattern}' under {cache_root}. "
            f"Available: {[p.name for p in cache_root.iterdir() if p.is_dir()]}"
        )
    if len(matches) > 1:
        print(f"  [warn] multiple cache dirs match; using {matches[0].name}")
    return matches[0]


def stream_usage_matrices(cache_dir: Path, max_parts: int = None,
                          split_seed: int = 0):
    """Stream the chunked cache and accumulate the channel x class usage
    matrix U WITHOUT ever holding more than one chunk in memory.

    At your scale (50k x 1024 x 14 x 14 fp32 ~= 40 GB), materializing the full
    activation tensor is impossible on a 64 GB box. But every quantity the
    three checks need is a *reduction* over samples:

        per-sample usage:  u_n[c] = sum_{i,j} L_GC_n[i,j] * A_{n,c}[i,j]   [C]

    and U[c,y] is just the per-class MEAN of u_n. So we keep running per-class
    SUMS (class_sum[C, n_class]) and COUNTS, updated one ~80 MB chunk at a time.

    We simultaneously maintain two disjoint-half accumulators (routing each
    sample by a hash of its global index) so Check 2 needs no second pass and
    no extra memory.

    Returns
    -------
    U        : [C, n_class]  full usage matrix (per-class mean)
    U_a, U_b : [C, n_class]  usage matrices on two disjoint sample halves
    classes  : [n_class]     sorted unique labels actually seen
    per_sample_usage : [N, C] float32  (small: 50k x 1024 x 4 = 205 MB)
                       kept because it's cheap and lets Check 2 resample
                       splits without re-reading the cache.
    sample_labels    : [N]   int64, aligned with per_sample_usage rows
    """
    meta = joblib.load(cache_dir / "metadata.pkl")
    n_parts = meta['num_chunks']
    if max_parts is not None:
        n_parts = min(n_parts, max_parts)
    C = meta['num_channels']

    # First pass over labels is unnecessary; we discover classes on the fly and
    # remap to a dense [0, n_class) index. ImageNet has <=1000 classes so a
    # fixed-size buffer is fine and avoids a remap pass.
    max_classes = 1000
    class_sum = np.zeros((C, max_classes), dtype=np.float64)
    class_cnt = np.zeros(max_classes, dtype=np.int64)
    class_sum_a = np.zeros((C, max_classes), dtype=np.float64)
    class_cnt_a = np.zeros(max_classes, dtype=np.int64)
    class_sum_b = np.zeros((C, max_classes), dtype=np.float64)
    class_cnt_b = np.zeros(max_classes, dtype=np.int64)

    per_sample_chunks = []   # small: [c, C] per chunk
    label_chunks = []
    seen = set()

    global_idx = 0
    from tqdm import tqdm as _tqdm
    for p in _tqdm(range(n_parts), desc="Streaming cache"):
        part = joblib.load(cache_dir / f"part_{p:04d}.pkl")
        a = part['activation'].numpy()              # [c, C, H, W] fp32
        g = part['gradcam_map'].numpy()             # [c, H, W]
        lbl = part['label'].numpy().astype(np.int64)  # [c]
        c_, C_, H_, W_ = a.shape

        np.clip(a, 0.0, None, out=a)                # defensive, in place
        # per-sample channel usage u_n[c] = sum_ij g_n * a_{n,c}
        # einsum keeps it in one BLAS call, no [N,C,H,W] intermediate copy
        u = np.einsum('nij,ncij->nc', g, a, optimize=True).astype(np.float32)  # [c, C]

        # accumulate full + split halves
        for i in range(c_):
            y = int(lbl[i]); seen.add(y)
            class_sum[:, y] += u[i]; class_cnt[y] += 1
            # deterministic split by hash of global index (independent of order)
            if (hash((split_seed, global_idx)) & 1) == 0:
                class_sum_a[:, y] += u[i]; class_cnt_a[y] += 1
            else:
                class_sum_b[:, y] += u[i]; class_cnt_b[y] += 1
            global_idx += 1

        per_sample_chunks.append(u)
        label_chunks.append(lbl)
        del a, g, part            # free the big chunk immediately

    classes = np.array(sorted(seen))
    def _to_U(csum, ccnt):
        U = np.zeros((C, classes.size), dtype=np.float64)
        for j, y in enumerate(classes):
            if ccnt[y] > 0:
                U[:, j] = csum[:, y] / ccnt[y]
        return np.clip(U, 0.0, None)

    U = _to_U(class_sum, class_cnt)
    U_a = _to_U(class_sum_a, class_cnt_a)
    U_b = _to_U(class_sum_b, class_cnt_b)

    per_sample_usage = np.concatenate(per_sample_chunks, axis=0)   # [N, C] fp32
    sample_labels = np.concatenate(label_chunks, axis=0)           # [N]

    print(f"  streamed {per_sample_usage.shape[0]} samples, "
          f"C={C}, classes={classes.size}, "
          f"per-sample-usage held in RAM = "
          f"{per_sample_usage.nbytes/1e9:.2f} GB")
    return U, U_a, U_b, classes, per_sample_usage, sample_labels


def usage_matrix_from_per_sample(per_sample_usage, sample_labels, idx=None):
    """Build U[C, n_class] from the small precomputed per-sample usage table.

    Used by Check 2 to resample splits cheaply (no cache re-read). `idx`
    restricts to a subset of samples (e.g. one data half).
    """
    if idx is not None:
        ps = per_sample_usage[idx]
        lab = sample_labels[idx]
    else:
        ps = per_sample_usage
        lab = sample_labels
    classes = np.unique(lab)
    C = ps.shape[1]
    U = np.zeros((C, classes.size), dtype=np.float64)
    for j, y in enumerate(classes):
        sel = ps[lab == y]
        if sel.shape[0] > 0:
            U[:, j] = sel.mean(axis=0)
    # guard tiny negatives from float error
    U = np.clip(U, 0.0, None)
    return U, classes


# ----------------------------------------------------------------------
# CHECK 1: NMF rank-vs-error plateau  (does r* exist?)
# ----------------------------------------------------------------------

def nmf_reconstruction_error(U, r, seed=0, max_iter=400):
    """Relative Frobenius reconstruction error of rank-r NMF of U.

    Uses NNDSVD init (deterministic) so the factorization is reproducible
    given U; the random_state only matters for the 'ar' tie-breaks.
    """
    model = NMF(n_components=r, init='nndsvd', solver='cd',
                max_iter=max_iter, random_state=seed)
    Wc = model.fit_transform(U)        # [C, r]
    Hc = model.components_             # [r, n_classes]
    rec = Wc @ Hc
    err = np.linalg.norm(U - rec) / (np.linalg.norm(U) + 1e-12)
    return err, Wc, Hc


def check1_rank_plateau(U, ranks):
    print("\n[CHECK 1] NMF rank-vs-error plateau ...")
    errs = []
    for r in ranks:
        err, _, _ = nmf_reconstruction_error(U, r)
        errs.append(err)
        print(f"    r={r:4d}   rel_err={err:.4f}")
    errs = np.array(errs)

    # Detect a plateau: first rank where the marginal error reduction per
    # added component drops below 10% of the initial marginal reduction.
    d = -np.diff(errs)                      # error reduction per rank step
    if d.size >= 2 and d[0] > 1e-9:
        rel_gain = d / d[0]
        plateau_idx = np.argmax(rel_gain < 0.10)  # first True
        r_star = ranks[plateau_idx + 1] if rel_gain[plateau_idx] < 0.10 else ranks[-1]
    else:
        r_star = ranks[-1]

    verdict = ("PLATEAU FOUND" if r_star < ranks[-1]
               else "NO CLEAR PLATEAU (gate at risk)")
    print(f"    -> estimated r* = {r_star}   [{verdict}]")
    return {'ranks': list(map(int, ranks)),
            'errors': errs.tolist(),
            'r_star': int(r_star),
            'verdict': verdict}


# ----------------------------------------------------------------------
# CHECK 2: NMF cross-split reproducibility (is the anchor data-stable?)
# ----------------------------------------------------------------------

def hungarian_match_cosine(Wa, Wb):
    """Match columns of Wa to columns of Wb maximizing total cosine, return
    the matched per-column cosines (sorted descending)."""
    from scipy.optimize import linear_sum_assignment
    a = Wa / (np.linalg.norm(Wa, axis=0, keepdims=True) + 1e-12)
    b = Wb / (np.linalg.norm(Wb, axis=0, keepdims=True) + 1e-12)
    S = a.T @ b                              # [r, r] cosine sim
    ri, ci = linear_sum_assignment(-S)       # maximize
    matched = S[ri, ci]
    return np.sort(matched)[::-1]


def check2_split_reproducibility(per_sample_usage, sample_labels, r_star,
                                 n_repeats=3):
    """Resample disjoint data halves from the small per-sample usage table
    (no cache re-read), NMF each half, Hungarian-match coalitions."""
    print("\n[CHECK 2] NMF coalition reproducibility across data halves ...")
    N = per_sample_usage.shape[0]
    rng = np.random.default_rng(0)
    all_matched = []
    for rep in range(n_repeats):
        perm = rng.permutation(N)
        half = N // 2
        idx_a, idx_b = perm[:half], perm[half:]
        Ua, _ = usage_matrix_from_per_sample(per_sample_usage, sample_labels, idx_a)
        Ub, _ = usage_matrix_from_per_sample(per_sample_usage, sample_labels, idx_b)
        # NMF factors are [C, r] regardless of how many class columns each half
        # has, so matching on W (channel-space coalitions) is well-defined even
        # if a half is missing some classes.
        _, Wa, _ = nmf_reconstruction_error(Ua, r_star)
        _, Wb, _ = nmf_reconstruction_error(Ub, r_star)
        matched = hungarian_match_cosine(Wa, Wb)
        all_matched.append(matched)
        print(f"    rep {rep}: matched-coalition cosine "
              f"median={np.median(matched):.3f}  "
              f"top-quartile={np.percentile(matched,75):.3f}  "
              f"min={matched.min():.3f}")
    matched = np.mean(all_matched, axis=0)
    frac_high = float((matched > 0.8).mean())
    verdict = ("ANCHOR DATA-STABLE" if np.median(matched) > 0.7
               else "ANCHOR NOT STABLE (NMF init/rank needs work)")
    print(f"    -> fraction of coalitions with cosine>0.8: {frac_high:.2f}  "
          f"[{verdict}]")
    return {'matched_cosine': matched.tolist(),
            'median': float(np.median(matched)),
            'frac_cos_gt_0.8': frac_high,
            'verdict': verdict}


# ----------------------------------------------------------------------
# CHECK 3: existing-seed recipe SUBSPACE angles vs per-atom matching
# ----------------------------------------------------------------------

def encoder_recipe_matrix(sae_pkl):
    """Pull the encoder weight as a [D, C] recipe matrix (1x1 conv)."""
    sae = joblib.load(sae_pkl)
    W = sae.encoder.weight.detach().cpu().numpy()   # [D, C, 1, 1]
    W = W.reshape(W.shape[0], W.shape[1])            # [D, C]
    return W                                          # rows = recipes


def principal_subspace_angles(Wa, Wb, r):
    """Principal angles between the top-r recipe subspaces of two seeds.

    We take the row-space of the r most-used recipes in each seed (by L2
    norm of the recipe, a proxy for usage independent of any image), build
    orthonormal bases via SVD, and return sin(theta) for each principal
    angle. Small sin(theta) == subspaces aligned == split/absorb-invariant
    stability.
    """
    def top_r_basis(W, r):
        usage = np.linalg.norm(W, axis=1)
        idx = np.argsort(usage)[::-1][:r]
        M = W[idx]                       # [r, C]
        # orthonormal basis of the row space
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        return Vt[:r]                    # [r, C] orthonormal rows
    Ba = top_r_basis(Wa, r)              # [r, C]
    Bb = top_r_basis(Wb, r)
    # principal angles: singular values of Ba Bb^T are cos(theta)
    M = Ba @ Bb.T                        # [r, r]
    cos_theta = np.clip(np.linalg.svd(M, compute_uv=False), -1, 1)
    sin_theta = np.sqrt(np.clip(1 - cos_theta**2, 0, 1))
    return cos_theta, sin_theta


def check3_subspace_vs_atom(sae_a, sae_b, r_star):
    print("\n[CHECK 3] existing-seed recipe subspace vs per-atom match ...")
    Wa = encoder_recipe_matrix(sae_a)    # [D, C]
    Wb = encoder_recipe_matrix(sae_b)
    C = Wa.shape[1]
    r = min(r_star, C, Wa.shape[0], Wb.shape[0])

    cos_theta, sin_theta = principal_subspace_angles(Wa, Wb, r)
    subspace_aligned = float(np.mean(cos_theta))   # 1 == perfectly aligned

    # Per-atom Hungarian match on the SAME top-r recipes (rows), for contrast.
    def top_r_recipes(W, r):
        usage = np.linalg.norm(W, axis=1)
        idx = np.argsort(usage)[::-1][:r]
        return W[idx]
    Ra = top_r_recipes(Wa, r)
    Rb = top_r_recipes(Wb, r)
    per_atom = hungarian_match_cosine(Ra.T, Rb.T)   # match recipes as vectors

    print(f"    subspace: mean cos(principal angle) = {subspace_aligned:.3f} "
          f"(median sin = {np.median(sin_theta):.3f})")
    print(f"    per-atom: Hungarian-matched recipe cosine "
          f"median={np.median(per_atom):.3f}  min={per_atom.min():.3f}")

    gap = subspace_aligned - float(np.median(per_atom))
    if subspace_aligned > 0.8 and np.median(per_atom) < 0.7:
        verdict = ("SPLIT/ABSORB CONFIRMED: subspace stable, atoms not "
                   "-> anchor is justified")
    elif subspace_aligned > 0.8:
        verdict = "BOTH STABLE: atoms already match (anchor may be unnecessary)"
    else:
        verdict = ("SUBSPACE ALSO DIVERGES: problem deeper than split/absorb "
                   "-> anchor alone won't fix it")
    print(f"    -> {verdict}")
    return {'subspace_mean_cos': subspace_aligned,
            'subspace_sin_angles': sin_theta.tolist(),
            'per_atom_matched_cosine': per_atom.tolist(),
            'per_atom_median': float(np.median(per_atom)),
            'subspace_minus_atom_gap': float(gap),
            'verdict': verdict}


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def make_figure(res, model, save_path):
    n_panels = 2 + (1 if 'check3' in res else 0)
    fig, axs = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axs = [axs]

    # Panel 1: rank-vs-error
    c1 = res['check1']
    axs[0].plot(c1['ranks'], c1['errors'], 'o-', color='navy')
    axs[0].axvline(c1['r_star'], ls='--', color='crimson',
                   label=f"r* = {c1['r_star']}")
    axs[0].set_xlabel('NMF rank r'); axs[0].set_ylabel('relative recon error')
    axs[0].set_title(f"Check 1: coalition count\n{c1['verdict']}")
    axs[0].legend(); axs[0].grid(alpha=0.3)

    # Panel 2: cross-split matched cosine
    c2 = res['check2']
    mc = np.array(c2['matched_cosine'])
    axs[1].plot(np.arange(mc.size), mc, color='darkgreen')
    axs[1].axhline(0.8, ls='--', color='gray')
    axs[1].set_xlabel('coalition (sorted)'); axs[1].set_ylabel('cross-split cosine')
    axs[1].set_ylim(0, 1.02)
    axs[1].set_title(f"Check 2: anchor reproducibility\n{c2['verdict']}")
    axs[1].grid(alpha=0.3)

    # Panel 3: subspace vs per-atom
    if 'check3' in res:
        c3 = res['check3']
        pa = np.sort(np.array(c3['per_atom_matched_cosine']))[::-1]
        axs[2].plot(np.arange(pa.size), pa, color='firebrick',
                    label='per-atom matched cos')
        axs[2].axhline(c3['subspace_mean_cos'], ls='-', color='royalblue',
                       label=f"subspace mean cos = {c3['subspace_mean_cos']:.2f}")
        axs[2].axhline(0.8, ls='--', color='gray')
        axs[2].set_xlabel('atom (sorted)'); axs[2].set_ylabel('cosine')
        axs[2].set_ylim(0, 1.02)
        axs[2].set_title("Check 3: subspace vs atom\n(gap => split/absorb)")
        axs[2].legend(fontsize=8); axs[2].grid(alpha=0.3)

    fig.suptitle(f"Coalition-stability gating  --  {model}",
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='resnet50')
    ap.add_argument('--target_layer', default='layer3')
    ap.add_argument('--cumulative_threshold', type=float, default=0.95,
                    help='must match the value used during extraction')
    ap.add_argument('--cache_root', default='cache_activations')
    ap.add_argument('--max_parts', type=int, default=None,
                    help='cap cache parts loaded (for a quick smoke test)')
    ap.add_argument('--ranks', type=int, nargs='+',
                    default=[4, 8, 16, 24, 32, 48, 64, 96, 128],
                    help='NMF ranks to sweep in Check 1')
    ap.add_argument('--sae_seed_a', default=None,
                    help='trained ConvSAE .pkl for seed A (enables Check 3)')
    ap.add_argument('--sae_seed_b', default=None,
                    help='trained ConvSAE .pkl for seed B (enables Check 3)')
    args = ap.parse_args()

    print("=" * 70)
    print(f"COALITION-STABILITY GATING EXPERIMENT  ({args.model})")
    print("=" * 70)

    cache_dir = find_cache_dir(Path(args.cache_root), args.model,
                               args.target_layer, args.cumulative_threshold)
    print(f"Cache: {cache_dir}")
    # Streaming pass: builds U (and split halves) one ~80 MB chunk at a time.
    # Never materializes the full [N, C, H, W] activation tensor (~40 GB).
    (U, U_a, U_b, classes,
     per_sample_usage, sample_labels) = stream_usage_matrices(
        cache_dir, max_parts=args.max_parts)
    print(f"Usage matrix U: {U.shape}  (channels x classes), "
          f"density={ (U>1e-8).mean():.3f}")

    # cap the rank sweep at C (can't have more coalitions than channels)
    C = U.shape[0]
    ranks = [r for r in args.ranks if r <= C]

    res = {}
    res['check1'] = check1_rank_plateau(U, ranks)
    r_star = res['check1']['r_star']
    res['check2'] = check2_split_reproducibility(
        per_sample_usage, sample_labels, r_star)

    if args.sae_seed_a and args.sae_seed_b:
        res['check3'] = check3_subspace_vs_atom(args.sae_seed_a,
                                                args.sae_seed_b, r_star)
    else:
        print("\n[CHECK 3] skipped (pass --sae_seed_a and --sae_seed_b "
              "with two trained seeds to enable).")

    out_png = f"coalition_verify_{args.model}.png"
    out_json = f"coalition_verify_{args.model}.json"
    make_figure(res, args.model, out_png)
    with open(out_json, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"Numbers saved: {out_json}")

    print("\n" + "=" * 70)
    print("VERDICT SUMMARY")
    print("=" * 70)
    print(f"  Check 1 (r* exists?)        : {res['check1']['verdict']}")
    print(f"  Check 2 (anchor stable?)    : {res['check2']['verdict']}")
    if 'check3' in res:
        print(f"  Check 3 (split/absorb?)     : {res['check3']['verdict']}")
    print("\nGate: if Check 1 finds NO plateau, the anchor approach is not")
    print("viable for this backbone/layer -- stop before reimplementing.")


if __name__ == "__main__":
    main()