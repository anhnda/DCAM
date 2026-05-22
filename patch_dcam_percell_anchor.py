"""
patch_dcam_percell_anchor.py
============================
Switches run_dcam_full.py from the averaged-S anchor to the per-cell NNDSVD
anchor, WITHOUT touching anything else in DCAM.

Why
---
The rank diagnostic showed run_dcam_full.compute_coactivation_matrix builds S
from the SPATIAL MEAN of activations, collapsing the layer (participation
ratio ~1.6, entropy rank ~5). The per-cell covariance the encoder actually
sees has participation ratio ~17, entropy rank ~217. The anchor Pi0 was
therefore built from a near-rank-5 shadow of a ~200-rank object.

The verification run further showed:
  * per-cell NMF with RANDOM init is seed-sensitive (cross-seed atom
    distance ~0.12, flat in D);
  * per-cell NMF with NNDSVD init is DETERMINISTIC (cross-seed distance
    0.0000) and non-degenerate (nu ~0.13).

So the corrected anchor is: per-cell covariance + NNDSVD-initialised
symmetric NMF. This module wires that into DCAM as a one-call replacement.

What this patch changes -- and what it deliberately does NOT
------------------------------------------------------------
CHANGES: only the anchor. Pi0 is now built from S_cell via NNDSVD-NMF.
UNCHANGED: the DCAM encoder ReLU(Pi A), the tied pinv decoder
Pi_dagger z, the three loss terms (recon + anchor + L1), the simplex
projection, the training loop, the seeds. Pi is still a [D, C] simplex
matrix; init_from_anchor still projects Pi0 to the simplex. Because only
the anchor changes, any change in DCAM's accuracy/faithfulness after the
retrain is attributable to the anchor and nothing else.

IMPORTANT -- this is the decisive experiment, not a guaranteed fix
------------------------------------------------------------------
DCAM-untied already showed the simplex encoder + a FREE decoder still gave
23% accuracy. A better anchor cannot repair an encoder that structurally
cannot reconstruct. This retrain answers one question:
  * if accuracy moves off 23% -> the anchor mattered for faithfulness too;
  * if accuracy stays ~23%   -> the simplex encoder is the binding
                                constraint, confirmed.
Either outcome is a publishable result. Do not assume the first.

============================================================================
HOW TO APPLY  (two-line edit to YOUR run_dcam_full.py)
============================================================================
In run_dcam_full.py, find this block in main() (the "anchor Pi0" section):

    # ---- anchor Pi0  (seed-free) ---------------------------------------
    print(f"\\n{'='*80}\\nBuilding seed-free anchor Pi0\\n{'='*80}")
    S = compute_coactivation_matrix(act_chunks)
    Pi0, nu, D_eff = build_anchor(
        S, D, nmf_iters=args.nmf_iters, merge_tol=args.merge_tol,
        seed=args.anchor_seed)
    Pi0 = Pi0.to(device)

Replace the THREE lines from `S = compute_coactivation_matrix(...)` through
`Pi0, nu, D_eff = build_anchor(...)` with:

    from patch_dcam_percell_anchor import build_dcam_anchor_percell
    Pi0, nu, D_eff = build_dcam_anchor_percell(
        act_chunks, D,
        nmf_iters=max(args.nmf_iters, 8000),
        merge_tol=args.merge_tol,
        device=str(device),
        subsample_cells=200000)

Keep the `Pi0 = Pi0.to(device)` line after it. Everything else in main()
stays exactly as it is.

Optionally also add a CLI flag so you can switch back for comparison:

    parser.add_argument('--anchor_mode', type=str, default='percell',
                        choices=['averaged', 'percell'],
                        help="'averaged' = original spatial-mean S anchor; "
                             "'percell' = corrected per-cell NNDSVD anchor.")

and branch on args.anchor_mode around the two anchor constructions. This
lets one code path run BOTH the baseline and the fix for a clean A/B.

============================================================================
"""

import torch
from dcam_percell_anchor import build_anchor_percell


def build_dcam_anchor_percell(act_chunks, D, nmf_iters=8000,
                              merge_tol=1e-2, device="cpu",
                              subsample_cells=200000, anchor_seed=0):
    """Build the DCAM anchor Pi0 from the per-cell covariance with a
    deterministic NNDSVD-initialised NMF.

    Drop-in for the
        S = compute_coactivation_matrix(act_chunks)
        Pi0, nu, D_eff = build_anchor(S, D, ...)
    pair in run_dcam_full.main().

    Args mirror what main() already has in scope:
        act_chunks: the cached activation chunks (already in memory).
        D: requested atom count (run_dcam_full caps D <= C; here D is
           whatever main() passes -- typically MODEL_CONFIGS default_D).
        nmf_iters: NMF iterations. The per-cell S is high-rank; >= 8000
           recommended. nmf_symmetric early-stops on a plateau anyway.
        merge_tol: non-degeneracy merge tolerance (same as build_anchor).
        device: 'cuda' / 'cpu' for the NMF math.
        subsample_cells: spatial cells kept per chunk when forming S_cell.
        anchor_seed: kept for signature parity with build_anchor; with
           NNDSVD init the result is seed-FREE, so this only affects the
           (irrelevant) cell-subsampling RNG.

    Returns:
        (Pi0, nu, D_eff) -- exactly what build_anchor returns, so the rest
        of run_dcam_full.main() needs no further change.
    """
    print(f"\n{'='*80}")
    print("Building DCAM anchor Pi0 -- PER-CELL covariance + NNDSVD NMF")
    print("  (corrected anchor: averaged-S collapsed the layer to rank ~5;")
    print("   per-cell S has effective rank ~200. NNDSVD init -> seed-free.)")
    print(f"{'='*80}")

    Pi0, nu, D_eff = build_anchor_percell(
        act_chunks, D,
        nmf_iters=nmf_iters,
        merge_tol=merge_tol,
        seed=anchor_seed,
        subsample_cells=subsample_cells,
        device=device,
        return_trace=False,
        init="nndsvd",          # deterministic anchor -- the verified fix
    )

    print(f"[anchor] per-cell NNDSVD anchor ready: D_eff={D_eff}, "
          f"nu={nu:.4e}")
    if nu < 1e-2:
        print(f"  [WARNING] nu={nu:.2e} is small -- anchor atoms are nearly "
              f"degenerate. Consider a smaller D.")
    return Pi0, nu, D_eff