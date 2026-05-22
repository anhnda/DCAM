"""
dcam_percell_anchor.py
======================
Fix for DCAM's anchor construction, justified by the rank diagnostic.

The problem (measured)
----------------------
run_dcam_full.compute_coactivation_matrix builds S from the SPATIAL MEAN of
each activation map (Abar = mean over H,W). The diagnostic showed this
collapses the layer's structure:

    averaged S : participation ratio 1.62,  entropy rank   5,  eigs->90%  32
    per-cell S : participation ratio 16.9,  entropy rank 217,  eigs->90% 614

The encoder ReLU(Pi A) operates PER SPATIAL CELL, so it must reconstruct a
~200-effective-rank signal. But Pi0 was built by NMF of a rank-~5 summary.
The anchor was stably reproducible (NMF of a near-rank-1 matrix is trivial)
but it was anchored to the wrong object -- a spatially-collapsed shadow that
discards most of what the encoder sees.

The fix
-------
Build the anchor from the PER-CELL channel covariance instead:

    S_cell[k,k'] = E_cell[ a_k a_k' ]      a = activations at one (h,w) cell

i.e. accumulate over every spatial cell, not over per-image spatial means.
Everything else in build_anchor (symmetric NMF, simplex projection,
non-degeneracy merge) is unchanged.

This module provides:
  * compute_coactivation_matrix_percell(): the corrected S.
  * build_anchor_percell(): thin wrapper that calls the original build_anchor
    on the corrected S, with stronger NMF defaults (the diagnostic showed
    build_anchor's 500 iters were undertrained; 4000 + normalization is safe).

How to use it in run_dcam_full.py
---------------------------------
In main(), replace the anchor block:

    # OLD:
    # S = compute_coactivation_matrix(act_chunks)
    # Pi0, nu, D_eff = build_anchor(S, D, nmf_iters=args.nmf_iters, ...)

    # NEW:
    from dcam_percell_anchor import build_anchor_percell
    Pi0, nu, D_eff = build_anchor_percell(
        act_chunks, D, nmf_iters=args.nmf_iters,
        merge_tol=args.merge_tol, seed=args.anchor_seed,
        subsample_cells=args.percell_subsample)

Nothing else in DCAM changes: the encoder, the tied pinv decoder, the three
loss terms, and the simplex projection are all untouched. This isolates the
anchor as the single variable, so any change in DCAM's accuracy is
attributable to the anchor and nothing else.

IMPORTANT -- this is a partial fix
----------------------------------
This corrects the anchor only. The DCAM-untied experiment showed the
simplex encoder + tied pinv decoder lost accuracy (23%) even with a FREE
decoder. A better anchor cannot fix a decoder/encoder that structurally
cannot reconstruct. Treat this as: "give DCAM the best possible anchor, then
re-measure." If accuracy is still far below CSAE's ~68%, the binding
constraint is the simplex encoder, not the anchor -- and the paper's claim
must move accordingly. The verification script reports this explicitly.
"""

import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

# Reuse the original, unchanged anchor machinery.
from run_dcam_full import build_anchor, project_rows_to_simplex


def compute_coactivation_matrix_percell(
        activation_chunks: List[torch.Tensor],
        subsample_cells: int = None,
        seed: int = 0) -> torch.Tensor:
    """Per-cell channel covariance S_cell = E_cell[ a a^T ].

    Unlike compute_coactivation_matrix (which averages each activation map
    over H,W first), this treats EVERY spatial cell as an independent
    C-vector. This is the covariance the per-cell encoder ReLU(Pi A) actually
    has to model.

    Memory: chunks are processed one at a time; each [n,C,H,W] chunk becomes
    [n*H*W, C] cells, optionally subsampled, accumulated into a C x C matrix,
    then freed. Never holds more than one chunk.

    Args:
        activation_chunks: list of [n, C, H, W] tensors (the cached A's).
        subsample_cells: if set, randomly keep at most this many cells per
            chunk. With H=W=14 and n=100 there are 19600 cells/chunk; the
            full set over 500 chunks is ~10M cells, which is fine, but
            subsampling speeds the accumulation with no real accuracy loss
            (S_cell is a population covariance).
        seed: rng seed for the subsampling (kept fixed for reproducibility).
    Returns:
        S_cell: [C, C] symmetric PSD covariance, float32.
    """
    g = torch.Generator().manual_seed(seed)
    C = activation_chunks[0].shape[1]
    S = torch.zeros(C, C, dtype=torch.float64)
    n_cells = 0

    for chunk in tqdm(activation_chunks, desc="Per-cell co-activation"):
        a = chunk
        if a.dim() == 4:
            n, Cc, H, W = a.shape
            # [n, C, H, W] -> [n*H*W, C]
            cells = a.permute(0, 2, 3, 1).reshape(-1, Cc).double()
        elif a.dim() == 2:
            cells = a.double()
        else:
            raise ValueError(f"unexpected activation ndim {a.dim()}")

        if subsample_cells is not None and cells.shape[0] > subsample_cells:
            idx = torch.randperm(cells.shape[0], generator=g)[:subsample_cells]
            cells = cells[idx]

        S += cells.t() @ cells
        n_cells += cells.shape[0]
        del cells

    S /= max(n_cells, 1)
    print(f"  [per-cell S] accumulated {n_cells} spatial cells, "
          f"S_cell shape {tuple(S.shape)}")
    return S.float()


def build_anchor_percell(activation_chunks: List[torch.Tensor], D: int,
                         nmf_iters: int = 4000, merge_tol: float = 1e-2,
                         seed: int = 0, subsample_cells: int = 200000
                         ) -> Tuple[torch.Tensor, float, int]:
    """Build the DCAM anchor Pi0 from the PER-CELL covariance.

    Drop-in replacement for the
        S = compute_coactivation_matrix(act_chunks)
        Pi0, nu, D_eff = build_anchor(S, D, ...)
    pair in run_dcam_full.main().

    Note on nmf_iters: the diagnostic showed build_anchor's default 500 was
    undertrained (error still descending). 4000 is the safe default here.
    build_anchor itself is called unchanged -- only the S it factorizes is
    corrected.

    Returns the same triple as build_anchor: (Pi0, nu, D_eff).
    """
    print(f"\n[build_anchor_percell] building per-cell covariance "
          f"(subsample_cells={subsample_cells})")
    S_cell = compute_coactivation_matrix_percell(
        activation_chunks, subsample_cells=subsample_cells, seed=seed)

    # Report the effective rank so the log makes the fix's rationale visible.
    with torch.no_grad():
        ev = torch.linalg.eigvalsh(S_cell.double())
        ev = ev[ev > 0]
        pr = float((ev.sum() ** 2 / (ev ** 2).sum()).item())
    print(f"[build_anchor_percell] per-cell S participation ratio = "
          f"{pr:.1f}  (averaged-S was ~1.6; higher = more structure kept)")

    print(f"[build_anchor_percell] running symmetric NMF "
          f"(D={D}, nmf_iters={nmf_iters})")
    Pi0, nu, D_eff = build_anchor(
        S_cell, D, nmf_iters=nmf_iters, merge_tol=merge_tol, seed=seed)
    return Pi0, nu, D_eff