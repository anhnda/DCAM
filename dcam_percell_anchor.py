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

Solver note (important)
-----------------------
build_anchor's symmetric NMF uses multiplicative updates with a [0,1] random
init and NO rescaling. On the per-cell S -- whose entries are much larger
than the averaged S's, because no spatial averaging shrinks them -- that NMF
STALLS: it reported rel_err ~285 (a relative error should be in [0,~1]; 285
means the factorization never converged). So this module does NOT call
build_anchor's NMF. It uses check_anchor_nonnegativity.nmf_symmetric, which
rescales the init to S's magnitude and correlation-normalizes S first --
the two things that make NMF converge on a large-magnitude matrix. The
non-degeneracy merge is reimplemented locally (_merge_near_duplicate_atoms)
so nothing depends on build_anchor's weak NMF path.

This module provides:
  * compute_coactivation_matrix_percell(): the corrected S.
  * build_anchor_percell(): per-cell S + converging NMF + simplex projection
    + non-degeneracy merge. Reports the NMF rel_err and warns if it did not
    converge -- so a stability verdict is never read off an unconverged fit.

How to use it in run_dcam_full.py
---------------------------------
In main(), replace the anchor block:

    # OLD:
    # S = compute_coactivation_matrix(act_chunks)
    # Pi0, nu, D_eff = build_anchor(S, D, nmf_iters=args.nmf_iters, ...)

    # NEW:
    from dcam_percell_anchor import build_anchor_percell
    Pi0, nu, D_eff = build_anchor_percell(
        act_chunks, D, nmf_iters=max(args.nmf_iters, 8000),
        merge_tol=args.merge_tol, seed=args.anchor_seed,
        subsample_cells=200000, device=str(device))

Note: bump nmf_iters to >= 8000 -- the per-cell S is large-magnitude and
high-rank, so it needs more iterations than the averaged S did. Pass
device=str(device) so the NMF runs on the same GPU as training. Nothing
else in DCAM changes: encoder, tied pinv decoder, three loss terms, simplex
projection -- all untouched. This isolates the anchor as the single variable.

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

# Reuse the original simplex projection from DCAM.
from run_dcam_full import project_rows_to_simplex
# Reuse the CONVERGING NMF solver from the diagnostic script. This solver
# rescales the init to S's magnitude and correlation-normalizes S before
# factorizing -- exactly the two things build_anchor's multiplicative-update
# NMF lacked, which is why build_anchor's NMF stalled at rel_err ~285 on the
# (large-magnitude) per-cell S. We must NOT use build_anchor's NMF here.
from check_anchor_nonnegativity import nmf_symmetric


def _merge_near_duplicate_atoms(Pi0: torch.Tensor, merge_tol: float
                                ) -> Tuple[torch.Tensor, float, int]:
    """Non-degeneracy check: merge anchor rows closer than merge_tol (L2),
    re-project the merged row to the simplex, and report nu = min pairwise
    atom distance. Reimplemented from build_anchor so this module does not
    depend on build_anchor's (weak-NMF) code path.

    Args:
        Pi0: [D, C] anchor, rows already on the L1-simplex.
        merge_tol: rows closer than this are merged.
    Returns:
        (Pi0_merged [D_eff, C], nu, D_eff).
    """
    D = Pi0.shape[0]
    keep = list(range(D))
    merged = True
    while merged:
        merged = False
        for a_i in range(len(keep)):
            for b_i in range(a_i + 1, len(keep)):
                da, db = keep[a_i], keep[b_i]
                if torch.norm(Pi0[da] - Pi0[db]).item() < merge_tol:
                    Pi0[da] = project_rows_to_simplex(
                        ((Pi0[da] + Pi0[db]) * 0.5).unsqueeze(0)).squeeze(0)
                    keep.pop(b_i)
                    merged = True
                    break
            if merged:
                break
    Pi0 = Pi0[keep]
    D_eff = Pi0.shape[0]
    if D_eff >= 2:
        dmat = torch.cdist(Pi0, Pi0) + torch.eye(D_eff) * 1e9
        nu = float(dmat.min().item())
    else:
        nu = float('inf')
    return Pi0, nu, D_eff


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
                         seed: int = 0, subsample_cells: int = 200000,
                         device: str = "cpu", return_trace: bool = False,
                         init: str = "random"):
    """Build the DCAM anchor Pi0 from the PER-CELL covariance.

    Drop-in replacement for the
        S = compute_coactivation_matrix(act_chunks)
        Pi0, nu, D_eff = build_anchor(S, D, ...)
    pair in run_dcam_full.main().

    Solver: this uses check_anchor_nonnegativity.nmf_symmetric, NOT
    build_anchor's NMF. nmf_symmetric (a) rescales the random init so
    W W^T starts at S's magnitude and (b) correlation-normalizes S before
    factorizing. build_anchor's NMF does neither, which is why it stalled at
    rel_err ~285 on the large-magnitude per-cell S.

    Args:
        init: 'random' -- seed-dependent NMF init (default).
              'nndsvd' -- deterministic SVD-based init; the anchor is then
                          seed-FREE by construction. Use this if the
                          random-init per-cell anchor is seed-sensitive and
                          you want to test whether a deterministic init
                          recovers reproducibility.
        return_trace: if True, also return (trace, nmf_err).
    Returns:
        (Pi0, nu, D_eff)  or  (Pi0, nu, D_eff, trace, nmf_err).
    """
    print(f"\n[build_anchor_percell] building per-cell covariance "
          f"(subsample_cells={subsample_cells})")
    S_cell = compute_coactivation_matrix_percell(
        activation_chunks, subsample_cells=subsample_cells, seed=seed)

    # Report effective rank so the log makes the fix's rationale visible.
    with torch.no_grad():
        ev = torch.linalg.eigvalsh(S_cell.double())
        ev = ev[ev > 0]
        pr = float((ev.sum() ** 2 / (ev ** 2).sum()).item())
    print(f"[build_anchor_percell] per-cell S participation ratio = "
          f"{pr:.1f}  (averaged-S was ~1.6; higher = more structure kept)")

    # --- symmetric NMF with the CONVERGING solver -----------------------
    print(f"[build_anchor_percell] running symmetric NMF via nmf_symmetric "
          f"(D={D}, iters={nmf_iters}, init={init}, "
          f"normalize=correlation, device={device})")
    W, nmf_err, trace = nmf_symmetric(
        S_cell.numpy(), D, iters=nmf_iters, restarts=1,
        normalize=True, seed=seed, verbose=True, device=device, init=init)

    # Convergence sanity check: nmf_err is rel_err vs the ORIGINAL S, in
    # [0, ~1] for a converged fit. build_anchor's stalled NMF reported ~285.
    print(f"[build_anchor_percell] NMF final rel_err = {nmf_err:.4f}")
    if nmf_err > 1.0:
        print(f"  [WARNING] rel_err {nmf_err:.2f} > 1.0 -- the NMF has NOT "
              f"converged. The anchor (and any stability verdict computed "
              f"from it) is NOT trustworthy. Raise --nmf_iters, or inspect "
              f"the convergence trace.")
    elif trace and len(trace) >= 2 and abs(trace[-1][1] - trace[-2][1]) > 1e-3:
        print(f"  [NOTE] NMF still descending at the last checkpoint "
              f"({trace[-2][1]:.4f} -> {trace[-1][1]:.4f}); consider more "
              f"iterations for a fully settled anchor.")

    # --- atoms = columns of W, L1-normalized as simplex rows ------------
    Pi0 = torch.as_tensor(W, dtype=torch.float32).t().clone()   # [D, C]
    Pi0 = project_rows_to_simplex(Pi0)

    # --- non-degeneracy merge ------------------------------------------
    Pi0, nu, D_eff = _merge_near_duplicate_atoms(Pi0, merge_tol)
    print(f"[build_anchor_percell] requested D={D}, after merge D_eff={D_eff}, "
          f"non-degeneracy nu={nu:.4e}")
    if D_eff < D:
        print(f"  [anchor] {D - D_eff} near-duplicate atom(s) merged "
              f"(merge_tol={merge_tol}).")

    if return_trace:
        return Pi0, nu, D_eff, trace, nmf_err
    return Pi0, nu, D_eff