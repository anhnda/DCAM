"""
test_unified_cam.py
===================
Self-test for unified_cam_joint.py. Runs on SYNTHETIC activations only -- no
backbone, no cache, no ImageNet -- so it verifies the math of the joint solver
in isolation before you spend GPU time on real models.

Each test checks a specific CLAIM from the paper:

  T1  Proposition 1     -- block coordinate descent gives a MONOTONE
                           non-increasing J sequence and converges.
  T2  Ky Fan / P-block  -- with alpha fixed, the P-block returns exactly the
                           top-D eigenspace of the blended Sigma (objective
                           equals the tail-eigenvalue sum).
  T3  corner: DCAM      -- phi=id, w=global, lambda=1, alpha free reproduces a
                           low-rank additive decomposition: ReLU(bias+sum beta z)
                           matches the true Grad-CAM to high cosine on a
                           genuinely low-rank synthetic signal.
  T4  additive identity -- for phi=id, bias + sum_d beta_d z_d equals
                           < alpha, mu + P(a-mu) > cell-for-cell (the exact
                           linear-slice identity of section 5).
  T5  scope caveat      -- for phi=rbf the result flags additive_exact=False
                           and decompose_query refuses to emit beta/z.
  T6  seed-invariance   -- on a signal with a clean eigengap, multi-restart
                           solving reports subspace_agreement ~ 1 and
                           seed_invariant=True (paper section 4 protocol).
  T7  Option A == B     -- Riemannian joint descent and block descent reach
                           the same objective value (up to tolerance) on a
                           convex-enough instance.

Run:
    python test_unified_cam.py
Exit code 0 = all claims hold.
"""

import sys
import numpy as np
import torch

sys.path.append('.')
from unified_cam_joint import (
    SolveConfig, solve, decompose_query, build_moments, solve_P_block,
    class_second_moment, objective_value, FeatureMap, locality_weights,
    config_for_corner,
)

torch.manual_seed(0)
np.random.seed(0)
DT = torch.float64


def make_synthetic(N=40, C=64, H=14, W=14, true_rank=8, noise=0.02):
    """Build a bank of activations with a KNOWN low-rank structure.

    Each image's activation cells live (mostly) in a shared `true_rank`
    subspace + small noise -> the solver SHOULD find a rank-~true_rank basis
    that reconstructs Grad-CAM well. Returns (acts_norm, alpha_bank).
    """
    # shared low-rank basis
    U = torch.linalg.qr(torch.randn(C, true_rank, dtype=DT))[0]  # [C, r]
    acts = torch.zeros(N, C, H, W, dtype=DT)
    for n in range(N):
        coeff = torch.randn(true_rank, H * W, dtype=DT) * (1.0 + 0.5 * n / N)
        cells = (U @ coeff).T                          # [HW, C]
        cells = cells + noise * torch.randn_like(cells)
        cells = cells.clamp(min=0.0)                   # activations are >=0
        acts[n] = cells.T.reshape(C, H, W)
    # normalize per channel to [0,1]-ish (mimics the cache normalization)
    for c in range(C):
        ch = acts[:, c]
        m = ch.max()
        if m > 1e-8:
            acts[:, c] = ch / m
    # per-image Grad-CAM alpha: random positive channel weights
    alpha_bank = torch.rand(N, C, dtype=DT) + 0.1
    return acts, alpha_bank


def _ok(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


def test_monotone():
    print("\nT1  Proposition 1: block descent is monotone non-increasing")
    acts, alpha_bank = make_synthetic()
    cfg = config_for_corner("dcam", D=8)
    cfg.dtype = "float64"; cfg.device = "cpu"; cfg.max_iter = 30
    res = solve(acts, alpha_bank, alpha_bank[0], 0, cfg, solver="block")
    J = res.J_history
    # allow tiny float slack
    monotone = all(J[i + 1] <= J[i] + 1e-6 * (abs(J[i]) + 1e-9)
                   for i in range(len(J) - 1))
    converged = abs(J[-1] - J[-2]) < 1e-4 * (abs(J[-2]) + 1e-9) if len(J) > 1 \
        else True
    return (_ok("J non-increasing", monotone,
                f"{len(J)} iters, J: {J[0]:.4e} -> {J[-1]:.4e}")
            and _ok("J converged", converged))


def test_kyfan():
    print("\nT2  Ky Fan: P-block returns the top-D eigenspace")
    acts, alpha_bank = make_synthetic()
    phi = FeatureMap("id", in_dim=acts.shape[1], dtype=DT)
    w = torch.ones(acts.shape[0], dtype=DT) / acts.shape[0]
    mb = build_moments(acts, alpha_bank, phi, w)
    D = 8
    V_D, evals, gap = solve_P_block(mb.Sigma_phi, D)
    # objective tr((I-P)Sigma(I-P)) should equal sum of tail eigenvalues
    P = V_D @ V_D.T
    I = torch.eye(P.shape[0], dtype=DT)
    ImP = I - P
    obj = float(torch.trace(ImP @ mb.Sigma_phi @ ImP).item())
    tail = float(evals[D:].sum().item())
    match = abs(obj - tail) < 1e-6 * (abs(tail) + 1e-9)
    return _ok("tr((I-P)Sigma(I-P)) == sum of tail eigenvalues", match,
               f"obj={obj:.6e}  tail-sum={tail:.6e}")


def test_dcam_corner():
    print("\nT3  DCAM corner: low-rank additive decomposition is faithful")
    acts, alpha_bank = make_synthetic(true_rank=8, noise=0.01)
    cfg = config_for_corner("dcam", D=12)
    cfg.dtype = "float64"; cfg.device = "cpu"
    res = solve(acts, alpha_bank, alpha_bank[0], 0, cfg, solver="block")
    dec = decompose_query(res, acts[0:1], alpha_bank[0])
    cos = dec["spatial_cos"]
    return _ok("ReLU(bias+sum beta_d z_d) ~ true Grad-CAM (cos>0.95)",
               cos > 0.95, f"spatial cos={cos:.4f}")


def test_additive_identity():
    print("\nT4  Exact additive identity for phi=id (paper section 5)")
    acts, alpha_bank = make_synthetic()
    cfg = config_for_corner("dcam", D=10)
    cfg.dtype = "float64"; cfg.device = "cpu"
    res = solve(acts, alpha_bank, alpha_bank[0], 0, cfg, solver="block")
    dec = decompose_query(res, acts[0:1], alpha_bank[0])

    # independently compute < alpha, mu + P(a-mu) > and compare to L_hat
    A = acts[0:1].to(DT)
    _, C, H, W = A.shape
    cells = A[0].permute(1, 2, 0).reshape(-1, C)
    mu = res.mu[:C]
    V_D = res.V_D[:C, :]
    centred = cells - mu
    proj = (centred @ V_D) @ V_D.T + mu                # mu + P(a-mu)
    L_hat_direct = (alpha_bank[0].to(DT).view(1, C) * proj).sum(dim=1)
    L_hat_decomp = dec["L_hat"].flatten().to(DT)
    err = float((L_hat_direct - L_hat_decomp).abs().max().item())
    return _ok("bias+sum beta_d z_d == <alpha, mu+P(a-mu)> cell-for-cell",
               err < 1e-8, f"max abs diff={err:.2e}")


def test_kernel_scope_caveat():
    print("\nT5  Scope caveat: kernel phi disables the additive identity")
    acts, alpha_bank = make_synthetic()
    cfg = config_for_corner("kpca_cam", D=8)
    cfg.dtype = "float64"; cfg.device = "cpu"; cfg.rff_dim = 256
    res = solve(acts, alpha_bank, alpha_bank[0], 0, cfg, solver="block")
    flagged = (res.diagnostics["additive_exact"] is False
               and "scope_caveat" in res.diagnostics)
    dec = decompose_query(res, acts[0:1], alpha_bank[0])
    no_beta = "beta" not in dec and dec["additive_exact"] is False
    return (_ok("result flags additive_exact=False + scope_caveat", flagged)
            and _ok("decompose_query emits no beta/z for kernel", no_beta))


def test_seed_invariance():
    print("\nT6  Seed-invariance: multi-restart agreement on a clean eigengap")
    # strong low-rank signal -> clean gap -> restarts should agree
    acts, alpha_bank = make_synthetic(true_rank=6, noise=0.005)
    cfg = config_for_corner("dcam", D=6)
    cfg.dtype = "float64"; cfg.device = "cpu"; cfg.n_restarts = 4
    res = solve(acts, alpha_bank, alpha_bank[0], 0, cfg, solver="block")
    agree = res.diagnostics["subspace_agreement"]
    gap = res.diagnostics["eigengap"]
    return (_ok("subspace agreement across restarts ~ 1", agree > 0.99,
                f"agreement={agree:.4f}")
            and _ok("positive eigengap certificate", gap > 1e-8,
                    f"gap={gap:.3e}")
            and _ok("seed_invariant verdict True",
                    res.diagnostics["seed_invariant"]))


def test_option_A_equals_B():
    print("\nT7  Option A (Riemannian) reaches the same J as Option B (block)")
    acts, alpha_bank = make_synthetic(true_rank=8, noise=0.01)
    cfg = config_for_corner("dcam", D=10)
    cfg.dtype = "float64"; cfg.device = "cpu"
    cfg.max_iter = 60; cfg.lr = 0.02
    res_b = solve(acts, alpha_bank, alpha_bank[0], 0, cfg, solver="block")
    res_a = solve(acts, alpha_bank, alpha_bank[0], 0, cfg, solver="joint")
    Jb = res_b.diagnostics["J_final"]
    Ja = res_a.diagnostics["J_final"]
    # joint descent is first-order; allow a loose relative tolerance
    rel = abs(Ja - Jb) / (abs(Jb) + 1e-9)
    return _ok("J_joint ~ J_block (rel diff < 5%)", rel < 0.05,
               f"J_block={Jb:.4e}  J_joint={Ja:.4e}  rel={rel:.3f}")


def main():
    print("=" * 72)
    print("Self-test: unified_cam_joint.py  (synthetic data, no backbone)")
    print("=" * 72)
    tests = [test_monotone, test_kyfan, test_dcam_corner,
             test_additive_identity, test_kernel_scope_caveat,
             test_seed_invariance, test_option_A_equals_B]
    results = []
    for t in tests:
        try:
            results.append(bool(t()))
        except Exception as e:
            print(f"  [FAIL] {t.__name__} raised: {type(e).__name__}: {e}")
            results.append(False)
    print("\n" + "=" * 72)
    n_pass = sum(results)
    print(f"  {n_pass}/{len(results)} test groups passed")
    print("=" * 72)
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()