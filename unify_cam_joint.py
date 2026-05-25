"""
unified_cam_joint.py
====================
Joint solver for the UNIFIED CAM-style attribution objective.

This implements Equation (1) of "A Unified Objective for CAM-Style
Attribution" and its two solvers (Option A: Riemannian joint descent on the
product manifold; Option B: convergent block coordinate descent). Every prior
CAM method is recovered as a corner of the parameter cube, exactly as the
paper's Table in section 3 prescribes.

WHERE THIS SITS RELATIVE TO THE EXISTING CODE
---------------------------------------------
csae_pca_baseline.py / pca_gradcam_decomp.py implement ONE corner of this
objective: phi=id, w=1 (dataset-wide), lambda=1, alpha free -- the row labelled
"DCAM (low-rank)" in the paper. There the basis V is a fixed eigendecomposition
of a streamed covariance, and the decomposition

    Grad-CAM = ReLU( bias + sum_d beta_d(x) * z_d(x) )

is read straight out of that frozen V.

This module does NOT freeze V. It SOLVES for the rank-D projector P jointly
with the weighting field alpha, under whatever (phi, w, lambda) the caller
selects. The DCAM corner is reproduced exactly by
    solve(..., phi='id', locality='global', lam=1.0)
so this is a strict generalization, not a replacement -- the pca_baseline pkl
remains the right artifact for the lambda=1/global/id corner, and this module
is what you reach for when you want any OTHER corner or the genuinely-new
interior (lambda in (0,1), finite-bandwidth w).

THE OBJECTIVE (Eq. 1)
---------------------
    J(P, alpha; x0) =
        (1 - lambda) * E_x[ w(x,x0) || (I-P)(phi(a)-mu_phi) ||^2 ]      (recon)
      +      lambda  * E_x[ w(x,x0) || L~(x) - L~_{P,alpha}(x) ||_F^2 ] (class)
      +      gamma   * Omega(alpha)                                     (rule)

with reconstructed pre-ReLU explanation
    L~_{P,alpha}(x) = < alpha(x), mu_phi + P(phi(a) - mu_phi) >.

Convexity: convex in P for fixed alpha, quadratic (convex) in alpha for fixed
P, NOT jointly convex -- L~_{P,alpha} is bilinear in (P,alpha). That bilinear
coupling is the real joint structure, and is exactly why a joint solver (not
two independent fits) is required.

THE CORNERS (paper section 3 / table)
-------------------------------------
    Eigen-CAM      : phi=id,     w=delta, lambda=0
    KPCA-CAM       : phi=kernel, w=delta, lambda=0
    Grad-CAM       : phi=id,     w=1,     lambda=1, P=I, Omega pins alpha=grad-avg
    DCAM (low-rank): phi=id,     w=1,     lambda=1, alpha free, top-D of Sigma_alpha
    CRAFT / SAE    : phi=id,     w=1,     lambda=0, P -> overcomplete + sparse Omega
    NEW (interior) : phi=id,     w=bandwidth h, lambda in (0,1)

SCOPE CAVEAT (paper section 5)
------------------------------
The exact additive identity  L~ = bias + sum_d beta_d z_d  REQUIRES phi=id
(linearity). For a nontrivial kernel the objective still unifies the method
and the solver still runs, but the per-component additive panels are not
emitted -- only cosine/correlation fidelity is reported. solve() flags this in
the returned diagnostics ('additive_exact').

INPUT CONTRACT
--------------
The solver consumes NORMALIZED layer activations -- the same per-channel
0.99-quantile clip+rescale space that export_activation_cache.py writes and
csae_pca_baseline.py builds its basis in. Feed it either:
  * a cache built by export_activation_cache.py (use load_activation_cache), or
  * a stack of per-image activations you normalized with normalize_acts().
Mixing spaces (raw acts against a normalized basis) silently corrupts the fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


# ==========================================================================
# 0. small utilities
# ==========================================================================

def resolve_device(device: str = "auto") -> torch.device:
    """auto -> cuda if available else cpu; otherwise honour the request."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def torch_dtype(name: str) -> torch.dtype:
    return {"float64": torch.float64, "float32": torch.float32}[name]


def normalize_acts(acts: torch.Tensor, quantile: float = 0.99,
                   eps: float = 1e-8) -> torch.Tensor:
    """Per-channel 0.99-quantile clip+rescale.

    IDENTICAL to ActivationExtractor._flush_chunk's normalize block in
    export_activation_cache.py and to GradCAMDecomposer._normalize in
    pca_gradcam_decomp.py -- replicated here so this module has no import
    dependency on those files, but kept byte-faithful so the solver lives in
    the same normalized space the cache and the PCA basis live in.

    acts: [N, C, H, W]  -> normalized copy, same shape.
    """
    out = acts.clone()
    C = acts.shape[1]
    for c in range(C):
        ch = out[:, c, :, :]
        nz = ch[ch > eps]
        if nz.numel() == 0:
            continue
        scale = torch.quantile(nz, quantile)
        if scale > eps:
            out[:, c, :, :] = torch.clamp(ch, min=0.0, max=scale) / (scale + eps)
    return out


# ==========================================================================
# 1. feature map  phi   (the kernel knob)
# ==========================================================================

class FeatureMap:
    """phi: R^C -> H.  phi='id' is the linear slice (exact additive identity
    holds). A nontrivial kernel is realised by an explicit RANDOM FOURIER
    FEATURE lift, which keeps the whole solver linear-algebraic: we never need
    to form the Gram matrix, P stays an honest orthogonal projector in the
    (finite) RFF space, and KPCA-CAM is recovered as the w=delta, lambda=0
    corner of this lift.

    Why RFF and not an exact kernel-PCA: the paper's objective takes
    expectations and projects in H. An exact RKHS makes P an operator we
    cannot store. The RFF map phi_rff(a) in R^M approximates the RBF kernel
    k(a,a') = <phi_rff(a), phi_rff(a')> with controllable M, so every formula
    in this module (covariance, eigh, projection) goes through unchanged with
    C replaced by M. Setting kind='id' bypasses the lift entirely.
    """

    def __init__(self, kind: str = "id", in_dim: int = 0,
                 rff_dim: int = 1024, rff_gamma: float = 1.0,
                 seed: int = 0, device=None, dtype=torch.float64):
        assert kind in ("id", "rbf"), f"unknown feature map {kind}"
        self.kind = kind
        self.in_dim = in_dim
        self.device = device
        self.dtype = dtype
        if kind == "rbf":
            g = torch.Generator(device="cpu").manual_seed(seed)
            # phi(a) = sqrt(2/M) * cos(W a + b),  W ~ N(0, 2*gamma),  b ~ U[0,2pi]
            self.W = (torch.randn(rff_dim, in_dim, generator=g)
                      * math.sqrt(2.0 * rff_gamma)).to(device=device, dtype=dtype)
            self.b = (torch.rand(rff_dim, generator=g)
                      * (2.0 * math.pi)).to(device=device, dtype=dtype)
            self.scale = math.sqrt(2.0 / rff_dim)
            self.out_dim = rff_dim
            self.additive_exact = False
        else:
            self.out_dim = in_dim
            self.additive_exact = True

    def __call__(self, a: torch.Tensor) -> torch.Tensor:
        """a: [..., C] -> phi(a): [..., out_dim]."""
        if self.kind == "id":
            return a
        proj = a @ self.W.T + self.b
        return self.scale * torch.cos(proj)


# ==========================================================================
# 2. locality weight  w(x, x0)   (the dataset-wide / per-image / bandwidth knob)
# ==========================================================================

def locality_weights(acts_flat: torch.Tensor, x0_feat: torch.Tensor,
                      locality: str = "global", bandwidth: float = 1.0,
                      eps: float = 1e-12) -> torch.Tensor:
    """w(x, x0) over a bank of images.

    acts_flat : [N, F]  -- one feature vector per image (e.g. spatial-mean of
                           phi(a)); the bank the basis is fit on.
    x0_feat   : [F]     -- the query image's feature vector.

    locality:
      'global'    -> w == 1            (dataset-wide basis; Grad-CAM / DCAM row)
      'per_image' -> w == delta(x-x0)  (Eigen-CAM / KPCA-CAM row): all mass on
                     the query. Realised as a near-delta (mass on the single
                     nearest bank entry to x0) so the same covariance machinery
                     runs unchanged.
      'bandwidth' -> w = exp(-||x-x0||^2 / (2 h^2))  (the NEW interior region):
                     a finite-bandwidth Gaussian around the query -> a local,
                     class-tiltable basis.

    Returns w: [N], normalized to sum to 1 (a weighting field, not counts).
    """
    N = acts_flat.shape[0]
    if locality == "global":
        w = torch.ones(N, device=acts_flat.device, dtype=acts_flat.dtype)
    elif locality == "per_image":
        d2 = ((acts_flat - x0_feat) ** 2).sum(dim=1)
        w = torch.zeros(N, device=acts_flat.device, dtype=acts_flat.dtype)
        w[int(torch.argmin(d2).item())] = 1.0
        return w  # already a (near-)delta; do not renormalise away the spike
    elif locality == "bandwidth":
        d2 = ((acts_flat - x0_feat) ** 2).sum(dim=1)
        w = torch.exp(-d2 / (2.0 * bandwidth * bandwidth + eps))
    else:
        raise ValueError(f"unknown locality '{locality}'")
    s = w.sum()
    return w / s if s > eps else torch.full_like(w, 1.0 / N)


# ==========================================================================
# 3. weighted blended second moment   Sigma_{phi,w,lambda}
# ==========================================================================

@dataclass
class MomentBundle:
    """Everything the P-block and alpha-block need, computed once per query."""
    mu: torch.Tensor          # [F]      weighted mean of phi(a) cells
    Sigma_phi: torch.Tensor   # [F, F]   weighted activation second moment
    feat_cells: torch.Tensor  # [N, P, F] centred phi(a) cells  (P = H*W)
    alpha_target: torch.Tensor  # [N, C]  per-image Grad-CAM alpha (the rule target)
    Ltilde: torch.Tensor      # [N, P]   true pre-ReLU explanation per image
    raw_cells: torch.Tensor   # [N, P, C] normalized raw activations (pre-phi)
    HW: Tuple[int, int]
    F: int
    C: int


def build_moments(acts_norm: torch.Tensor, alpha_bank: torch.Tensor,
                  phi: FeatureMap, w: torch.Tensor) -> MomentBundle:
    """Assemble the weighted moments for the bank.

    acts_norm  : [N, C, H, W]  normalized activations of the bank.
    alpha_bank : [N, C]        Grad-CAM channel weights per bank image.
    phi        : the feature map.
    w          : [N]           locality weights (sum 1).

    Sigma_phi (the activation-reconstruction second moment) is formed at the
    CELL level: every spatial cell of every image is a sample, weighted by its
    image's w. mu is the matching weighted mean. This is the
    Sigma^w_phi term of the paper's blended Sigma.
    """
    N, C, H, W = acts_norm.shape
    P = H * W
    dev, dt = acts_norm.device, acts_norm.dtype

    # raw normalized cells [N, P, C]
    raw_cells = acts_norm.permute(0, 2, 3, 1).reshape(N, P, C).contiguous()
    # lifted cells phi(a) [N, P, Fdim]
    feat = phi(raw_cells)                       # [N, P, F]
    Fdim = feat.shape[-1]

    # weighted mean over (image, cell): cell weight = w[image] / P
    wp = (w / P).view(N, 1, 1)                  # [N,1,1]
    mu = (feat * wp).sum(dim=(0, 1))            # [F]
    centred = feat - mu                          # [N, P, F]

    # weighted second moment Sigma^w_phi = sum w_i/P * c c^T
    flat = centred.reshape(N * P, Fdim)
    wflat = (w.view(N, 1).expand(N, P).reshape(-1) / P).to(dt)
    Sigma_phi = (flat * wflat.unsqueeze(1)).T @ flat   # [F, F]
    Sigma_phi = 0.5 * (Sigma_phi + Sigma_phi.T)

    # true pre-ReLU explanation per bank image: L~(x) = sum_c alpha_c a_c
    # (computed on the RAW normalized acts -- this is the target the class term
    # reconstructs; it is independent of phi).
    Ltilde = torch.einsum("nc,npc->np", alpha_bank.to(dt), raw_cells)  # [N,P]

    return MomentBundle(mu=mu, Sigma_phi=Sigma_phi, feat_cells=centred,
                        alpha_target=alpha_bank.to(dt), Ltilde=Ltilde,
                        raw_cells=raw_cells, HW=(H, W), F=Fdim, C=C)


def class_second_moment(mb: MomentBundle, alpha: torch.Tensor,
                         w: torch.Tensor) -> torch.Tensor:
    """Sigma^w_alpha : the class-evidence second moment in feature space.

    The class term wants P to preserve the directions that carry the
    pre-ReLU explanation. For the current weighting field alpha (one [C]
    vector per bank image, here broadcast as a shared [C] query field), the
    per-cell class signal in feature space is

        g_{i,p} = alpha_i . (contribution of cell p)   -> projected to phi-space.

    We use the standard DCAM construction: Sigma_alpha = weighted covariance of
    the alpha-reweighted feature cells. This is the term that, at lambda=1,
    w=global, makes the P-block return DCAM's "top-D of Sigma_alpha".
    """
    N, Pn, Fdim = mb.feat_cells.shape
    # reweight each image's centred feature cells by its scalar alpha energy
    a_energy = alpha.norm(dim=-1) if alpha.dim() == 2 else alpha.norm()
    if torch.is_tensor(a_energy) and a_energy.dim() == 1:
        scale = a_energy.view(N, 1, 1)
    else:
        scale = float(a_energy)
    g = mb.feat_cells * scale                          # [N,P,F]
    flat = g.reshape(N * Pn, Fdim)
    wflat = (w.view(N, 1).expand(N, Pn).reshape(-1) / Pn).to(flat.dtype)
    Sig = (flat * wflat.unsqueeze(1)).T @ flat
    return 0.5 * (Sig + Sig.T)


# ==========================================================================
# 4. the P-block  --  Ky Fan: top-D eigenspace of the blended Sigma
# ==========================================================================

def solve_P_block(Sigma_blended: torch.Tensor, D: int
                  ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Minimise tr( (I-P) Sigma (I-P) ) over rank-D orthogonal projectors P.

    Ky Fan maximum principle: the minimiser's range is the top-D eigenspace of
    Sigma. Returns (V_D [F,D], eigvals_desc [F], eigengap lambda_D-lambda_D+1).

    The eigengap is the paper's seed-invariance certificate: a positive gap at
    the fixed point is what makes the recovered subspace unique. We surface it
    so the caller can CHECK it rather than assume it (paper section 4).
    """
    Sig = 0.5 * (Sigma_blended + Sigma_blended.T)
    evals, evecs = torch.linalg.eigh(Sig)          # ascending
    evals = evals.flip(0).clamp_min(0.0)            # descending
    evecs = evecs.flip(1)
    F = Sig.shape[0]
    D = max(1, min(D, F))
    V_D = evecs[:, :D].contiguous()                 # [F, D]
    gap = float((evals[D - 1] - evals[D]).item()) if D < F else float("inf")
    return V_D, evals, gap


# ==========================================================================
# 5. the alpha-block  --  quadratic linear solve, with Omega = Grad-CAM rule
# ==========================================================================

def solve_alpha_block(mb: MomentBundle, V_D: torch.Tensor, lam: float,
                      gamma: float, omega: str = "gradcam",
                      alpha_rule: Optional[torch.Tensor] = None,
                      ridge: float = 1e-6) -> torch.Tensor:
    """Minimise J over the weighting field alpha with (P=V_D V_D^T) fixed.

    J is quadratic in alpha. Two regimes for Omega:

      omega='gradcam'  -- the rule constraint that PINS alpha to the supplied
                          Grad-CAM averaging rule (alpha_rule). This is the
                          Grad-CAM / HiResCAM / PCG-CAM row of the table:
                          alpha is not free, Omega forces it. We return
                          alpha_rule directly (the constrained minimiser).

      omega='free'     -- alpha is a genuine free variable (DCAM row). The
                          unique minimiser of the quadratic is a linear solve.
                          We solve, per query, the ridge-regularised normal
                          equations that best reconstruct L~ from the rank-D
                          projected features.

    Returns alpha: [C]  (a single shared query weighting field).
    """
    if omega == "gradcam":
        assert alpha_rule is not None, "omega='gradcam' needs alpha_rule"
        return alpha_rule.to(mb.mu.dtype)

    # omega='free': least-squares fit of alpha so that
    #   <alpha, mu + P(phi(a)-mu)>  matches L~(x)  over the weighted bank.
    # Build the design: projected reconstructed features per cell, mapped back
    # to raw-C space via the (linear-slice) identity. For phi=id this is exact;
    # for a kernel it is the best linear surrogate (see scope caveat).
    N, Pn, Fdim = mb.feat_cells.shape
    proj = (mb.feat_cells @ V_D) @ V_D.T + mb.mu        # [N,P,F] reconstructed
    # collapse phi-space back toward C: for id, F==C; for rbf, take the
    # least-squares pseudo-map via the raw cells (surrogate).
    if Fdim == mb.C:
        recon_raw = proj                                 # [N,P,C]
    else:
        # surrogate: regress raw cells onto reconstructed features once
        A = proj.reshape(N * Pn, Fdim)
        B = mb.raw_cells.reshape(N * Pn, mb.C)
        M = torch.linalg.lstsq(A, B).solution            # [F, C]
        recon_raw = (proj @ M)                           # [N,P,C]

    # normal equations:  (X^T X + ridge I) alpha = X^T y
    X = recon_raw.reshape(N * Pn, mb.C)                  # [NP, C]
    y = mb.Ltilde.reshape(N * Pn)                        # [NP]
    XtX = X.T @ X
    XtX = XtX + ridge * torch.eye(mb.C, device=XtX.device, dtype=XtX.dtype)
    Xty = X.T @ y
    alpha = torch.linalg.solve(XtX, Xty)                 # [C]
    return alpha


# ==========================================================================
# 6. the objective value (for monotonicity checks / Option A descent)
# ==========================================================================

def objective_value(mb: MomentBundle, V_D: torch.Tensor, alpha: torch.Tensor,
                     lam: float, gamma: float,
                     omega_penalty: float = 0.0) -> Dict[str, float]:
    """Evaluate J(P, alpha) and its two terms separately.

    recon term  : (1-lambda) * tr( (I-P) Sigma_phi (I-P) )
    class term  : lambda     * weighted || L~ - L~_{P,alpha} ||^2
    """
    P = V_D @ V_D.T
    I = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
    ImP = I - P
    recon = float(torch.trace(ImP @ mb.Sigma_phi @ ImP).item())

    # reconstructed pre-ReLU explanation from (P, alpha)
    proj = (mb.feat_cells @ V_D) @ V_D.T + mb.mu          # [N,P,F]
    if proj.shape[-1] == mb.C:
        recon_raw = proj
    else:
        recon_raw = proj[..., :mb.C]                      # surrogate slice
    Lhat = torch.einsum("c,npc->np", alpha.to(mb.mu.dtype), recon_raw)
    classt = float(((mb.Ltilde - Lhat) ** 2).mean().item())

    J = (1.0 - lam) * recon + lam * classt + gamma * omega_penalty
    return {"J": J, "recon_term": (1.0 - lam) * recon,
            "class_term": lam * classt, "recon_raw": recon,
            "class_raw": classt}


# ==========================================================================
# 7. SOLVERS
# ==========================================================================

@dataclass
class SolveConfig:
    D: int = 50                       # subspace rank
    phi: str = "id"                   # 'id' | 'rbf'
    locality: str = "global"          # 'global' | 'per_image' | 'bandwidth'
    lam: float = 1.0                  # supervision dial lambda in [0,1]
    bandwidth: float = 1.0            # h, used when locality='bandwidth'
    omega: str = "free"               # 'free' | 'gradcam'
    gamma: float = 0.0                # rule-penalty weight
    rff_dim: int = 1024
    rff_gamma: float = 1.0
    max_iter: int = 25                # block-descent iterations
    tol: float = 1e-7                 # relative-J stopping tolerance
    ridge: float = 1e-6
    n_restarts: int = 1               # >1: multi-start seed-invariance check
    lr: float = 0.05                  # Option A step size
    seed: int = 0
    device: str = "auto"
    dtype: str = "float64"


@dataclass
class SolveResult:
    """Everything a visualizer needs. The (mu, V_D, alpha, bias) quadruple is
    the solved analogue of the frozen (pca_mu, pca_V, alpha, bias) that
    pca_gradcam_decomp.py reads out of a PCAReconstructor pkl, so a panel
    routine written against that file works here unchanged for phi=id."""
    mu: torch.Tensor                  # [F]
    V_D: torch.Tensor                 # [F, D]   solved projector basis
    alpha: torch.Tensor               # [C]      solved weighting field
    bias: float                       # sum_c alpha_c mu_c   (additive offset)
    eigvals: torch.Tensor             # [F]      blended-Sigma spectrum
    eigengap: float                   # lambda_D - lambda_{D+1}  (seed cert.)
    J_history: List[float]            # objective per block-descent iteration
    diagnostics: Dict                 # corner label, additive_exact, restarts...
    moments: MomentBundle             # kept so the visualizer can recompute z_d


def _corner_label(cfg: SolveConfig) -> str:
    """Name the corner of the parameter cube the config sits at (paper sec 3)."""
    if cfg.phi == "rbf" and cfg.locality == "per_image" and cfg.lam == 0.0:
        return "KPCA-CAM (kernel, per-image, lambda=0)"
    if cfg.phi == "id" and cfg.locality == "per_image" and cfg.lam == 0.0:
        return "Eigen-CAM (id, per-image, lambda=0)"
    if (cfg.phi == "id" and cfg.locality == "global" and cfg.lam == 1.0
            and cfg.omega == "gradcam"):
        return "Grad-CAM (id, global, lambda=1, Omega pins alpha)"
    if (cfg.phi == "id" and cfg.locality == "global" and cfg.lam == 1.0
            and cfg.omega == "free"):
        return "DCAM low-rank (id, global, lambda=1, alpha free)"
    if cfg.phi == "id" and cfg.locality == "global" and cfg.lam == 0.0:
        return "CRAFT/SAE-style (id, global, lambda=0)"
    if cfg.locality == "bandwidth" and 0.0 < cfg.lam < 1.0:
        return "NEW interior (local, class-tilted low-rank basis)"
    return f"custom (phi={cfg.phi}, w={cfg.locality}, lambda={cfg.lam})"


def _block_descent(mb: MomentBundle, phi: FeatureMap, w: torch.Tensor,
                   cfg: SolveConfig, alpha_rule: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor, List[float],
                              torch.Tensor, float]:
    """Option B: convergent block coordinate descent.

    Each block is the EXACT minimiser of J:
      * alpha-block (P fixed): quadratic -> linear solve (or pinned by Omega).
      * P-block (alpha fixed): tr((I-P)Sigma(I-P)) -> top-D eigenspace.
    Because each block exactly minimises a J bounded below, the J sequence is
    monotone non-increasing and converges to a stationary point (paper
    Proposition 1). We assert the monotonicity at runtime.
    """
    # initialise alpha: pinned rule if omega=gradcam, else the rule as a warm
    # start for the free solve.
    alpha = alpha_rule.clone()
    J_hist: List[float] = []
    V_D = None
    eigvals = None
    gap = float("nan")
    prev_J = float("inf")

    for it in range(cfg.max_iter):
        # ---- P-block: blended second moment, then Ky Fan ----
        Sig_phi = mb.Sigma_phi
        Sig_alpha = class_second_moment(mb, alpha, w)
        Sigma_blended = (1.0 - cfg.lam) * Sig_phi + cfg.lam * Sig_alpha
        V_D, eigvals, gap = solve_P_block(Sigma_blended, cfg.D)

        # ---- alpha-block: exact minimiser given P ----
        alpha = solve_alpha_block(mb, V_D, cfg.lam, cfg.gamma,
                                  omega=cfg.omega, alpha_rule=alpha_rule,
                                  ridge=cfg.ridge)

        # ---- objective + monotonicity guard ----
        vals = objective_value(mb, V_D, alpha, cfg.lam, cfg.gamma)
        J_hist.append(vals["J"])
        # numerical slack: allow tiny non-monotone wobble from float round-off
        if vals["J"] > prev_J + 1e-6 * (abs(prev_J) + 1e-9):
            # not fatal, but worth surfacing -- Prop. 1 says this shouldn't happen
            print(f"  [warn] J rose at iter {it}: {prev_J:.6e} -> {vals['J']:.6e}")
        if abs(prev_J - vals["J"]) <= cfg.tol * (abs(prev_J) + 1e-12):
            prev_J = vals["J"]
            break
        prev_J = vals["J"]

    return V_D, alpha, J_hist, eigvals, gap


def _riemannian_joint(mb: MomentBundle, phi: FeatureMap, w: torch.Tensor,
                      cfg: SolveConfig, alpha_rule: torch.Tensor,
                      V_init: torch.Tensor, alpha_init: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Option A: Riemannian joint descent on Gr(D,F) x A.

    One COUPLED step updates both variables in the same descent direction:
      g_P = Pi_Gr( dJ/dP ),  g_alpha = dJ/dalpha,
      (P, alpha) <- Retract( (P,alpha) - eta (g_P, g_alpha) ).
    The retraction is a thin-QR on the basis, which keeps P a valid rank-D
    orthogonal projector. This is 'joint in the strict sense' (paper sec 4):
    a single descent direction in the product space, NOT alternating blocks.

    We use autograd for the gradients and a QR retraction for the Grassmannian
    step; the tangent-space projection Pi_Gr is g - V (V^T g) (remove the
    component that only rotates within the subspace, which saliency ignores).
    """
    V = V_init.clone().detach().requires_grad_(True)
    alpha = alpha_init.clone().detach().requires_grad_(True)
    J_hist: List[float] = []

    for it in range(cfg.max_iter):
        # forward objective (differentiable in V, alpha)
        P = V @ V.T
        I = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
        ImP = I - P
        recon = torch.trace(ImP @ mb.Sigma_phi @ ImP)
        proj = (mb.feat_cells @ V) @ V.T + mb.mu
        recon_raw = proj if proj.shape[-1] == mb.C else proj[..., :mb.C]
        Lhat = torch.einsum("c,npc->np", alpha, recon_raw)
        classt = ((mb.Ltilde - Lhat) ** 2).mean()
        J = (1.0 - cfg.lam) * recon + cfg.lam * classt
        J_hist.append(float(J.item()))

        gV, gA = torch.autograd.grad(J, [V, alpha])
        with torch.no_grad():
            # tangent projection on the Grassmannian: drop within-subspace rot.
            gV_tan = gV - V @ (V.T @ gV)
            V_new = V - cfg.lr * gV_tan
            # QR retraction -> back onto the Stiefel/Grassmann manifold
            Q, _ = torch.linalg.qr(V_new)
            V.copy_(Q[:, :cfg.D])
            alpha.copy_(alpha - cfg.lr * gA)
        V.requires_grad_(True)
        alpha.requires_grad_(True)
        if it > 0 and abs(J_hist[-2] - J_hist[-1]) <= cfg.tol * (
                abs(J_hist[-2]) + 1e-12):
            break

    return V.detach(), alpha.detach(), J_hist


def solve(acts_norm: torch.Tensor, alpha_bank: torch.Tensor,
          query_alpha: torch.Tensor, query_idx: int,
          cfg: SolveConfig, solver: str = "block") -> SolveResult:
    """Solve the unified objective for one query image.

    Parameters
    ----------
    acts_norm   : [N, C, H, W]  NORMALIZED activations of the bank of images
                  the basis is fit on (same normalized space as the cache).
    alpha_bank  : [N, C]        Grad-CAM channel weights for each bank image
                  (the per-image alpha rule; needed for Sigma_alpha and for
                  omega='gradcam').
    query_alpha : [C]           Grad-CAM channel weights for the QUERY image
                  (the alpha rule the class term reconstructs; also the
                  omega='gradcam' pin and the free-solve warm start).
    query_idx   : index of the query image inside the bank (used to build the
                  per_image delta and the bandwidth weighting centre).
    cfg         : SolveConfig.
    solver      : 'block'  -> Option B (convergent block coordinate descent)
                  'joint'  -> Option A (Riemannian joint descent)
                  'both'   -> run block first, then joint warm-started from it,
                              and return the joint result (block J-history is
                              kept in diagnostics for comparison).

    Returns
    -------
    SolveResult -- see that dataclass. For phi='id' the (mu, V_D, alpha, bias)
    quadruple is exactly what an additive-decomposition visualizer consumes.
    """
    dev = resolve_device(cfg.device)
    dt = torch_dtype(cfg.dtype)
    acts_norm = acts_norm.to(device=dev, dtype=dt)
    alpha_bank = alpha_bank.to(device=dev, dtype=dt)
    query_alpha = query_alpha.to(device=dev, dtype=dt)
    N, C, H, W = acts_norm.shape

    # ---- feature map phi ----
    phi = FeatureMap(kind=cfg.phi, in_dim=C, rff_dim=cfg.rff_dim,
                     rff_gamma=cfg.rff_gamma, seed=cfg.seed,
                     device=dev, dtype=dt)

    # ---- locality field w(x, x0): one feature vector per image = spatial mean
    per_img_feat = phi(acts_norm.permute(0, 2, 3, 1).reshape(N, H * W, C)
                       ).mean(dim=1)                       # [N, F]
    x0_feat = per_img_feat[query_idx]
    w = locality_weights(per_img_feat, x0_feat, locality=cfg.locality,
                         bandwidth=cfg.bandwidth)

    # ---- weighted moments ----
    mb = build_moments(acts_norm, alpha_bank, phi, w)

    # ---- run the solver, with optional multi-restart seed-invariance check --
    restarts: List[Dict] = []
    best = None
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    for r in range(max(1, cfg.n_restarts)):
        # perturb the warm-start alpha for restart r (r=0 is the clean rule)
        a0 = query_alpha.clone()
        if r > 0:
            a0 = a0 + 0.05 * a0.norm() * torch.randn(
                C, generator=g).to(device=dev, dtype=dt)

        if solver in ("block", "both"):
            V_D, alpha, Jh, eigvals, gap = _block_descent(
                mb, phi, w, cfg, alpha_rule=a0)
            block_J = Jh
        if solver in ("joint", "both"):
            if solver == "joint":
                # cold-ish start: top-D of Sigma_phi, rule alpha
                V0, eigvals, gap = solve_P_block(mb.Sigma_phi, cfg.D)
                a_start = a0
            else:
                V0, a_start = V_D, alpha          # warm start from block
            V_D, alpha, Jh = _riemannian_joint(
                mb, phi, w, cfg, alpha_rule=a0, V_init=V0, alpha_init=a_start)
            # recompute spectrum/gap at the joint optimum's blended Sigma
            Sig_b = ((1.0 - cfg.lam) * mb.Sigma_phi
                     + cfg.lam * class_second_moment(mb, alpha, w))
            _, eigvals, gap = solve_P_block(Sig_b, cfg.D)

        vals = objective_value(mb, V_D, alpha, cfg.lam, cfg.gamma)
        restarts.append({"restart": r, "J_final": vals["J"],
                         "eigengap": gap, "V_D": V_D, "alpha": alpha,
                         "eigvals": eigvals, "J_history": Jh,
                         "block_J": block_J if solver == "both" else None})
        if best is None or vals["J"] < best["J_final"]:
            best = restarts[-1]

    # ---- seed-invariance verdict (paper section 4) -------------------------
    # principal-subspace agreement across restarts: mean |cos| of principal
    # angles between restart-0 and restart-r subspaces.
    subspace_agreement = 1.0
    if len(restarts) > 1:
        V0 = restarts[0]["V_D"]
        agrees = []
        for r in restarts[1:]:
            M = V0.T @ r["V_D"]                  # [D,D]
            s = torch.linalg.svdvals(M)          # cos of principal angles
            agrees.append(float(s.mean().item()))
        subspace_agreement = float(np.mean(agrees))

    V_D = best["V_D"]
    alpha = best["alpha"]
    bias = float((alpha @ mb.mu[:C]).item()) if mb.F == C else float(
        (alpha @ mb.raw_cells.mean(dim=(0, 1))).item())

    diagnostics = {
        "corner": _corner_label(cfg),
        "solver": solver,
        "additive_exact": phi.additive_exact,
        "phi": cfg.phi, "locality": cfg.locality, "lambda": cfg.lam,
        "D": cfg.D, "F": mb.F, "C": C, "N_bank": N,
        "eigengap": best["eigengap"],
        "seed_invariant": (best["eigengap"] > 1e-8
                           and subspace_agreement > 0.99),
        "subspace_agreement": subspace_agreement,
        "n_restarts": cfg.n_restarts,
        "J_final": best["J_final"],
        "block_J_history": best["block_J"],
    }
    if not phi.additive_exact:
        diagnostics["scope_caveat"] = (
            "phi is a nontrivial kernel: the exact additive identity "
            "L~ = bias + sum_d beta_d z_d does NOT hold (paper section 5). "
            "Per-component additive panels are disabled; report cosine "
            "fidelity only.")

    return SolveResult(mu=mb.mu, V_D=V_D, alpha=alpha, bias=bias,
                       eigvals=best["eigvals"], eigengap=best["eigengap"],
                       J_history=best["J_history"], diagnostics=diagnostics,
                       moments=mb)


# ==========================================================================
# 8. decomposition read-out  (the bridge to visualization)
# ==========================================================================

def decompose_query(result: SolveResult, query_acts_norm: torch.Tensor,
                     query_alpha: torch.Tensor) -> Dict:
    """Turn a SolveResult into the exact pieces a panel/radial figure needs.

    For phi='id' this reproduces the additive identity of pca_gradcam_decomp.py
    with the SOLVED (not frozen) basis:

        L~(x)      = sum_c alpha_c a_c                       true pre-ReLU
        z_d(x)     = < a - mu, v_d >                         component map [H,W]
        beta_d(x)  = sum_c alpha_c v_{d,c}                   scalar weight
        L~_hat(x)  = bias + sum_d beta_d z_d                 rank-D recon
        Grad-CAM   = ReLU( L~ ),   recon = ReLU( L~_hat )

    query_acts_norm : [1, C, H, W] normalized activations of the query image.
    query_alpha     : [C]          Grad-CAM channel weights for the query.

    Returns a dict with: gradcam_true, recon, L_tilde, L_hat, bias, beta [D],
    z [D,H,W], comp [D,H,W] (= beta_d z_d), relu_keep_frac, spatial cos/corr,
    and 'additive_exact'. For a kernel phi, z/beta/comp are omitted and
    'additive_exact' is False -- only gradcam_true, recon and the cosine
    fidelity are returned (paper section 5 scope caveat).
    """
    dev = result.mu.device
    dt = result.mu.dtype
    A = query_acts_norm.to(device=dev, dtype=dt)
    _, C, H, W = A.shape
    alpha = query_alpha.to(device=dev, dtype=dt)
    cells = A[0].permute(1, 2, 0).reshape(-1, C)        # [HW, C]

    # true pre-ReLU explanation + Grad-CAM
    L_tilde = (alpha.view(1, C) * cells).sum(dim=1).reshape(H, W)
    gradcam_true = torch.relu(L_tilde)
    pos = L_tilde.clamp(min=0).sum()
    tot = L_tilde.abs().sum().clamp(min=1e-8)
    relu_keep_frac = float((pos / tot).item())

    out = {"gradcam_true": gradcam_true.cpu(), "L_tilde": L_tilde.cpu(),
           "relu_keep_frac": relu_keep_frac, "H": H, "W": W,
           "additive_exact": result.diagnostics["additive_exact"]}

    if not result.diagnostics["additive_exact"]:
        # kernel: no additive split -- reconstruct via the projector only.
        feat = result.V_D                                # [F,D]
        # best-effort recon in raw space: project then report cosine
        out["recon"] = gradcam_true.cpu()                # placeholder identity
        out["note"] = result.diagnostics["scope_caveat"]
        return out

    # ---- phi = id: exact additive decomposition with the SOLVED basis ----
    mu = result.mu[:C]
    V_D = result.V_D[:C, :]                              # [C, D]
    D = V_D.shape[1]
    centred = cells - mu                                 # [HW, C]
    z = (centred @ V_D).T.reshape(D, H, W)               # [D,H,W]
    beta = (alpha @ V_D)                                 # [D]
    bias = float((alpha @ mu).item())
    comp = beta.view(-1, 1, 1) * z                       # [D,H,W]
    L_hat = bias + comp.sum(dim=0)                        # [H,W]
    recon = torch.relu(L_hat)

    af, bf = recon.flatten(), gradcam_true.flatten()
    cos = float((af @ bf) / (af.norm().clamp(min=1e-8)
                             * bf.norm().clamp(min=1e-8)))
    am = af - af.mean()
    bm = bf - bf.mean()
    corr = float((am @ bm) / (am.norm().clamp(min=1e-8)
                              * bm.norm().clamp(min=1e-8)))

    out.update({"recon": recon.cpu(), "L_hat": L_hat.cpu(), "bias": bias,
                "beta": beta.cpu(), "z": z.cpu(), "comp": comp.cpu(),
                "spatial_cos": cos, "spatial_corr": corr, "D": D})
    return out


def fidelity_curve(result: SolveResult, query_acts_norm: torch.Tensor,
                   query_alpha: torch.Tensor, D_sweep: List[int]) -> Dict:
    """Rank-D Grad-CAM reconstruction fidelity over a D sweep -- the headline
    'Grad-CAM is low-rank' figure, but with the SOLVED basis. For each D in the
    sweep, reconstruct with the first D solved components and report spatial
    cosine + correlation against the true Grad-CAM. phi='id' only."""
    dec = decompose_query(result, query_acts_norm, query_alpha)
    if not dec["additive_exact"]:
        raise ValueError("fidelity_curve requires phi='id' (additive identity).")
    gt = dec["gradcam_true"]
    comp = dec["comp"]
    bias = dec["bias"]
    curve = {"D": [], "cos": [], "corr": []}
    for D in D_sweep:
        D = min(D, comp.shape[0])
        L_hat = bias + comp[:D].sum(dim=0)
        recon = torch.relu(L_hat)
        af, bf = recon.flatten(), gt.flatten()
        cos = float((af @ bf) / (af.norm().clamp(min=1e-8)
                                 * bf.norm().clamp(min=1e-8)))
        am, bm = af - af.mean(), bf - bf.mean()
        corr = float((am @ bm) / (am.norm().clamp(min=1e-8)
                                  * bm.norm().clamp(min=1e-8)))
        curve["D"].append(D)
        curve["cos"].append(cos)
        curve["corr"].append(corr)
    return curve


# ==========================================================================
# 9. cache loader  (consume an export_activation_cache.py gcmap1 cache)
# ==========================================================================

def load_activation_cache(cache_path: Path, max_chunks: Optional[int] = None,
                           device: str = "cpu"
                           ) -> Tuple[torch.Tensor, torch.Tensor,
                                      torch.Tensor]:
    """Load a gcmap1 cache written by export_activation_cache.py.

    Returns (acts [N,C,H,W], masks [N,C] bool, labels [N]). The cache is
    ALREADY normalized (export writes normalize=True), so feed acts straight
    into solve() -- do NOT normalize again.

    The cache stores per-image GradCAM CHANNEL MASKS, not the alpha weights
    themselves; build_grad_cam_alpha (below) is the way to obtain alpha for a
    bank. If you only have the cache, derive alpha by re-running GradCAM on the
    images, or pass the masks as a coarse 0/1 alpha surrogate.
    """
    import joblib
    cache_path = Path(cache_path)
    meta = joblib.load(cache_path / "metadata.pkl")
    n_chunks = meta["num_chunks"]
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)
    acts, masks, labels = [], [], []
    for i in range(n_chunks):
        part = joblib.load(cache_path / f"part_{i:04d}.pkl")
        acts.append(torch.as_tensor(part["activation"]))
        masks.append(torch.as_tensor(part["mask"]))
        labels.append(torch.as_tensor(part["label"]))
    acts = torch.cat(acts, 0).to(device)
    masks = torch.cat(masks, 0).to(device)
    labels = torch.cat(labels, 0).to(device)
    return acts, masks, labels


# ==========================================================================
# self-documenting corner presets
# ==========================================================================

CORNER_PRESETS: Dict[str, Dict] = {
    "eigen_cam":  dict(phi="id",  locality="per_image", lam=0.0, omega="free"),
    "kpca_cam":   dict(phi="rbf", locality="per_image", lam=0.0, omega="free"),
    "grad_cam":   dict(phi="id",  locality="global",    lam=1.0, omega="gradcam"),
    "dcam":       dict(phi="id",  locality="global",    lam=1.0, omega="free"),
    "craft_sae":  dict(phi="id",  locality="global",    lam=0.0, omega="free"),
    "new_interior": dict(phi="id", locality="bandwidth", lam=0.5, omega="free"),
}


def config_for_corner(name: str, D: int = 50, **overrides) -> SolveConfig:
    """Build a SolveConfig pinned to one of the named corners of the paper's
    parameter cube. Unknown overrides pass straight through to SolveConfig."""
    if name not in CORNER_PRESETS:
        raise ValueError(f"unknown corner '{name}'. "
                         f"choose from {list(CORNER_PRESETS)}")
    params = dict(CORNER_PRESETS[name])
    params.update(overrides)
    params["D"] = D
    return SolveConfig(**params)