"""
Empirical verification of Proposition 5.6 (Top-K Support Stability)
====================================================================

Prop 5.6: If L_f * L_Phi * eps < gamma_K / (2 sqrt(HW)), the Top-K
channel selection is unchanged under input perturbation, where
L_f = ||W_e||_2 (encoder operator norm) and
gamma_K = s_(K) - s_(K+1) (gap between K-th and (K+1)-th importance scores).

Claims tested in this script (single trained SAE):
  (A) Top-K threshold: when eps < gamma_K(x) / (2 sqrt(HW) L_f L_Phi),
      the support is unchanged.
  (B) Top-K Jaccard scales with eps; per-image gamma_K(x) predicts
      per-image stability.

Claim C (the Remark: L_gc widens gamma_K) requires a separate ablation
script comparing two checkpoints.

Predictions:
  (P1) Top-K Jaccard decreases monotonically in eps.
  (P2) Sym-diff ratio |S Δ S'| / K stays small even when support flips.
  (P3) Per-image gamma_K(x) predicts per-image Top-K stability:
       larger gap => more stable support (positive Spearman).
  (P4) The threshold eps* = gamma_K(x) / (2 sqrt(HW) L_f L_Phi) rank-orders
       images by stability (even if vacuous as an absolute predictor).

Outputs:
  - topk_stability_<model>_topk<K>.json
  - topk_stability_<model>_topk<K>.png  (6 panels)
  - topk_stability_<model>_topk<K>_report.txt

Usage:
    python eval_topk_stability.py --model resnet50 \
        --csae_checkpoint imagenet1k_csae_resnet50_gcsum_model.pth \
        --csae_top_k 32 --num_images 500
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.append('.')
from run_xcsae_full import (
    MODEL_CONFIGS,
    ImageNet1kSampledDataset,
    MultiModelActivationExtractor,
    MultiChannelConvSAE,
    IMAGENET_RAW_DIR,
    IMAGENET_SAMPLED_DIR,
    IMAGES_PER_CLASS,
)


# ==========================================
# Core: extract activations, compute Top-K support and gamma_K
# ==========================================

@torch.no_grad()
def compute_topk_support(csae_model: MultiChannelConvSAE,
                         A: torch.Tensor,
                         K: int):
    """Compute Top-K support set, all importance scores, and gamma_K.

    Args:
        csae_model:  trained ConvSAE
        A:           [1, C, H, W] activation tensor (target-layer)
        K:           Top-K parameter (== csae_model.top_k typically)

    Returns:
        support:        bool tensor [D], True for top-K channels
        scores:         tensor [D], importance scores s_d = sum_{i,j} z'_{d,i,j}
        gamma_K:        float, s_(K) - s_(K+1)
        z_pre_topk:     tensor [1, D, H, W], features before Top-K (post-ReLU)
    """
    # Run encoder only
    z_pre_topk = csae_model.encoder(A)
    z_pre_topk = F.relu(z_pre_topk)                  # [1, D, H, W]

    # Channel importance scores
    scores = z_pre_topk[0].sum(dim=(1, 2))           # [D]
    D = scores.shape[0]

    sorted_vals, sorted_idx = torch.sort(scores, descending=True)
    support = torch.zeros(D, dtype=torch.bool, device=A.device)
    support[sorted_idx[:K]] = True

    if K < D:
        gamma_K = float(sorted_vals[K - 1].item() - sorted_vals[K].item())
    else:
        gamma_K = float('inf')

    return support, scores, gamma_K, z_pre_topk


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union > 0 else 1.0


# ==========================================
# Estimate L_f = ||W_e||_2 (operator norm of encoder)
# ==========================================

@torch.no_grad()
def estimate_L_f(csae_model: MultiChannelConvSAE) -> float:
    """For a 1x1 convolution, ||W_e||_2 is the spectral norm of the
    [D, C] weight matrix (with the kernel dim squeezed out)."""
    W = csae_model.encoder.weight.detach()
    # Conv2d weight shape: [D, C, kh, kw]. For 1x1 it's [D, C, 1, 1].
    if W.dim() == 4:
        W = W.view(W.shape[0], -1)                   # [D, C * kh * kw]
    return float(torch.linalg.matrix_norm(W, ord=2).item())


# ==========================================
# Estimate L_Phi via finite differences on x -> A
# ==========================================

@torch.no_grad()
def estimate_L_phi(extractor: MultiModelActivationExtractor,
                   images: torch.Tensor,
                   n_probes: int = 5,
                   probe_eps: float = 1e-3) -> float:
    """Estimate L_Phi = sup ||A(x) - A(x')||_F / ||x - x'||
    via random small perturbations. Returns 95th percentile.
    """
    ratios = []
    for i in range(images.size(0)):
        x = images[i:i+1]
        _ = extractor.model(x)
        A_x = extractor.activations.clone()

        for _ in range(n_probes):
            noise = torch.randn_like(x)
            noise = noise * (probe_eps / (noise.norm() + 1e-12))
            x_pert = x + noise

            _ = extractor.model(x_pert)
            A_xp = extractor.activations.clone()

            num = (A_x - A_xp).norm().item()
            den = noise.norm().item()
            if den > 1e-12:
                ratios.append(num / den)

    if not ratios:
        return float('nan')
    return float(np.percentile(ratios, 95))


# ==========================================
# Load CSAE
# ==========================================

def load_csae(checkpoint: str, in_channels: int, top_k: int, device):
    hidden_dim = in_channels * 8
    csae = MultiChannelConvSAE(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        kernel_size=1,
        top_k=top_k,
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    csae.load_state_dict(state)
    csae.eval()
    return csae


# ==========================================
# Main evaluation
# ==========================================

def run_evaluation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---------- Dataset ----------
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageNet1kSampledDataset(
        raw_dir=IMAGENET_RAW_DIR,
        sampled_dir=IMAGENET_SAMPLED_DIR,
        images_per_class=IMAGES_PER_CLASS,
        transform=data_transform,
        force_resample=False,
    )

    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(dataset), size=args.num_images, replace=False)
    eval_dataset = Subset(dataset, indices.tolist())
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                             num_workers=2)

    # ---------- Extractor + SAE ----------
    extractor = MultiModelActivationExtractor(
        model_name=args.model,
        target_layer=args.target_layer,
        device=device,
        cumulative_threshold=args.tau,  # not used directly here, but kept consistent
    )
    csae_model = load_csae(args.csae_checkpoint, extractor.num_channels,
                            args.csae_top_k, device)
    print(f"\nLoaded CSAE from {args.csae_checkpoint}")
    print(f"  in_channels={extractor.num_channels}, "
          f"hidden_dim={extractor.num_channels * 8}, top_k={args.csae_top_k}")

    H = extractor.spatial_size
    W = H   # square spatial maps
    K = args.csae_top_k

    # ---------- Estimate Lipschitz constants ----------
    print("\n" + "=" * 70)
    print("Estimating Lipschitz constants...")
    print("=" * 70)
    L_f = estimate_L_f(csae_model)
    print(f"  L_f = ||W_e||_2 = {L_f:.4f}  (spectral norm of encoder)")

    probe_imgs = torch.stack([eval_dataset[i][0]
                              for i in range(min(20, len(eval_dataset)))])
    probe_imgs = probe_imgs.to(device)
    L_phi = estimate_L_phi(extractor, probe_imgs, n_probes=5, probe_eps=1e-3)
    print(f"  L_Phi = sup ||dA||/||dx|| = {L_phi:.4f}  (95th pctile)")

    composite_L = L_f * L_phi
    print(f"  L_f * L_Phi = {composite_L:.4f}")
    print(f"  Theoretical threshold prefactor: gamma_K / "
          f"(2 sqrt(HW) * L_f * L_Phi) = gamma_K / {2 * np.sqrt(H*W) * composite_L:.4f}")

    # ---------- Main loop ----------
    print("\n" + "=" * 70)
    print(f"Top-K stability over {len(eval_dataset)} images, "
          f"K={K}, epsilons={args.epsilons}")
    print("=" * 70)

    results = {str(eps): [] for eps in args.epsilons}

    for img_idx, (image, label) in enumerate(tqdm(eval_loader, desc="Images")):
        image = image.to(device)

        # Original activations + Top-K support
        _ = extractor.model(image)
        A_orig = extractor.activations.clone()
        support_orig, scores_orig, gamma_K, _ = compute_topk_support(
            csae_model, A_orig, K)

        # Per-image safe radius from Prop 5.6
        if composite_L > 1e-12:
            eps_star = gamma_K / (2 * np.sqrt(H * W) * composite_L)
        else:
            eps_star = float('inf')

        for eps in args.epsilons:
            j_trials, flipped_trials = [], []
            sym_diff_trials = []

            for trial in range(args.trials_per_eps):
                if eps == 0.0:
                    image_pert = image
                else:
                    noise = torch.randn_like(image)
                    noise = noise * (eps / (noise.norm() + 1e-12))
                    image_pert = image + noise

                _ = extractor.model(image_pert)
                A_pert = extractor.activations.clone()
                support_pert, _, _, _ = compute_topk_support(
                    csae_model, A_pert, K)

                j = jaccard(support_orig, support_pert)
                j_trials.append(j)
                flipped_trials.append(int(j < 1.0))
                sym_diff_trials.append(
                    int((support_orig ^ support_pert).sum().item()))

            entry = {
                'img_idx':            int(indices[img_idx]),
                'label':              int(label.item()),
                'gamma_K':            float(gamma_K),
                'eps_star':           float(eps_star),
                'jaccard_mean':       float(np.mean(j_trials)),
                'jaccard_std':        float(np.std(j_trials)),
                'flip_rate':          float(np.mean(flipped_trials)),
                'sym_diff_mean':      float(np.mean(sym_diff_trials)),
                'sym_diff_ratio':     float(np.mean(sym_diff_trials)) / max(K, 1),
                'below_threshold':    bool(eps < eps_star),
            }
            results[str(eps)].append(entry)

    # ---------- Aggregate ----------
    summary = {
        'config': {
            'model':            args.model,
            'target_layer':     extractor.target_layer_name,
            'csae_checkpoint':  args.csae_checkpoint,
            'csae_top_k':       K,
            'num_images':       args.num_images,
            'epsilons':         args.epsilons,
            'trials_per_eps':   args.trials_per_eps,
            'L_f':              L_f,
            'L_phi':            L_phi,
            'L_f_times_L_phi':  composite_L,
            'spatial_size':     H,
        },
        'per_epsilon': {},
        'per_image':   results,
    }

    rs0 = results[str(args.epsilons[0])]
    gamma_all = [r['gamma_K'] for r in rs0]
    eps_star_all = [r['eps_star'] for r in rs0 if np.isfinite(r['eps_star'])]

    summary['diagnostics'] = {
        'L_f':                  L_f,
        'L_phi':                L_phi,
        'median_gamma_K':       float(np.median(gamma_all)),
        'q5_gamma_K':           float(np.percentile(gamma_all, 5)),
        'q95_gamma_K':          float(np.percentile(gamma_all, 95)),
        'min_gamma_K':          float(np.min(gamma_all)),
        'median_eps_star':      float(np.median(eps_star_all)) if eps_star_all
                                else float('nan'),
        'max_eps_star':         float(np.max(eps_star_all)) if eps_star_all
                                else float('nan'),
    }

    print("\n" + "=" * 70)
    print("Diagnostics (image-level)")
    print("=" * 70)
    for k, v in summary['diagnostics'].items():
        print(f"  {k:25s}: {v:.6g}")

    print("\n" + "=" * 70)
    print("Per-epsilon summary")
    print("=" * 70)
   # Define the problematic string separately
    sds_header = "|SΔS\\|/K"

    print(f"{'eps':>8} {'Jaccard':>9} {'flip':>8} "
        f"{sds_header:>12} {'<eps*':>8} {'rho(γ,J)':>11} {'rho(e*,J)':>11}")  print('-' * 90)

    for eps in args.epsilons:
        rs = results[str(eps)]
        j_mean = np.mean([r['jaccard_mean'] for r in rs])
        flip = np.mean([r['flip_rate'] for r in rs])
        sd_ratio = np.mean([r['sym_diff_ratio'] for r in rs])
        below = np.mean([r['below_threshold'] for r in rs])

        gammas = [r['gamma_K'] for r in rs]
        eps_stars = [r['eps_star'] for r in rs]
        jaccs = [r['jaccard_mean'] for r in rs]

        # P3: rho(gamma_K, Jaccard)
        if len(gammas) >= 5 and len(set(jaccs)) > 1:
            rho_g, _ = spearmanr(gammas, jaccs)
        else:
            rho_g = float('nan')

        # P4: rho(eps_star, Jaccard)
        finite = [(e, j) for e, j in zip(eps_stars, jaccs) if np.isfinite(e)]
        if len(finite) >= 5 and len(set(j for _, j in finite)) > 1:
            rho_e, _ = spearmanr([e for e, _ in finite],
                                  [j for _, j in finite])
        else:
            rho_e = float('nan')

        print(f"{eps:>8.4f} {j_mean:>9.4f} {flip:>8.4f} "
              f"{sd_ratio:>12.4f} {below:>8.3f} {rho_g:>11.4f} {rho_e:>11.4f}")

        summary['per_epsilon'][str(eps)] = {
            'mean_jaccard':                float(j_mean),
            'flip_rate':                   float(flip),
            'mean_sym_diff_ratio':         float(sd_ratio),
            'frac_below_threshold':        float(below),
            'spearman_gamma_K_vs_jaccard': float(rho_g),
            'spearman_eps_star_vs_jaccard': float(rho_e),
        }

    # ---------- Save ----------
    out_prefix = f"topk_stability_{args.model}_topk{K}"
    json_path = f"{out_prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nRaw results: {json_path}")

    plot_path = f"{out_prefix}.png"
    make_plots(summary, plot_path, args.model)
    print(f"Plot: {plot_path}")

    report_path = f"{out_prefix}_report.txt"
    write_report(summary, report_path, args.model)
    print(f"Report: {report_path}")

    return summary


# ==========================================
# Plots
# ==========================================

def make_plots(summary, save_path, model_name):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Top-K Support Stability ({model_name.upper()}, "
                 f"K={summary['config']['csae_top_k']}) -- Prop. 5.6",
                 fontsize=13, fontweight='bold')

    eps_strs = list(summary['per_epsilon'].keys())
    epsilons = sorted([float(e) for e in eps_strs])
    eps_str_map = {float(e): e for e in eps_strs}

    # (a) Top-K Jaccard + flip rate vs eps -- P1
    ax = axes[0, 0]
    means = [summary['per_epsilon'][eps_str_map[e]]['mean_jaccard']
             for e in epsilons]
    flips = [summary['per_epsilon'][eps_str_map[e]]['flip_rate']
             for e in epsilons]
    ax.plot(epsilons, means, 'o-', color='C0', label='Top-K Jaccard',
            linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(epsilons, flips, 's--', color='C3', label='Flip rate',
             linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Top-K Jaccard', color='C0')
    ax2.set_ylabel('Flip rate', color='C3')
    ax.set_title('(a) P1: Top-K Jaccard & flip rate')
    ax.grid(True, alpha=0.3)
    j_min = min(means) - 0.05
    ax.set_ylim(max(0, j_min), 1.005)
    ax2.set_ylim(0, 1.05)

    # (b) sym-diff ratio vs eps -- P2
    ax = axes[0, 1]
    ratios = [summary['per_epsilon'][eps_str_map[e]]['mean_sym_diff_ratio']
              for e in epsilons]
    ax.plot(epsilons, ratios, 'o-', color='C4', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r"Mean $|S \triangle S'| / K$")
    ax.set_title('(b) P2: Boundary-churn ratio')
    ax.grid(True, alpha=0.3)

    # (c) gamma_K vs Jaccard scatter -- P3
    mid_eps = epsilons[len(epsilons) // 2]
    rs = summary['per_image'][eps_str_map[mid_eps]]
    gammas = [r['gamma_K'] for r in rs]
    j_scatter = [r['jaccard_mean'] for r in rs]
    ax = axes[0, 2]
    ax.scatter(gammas, j_scatter, alpha=0.4, s=15, c='C2')
    ax.set_xlabel(r'$\gamma_K(x)$')
    ax.set_ylabel('Top-K Jaccard')
    ax.set_title(f'(c) P3: $\\gamma_K(x)$ vs Jaccard '
                 f'($\\epsilon$={mid_eps:.3f})')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    if len(gammas) >= 5:
        rho, _ = spearmanr(gammas, j_scatter)
        ax.text(0.05, 0.05, f'Spearman $\\rho$ = {rho:.3f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # (d) gamma_K distribution
    ax = axes[1, 0]
    rs = summary['per_image'][eps_str_map[epsilons[0]]]
    gammas_all = [r['gamma_K'] for r in rs]
    ax.hist(np.log10(np.clip(gammas_all, 1e-10, None)),
            bins=40, alpha=0.7, color='C0')
    med = np.median(gammas_all)
    ax.axvline(np.log10(max(med, 1e-10)), color='r', linestyle='--',
               label=f'median = {med:.3e}')
    ax.set_xlabel(r'$\log_{10} \gamma_K(x)$')
    ax.set_ylabel('Count')
    ax.set_title(r'(d) Distribution of $\gamma_K(x)$')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (e) eps* distribution + tested epsilons
    ax = axes[1, 1]
    rs = summary['per_image'][eps_str_map[epsilons[0]]]
    eps_stars = [r['eps_star'] for r in rs if np.isfinite(r['eps_star'])]
    if eps_stars:
        ax.hist(np.log10(np.clip(eps_stars, 1e-20, None)),
                bins=40, alpha=0.7, color='C1', label=r'$\epsilon^*(x)$')
        for e in epsilons:
            ax.axvline(np.log10(e), color='gray', alpha=0.5,
                       linestyle=':')
        ax.axvline(np.log10(epsilons[0]), color='red',
                   alpha=0.7, linestyle='--',
                   label=f'min tested $\\epsilon$={epsilons[0]:.4f}')
    ax.set_xlabel(r'$\log_{10} \epsilon^*(x)$')
    ax.set_ylabel('Count')
    ax.set_title(r'(e) Distribution of $\epsilon^*(x)$ vs tested $\epsilon$')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (f) Spearman correlations across eps -- P3 + P4
    ax = axes[1, 2]
    rho_g = [summary['per_epsilon'][eps_str_map[e]]['spearman_gamma_K_vs_jaccard']
             for e in epsilons]
    rho_e = [summary['per_epsilon'][eps_str_map[e]]['spearman_eps_star_vs_jaccard']
             for e in epsilons]
    ax.plot(epsilons, rho_g, 'o-', color='C2', linewidth=2,
            label=r'$\rho(\gamma_K, J)$ (P3)')
    ax.plot(epsilons, rho_e, 's--', color='C5', linewidth=2,
            label=r'$\rho(\epsilon^*, J)$ (P4)')
    ax.axhline(0, color='k', alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Spearman correlation')
    ax.set_title('(f) Do $\\gamma_K$ and $\\epsilon^*$ rank-order stability?')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()


# ==========================================
# Report
# ==========================================

def write_report(summary, save_path, model_name):
    lines = []
    lines.append("=" * 72)
    lines.append(f"Top-K Stability Report -- {model_name.upper()} "
                 f"(K={summary['config']['csae_top_k']})")
    lines.append("(Empirical verification of Proposition 5.6, Claims A and B)")
    lines.append("=" * 72)
    lines.append("")

    cfg = summary['config']
    lines.append("Configuration:")
    for k, v in cfg.items():
        lines.append(f"  {k:25s} : {v}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("Image-level diagnostics:")
    lines.append("-" * 72)
    for k, v in summary['diagnostics'].items():
        lines.append(f"  {k:25s} : {v:.6g}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(f"  - L_f * L_Phi = {cfg['L_f_times_L_phi']:.4g} -- this is")
    lines.append(f"    the composite Lipschitz constant from Prop 5.6.")
    lines.append(f"  - gamma_K(x) values vary across images:")
    lines.append(f"      median = {summary['diagnostics']['median_gamma_K']:.4g}")
    lines.append(f"      5th pctile = {summary['diagnostics']['q5_gamma_K']:.4g}")
    lines.append(f"      min = {summary['diagnostics']['min_gamma_K']:.4g}")
    lines.append(f"    Small gamma_K means K-th and (K+1)-th channels are")
    lines.append(f"    close in importance: small perturbation may swap them.")
    lines.append(f"  - eps*(x) = gamma_K(x) / (2 sqrt(HW) L_f L_Phi):")
    lines.append(f"      median = {summary['diagnostics']['median_eps_star']:.4g}")
    lines.append(f"    Below this radius, Prop 5.6 guarantees support stability.")
    lines.append("")

    lines.append("-" * 72)
    lines.append("Per-epsilon summary:")
    lines.append("-" * 72)
    lines.append(f"{'eps':>8} {'Jaccard':>9} {'flip':>8} "
                 f"{'sd_ratio':>10} {'<eps*':>8} "
                 f"{'rho(γ,J)':>11} {'rho(e*,J)':>11}")
    for eps_str, m in summary['per_epsilon'].items():
        lines.append(
            f"{float(eps_str):>8.4f} "
            f"{m['mean_jaccard']:>9.4f} "
            f"{m['flip_rate']:>8.4f} "
            f"{m['mean_sym_diff_ratio']:>10.4f} "
            f"{m['frac_below_threshold']:>8.3f} "
            f"{m['spearman_gamma_K_vs_jaccard']:>11.4f} "
            f"{m['spearman_eps_star_vs_jaccard']:>11.4f}"
        )
    lines.append("")

    lines.append("Reading the table:")
    lines.append("  - Jaccard    : Top-K support Jaccard under perturbation.")
    lines.append("  - flip       : fraction of trials where support changed.")
    lines.append("  - sd_ratio   : |S Δ S'|/K, fraction of Top-K channels swapped.")
    lines.append("                 Small => boundary churn only (good).")
    lines.append("  - <eps*      : fraction of samples below the Prop 5.6 threshold.")
    lines.append("  - rho(γ, J)  : P3 test -- positive means larger gap predicts")
    lines.append("                 more stable support (theoretically expected).")
    lines.append("  - rho(e*, J) : P4 test -- does the per-image threshold")
    lines.append("                 rank-order stability? (often equivalent to P3.)")
    lines.append("")
    lines.append("What to look for:")
    lines.append("  P1: Jaccard decreases monotonically in eps.")
    lines.append("  P2: sd_ratio stays small even when flip rate is high.")
    lines.append("  P3: rho(gamma_K, Jaccard) > 0 across all eps.")
    lines.append("  P4: rho(eps*, Jaccard) > 0; magnitude similar to P3 since")
    lines.append("      eps* is a monotone rescaling of gamma_K.")
    lines.append("")
    lines.append("Note: this script verifies Claims A and B of Prop 5.6.")
    lines.append("Claim C (L_gc widens gamma_K) requires the separate")
    lines.append("ablation script compare_gc_ablation.py which contrasts")
    lines.append("two checkpoints trained with lambda_gc = 0 and lambda_gc > 0.")

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Empirical verification of Prop 5.6 (Top-K stability)"
    )
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--target_layer', type=str, default=None)
    parser.add_argument('--tau', type=float, default=0.95,
                        help='(Unused here, kept for extractor compatibility.)')
    parser.add_argument('--num_images', type=int, default=500)
    parser.add_argument('--epsilons', type=float, nargs='+',
                        default=[0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument('--trials_per_eps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--csae_checkpoint', type=str, required=True,
                        help='Path to trained ConvSAE .pth state_dict')
    parser.add_argument('--csae_top_k', type=int, default=32,
                        help='Top-K parameter used at training time')

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()