"""
Empirical verification of Proposition 5.2 (Grad-CAM Mask Stability)
====================================================================

Verifies that the channel mask M_tau(x) is stable under input perturbations,
and that images with larger margin delta(x) are more stable.

Predictions tested:
  (P1) Mask Jaccard similarity decreases monotonically with epsilon.
  (P2) Per-image margin delta(x) predicts mask stability.
  (P3) Theoretical threshold eps* = delta(x) * min_k alpha_(k) / L_alpha
       separates stable from unstable samples (when below eps*, mask preserved).
  (P4) The Lipschitz constant L_alpha = L_H * L_Phi / (HW) is non-vacuous
       (i.e., empirical alpha perturbations respect the bound).

Outputs:
  - mask_stability_<model>.json   : raw measurements
  - mask_stability_<model>.png    : 4-panel diagnostic plot
  - mask_stability_<model>_report.txt : numerical summary

Usage:
    python eval_mask_stability.py --model resnet50 --num_images 500
    python eval_mask_stability.py --model resnet18 --num_images 1000 --epsilons 0.001 0.005 0.01 0.05 0.1
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

sys.path.append('.')
from src.gradcam import GradCAM
from run_xcsae_full import (
    MODEL_CONFIGS,
    ImageNet1kSampledDataset,
    MultiModelActivationExtractor,
    IMAGENET_RAW_DIR,
    IMAGENET_SAMPLED_DIR,
    IMAGES_PER_CLASS,
)


# ==========================================
# Core mask + margin computation
# ==========================================

@torch.no_grad()
def compute_mask_and_margin(extractor: MultiModelActivationExtractor,
                            image: torch.Tensor,
                            class_idx: int = None):
    """Compute Grad-CAM mask M_tau(x), margin delta(x), and the sorted
    normalized weights bar_alpha (needed for the theoretical threshold).

    The forward pass through gradcam.forward populates self.activations
    via the hook, but we don't need them here -- only the alpha weights.

    Returns:
        mask:           bool tensor [C], True for selected channels
        margin:         float, |cumsum[K_tau] - tau|
        min_bar_alpha:  float, min of the *selected* normalized alphas
        K_tau:          int, number of selected channels
        alpha:          tensor [C], raw alpha values (for Lipschitz check)
    """
    # gradcam.forward needs gradients enabled to backprop
    with torch.enable_grad():
        weights, _, _ = extractor.gradcam.forward(image, class_idx=class_idx,
                                                   verbose=False)
    # weights: [C], the alpha_k values

    sorted_w, sorted_idx = torch.sort(weights, descending=True)
    pos_mask = sorted_w > 0
    if pos_mask.sum() == 0:
        # Degenerate: no positive weights. Select top 10%.
        num_selected = max(1, int(0.1 * len(sorted_idx)))
        mask = torch.zeros(extractor.num_channels, dtype=torch.bool,
                           device=image.device)
        mask[sorted_idx[:num_selected]] = True
        return mask, 0.0, 0.0, num_selected, weights

    pos_w = sorted_w[pos_mask]
    total = pos_w.sum()
    bar_alpha = pos_w / total  # normalized, sorted descending

    cumsum = torch.cumsum(bar_alpha, dim=0)
    # K_tau = smallest K such that cumsum[K-1] >= tau
    above = cumsum >= extractor.cumulative_threshold
    if above.any():
        K_tau = int(above.nonzero()[0].item()) + 1
    else:
        K_tau = len(bar_alpha)

    K_tau = min(K_tau, len(sorted_idx))
    mask = torch.zeros(extractor.num_channels, dtype=torch.bool,
                       device=image.device)
    mask[sorted_idx[:K_tau]] = True

    # Margin: how much cumsum at K_tau exceeds tau
    margin = float(cumsum[K_tau - 1].item() - extractor.cumulative_threshold)

    # Min of the *selected* normalized alphas
    min_bar_alpha = float(bar_alpha[:K_tau].min().item())

    return mask, margin, min_bar_alpha, K_tau, weights


def jaccard(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    """Jaccard similarity between two boolean channel masks."""
    inter = (mask_a & mask_b).sum().item()
    union = (mask_a | mask_b).sum().item()
    if union == 0:
        return 1.0
    return inter / union


# ==========================================
# Lipschitz constant L_alpha (empirical estimate)
# ==========================================

def estimate_L_alpha(extractor: MultiModelActivationExtractor,
                     images: torch.Tensor,
                     n_probes: int = 5,
                     probe_eps: float = 1e-3) -> float:
    """Estimate L_alpha = sup ||alpha(x) - alpha(x')|| / ||x - x'||
    via random small perturbations on a batch of images.

    The theory says L_alpha = L_H * L_Phi / (HW). We bypass estimating
    L_H and L_Phi separately and directly measure the effective Lipschitz
    constant of x -> alpha(x).
    """
    ratios = []
    for i in range(images.size(0)):
        x = images[i:i+1]
        _, _, _, _, alpha_x = compute_mask_and_margin(extractor, x)
        alpha_x = alpha_x.detach()

        for _ in range(n_probes):
            noise = torch.randn_like(x) * probe_eps
            # Normalize noise to have exact norm probe_eps
            noise = noise * (probe_eps / (noise.norm() + 1e-12))
            x_pert = x + noise

            _, _, _, _, alpha_xp = compute_mask_and_margin(extractor, x_pert)
            alpha_xp = alpha_xp.detach()

            num = (alpha_x - alpha_xp).norm().item()
            den = noise.norm().item()
            if den > 1e-12:
                ratios.append(num / den)

    if len(ratios) == 0:
        return float('nan')
    # Use 95th percentile as a robust estimate of the Lipschitz sup
    return float(np.percentile(ratios, 95))


# ==========================================
# Main evaluation
# ==========================================

def run_evaluation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---------- Setup dataset ----------
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

    # Subsample for evaluation
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(dataset), size=args.num_images, replace=False)
    eval_dataset = Subset(dataset, indices.tolist())
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                             num_workers=2)

    # ---------- Setup extractor ----------
    extractor = MultiModelActivationExtractor(
        model_name=args.model,
        target_layer=args.target_layer,
        device=device,
        cumulative_threshold=args.tau,
    )

    # ---------- Estimate L_alpha on a small probe batch ----------
    print("\n" + "=" * 70)
    print("Estimating L_alpha (effective Lipschitz of x -> alpha(x))...")
    print("=" * 70)
    probe_imgs = torch.stack([eval_dataset[i][0]
                              for i in range(min(20, len(eval_dataset)))])
    probe_imgs = probe_imgs.to(device)
    L_alpha = estimate_L_alpha(extractor, probe_imgs,
                               n_probes=5, probe_eps=1e-3)
    print(f"  Estimated L_alpha (95th pctile of ||dalpha||/||dx||): {L_alpha:.4f}")

    # ---------- Main loop: per-image, per-epsilon mask stability ----------
    print("\n" + "=" * 70)
    print(f"Measuring mask stability over {len(eval_dataset)} images, "
          f"epsilons={args.epsilons}")
    print("=" * 70)

    # Storage: results[eps] = list of dicts
    results = {eps: [] for eps in args.epsilons}

    for img_idx, (image, label) in enumerate(tqdm(eval_loader, desc="Images")):
        image = image.to(device)

        # Original mask + margin
        mask_orig, margin, min_bar_alpha, K_tau, _ = \
            compute_mask_and_margin(extractor, image)

        # Theoretical threshold per Prop 5.2:
        # eps* = delta(x) * min_k bar_alpha_(k) / L_alpha
        if L_alpha > 1e-12:
            eps_star = (margin * min_bar_alpha) / L_alpha
        else:
            eps_star = float('inf')

        for eps in args.epsilons:
            # Average over multiple noise realizations for a stable estimate
            jaccards = []
            flipped = []
            for trial in range(args.trials_per_eps):
                noise = torch.randn_like(image)
                noise = noise * (eps / (noise.norm() + 1e-12))
                image_pert = image + noise

                mask_pert, _, _, _, _ = compute_mask_and_margin(
                    extractor, image_pert)

                j = jaccard(mask_orig, mask_pert)
                jaccards.append(j)
                flipped.append(int(j < 1.0))

            results[eps].append({
                'img_idx': int(indices[img_idx]),
                'label': int(label.item()),
                'margin': margin,
                'min_bar_alpha': min_bar_alpha,
                'K_tau': K_tau,
                'eps_star': eps_star,
                'jaccard_mean': float(np.mean(jaccards)),
                'jaccard_std': float(np.std(jaccards)),
                'flip_rate': float(np.mean(flipped)),
                'below_threshold': bool(eps < eps_star),
            })

    # ---------- Aggregate + report ----------
    print("\n" + "=" * 70)
    print("Aggregate results")
    print("=" * 70)

    summary = {
        'config': {
            'model': args.model,
            'target_layer': extractor.target_layer_name,
            'tau': args.tau,
            'num_images': args.num_images,
            'epsilons': args.epsilons,
            'trials_per_eps': args.trials_per_eps,
            'L_alpha_est': L_alpha,
        },
        'per_epsilon': {},
        'per_image': results,
    }

    print(f"\n{'eps':>10} {'mean Jaccard':>14} {'flip rate':>11} "
          f"{'frac < eps*':>13} {'cond. Jaccard':>15}")
    print("-" * 70)
    for eps in args.epsilons:
        rs = results[eps]
        j_mean = np.mean([r['jaccard_mean'] for r in rs])
        flip = np.mean([r['flip_rate'] for r in rs])
        below = np.mean([r['below_threshold'] for r in rs])

        # Conditional Jaccard: among images below the theoretical threshold,
        # what's the Jaccard? The theory predicts this should be ~1.0.
        below_jaccards = [r['jaccard_mean'] for r in rs
                          if r['below_threshold']]
        cond_j = np.mean(below_jaccards) if len(below_jaccards) > 0 \
            else float('nan')

        summary['per_epsilon'][str(eps)] = {
            'mean_jaccard': float(j_mean),
            'flip_rate': float(flip),
            'frac_below_threshold': float(below),
            'conditional_jaccard_below_threshold': float(cond_j),
            'n_below_threshold': int(len(below_jaccards)),
        }

        print(f"{eps:>10.4f} {j_mean:>14.4f} {flip:>11.4f} "
              f"{below:>13.4f} {cond_j:>15.4f}")

    # ---------- Save raw results ----------
    out_prefix = f"mask_stability_{args.model}"
    json_path = f"{out_prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nRaw results: {json_path}")

    # ---------- Plot ----------
    plot_path = f"{out_prefix}.png"
    make_plots(summary, plot_path, args.model)
    print(f"Plot: {plot_path}")

    # ---------- Text report ----------
    report_path = f"{out_prefix}_report.txt"
    write_report(summary, report_path, args.model)
    print(f"Report: {report_path}")

    return summary


def make_plots(summary, save_path, model_name):
    """4-panel diagnostic figure verifying the four predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Grad-CAM Mask Stability ({model_name.upper()}) "
                 f"-- Prop. 5.2 verification",
                 fontsize=13, fontweight='bold')

    epsilons = sorted([float(e) for e in summary['per_epsilon'].keys()])

    # ----- Panel (a): Jaccard vs epsilon (Prediction P1) -----
    ax = axes[0, 0]
    means = [summary['per_epsilon'][str(e)]['mean_jaccard']
             for e in epsilons]
    flips = [summary['per_epsilon'][str(e)]['flip_rate']
             for e in epsilons]
    ax.plot(epsilons, means, 'o-', color='C0', label='Mean Jaccard', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(epsilons, flips, 's--', color='C3',
             label='Flip rate', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel(r'Perturbation magnitude $\epsilon$')
    ax.set_ylabel('Mean mask Jaccard', color='C0')
    ax2.set_ylabel('Flip rate', color='C3')
    ax.set_title('(a) P1: Monotone Jaccard decay with $\\epsilon$')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)

    # ----- Panel (b): margin vs Jaccard scatter (Prediction P2) -----
    # Use the median epsilon for the scatter
    mid_eps = epsilons[len(epsilons) // 2]
    rs = summary['per_image'][mid_eps] if mid_eps in summary['per_image'] \
        else summary['per_image'][list(summary['per_image'].keys())[len(epsilons) // 2]]
    margins = [r['margin'] for r in rs]
    j_scatter = [r['jaccard_mean'] for r in rs]
    ax = axes[0, 1]
    ax.scatter(margins, j_scatter, alpha=0.4, s=15, c='C2')
    ax.set_xlabel(r'Per-image margin $\delta(x)$')
    ax.set_ylabel('Mask Jaccard')
    ax.set_title(f'(b) P2: Larger $\\delta(x)$ implies more stability '
                 f'($\\epsilon$={mid_eps:.3f})')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Add binned mean overlay
    bins = np.linspace(0, max(margins), 8)
    bin_centers, bin_means = [], []
    for i in range(len(bins) - 1):
        in_bin = [j for m, j in zip(margins, j_scatter)
                  if bins[i] <= m < bins[i + 1]]
        if len(in_bin) > 0:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_means.append(np.mean(in_bin))
    if bin_centers:
        ax.plot(bin_centers, bin_means, 'r-', linewidth=2.5,
                label='Binned mean', zorder=10)
        ax.legend()

    # ----- Panel (c): theoretical threshold separation (Prediction P3) -----
    ax = axes[1, 0]
    # For each epsilon, plot conditional Jaccard below vs above eps*
    cond_below, cond_above = [], []
    for eps in epsilons:
        rs = summary['per_image'][eps]
        below = [r['jaccard_mean'] for r in rs if r['below_threshold']]
        above = [r['jaccard_mean'] for r in rs if not r['below_threshold']]
        cond_below.append(np.mean(below) if below else np.nan)
        cond_above.append(np.mean(above) if above else np.nan)

    width = 0.35
    x = np.arange(len(epsilons))
    ax.bar(x - width / 2, cond_below, width,
           label=r'$\epsilon < \epsilon^*(x)$ (theory: stable)',
           color='C2', alpha=0.8)
    ax.bar(x + width / 2, cond_above, width,
           label=r'$\epsilon \geq \epsilon^*(x)$ (theory: may flip)',
           color='C3', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{e:.3f}' for e in epsilons])
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Mean Jaccard')
    ax.set_title('(c) P3: Theoretical threshold separates stable / unstable')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    # ----- Panel (d): margin distribution + K_tau distribution -----
    ax = axes[1, 1]
    # Use first epsilon's results (margin is per-image, doesn't depend on eps)
    rs = summary['per_image'][epsilons[0]]
    margins_all = [r['margin'] for r in rs]
    eps_stars = [r['eps_star'] for r in rs if np.isfinite(r['eps_star'])]
    ax.hist(margins_all, bins=40, alpha=0.7, color='C0',
            label=r'$\delta(x)$ distribution')
    ax.set_xlabel(r'Margin $\delta(x)$')
    ax.set_ylabel('Count')
    ax.set_title(r'(d) Distribution of mask margins $\delta(x)$')
    ax.grid(True, alpha=0.3)
    ax.axvline(np.median(margins_all), color='r', linestyle='--',
               label=f'median = {np.median(margins_all):.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()


def write_report(summary, save_path, model_name):
    """Plain-text numerical summary suitable for the paper appendix."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"Mask Stability Report -- {model_name.upper()}")
    lines.append(f"(Empirical verification of Proposition 5.2)")
    lines.append("=" * 72)
    lines.append("")
    cfg = summary['config']
    lines.append("Configuration:")
    for k, v in cfg.items():
        lines.append(f"  {k:25s} : {v}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("Per-epsilon summary:")
    lines.append("-" * 72)
    lines.append(f"{'epsilon':>10} {'mean Jaccard':>14} {'flip rate':>11} "
                 f"{'<eps*':>8} {'Jaccard|<eps*':>15} {'n_below':>9}")
    for eps_str, m in summary['per_epsilon'].items():
        lines.append(
            f"{float(eps_str):>10.4f} "
            f"{m['mean_jaccard']:>14.4f} "
            f"{m['flip_rate']:>11.4f} "
            f"{m['frac_below_threshold']:>8.3f} "
            f"{m['conditional_jaccard_below_threshold']:>15.4f} "
            f"{m['n_below_threshold']:>9d}"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - Mean Jaccard should decrease monotonically in epsilon (P1).")
    lines.append("  - 'Jaccard|<eps*' is the conditional Jaccard among samples")
    lines.append("    where epsilon is below the theoretical threshold eps*(x).")
    lines.append("    The theory predicts this should be ~1.0 (mask preserved).")
    lines.append("  - 'flip rate' = fraction of (image, trial) pairs where the")
    lines.append("    mask changed at all (Jaccard < 1).")
    lines.append("")

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Empirical verification of Prop 5.2 (mask stability)"
    )
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--target_layer', type=str, default=None)
    parser.add_argument('--tau', type=float, default=0.85,
                        help='Cumulative threshold for the mask.')
    parser.add_argument('--num_images', type=int, default=500,
                        help='Number of images to evaluate on.')
    parser.add_argument('--epsilons', type=float, nargs='+',
                        default=[0.001, 0.005, 0.01, 0.05, 0.1],
                        help='Perturbation magnitudes (L2 norm) to test.')
    parser.add_argument('--trials_per_eps', type=int, default=3,
                        help='Number of noise realizations per (image, eps).')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_evaluation(args)


if __name__ == "__main__":
    main()