"""
Empirical verification of Proposition 5.2 (Grad-CAM Mask Stability)
====================================================================

REVISED VERSION: focuses on the *excess-loss bound* (Claim B) which is the
testable part of Prop. 5.2 at practical perturbation magnitudes. The
worst-case Lipschitz threshold (Claim A) is reported as a diagnostic
but is typically vacuous for tau >= 0.85 due to small boundary alphas.

Predictions tested:
  (P1) Mask Jaccard similarity decreases monotonically with epsilon.
  (P2) Per-image margin delta(x) predicts mask stability (rank order).
  (P3) Symmetric-difference ratio |M Δ M'| / |M| stays small even when
       mask "flips" -- boundary churn is bounded.
  (P4) Excess-loss bound from Claim B holds:
       |L_M - L_M'|  <=  (|M Δ M'| / |M|) * max_c ||A_c||_F^2
       and is non-vacuous in practice.

Diagnostics reported:
  - L_alpha estimate (empirical Lipschitz of x -> alpha(x))
  - Median delta(x), min bar_alpha, eps*(x)  --  shows why Claim A is loose
  - Spearman correlation between eps*(x) and Jaccard at fixed eps
    (does the worst-case bound at least rank-order images correctly?)

Outputs:
  - mask_stability_<model>_tau<...>.json
  - mask_stability_<model>_tau<...>.png   (6 panels)
  - mask_stability_<model>_tau<...>_report.txt

Usage:
    python eval_mask_stability.py --model resnet50 --tau 0.95 --num_images 500
    python eval_mask_stability.py --model resnet18 --tau 0.85 --num_images 1000 \
        --epsilons 0.001 0.005 0.01 0.05 0.1 \
        --csae_checkpoint imagenet1k_csae_resnet18_gcsum_model.pth
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
from src.gradcam import GradCAM
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
# Core mask + margin computation
# ==========================================

def compute_mask_and_margin(extractor: MultiModelActivationExtractor,
                            image: torch.Tensor,
                            class_idx: int = None):
    """Compute Grad-CAM mask M_tau(x), margin delta(x), normalized alpha info.

    Side effect: extractor.activations holds the target-layer activations
    after this call (the gradcam.forward triggers a forward pass through
    the hooked layer). Caller should clone before the next invocation.

    Returns:
        mask:           bool tensor [C]
        margin:         float, cumsum[K_tau] - tau (>=0)
        min_bar_alpha:  float, min of the *selected* normalized alphas
        K_tau:          int
        alpha:          tensor [C], raw alpha values (detached)
    """
    with torch.enable_grad():
        weights, _, _ = extractor.gradcam.forward(image, class_idx=class_idx,
                                                   verbose=False)
    weights = weights.detach()

    sorted_w, sorted_idx = torch.sort(weights, descending=True)
    pos_mask = sorted_w > 0
    if pos_mask.sum() == 0:
        num_selected = max(1, int(0.1 * len(sorted_idx)))
        mask = torch.zeros(extractor.num_channels, dtype=torch.bool,
                           device=image.device)
        mask[sorted_idx[:num_selected]] = True
        return mask, 0.0, 0.0, num_selected, weights

    pos_w = sorted_w[pos_mask]
    total = pos_w.sum()
    bar_alpha = pos_w / total

    cumsum = torch.cumsum(bar_alpha, dim=0)
    above = cumsum >= extractor.cumulative_threshold
    if above.any():
        K_tau = int(above.nonzero()[0].item()) + 1
    else:
        K_tau = len(bar_alpha)

    K_tau = min(K_tau, len(sorted_idx))
    mask = torch.zeros(extractor.num_channels, dtype=torch.bool,
                       device=image.device)
    mask[sorted_idx[:K_tau]] = True

    margin = float(cumsum[K_tau - 1].item() - extractor.cumulative_threshold)
    min_bar_alpha = float(bar_alpha[:K_tau].min().item())

    return mask, margin, min_bar_alpha, K_tau, weights


def jaccard(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    inter = (mask_a & mask_b).sum().item()
    union = (mask_a | mask_b).sum().item()
    return inter / union if union > 0 else 1.0


# ==========================================
# Excess-loss bound verification (Claim B)
# ==========================================

@torch.no_grad()
def measure_excess_loss_bound(A_orig: torch.Tensor,
                              mask_orig: torch.Tensor,
                              mask_pert: torch.Tensor,
                              csae_model=None):
    """Verify Claim B of Prop 5.2:
       |L_M - L_M'|  <=  (|M Δ M'| / |M|) * max_c ||A_c||_F^2

    If a CSAE checkpoint is provided, also measure |L_M - L_M'| directly
    by reconstructing A_orig and computing the masked reconstruction loss
    under both masks.

    Args:
        A_orig:      [1, C, H, W] target-layer activations (not perturbed --
                     Prop 5.2 holds A fixed and changes the mask)
        mask_orig:   [C] bool
        mask_pert:   [C] bool
        csae_model:  optional trained MultiChannelConvSAE

    Returns dict with diagnostic quantities.
    """
    sym_diff = int((mask_orig ^ mask_pert).sum().item())
    mask_size = max(int(mask_orig.sum().item()), 1)
    mask_size_p = max(int(mask_pert.sum().item()), 1)
    ratio = sym_diff / mask_size

    channel_norms_sq = (A_orig[0] ** 2).sum(dim=(1, 2))   # [C]
    max_norm_sq = float(channel_norms_sq.max().item())
    theoretical_bound = ratio * max_norm_sq

    out = {
        'sym_diff': sym_diff,
        'mask_size': mask_size,
        'mask_size_pert': mask_size_p,
        'ratio': float(ratio),
        'max_norm_sq': max_norm_sq,
        'theoretical_bound': float(theoretical_bound),
    }

    if csae_model is not None:
        recon, _ = csae_model(A_orig, use_topk=True)
        per_c_mse = ((recon - A_orig) ** 2).mean(dim=(2, 3))[0]   # [C]
        L_M = float(per_c_mse[mask_orig].mean().item()) if mask_size > 0 else 0.0
        L_Mp = float(per_c_mse[mask_pert].mean().item()) if mask_size_p > 0 else 0.0
        out['L_M'] = L_M
        out['L_M_pert'] = L_Mp
        out['measured_excess_loss'] = abs(L_M - L_Mp)

    return out


# ==========================================
# Lipschitz constant estimate (diagnostic)
# ==========================================

def estimate_L_alpha(extractor: MultiModelActivationExtractor,
                     images: torch.Tensor,
                     n_probes: int = 5,
                     probe_eps: float = 1e-3) -> float:
    ratios = []
    for i in range(images.size(0)):
        x = images[i:i+1]
        _, _, _, _, alpha_x = compute_mask_and_margin(extractor, x)

        for _ in range(n_probes):
            noise = torch.randn_like(x)
            noise = noise * (probe_eps / (noise.norm() + 1e-12))
            x_pert = x + noise

            _, _, _, _, alpha_xp = compute_mask_and_margin(extractor, x_pert)

            num = (alpha_x - alpha_xp).norm().item()
            den = noise.norm().item()
            if den > 1e-12:
                ratios.append(num / den)

    if not ratios:
        return float('nan')
    return float(np.percentile(ratios, 95))


# ==========================================
# Optional: load a trained CSAE matching the backbone
# ==========================================

def maybe_load_csae(args, extractor, device):
    if args.csae_checkpoint is None:
        return None
    if not os.path.exists(args.csae_checkpoint):
        print(f"WARNING: CSAE checkpoint not found: {args.csae_checkpoint}")
        return None

    in_channels = extractor.num_channels
    hidden_dim = in_channels * 8
    top_k = args.csae_top_k

    csae = MultiChannelConvSAE(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        kernel_size=1,
        top_k=top_k,
    ).to(device)
    state = torch.load(args.csae_checkpoint, map_location=device)
    csae.load_state_dict(state)
    csae.eval()
    print(f"  Loaded CSAE: in_ch={in_channels}, hidden={hidden_dim}, "
          f"top_k={top_k}")
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

    # ---------- Extractor ----------
    extractor = MultiModelActivationExtractor(
        model_name=args.model,
        target_layer=args.target_layer,
        device=device,
        cumulative_threshold=args.tau,
    )

    csae_model = maybe_load_csae(args, extractor, device)
    if csae_model is None:
        print("  (No CSAE checkpoint -- skipping measured excess loss.)")

    # ---------- L_alpha estimate ----------
    print("\n" + "=" * 70)
    print("Estimating L_alpha (effective Lipschitz of x -> alpha(x))...")
    print("=" * 70)
    probe_imgs = torch.stack([eval_dataset[i][0]
                              for i in range(min(20, len(eval_dataset)))])
    probe_imgs = probe_imgs.to(device)
    L_alpha = estimate_L_alpha(extractor, probe_imgs,
                               n_probes=5, probe_eps=1e-3)
    print(f"  L_alpha (95th pctile of ||dalpha||/||dx||): {L_alpha:.4f}")

    # ---------- Main loop ----------
    print("\n" + "=" * 70)
    print(f"Mask stability over {len(eval_dataset)} images, "
          f"tau={args.tau}, epsilons={args.epsilons}")
    print("=" * 70)

    results = {eps: [] for eps in args.epsilons}

    for img_idx, (image, label) in enumerate(tqdm(eval_loader, desc="Images")):
        image = image.to(device)

        mask_orig, margin, min_bar_alpha, K_tau, _ = \
            compute_mask_and_margin(extractor, image)
        A_orig = extractor.activations.clone()

        eps_star = (margin * min_bar_alpha / L_alpha) if L_alpha > 1e-12 \
            else float('inf')

        for eps in args.epsilons:
            j_trials, flipped_trials = [], []
            ratio_trials, theo_bound_trials, measured_trials = [], [], []

            for trial in range(args.trials_per_eps):
                noise = torch.randn_like(image)
                noise = noise * (eps / (noise.norm() + 1e-12))
                image_pert = image + noise

                mask_pert, _, _, _, _ = compute_mask_and_margin(
                    extractor, image_pert)

                j = jaccard(mask_orig, mask_pert)
                j_trials.append(j)
                flipped_trials.append(int(j < 1.0))

                bound_info = measure_excess_loss_bound(
                    A_orig, mask_orig, mask_pert, csae_model=csae_model)
                ratio_trials.append(bound_info['ratio'])
                theo_bound_trials.append(bound_info['theoretical_bound'])
                if 'measured_excess_loss' in bound_info:
                    measured_trials.append(bound_info['measured_excess_loss'])

            entry = {
                'img_idx': int(indices[img_idx]),
                'label': int(label.item()),
                'margin': margin,
                'min_bar_alpha': min_bar_alpha,
                'K_tau': K_tau,
                'eps_star': eps_star,
                'jaccard_mean': float(np.mean(j_trials)),
                'jaccard_std': float(np.std(j_trials)),
                'flip_rate': float(np.mean(flipped_trials)),
                'below_threshold': bool(eps < eps_star),
                'sym_diff_ratio_mean': float(np.mean(ratio_trials)),
                'theoretical_bound_mean': float(np.mean(theo_bound_trials)),
            }
            if measured_trials:
                entry['measured_excess_loss_mean'] = float(np.mean(measured_trials))
                entry['bound_ratio'] = (
                    float(np.mean(theo_bound_trials)) /
                    max(float(np.mean(measured_trials)), 1e-12)
                )
            results[eps].append(entry)

    # ---------- Aggregate ----------
    summary = {
        'config': {
            'model': args.model,
            'target_layer': extractor.target_layer_name,
            'tau': args.tau,
            'num_images': args.num_images,
            'epsilons': args.epsilons,
            'trials_per_eps': args.trials_per_eps,
            'L_alpha_est': L_alpha,
            'csae_checkpoint': args.csae_checkpoint,
        },
        'per_epsilon': {},
        'per_image': results,
    }

    # Image-level diagnostics
    rs0 = results[args.epsilons[0]]
    margins_all = [r['margin'] for r in rs0]
    minba_all = [r['min_bar_alpha'] for r in rs0]
    eps_star_all = [r['eps_star'] for r in rs0 if np.isfinite(r['eps_star'])]
    summary['diagnostics'] = {
        'L_alpha_est': L_alpha,
        'median_margin': float(np.median(margins_all)),
        'median_min_bar_alpha': float(np.median(minba_all)),
        'median_eps_star': float(np.median(eps_star_all)) if eps_star_all
                          else float('nan'),
        'max_eps_star': float(np.max(eps_star_all)) if eps_star_all
                       else float('nan'),
        'median_K_tau': float(np.median([r['K_tau'] for r in rs0])),
    }

    print("\n" + "=" * 70)
    print("Diagnostics (image-level, eps-independent)")
    print("=" * 70)
    for k, v in summary['diagnostics'].items():
        print(f"  {k:25s}: {v:.6g}")

    print("\n" + "=" * 70)
    print("Per-epsilon summary")
    print("=" * 70)
    header_cols = [
        ('eps',          '{:>8.4f}'),
        ('Jaccard',      '{:>9.4f}'),
        ('flip rate',    '{:>10.4f}'),
        ('|MΔM\'|/|M|',  '{:>11.4f}'),
        ('theo. bound',  '{:>12.4e}'),
    ]
    has_csae = csae_model is not None
    if has_csae:
        header_cols += [
            ('measured',   '{:>11.4e}'),
            ('bound/meas', '{:>11.2f}'),
        ]
    header_cols += [('rho(eps*,J)', '{:>12.4f}')]
    print(' '.join(f"{c[0]:>{int(c[1].split(':')[1].split('.')[0].lstrip('>'))}}"
                   for c in header_cols))
    print('-' * 110)

    for eps in args.epsilons:
        rs = results[eps]
        j_mean = np.mean([r['jaccard_mean'] for r in rs])
        flip = np.mean([r['flip_rate'] for r in rs])
        ratio_mean = np.mean([r['sym_diff_ratio_mean'] for r in rs])
        theo_mean = np.mean([r['theoretical_bound_mean'] for r in rs])

        eps_stars = [r['eps_star'] for r in rs]
        jaccs = [r['jaccard_mean'] for r in rs]
        finite = [(e, j) for e, j in zip(eps_stars, jaccs) if np.isfinite(e)]
        if len(finite) >= 5 and len(set(j for _, j in finite)) > 1:
            rho, _ = spearmanr([e for e, _ in finite], [j for _, j in finite])
        else:
            rho = float('nan')

        row_vals = [eps, j_mean, flip, ratio_mean, theo_mean]
        per_eps = {
            'mean_jaccard': float(j_mean),
            'flip_rate': float(flip),
            'mean_sym_diff_ratio': float(ratio_mean),
            'mean_theoretical_bound': float(theo_mean),
            'spearman_eps_star_vs_jaccard': float(rho),
        }
        if has_csae:
            meas_mean = np.mean([r['measured_excess_loss_mean'] for r in rs])
            br = theo_mean / max(meas_mean, 1e-12)
            row_vals += [meas_mean, br]
            per_eps['mean_measured_excess_loss'] = float(meas_mean)
            per_eps['mean_bound_over_measured'] = float(br)
        row_vals += [rho]

        row_str = ' '.join(c[1].format(v) for c, v in zip(header_cols, row_vals))
        print(row_str)
        summary['per_epsilon'][str(eps)] = per_eps

    # ---------- Save ----------
    tau_str = f"{args.tau:.2f}".replace('.', 'p')
    out_prefix = f"mask_stability_{args.model}_tau{tau_str}"

    json_path = f"{out_prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nRaw results: {json_path}")

    plot_path = f"{out_prefix}.png"
    make_plots(summary, plot_path, args.model, has_csae=has_csae)
    print(f"Plot: {plot_path}")

    report_path = f"{out_prefix}_report.txt"
    write_report(summary, report_path, args.model)
    print(f"Report: {report_path}")

    return summary


# ==========================================
# Plots
# ==========================================

def make_plots(summary, save_path, model_name, has_csae=False):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Grad-CAM Mask Stability ({model_name.upper()}, "
                 f"tau={summary['config']['tau']}) -- Prop. 5.2",
                 fontsize=13, fontweight='bold')

    eps_strs = list(summary['per_epsilon'].keys())
    epsilons = sorted([float(e) for e in eps_strs])
    eps_str_map = {float(e): e for e in eps_strs}

    # (a) Jaccard + flip rate vs eps -- P1
    ax = axes[0, 0]
    means = [summary['per_epsilon'][eps_str_map[e]]['mean_jaccard']
             for e in epsilons]
    flips = [summary['per_epsilon'][eps_str_map[e]]['flip_rate']
             for e in epsilons]
    ax.plot(epsilons, means, 'o-', color='C0', label='Mean Jaccard',
            linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(epsilons, flips, 's--', color='C3', label='Flip rate',
             linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Mean Jaccard', color='C0')
    ax2.set_ylabel('Flip rate', color='C3')
    ax.set_title('(a) P1: Jaccard & flip rate vs $\\epsilon$')
    ax.grid(True, alpha=0.3)
    j_min = min(means) - 0.02
    ax.set_ylim(max(0, j_min), 1.005)
    ax2.set_ylim(0, 1.05)

    # (b) sym diff ratio vs eps -- P3
    ax = axes[0, 1]
    ratios = [summary['per_epsilon'][eps_str_map[e]]['mean_sym_diff_ratio']
              for e in epsilons]
    ax.plot(epsilons, ratios, 'o-', color='C4', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r"Mean $|M \triangle M'| / |M|$")
    ax.set_title('(b) P3: Boundary-churn ratio')
    ax.grid(True, alpha=0.3)

    # (c) margin vs Jaccard at median eps -- P2
    mid_eps = epsilons[len(epsilons) // 2]
    rs = summary['per_image'][eps_str_map[mid_eps]]
    ax = axes[0, 2]
    margins = [r['margin'] for r in rs]
    j_scatter = [r['jaccard_mean'] for r in rs]
    ax.scatter(margins, j_scatter, alpha=0.4, s=15, c='C2')
    ax.set_xlabel(r'$\delta(x)$')
    ax.set_ylabel('Mask Jaccard')
    ax.set_title(f'(c) P2: $\\delta(x)$ vs Jaccard '
                 f'($\\epsilon$={mid_eps:.3f})')
    ax.grid(True, alpha=0.3)
    if len(margins) >= 5:
        rho, _ = spearmanr(margins, j_scatter)
        ax.text(0.05, 0.05, f'Spearman $\\rho$ = {rho:.3f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # (d) excess-loss bound: theoretical vs measured -- P4
    ax = axes[1, 0]
    if has_csae:
        eps_for_scatter = epsilons[-1]
        rs = summary['per_image'][eps_str_map[eps_for_scatter]]
        theo = [r['theoretical_bound_mean'] for r in rs
                if r['theoretical_bound_mean'] > 0]
        meas = [r['measured_excess_loss_mean'] for r in rs
                if r['theoretical_bound_mean'] > 0
                and r.get('measured_excess_loss_mean', 0) > 0]
        if theo and meas and len(theo) == len(meas):
            ax.scatter(meas, theo, alpha=0.4, s=15, c='C1')
            lo = min(min(theo), min(meas)) * 0.5 + 1e-12
            hi = max(max(theo), max(meas)) * 2
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5,
                    label='bound = measured')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r"Measured $|L_M - L_{M'}|$")
            ax.set_ylabel('Theoretical bound')
            ax.set_title(f'(d) P4: Bound vs measured '
                         f'($\\epsilon$={eps_for_scatter:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3, which='both')
        else:
            ax.text(0.5, 0.5, 'No valid bound/measured pairs',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(d) P4: Excess-loss bound')
    else:
        ax.text(0.5, 0.5,
                'CSAE checkpoint not provided.\n'
                'Pass --csae_checkpoint to enable\n'
                'measured excess-loss verification.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='gray')
        ax.set_title('(d) P4: Excess-loss bound (disabled)')
        ax.set_xticks([]); ax.set_yticks([])

    # (e) margin distribution
    ax = axes[1, 1]
    rs = summary['per_image'][eps_str_map[epsilons[0]]]
    margins_all = [r['margin'] for r in rs]
    ax.hist(margins_all, bins=40, alpha=0.7, color='C0')
    med = np.median(margins_all)
    ax.axvline(med, color='r', linestyle='--', label=f'median = {med:.4f}')
    ax.set_xlabel(r'$\delta(x)$')
    ax.set_ylabel('Count')
    ax.set_title(r'(e) Distribution of $\delta(x)$')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (f) Spearman(eps*, Jaccard) across eps
    ax = axes[1, 2]
    rhos = [summary['per_epsilon'][eps_str_map[e]]['spearman_eps_star_vs_jaccard']
            for e in epsilons]
    ax.plot(epsilons, rhos, 'o-', color='C5', linewidth=2)
    ax.axhline(0, color='k', alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r"Spearman $\rho$($\epsilon^*$, Jaccard)")
    ax.set_title('(f) Does $\\epsilon^*(x)$ rank-order stability?')
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
    lines.append(f"Mask Stability Report -- {model_name.upper()} "
                 f"(tau={summary['config']['tau']})")
    lines.append("(Empirical verification of Proposition 5.2)")
    lines.append("=" * 72)
    lines.append("")

    cfg = summary['config']
    lines.append("Configuration:")
    for k, v in cfg.items():
        lines.append(f"  {k:25s} : {v}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("Image-level diagnostics (eps-independent):")
    lines.append("-" * 72)
    for k, v in summary['diagnostics'].items():
        lines.append(f"  {k:25s} : {v:.6g}")
    lines.append("")
    lines.append(f"  Interpretation:")
    lines.append(f"    The worst-case Lipschitz threshold eps* has median")
    lines.append(f"    {summary['diagnostics']['median_eps_star']:.2e} across the dataset.")
    lines.append(f"    Tested epsilons range from {min(cfg['epsilons']):.4f} to "
                 f"{max(cfg['epsilons']):.4f}.")
    lines.append(f"    If all tested epsilons exceed eps*, Claim A is not")
    lines.append(f"    exercised at these scales. Claim B (excess-loss bound)")
    lines.append(f"    is the testable part at practical perturbation magnitudes.")
    lines.append("")

    lines.append("-" * 72)
    lines.append("Per-epsilon summary:")
    lines.append("-" * 72)
    has_meas = any('mean_measured_excess_loss' in m
                   for m in summary['per_epsilon'].values())

    hdr = (f"{'eps':>8} {'Jaccard':>9} {'flip':>8} "
           f"{'sd_ratio':>10} {'theo_bound':>14}")
    if has_meas:
        hdr += f" {'measured':>13} {'bnd/meas':>10}"
    hdr += f" {'rho(e*,J)':>11}"
    lines.append(hdr)

    for eps_str, m in summary['per_epsilon'].items():
        row = (f"{float(eps_str):>8.4f} "
               f"{m['mean_jaccard']:>9.4f} "
               f"{m['flip_rate']:>8.4f} "
               f"{m['mean_sym_diff_ratio']:>10.4f} "
               f"{m['mean_theoretical_bound']:>14.4e}")
        if 'mean_measured_excess_loss' in m:
            row += (f" {m['mean_measured_excess_loss']:>13.4e} "
                    f"{m['mean_bound_over_measured']:>10.2f}")
        row += f" {m['spearman_eps_star_vs_jaccard']:>11.4f}"
        lines.append(row)
    lines.append("")

    lines.append("Reading the table:")
    lines.append("  - 'Jaccard'    : mean mask Jaccard under perturbation (1=stable).")
    lines.append("  - 'flip'       : fraction of trials where Jaccard < 1.")
    lines.append("  - 'sd_ratio'   : |M Δ M'|/|M|. Small => boundary churn only.")
    lines.append("  - 'theo_bound' : Claim B upper bound on |L_M - L_M'|.")
    lines.append("  - 'measured'   : actual |L_M - L_M'| using the CSAE (if loaded).")
    lines.append("  - 'bnd/meas'   : Claim B tightness (1=tight, >>1=loose).")
    lines.append("  - 'rho(e*, J)' : Spearman corr. between per-image eps*(x)")
    lines.append("                   and observed Jaccard. > 0 means eps*(x) is")
    lines.append("                   a useful relative predictor even if loose.")
    lines.append("")
    lines.append("What to look for:")
    lines.append("  P1: Mean Jaccard should decrease monotonically in eps.")
    lines.append("  P2: rho(margin, Jaccard) > 0 at fixed eps (panel c in figure).")
    lines.append("  P3: sd_ratio stays small (e.g. < 0.05) even at large eps,")
    lines.append("      meaning only boundary channels churn.")
    lines.append("  P4: bnd/meas is finite and not absurdly large (< ~100).")
    lines.append("      A constant offset between bound and measured is OK --")
    lines.append("      it just means the bound has a known slack factor.")

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Empirical verification of Prop 5.2 (mask stability)"
    )
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--target_layer', type=str, default=None)
    parser.add_argument('--tau', type=float, default=0.95)
    parser.add_argument('--num_images', type=int, default=500)
    parser.add_argument('--epsilons', type=float, nargs='+',
                        default=[0.001, 0.005, 0.01, 0.05, 0.1])
    parser.add_argument('--trials_per_eps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--csae_checkpoint', type=str, default=None,
                        help='Path to .pth state_dict of trained CSAE')
    parser.add_argument('--csae_top_k', type=int, default=32,
                        help='Top-K used at CSAE training time')

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()