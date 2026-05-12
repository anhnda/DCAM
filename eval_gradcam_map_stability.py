"""
Empirical verification of Proposition 5.3 (Grad-CAM Spatial Map Stability)
==========================================================================

Verifies that the Grad-CAM heatmap L_GC(x) is stable under input perturbations.

Prop 5.3 (raw map, unnormalized):
  ||~L_GC(x) - ~L_GC(x')||_F <= L_~L * eps
  with L_~L = L_Phi * ( sqrt(C) * B_A * L_H / (HW) + ||alpha||_2 )

Prop 5.3 (normalized map):
  ||L_GC(x) - L_GC(x')||_F <= (2 * L_~L / m) * eps
  where m = min ||~L_GC(.)||_1 over x and x' (the degeneracy bound).

Predictions tested:
  (P1) Raw map perturbation is bounded linearly in eps.
  (P2) Normalized map perturbation is also bounded linearly in eps,
       with the prefactor 2*L_~L/m increasing as m -> 0 (degeneracy).
  (P3) Per-image L_~L (computed from alpha and bA) predicts the
       observed map perturbation across the dataset (Spearman rank).
  (P4) Per-image m(x) = ||~L_GC(x)||_1 predicts normalized-map
       sensitivity: small m => large perturbation under same eps.

Why this matters for DCAM:
  The decomposition loss L_gc uses L_GC(x) as a *target* for sum_d z_d.
  If L_GC is highly sensitive to input perturbations, then the training
  target itself is noisy, and the learned decomposition cannot be more
  stable than its target. This experiment quantifies that noise floor.

Outputs:
  - gradcam_map_stability_<model>_tau<...>.json
  - gradcam_map_stability_<model>_tau<...>.png  (6 panels)
  - gradcam_map_stability_<model>_tau<...>_report.txt

Usage:
    python eval_gradcam_map_stability.py --model resnet50 --tau 0.95 --num_images 500
    python eval_gradcam_map_stability.py --model resnet18 --tau 0.85 --num_images 1000 \
        --epsilons 0.001 0.005 0.01 0.05 0.1
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
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
# Compute alpha, raw map, normalized map, and per-image L_~L proxy
# ==========================================

def compute_alpha_and_maps(extractor: MultiModelActivationExtractor,
                            image: torch.Tensor,
                            class_idx: int = None,
                            eps_norm: float = 1e-8):
    """Run gradcam.forward, then build:
       - alpha vector             [C]
       - activations bA           [1, C, H, W]
       - raw map ~L_GC            [H, W]   = ReLU(sum_k alpha_k * A_k)
       - normalized map L_GC      [H, W]   = ~L_GC / sum(~L_GC)
       - per-image L_tilde_L proxy (the theoretical Lipschitz of ~L_GC):
            L_proxy = L_Phi * ( sqrt(C) * ||bA||_F * L_H / (HW) + ||alpha||_2 )
       Since L_Phi and L_H are not directly available, we report:
            (a) ||alpha||_2  -- a *known* lower bound on L_proxy
            (b) ||bA||_F     -- needed for the bA*L_H/(HW) term
       The actual L_proxy can be estimated empirically by finite-difference
       probes; we do that for a small subset in estimate_L_tilde().

       Returns dict with:
         'alpha':            [C]
         'A':                [1, C, H, W]   (NOT cloned -- caller owns)
         'raw_map':          [H, W]
         'norm_map':         [H, W]
         'm':                float, ||raw_map||_1 (the degeneracy lower bound)
         'alpha_norm':       float, ||alpha||_2
         'A_frob':           float, ||A||_F
    """
    with torch.enable_grad():
        weights, _, _ = extractor.gradcam.forward(image, class_idx=class_idx,
                                                   verbose=False)
    weights = weights.detach()                       # [C]
    A = extractor.activations.detach()               # [1, C, H, W]

    # Raw map
    # weighted = sum_k alpha_k * A_k  : [H, W]
    weighted = (weights.view(-1, 1, 1) * A[0]).sum(dim=0)
    raw_map = F.relu(weighted)
    s = raw_map.sum()
    if s > eps_norm:
        norm_map = raw_map / s
    else:
        # Degenerate: uniform fallback
        norm_map = torch.full_like(raw_map, 1.0 / raw_map.numel())

    return {
        'alpha':      weights,
        'A':          A,
        'raw_map':    raw_map,
        'norm_map':   norm_map,
        'm':          float(s.item()),
        'alpha_norm': float(weights.norm(p=2).item()),
        'A_frob':     float(A.norm(p='fro').item()),
    }


# ==========================================
# Empirical estimate of L_tilde_L
# (Lipschitz of x -> ~L_GC(x), the raw heatmap)
# ==========================================

def estimate_L_tilde(extractor: MultiModelActivationExtractor,
                     images: torch.Tensor,
                     n_probes: int = 5,
                     probe_eps: float = 1e-3) -> float:
    """Estimate L_~L = sup ||~L_GC(x) - ~L_GC(x')||_F / ||x - x'||
    via random small perturbations on a batch of images.
    Returns 95th-percentile estimate.
    """
    ratios = []
    for i in range(images.size(0)):
        x = images[i:i+1]
        info_x = compute_alpha_and_maps(extractor, x)
        rmap_x = info_x['raw_map']

        for _ in range(n_probes):
            noise = torch.randn_like(x)
            noise = noise * (probe_eps / (noise.norm() + 1e-12))
            x_pert = x + noise

            info_xp = compute_alpha_and_maps(extractor, x_pert)
            rmap_xp = info_xp['raw_map']

            num = (rmap_x - rmap_xp).norm().item()
            den = noise.norm().item()
            if den > 1e-12:
                ratios.append(num / den)

    if not ratios:
        return float('nan')
    return float(np.percentile(ratios, 95))


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

    # ---------- Estimate L_tilde ----------
    print("\n" + "=" * 70)
    print("Estimating L_~L (effective Lipschitz of x -> ~L_GC(x))...")
    print("=" * 70)
    probe_imgs = torch.stack([eval_dataset[i][0]
                              for i in range(min(20, len(eval_dataset)))])
    probe_imgs = probe_imgs.to(device)
    L_tilde = estimate_L_tilde(extractor, probe_imgs,
                               n_probes=5, probe_eps=1e-3)
    print(f"  L_~L (95th pctile of ||d~L||_F / ||dx||): {L_tilde:.4f}")

    # ---------- Main loop ----------
    print("\n" + "=" * 70)
    print(f"Grad-CAM map stability over {len(eval_dataset)} images, "
          f"tau={args.tau}, epsilons={args.epsilons}")
    print("=" * 70)

    # Use string keys (consistent with JSON serialization)
    results = {str(eps): [] for eps in args.epsilons}

    for img_idx, (image, label) in enumerate(tqdm(eval_loader, desc="Images")):
        image = image.to(device)

        info_orig = compute_alpha_and_maps(extractor, image)
        rmap_orig = info_orig['raw_map']
        nmap_orig = info_orig['norm_map']
        m_orig = info_orig['m']

        for eps in args.epsilons:
            raw_perts, norm_perts, m_perts = [], [], []

            for trial in range(args.trials_per_eps):
                noise = torch.randn_like(image)
                noise = noise * (eps / (noise.norm() + 1e-12))
                image_pert = image + noise

                info_pert = compute_alpha_and_maps(extractor, image_pert)
                rmap_pert = info_pert['raw_map']
                nmap_pert = info_pert['norm_map']

                raw_perts.append(
                    (rmap_orig - rmap_pert).norm().item())
                norm_perts.append(
                    (nmap_orig - nmap_pert).norm().item())
                m_perts.append(info_pert['m'])

            m_min = min(m_orig, min(m_perts))   # per-eps degeneracy lower bound
            entry = {
                'img_idx':              int(indices[img_idx]),
                'label':                int(label.item()),
                'm_orig':               float(m_orig),
                'm_min_pair':           float(m_min),
                'alpha_norm':           float(info_orig['alpha_norm']),
                'A_frob':               float(info_orig['A_frob']),
                'raw_map_pert_mean':    float(np.mean(raw_perts)),
                'raw_map_pert_std':     float(np.std(raw_perts)),
                'norm_map_pert_mean':   float(np.mean(norm_perts)),
                'norm_map_pert_std':    float(np.std(norm_perts)),

                # Theoretical bounds (per-image, with empirical L_~L)
                'raw_theo_bound':       float(L_tilde * eps),
                'norm_theo_bound':      float(
                    (2.0 * L_tilde / max(m_min, 1e-12)) * eps),
            }
            results[str(eps)].append(entry)

    # ---------- Aggregate ----------
    summary = {
        'config': {
            'model': args.model,
            'target_layer': extractor.target_layer_name,
            'tau': args.tau,
            'num_images': args.num_images,
            'epsilons': args.epsilons,
            'trials_per_eps': args.trials_per_eps,
            'L_tilde_est': L_tilde,
        },
        'per_epsilon': {},
        'per_image': results,
        'diagnostics': {},
    }

    rs0 = results[str(args.epsilons[0])]
    m_all = [r['m_orig'] for r in rs0]
    summary['diagnostics'] = {
        'L_tilde_est':           L_tilde,
        'median_m':              float(np.median(m_all)),
        'q5_m':                  float(np.percentile(m_all, 5)),
        'q95_m':                 float(np.percentile(m_all, 95)),
        'min_m':                 float(np.min(m_all)),
        'frac_m_below_1':        float(np.mean([m < 1.0 for m in m_all])),
        'mean_alpha_norm':       float(np.mean([r['alpha_norm'] for r in rs0])),
        'mean_A_frob':           float(np.mean([r['A_frob'] for r in rs0])),
    }

    print("\n" + "=" * 70)
    print("Diagnostics (image-level)")
    print("=" * 70)
    for k, v in summary['diagnostics'].items():
        print(f"  {k:25s}: {v:.6g}")

    print("\n" + "=" * 70)
    print("Per-epsilon summary")
    print("=" * 70)
    header = (f"{'eps':>8} {'raw_pert':>12} {'raw_bound':>12} "
              f"{'raw_b/m':>9} {'norm_pert':>12} {'norm_bound':>12} "
              f"{'norm_b/m':>10} {'rho(m,np)':>11}")
    print(header)
    print('-' * len(header))

    for eps in args.epsilons:
        rs = results[str(eps)]
        rp_mean = np.mean([r['raw_map_pert_mean'] for r in rs])
        rb_mean = np.mean([r['raw_theo_bound'] for r in rs])
        np_mean = np.mean([r['norm_map_pert_mean'] for r in rs])
        nb_mean = np.mean([r['norm_theo_bound'] for r in rs])

        raw_ratio = rb_mean / max(rp_mean, 1e-12)
        norm_ratio = nb_mean / max(np_mean, 1e-12)

        # Does m(x) rank-order normalized-map perturbations? Theory says
        # smaller m -> larger normalized perturbation (inverse rank).
        ms = [r['m_orig'] for r in rs]
        nps = [r['norm_map_pert_mean'] for r in rs]
        if len(ms) >= 5 and len(set(nps)) > 1:
            rho_m_np, _ = spearmanr(ms, nps)
        else:
            rho_m_np = float('nan')

        print(f"{eps:>8.4f} {rp_mean:>12.4e} {rb_mean:>12.4e} "
              f"{raw_ratio:>9.2f} {np_mean:>12.4e} {nb_mean:>12.4e} "
              f"{norm_ratio:>10.2f} {rho_m_np:>11.4f}")

        summary['per_epsilon'][str(eps)] = {
            'mean_raw_map_pert':     float(rp_mean),
            'mean_raw_theo_bound':   float(rb_mean),
            'raw_bound_over_meas':   float(raw_ratio),
            'mean_norm_map_pert':    float(np_mean),
            'mean_norm_theo_bound':  float(nb_mean),
            'norm_bound_over_meas':  float(norm_ratio),
            'spearman_m_vs_norm_pert': float(rho_m_np),
        }

    # ---------- Save ----------
    tau_str = f"{args.tau:.2f}".replace('.', 'p')
    out_prefix = f"gradcam_map_stability_{args.model}_tau{tau_str}"

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
    fig.suptitle(f"Grad-CAM Map Stability ({model_name.upper()}, "
                 f"tau={summary['config']['tau']}) -- Prop. 5.3",
                 fontsize=13, fontweight='bold')

    eps_strs = list(summary['per_epsilon'].keys())
    epsilons = sorted([float(e) for e in eps_strs])
    eps_str_map = {float(e): e for e in eps_strs}

    # ---------- (a) raw-map perturbation vs eps (P1) ----------
    ax = axes[0, 0]
    rp = [summary['per_epsilon'][eps_str_map[e]]['mean_raw_map_pert']
          for e in epsilons]
    rb = [summary['per_epsilon'][eps_str_map[e]]['mean_raw_theo_bound']
          for e in epsilons]
    ax.plot(epsilons, rp, 'o-', color='C0', linewidth=2,
            label='Measured raw-map perturbation')
    ax.plot(epsilons, rb, 's--', color='C3', linewidth=2,
            label='Theoretical bound $L_{\\tilde L}\\epsilon$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r"$\|\tilde{L}_{GC}(x) - \tilde{L}_{GC}(x')\|_F$")
    ax.set_title('(a) P1: raw-map perturbation scaling')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # ---------- (b) normalized-map perturbation vs eps (P2) ----------
    ax = axes[0, 1]
    np_p = [summary['per_epsilon'][eps_str_map[e]]['mean_norm_map_pert']
            for e in epsilons]
    nb = [summary['per_epsilon'][eps_str_map[e]]['mean_norm_theo_bound']
          for e in epsilons]
    ax.plot(epsilons, np_p, 'o-', color='C0', linewidth=2,
            label='Measured normalized-map perturbation')
    ax.plot(epsilons, nb, 's--', color='C3', linewidth=2,
            label=r'Theoretical bound $(2L_{\tilde L}/m)\epsilon$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r"$\|L_{GC}(x) - L_{GC}(x')\|_F$")
    ax.set_title('(b) P2: normalized-map perturbation scaling')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # ---------- (c) m(x) vs normalized-map perturbation (P4) ----------
    mid_eps = epsilons[len(epsilons) // 2]
    rs = summary['per_image'][eps_str_map[mid_eps]]
    ms = [r['m_orig'] for r in rs]
    nps = [r['norm_map_pert_mean'] for r in rs]
    ax = axes[0, 2]
    ax.scatter(ms, nps, alpha=0.4, s=15, c='C2')
    ax.set_xlabel(r'$m(x) = \|\tilde{L}_{GC}(x)\|_1$')
    ax.set_ylabel('Normalized-map perturbation')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'(c) P4: small $m(x)$ predicts higher sensitivity '
                 f'($\\epsilon$={mid_eps:.3f})')
    ax.grid(True, alpha=0.3, which='both')
    if len(ms) >= 5:
        rho, _ = spearmanr(ms, nps)
        ax.text(0.05, 0.95, f'Spearman $\\rho$ = {rho:.3f}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ---------- (d) per-image bound vs measured (raw map) ----------
    ax = axes[1, 0]
    largest_eps = epsilons[-1]
    rs = summary['per_image'][eps_str_map[largest_eps]]
    pts_x = [r['raw_map_pert_mean'] for r in rs if r['raw_map_pert_mean'] > 0]
    pts_y = [r['raw_theo_bound'] for r in rs if r['raw_map_pert_mean'] > 0]
    if pts_x and pts_y:
        ax.scatter(pts_x, pts_y, alpha=0.4, s=15, c='C1')
        lo = min(min(pts_x), min(pts_y)) * 0.5 + 1e-12
        hi = max(max(pts_x), max(pts_y)) * 2
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5,
                label='bound = measured')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Measured raw-map perturbation')
    ax.set_ylabel('Theoretical bound')
    ax.set_title(f'(d) Raw-map bound vs measured '
                 f'($\\epsilon$={largest_eps:.3f})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # ---------- (e) distribution of m(x) ----------
    ax = axes[1, 1]
    rs = summary['per_image'][eps_str_map[epsilons[0]]]
    m_all = [r['m_orig'] for r in rs]
    ax.hist(m_all, bins=40, alpha=0.7, color='C0',
            label=r'$m(x) = \|\tilde{L}_{GC}(x)\|_1$')
    med = np.median(m_all)
    p5 = np.percentile(m_all, 5)
    ax.axvline(med, color='r', linestyle='--', label=f'median={med:.3f}')
    ax.axvline(p5, color='orange', linestyle='--', label=f'5th pctile={p5:.3f}')
    ax.set_xlabel(r'$m(x)$')
    ax.set_ylabel('Count')
    ax.set_title(r'(e) Distribution of $m(x)$ (degeneracy diagnostic)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # ---------- (f) Spearman(m, norm_pert) across eps ----------
    ax = axes[1, 2]
    rhos = [summary['per_epsilon'][eps_str_map[e]]['spearman_m_vs_norm_pert']
            for e in epsilons]
    ax.plot(epsilons, rhos, 'o-', color='C5', linewidth=2)
    ax.axhline(0, color='k', alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r"Spearman $\rho(m(x),$ norm. perturbation$)$")
    ax.set_title('(f) Does $m(x)$ rank-order sensitivity?')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()


# ==========================================
# Report
# ==========================================

def write_report(summary, save_path, model_name):
    lines = []
    lines.append("=" * 72)
    lines.append(f"Grad-CAM Map Stability Report -- {model_name.upper()} "
                 f"(tau={summary['config']['tau']})")
    lines.append("(Empirical verification of Proposition 5.3)")
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
    lines.append(f"  - m(x) = ||~L_GC(x)||_1 governs the normalized-map")
    lines.append(f"    perturbation bound. The bound scales as (2 L_~L / m) * eps.")
    lines.append(f"  - Median m = {summary['diagnostics']['median_m']:.4g}, "
                 f"5th pctile = {summary['diagnostics']['q5_m']:.4g}.")
    lines.append(f"  - L_~L estimate = {summary['diagnostics']['L_tilde_est']:.4f}")
    lines.append(f"  - Small m(x) (degenerate samples) inflates the bound.")
    lines.append("")

    lines.append("-" * 72)
    lines.append("Per-epsilon summary:")
    lines.append("-" * 72)
    lines.append(f"{'eps':>8} {'raw_pert':>12} {'raw_bnd':>12} "
                 f"{'raw_b/m':>9} {'norm_pert':>12} {'norm_bnd':>12} "
                 f"{'norm_b/m':>10} {'rho(m,np)':>11}")
    for eps_str, m in summary['per_epsilon'].items():
        lines.append(
            f"{float(eps_str):>8.4f} "
            f"{m['mean_raw_map_pert']:>12.4e} "
            f"{m['mean_raw_theo_bound']:>12.4e} "
            f"{m['raw_bound_over_meas']:>9.2f} "
            f"{m['mean_norm_map_pert']:>12.4e} "
            f"{m['mean_norm_theo_bound']:>12.4e} "
            f"{m['norm_bound_over_meas']:>10.2f} "
            f"{m['spearman_m_vs_norm_pert']:>11.4f}"
        )
    lines.append("")

    lines.append("Reading the table:")
    lines.append("  - raw_pert  : ||~L_GC(x) - ~L_GC(x')||_F, measured.")
    lines.append("  - raw_bnd   : Prop 5.3 raw-map bound L_~L * eps.")
    lines.append("  - raw_b/m   : tightness ratio (1 = tight, >1 = loose).")
    lines.append("  - norm_pert : ||L_GC(x) - L_GC(x')||_F (normalized), measured.")
    lines.append("  - norm_bnd  : Prop 5.3 normalized bound (2 L_~L / m) * eps,")
    lines.append("                using the per-image worst-case m_min_pair.")
    lines.append("  - rho(m,np) : Spearman corr between m(x) and norm-map")
    lines.append("                perturbation. Negative => smaller m, larger")
    lines.append("                perturbation, exactly as theory predicts.")
    lines.append("")
    lines.append("What to look for:")
    lines.append("  P1: raw_pert grows linearly in eps (slope ~ 1 on log-log).")
    lines.append("  P2: norm_pert grows linearly in eps (also slope ~ 1).")
    lines.append("  P3: raw_bnd >= raw_pert always (raw bound holds).")
    lines.append("  P4: rho(m, np) < 0 (smaller m => larger normalized pert).")

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Empirical verification of Prop 5.3 (Grad-CAM map stability)"
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
    args = parser.parse_args()

    run_evaluation(args)


if __name__ == "__main__":
    main()