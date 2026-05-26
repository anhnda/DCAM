"""
Ablation: does L_gc widen gamma_K and improve Top-K stability?
================================================================

This script verifies Claim C of the Remark following Proposition 5.6:

    "The decomposition loss L_gc rewards active features whose spatial
     sum tracks L_GC, encouraging the top-K features to carry concentrated,
     high-mass activations and pushing weak features toward zero.
     Heuristically, this widens gamma_K and improves Top-K stability."

We compare two ConvSAE checkpoints that share the same architecture,
backbone, dataset, optimizer, schedule, and seed, differing only in
whether lambda_gc = 0 (baseline) or lambda_gc > 0 (with decomposition
loss). For each checkpoint we measure:

  (M1) Distribution of gamma_K(x) across a held-out image set.
       Claim C predicts the L_gc distribution is shifted to LARGER values.

  (M2) Top-K Jaccard under random L_2 input perturbations across the
       same epsilons used in eval_topk_stability.py.
       Claim C predicts L_gc has HIGHER Jaccard at matched epsilon.

  (M3) Per-image paired comparison: matched (x, epsilon) trial gives
       (gamma_K_baseline, gamma_K_gc, Jaccard_baseline, Jaccard_gc).
       We report paired tests (Wilcoxon signed-rank) on these matched
       differences -- the strongest evidence available without retraining
       multiple seeds.

  (M4) [Optional] If training logs (gamma_K vs epoch) are provided via
       --baseline_log and --gc_log (JSON files with a "gamma_K_history"
       key), plot the two trajectories.

Outputs:
  - gc_ablation_<model>_topk<K>.json
  - gc_ablation_<model>_topk<K>.png      (multi-panel)
  - gc_ablation_<model>_topk<K>_report.txt

Usage:
    python compare_gc_ablation.py --model resnet50 \
        --baseline_checkpoint imagenet1k_csae_resnet50_baseline_model.pth \
        --gc_checkpoint       imagenet1k_csae_resnet50_gcsum_model.pth \
        --csae_top_k 32 --num_images 500

Notes:
  - Both checkpoints MUST share architecture (in_channels, hidden_dim,
    kernel_size, top_k). The script asserts this.
  - The input perturbation seed is fixed across the two checkpoints so
    that comparisons are paired on identical noise realizations.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr

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

# Re-use core utilities from the single-checkpoint script if it's on PYTHONPATH;
# otherwise inline them here for self-containedness.
try:
    from misc.eval_topk_stability import (
        compute_topk_support, jaccard, estimate_L_f, estimate_L_phi, load_csae,
    )
except ImportError:
    # Inline fallback ----------------------------------------------------
    @torch.no_grad()
    def compute_topk_support(csae_model, A, K):
        z_pre = F.relu(csae_model.encoder(A))
        scores = z_pre[0].sum(dim=(1, 2))
        D = scores.shape[0]
        sorted_vals, sorted_idx = torch.sort(scores, descending=True)
        support = torch.zeros(D, dtype=torch.bool, device=A.device)
        support[sorted_idx[:K]] = True
        gamma_K = (float(sorted_vals[K - 1].item() - sorted_vals[K].item())
                   if K < D else float('inf'))
        return support, scores, gamma_K, z_pre

    def jaccard(a, b):
        inter = (a & b).sum().item()
        union = (a | b).sum().item()
        return inter / union if union > 0 else 1.0

    @torch.no_grad()
    def estimate_L_f(csae_model):
        W = csae_model.encoder.weight.detach()
        if W.dim() == 4:
            W = W.view(W.shape[0], -1)
        return float(torch.linalg.matrix_norm(W, ord=2).item())

    @torch.no_grad()
    def estimate_L_phi(extractor, images, n_probes=5, probe_eps=1e-3):
        ratios = []
        for i in range(images.size(0)):
            x = images[i:i + 1]
            _ = extractor.model(x)
            A_x = extractor.activations.clone()
            for _ in range(n_probes):
                noise = torch.randn_like(x)
                noise = noise * (probe_eps / (noise.norm() + 1e-12))
                _ = extractor.model(x + noise)
                A_xp = extractor.activations.clone()
                num = (A_x - A_xp).norm().item()
                den = noise.norm().item()
                if den > 1e-12:
                    ratios.append(num / den)
        return float(np.percentile(ratios, 95)) if ratios else float('nan')

    def load_csae(checkpoint, in_channels, top_k, device):
        csae = MultiChannelConvSAE(
            in_channels=in_channels,
            hidden_dim=in_channels * 8,
            kernel_size=1,
            top_k=top_k,
        ).to(device)
        state = torch.load(checkpoint, map_location=device)
        csae.load_state_dict(state)
        csae.eval()
        return csae


# ==========================================================================
# Paired per-image measurement
# ==========================================================================

def measure_checkpoint(extractor, csae_model, eval_dataset, indices,
                       K, epsilons, trials_per_eps, noise_seed, device):
    """Run M1+M2+M3 measurements for one checkpoint.

    Returns a dict keyed by str(eps) -> list of per-image entries, each
    with gamma_K, jaccard_mean (across trials), flip_rate, sym_diff_ratio.

    noise_seed: the SAME seed is used for both checkpoints so that
    noise realizations match across them (proper pairing).
    """
    loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=2)
    results = {str(eps): [] for eps in epsilons}

    # Deterministic noise: pre-generate per (image_idx, trial) to enforce
    # exact pairing between the two checkpoints.
    gen = torch.Generator(device='cpu').manual_seed(noise_seed)

    for img_idx, (image, label) in enumerate(tqdm(loader, desc="Images")):
        image = image.to(device)

        _ = extractor.model(image)
        A_orig = extractor.activations.clone()
        support_orig, _, gamma_K, _ = compute_topk_support(csae_model, A_orig, K)

        for eps in epsilons:
            j_trials, flip_trials, sd_trials = [], [], []
            for t in range(trials_per_eps):
                if eps == 0.0:
                    image_pert = image
                else:
                    noise = torch.randn(image.shape, generator=gen).to(device)
                    noise = noise * (eps / (noise.norm() + 1e-12))
                    image_pert = image + noise

                _ = extractor.model(image_pert)
                A_pert = extractor.activations.clone()
                support_pert, _, _, _ = compute_topk_support(
                    csae_model, A_pert, K)

                j = jaccard(support_orig, support_pert)
                j_trials.append(j)
                flip_trials.append(int(j < 1.0))
                sd_trials.append(int((support_orig ^ support_pert).sum().item()))

            results[str(eps)].append({
                'img_idx':         int(indices[img_idx]),
                'label':           int(label.item()),
                'gamma_K':         float(gamma_K),
                'jaccard_mean':    float(np.mean(j_trials)),
                'flip_rate':       float(np.mean(flip_trials)),
                'sym_diff_mean':   float(np.mean(sd_trials)),
                'sym_diff_ratio':  float(np.mean(sd_trials)) / max(K, 1),
            })
    return results


# ==========================================================================
# Statistical comparison
# ==========================================================================

def compare_distributions(baseline_results, gc_results, epsilons):
    """Per-eps comparison + paired tests."""
    comparison = {}

    eps0 = str(epsilons[0])
    gamma_b = np.array([r['gamma_K'] for r in baseline_results[eps0]])
    gamma_g = np.array([r['gamma_K'] for r in gc_results[eps0]])

    # M1: distribution-level test on gamma_K. Use Mann-Whitney U with
    # one-sided alternative (gc > baseline). Also report Wilcoxon paired,
    # since the same images appear in both groups (matched on img_idx).
    try:
        mwu = mannwhitneyu(gamma_g, gamma_b, alternative='greater')
        mwu_stat, mwu_p = float(mwu.statistic), float(mwu.pvalue)
    except Exception:
        mwu_stat, mwu_p = float('nan'), float('nan')

    diff_gamma = gamma_g - gamma_b
    try:
        w = wilcoxon(diff_gamma, alternative='greater')
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    except Exception:
        w_stat, w_p = float('nan'), float('nan')

    cohens_d = (float(np.mean(diff_gamma) / (np.std(diff_gamma) + 1e-12))
                if len(diff_gamma) > 1 else float('nan'))

    comparison['gamma_K'] = {
        'baseline_median':    float(np.median(gamma_b)),
        'baseline_mean':      float(np.mean(gamma_b)),
        'baseline_q5':        float(np.percentile(gamma_b, 5)),
        'baseline_q95':       float(np.percentile(gamma_b, 95)),
        'gc_median':          float(np.median(gamma_g)),
        'gc_mean':            float(np.mean(gamma_g)),
        'gc_q5':              float(np.percentile(gamma_g, 5)),
        'gc_q95':             float(np.percentile(gamma_g, 95)),
        'median_ratio_gc_to_baseline':
            float(np.median(gamma_g) / (np.median(gamma_b) + 1e-12)),
        'mean_paired_diff':   float(np.mean(diff_gamma)),
        'frac_gc_larger':     float(np.mean(diff_gamma > 0)),
        'mannwhitney_u':      mwu_stat,
        'mannwhitney_p':      mwu_p,
        'wilcoxon_stat':      w_stat,
        'wilcoxon_p':         w_p,
        'cohens_d_paired':    cohens_d,
    }

    # M2 + M3: per-eps Jaccard comparison (paired by image and noise).
    comparison['per_epsilon'] = {}
    for eps in epsilons:
        ek = str(eps)
        j_b = np.array([r['jaccard_mean'] for r in baseline_results[ek]])
        j_g = np.array([r['jaccard_mean'] for r in gc_results[ek]])
        diff_j = j_g - j_b
        try:
            wj = wilcoxon(diff_j, alternative='greater')
            wj_stat, wj_p = float(wj.statistic), float(wj.pvalue)
        except Exception:
            wj_stat, wj_p = float('nan'), float('nan')

        comparison['per_epsilon'][ek] = {
            'baseline_mean_jaccard': float(np.mean(j_b)),
            'gc_mean_jaccard':       float(np.mean(j_g)),
            'jaccard_diff_mean':     float(np.mean(diff_j)),
            'frac_gc_better':        float(np.mean(diff_j > 0)),
            'frac_tied':             float(np.mean(diff_j == 0)),
            'wilcoxon_stat':         wj_stat,
            'wilcoxon_p':            wj_p,
        }

    return comparison


# ==========================================================================
# Plots
# ==========================================================================

def make_plots(summary, save_path, model_name, baseline_log=None, gc_log=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"L_gc ablation: does it widen gamma_K and improve Top-K stability?\n"
        f"({model_name.upper()}, K={summary['config']['csae_top_k']}) "
        f"-- Remark following Prop. 5.6",
        fontsize=12, fontweight='bold',
    )

    eps_strs = list(summary['comparison']['per_epsilon'].keys())
    epsilons = sorted([float(e) for e in eps_strs])
    eps_str_map = {float(e): e for e in eps_strs}

    eps0 = str(epsilons[0])
    gamma_b = np.array([r['gamma_K'] for r in summary['baseline'][eps0]])
    gamma_g = np.array([r['gamma_K'] for r in summary['gc'][eps0]])

    # ---- (a) gamma_K distribution overlay (log scale) -- M1 ----
    ax = axes[0, 0]
    lo = max(min(gamma_b.min(), gamma_g.min()), 1e-12)
    hi = max(gamma_b.max(), gamma_g.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)
    ax.hist(np.clip(gamma_b, lo, None), bins=bins, alpha=0.55,
            color='C0', label=f'baseline (median={np.median(gamma_b):.3g})')
    ax.hist(np.clip(gamma_g, lo, None), bins=bins, alpha=0.55,
            color='C3', label=f'with $L_{{gc}}$ (median={np.median(gamma_g):.3g})')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma_K(x)$')
    ax.set_ylabel('Count')
    ax.set_title(r'(a) M1: Distribution of $\gamma_K(x)$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    gd = summary['comparison']['gamma_K']
    txt = (f"median ratio = {gd['median_ratio_gc_to_baseline']:.2f}×\n"
           f"frac (gc>base) = {gd['frac_gc_larger']:.2f}\n"
           f"Wilcoxon p = {gd['wilcoxon_p']:.2e}\n"
           f"Cohen's d = {gd['cohens_d_paired']:.3f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8,
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # ---- (b) Paired gamma_K scatter (each point = one image) ----
    ax = axes[0, 1]
    ax.scatter(gamma_b, gamma_g, alpha=0.4, s=12, c='C2')
    lo2 = max(min(gamma_b.min(), gamma_g.min()), 1e-12)
    hi2 = max(gamma_b.max(), gamma_g.max())
    ax.plot([lo2, hi2], [lo2, hi2], 'k--', alpha=0.6, label='y = x')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\gamma_K(x)$, baseline')
    ax.set_ylabel(r'$\gamma_K(x)$, with $L_{gc}$')
    ax.set_title(r'(b) Paired $\gamma_K$ (above diagonal = $L_{gc}$ widens gap)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # ---- (c) Top-K Jaccard vs eps, two curves -- M2 ----
    ax = axes[0, 2]
    j_b = [summary['comparison']['per_epsilon'][eps_str_map[e]]['baseline_mean_jaccard']
           for e in epsilons]
    j_g = [summary['comparison']['per_epsilon'][eps_str_map[e]]['gc_mean_jaccard']
           for e in epsilons]
    ax.plot(epsilons, j_b, 'o-', color='C0', linewidth=2, label='baseline')
    ax.plot(epsilons, j_g, 's-', color='C3', linewidth=2, label='with $L_{gc}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Mean Top-K Jaccard')
    ax.set_title('(c) M2: Top-K stability vs $\\epsilon$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- (d) Jaccard difference (gc - baseline) per eps -- M3 ----
    ax = axes[1, 0]
    diffs = [summary['comparison']['per_epsilon'][eps_str_map[e]]['jaccard_diff_mean']
             for e in epsilons]
    pvals = [summary['comparison']['per_epsilon'][eps_str_map[e]]['wilcoxon_p']
             for e in epsilons]
    colors = ['C3' if p is not None and not np.isnan(p) and p < 0.05 else 'C7'
              for p in pvals]
    bars = ax.bar(range(len(epsilons)), diffs, color=colors)
    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([f'{e:.0e}' if e > 0 else '0' for e in epsilons],
                       rotation=30)
    ax.axhline(0, color='k', alpha=0.5)
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'Mean $J_{gc} - J_{base}$')
    ax.set_title('(d) M3: Paired Jaccard improvement (red = $p < 0.05$)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, p in zip(bars, pvals):
        h = bar.get_height()
        if not np.isnan(p):
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f'p={p:.1e}' if p < 0.01 else f'p={p:.2f}',
                    ha='center', va='bottom' if h >= 0 else 'top', fontsize=7)

    # ---- (e) gamma_K vs Jaccard, both checkpoints, mid eps ----
    ax = axes[1, 1]
    mid_eps = epsilons[len(epsilons) // 2]
    rs_b = summary['baseline'][eps_str_map[mid_eps]]
    rs_g = summary['gc'][eps_str_map[mid_eps]]
    ax.scatter([r['gamma_K'] for r in rs_b], [r['jaccard_mean'] for r in rs_b],
               alpha=0.4, s=12, c='C0', label='baseline')
    ax.scatter([r['gamma_K'] for r in rs_g], [r['jaccard_mean'] for r in rs_g],
               alpha=0.4, s=12, c='C3', label='with $L_{gc}$')
    if min(gamma_b.min(), gamma_g.min()) > 0:
        ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma_K(x)$')
    ax.set_ylabel('Top-K Jaccard')
    ax.set_title(f'(e) $\\gamma_K$ vs Jaccard ($\\epsilon = {mid_eps:.3g}$)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # ---- (f) Optional: gamma_K trajectory during training ----
    ax = axes[1, 2]
    if baseline_log is not None and gc_log is not None:
        try:
            with open(baseline_log) as f:
                blog = json.load(f)
            with open(gc_log) as f:
                glog = json.load(f)
            b_hist = blog.get('gamma_K_history', [])
            g_hist = glog.get('gamma_K_history', [])
            if b_hist and g_hist:
                ax.plot(b_hist, 'o-', color='C0', label='baseline', linewidth=2)
                ax.plot(g_hist, 's-', color='C3', label='with $L_{gc}$',
                        linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(r'$\gamma_K$ (mean over batch)')
                ax.set_title('(f) M4: Training trajectory of $\\gamma_K$')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No "gamma_K_history" key\nin training logs',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, alpha=0.6)
                ax.set_title('(f) M4: Training trajectory (unavailable)')
                ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f'Log read error:\n{e}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, alpha=0.6)
            ax.set_title('(f) M4: Training trajectory (error)')
            ax.axis('off')
    else:
        ax.text(0.5, 0.5,
                'Pass --baseline_log and --gc_log\nto show training trajectories',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, alpha=0.6)
        ax.set_title('(f) M4: Training trajectory (skipped)')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()


# ==========================================================================
# Report
# ==========================================================================

def write_report(summary, save_path, model_name):
    cmp_ = summary['comparison']
    cfg = summary['config']
    L = []
    L.append("=" * 72)
    L.append(f"L_gc Ablation Report -- {model_name.upper()} "
             f"(K={cfg['csae_top_k']})")
    L.append("(Claim C of the Remark after Prop 5.6: L_gc widens gamma_K)")
    L.append("=" * 72)
    L.append("")
    L.append("Configuration:")
    for k, v in cfg.items():
        L.append(f"  {k:25s} : {v}")
    L.append("")
    L.append("-" * 72)
    L.append("M1: gamma_K(x) distribution -- does L_gc shift it right?")
    L.append("-" * 72)
    g = cmp_['gamma_K']
    L.append(f"  baseline:   median={g['baseline_median']:.4g}  "
             f"mean={g['baseline_mean']:.4g}  "
             f"[Q5={g['baseline_q5']:.4g}, Q95={g['baseline_q95']:.4g}]")
    L.append(f"  with L_gc:  median={g['gc_median']:.4g}  "
             f"mean={g['gc_mean']:.4g}  "
             f"[Q5={g['gc_q5']:.4g}, Q95={g['gc_q95']:.4g}]")
    L.append(f"  median ratio (gc / baseline)  : "
             f"{g['median_ratio_gc_to_baseline']:.3f}")
    L.append(f"  frac samples gc > baseline    : {g['frac_gc_larger']:.3f}")
    L.append(f"  Mann-Whitney U (gc > base)    : "
             f"U={g['mannwhitney_u']:.4g}  p={g['mannwhitney_p']:.4g}")
    L.append(f"  Wilcoxon paired (gc > base)   : "
             f"W={g['wilcoxon_stat']:.4g}  p={g['wilcoxon_p']:.4g}")
    L.append(f"  Cohen's d (paired diff)       : {g['cohens_d_paired']:.4g}")
    L.append("")
    if g['wilcoxon_p'] < 0.05 and g['median_ratio_gc_to_baseline'] > 1:
        L.append("  >>> SUPPORTED: L_gc significantly widens gamma_K.")
    elif g['wilcoxon_p'] >= 0.05:
        L.append("  >>> NOT SUPPORTED: paired Wilcoxon does not reject the null;")
        L.append("      the headline claim of the Remark fails on this data.")
    else:
        L.append("  >>> EQUIVOCAL: significant but median ratio <= 1, or vice versa.")
    L.append("")

    L.append("-" * 72)
    L.append("M2+M3: Top-K Jaccard under perturbation, paired by (image, noise)")
    L.append("-" * 72)
    L.append(f"{'eps':>10} {'J_base':>9} {'J_gc':>9} {'Δ J':>9} "
             f"{'frac>':>8} {'Wilcoxon p':>12}")
    sig_eps = []
    for eps_str, m in cmp_['per_epsilon'].items():
        marker = ' *' if m['wilcoxon_p'] < 0.05 and m['jaccard_diff_mean'] > 0 else ''
        L.append(f"{float(eps_str):>10.4g} "
                 f"{m['baseline_mean_jaccard']:>9.4f} "
                 f"{m['gc_mean_jaccard']:>9.4f} "
                 f"{m['jaccard_diff_mean']:>+9.4f} "
                 f"{m['frac_gc_better']:>8.3f} "
                 f"{m['wilcoxon_p']:>12.4g}{marker}")
        if m['wilcoxon_p'] < 0.05 and m['jaccard_diff_mean'] > 0:
            sig_eps.append(float(eps_str))
    L.append("  (*) Wilcoxon p < 0.05 AND mean improvement is positive.")
    L.append("")

    if sig_eps:
        L.append(f"  >>> SUPPORTED at eps in: {sig_eps}")
        L.append("      L_gc yields statistically significant Top-K stability gains.")
    else:
        L.append("  >>> NOT SUPPORTED: no epsilon shows a significant Jaccard")
        L.append("      improvement. The 'improves Top-K stability' part of the")
        L.append("      claim is not borne out on this data, even if M1 passes.")
    L.append("")

    L.append("Interpretation:")
    L.append("  The Remark makes a single conjunction: 'widens gamma_K AND")
    L.append("  improves Top-K stability'. Both M1 and M2/M3 must show the")
    L.append("  predicted direction for the Remark to be empirically supported.")
    L.append("  If M1 passes but M2/M3 fails, the gap widening is real but does")
    L.append("  not translate into observable stability gains at the perturbation")
    L.append("  scales we tested -- the second half of the Remark needs to be")
    L.append("  weakened or qualified in the paper.")
    L.append("")
    L.append("Caveats:")
    L.append("  - Single-seed comparison. To rule out training-noise effects,")
    L.append("    rerun with multiple seeds per arm and add a mixed-effects")
    L.append("    model with seed as a random effect.")
    L.append("  - Both checkpoints share the same dataset and noise realizations,")
    L.append("    making this a tight paired comparison given the constraint.")
    L.append("  - L_gc may also change L_f (encoder weights). The composite")
    L.append("    Lipschitz constant is reported per-checkpoint in config.")

    with open(save_path, 'w') as f:
        f.write('\n'.join(L))


# ==========================================================================
# Driver
# ==========================================================================

def run_comparison(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---------- Dataset (shared) ----------
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
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

    # ---------- Extractor (shared backbone) ----------
    extractor = MultiModelActivationExtractor(
        model_name=args.model,
        target_layer=args.target_layer,
        device=device,
        cumulative_threshold=args.tau,
    )
    in_channels = extractor.num_channels
    K = args.csae_top_k

    # ---------- Load both checkpoints ----------
    print(f"\nLoading baseline checkpoint: {args.baseline_checkpoint}")
    csae_baseline = load_csae(args.baseline_checkpoint, in_channels, K, device)
    print(f"Loading GC checkpoint:       {args.gc_checkpoint}")
    csae_gc = load_csae(args.gc_checkpoint, in_channels, K, device)

    # Sanity: architectures must match
    assert csae_baseline.encoder.weight.shape == csae_gc.encoder.weight.shape, \
        "Baseline and GC checkpoints must share architecture."

    # Per-checkpoint Lipschitz diagnostics (only need a few probe images)
    probe_imgs = torch.stack([eval_dataset[i][0]
                              for i in range(min(20, len(eval_dataset)))]
                             ).to(device)
    L_phi = estimate_L_phi(extractor, probe_imgs, n_probes=5, probe_eps=1e-3)
    L_f_base = estimate_L_f(csae_baseline)
    L_f_gc = estimate_L_f(csae_gc)
    print(f"\nLipschitz diagnostics:")
    print(f"  L_Phi                  = {L_phi:.4f}")
    print(f"  L_f (baseline)         = {L_f_base:.4f}")
    print(f"  L_f (with L_gc)        = {L_f_gc:.4f}")

    # ---------- Measure both checkpoints with paired noise ----------
    print(f"\n{'='*72}")
    print(f"Measuring baseline checkpoint over {len(eval_dataset)} images")
    print(f"{'='*72}")
    baseline_results = measure_checkpoint(
        extractor, csae_baseline, eval_dataset, indices,
        K, args.epsilons, args.trials_per_eps,
        noise_seed=args.noise_seed, device=device,
    )

    print(f"\n{'='*72}")
    print(f"Measuring GC checkpoint over {len(eval_dataset)} images")
    print(f"  (same image subset, same noise seed -> paired comparison)")
    print(f"{'='*72}")
    gc_results = measure_checkpoint(
        extractor, csae_gc, eval_dataset, indices,
        K, args.epsilons, args.trials_per_eps,
        noise_seed=args.noise_seed, device=device,
    )

    # ---------- Statistical comparison ----------
    comparison = compare_distributions(baseline_results, gc_results, args.epsilons)

    summary = {
        'config': {
            'model':                 args.model,
            'target_layer':          extractor.target_layer_name,
            'baseline_checkpoint':   args.baseline_checkpoint,
            'gc_checkpoint':         args.gc_checkpoint,
            'csae_top_k':            K,
            'num_images':            args.num_images,
            'epsilons':              args.epsilons,
            'trials_per_eps':        args.trials_per_eps,
            'noise_seed':            args.noise_seed,
            'L_phi':                 L_phi,
            'L_f_baseline':          L_f_base,
            'L_f_gc':                L_f_gc,
            'spatial_size':          extractor.spatial_size,
        },
        'baseline':   baseline_results,
        'gc':         gc_results,
        'comparison': comparison,
    }

    # ---------- Save outputs ----------
    out_prefix = f"gc_ablation_{args.model}_topk{K}"
    json_path = f"{out_prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nRaw results: {json_path}")

    plot_path = f"{out_prefix}.png"
    make_plots(summary, plot_path, args.model,
               baseline_log=args.baseline_log, gc_log=args.gc_log)
    print(f"Plot:        {plot_path}")

    report_path = f"{out_prefix}_report.txt"
    write_report(summary, report_path, args.model)
    print(f"Report:      {report_path}")

    # ---------- Console headline ----------
    print(f"\n{'='*72}")
    print("HEADLINE")
    print(f"{'='*72}")
    g = comparison['gamma_K']
    print(f"  gamma_K median: baseline = {g['baseline_median']:.4g}, "
          f"with L_gc = {g['gc_median']:.4g}  "
          f"({g['median_ratio_gc_to_baseline']:.2f}×)")
    print(f"  Wilcoxon paired (gc > baseline): p = {g['wilcoxon_p']:.4g}")
    sig = [e for e in args.epsilons
           if comparison['per_epsilon'][str(e)]['wilcoxon_p'] < 0.05 and
              comparison['per_epsilon'][str(e)]['jaccard_diff_mean'] > 0]
    if sig:
        print(f"  Top-K stability significantly improved at eps in: {sig}")
    else:
        print("  No epsilon shows a significant Top-K Jaccard improvement.")
    return summary


def main():
    p = argparse.ArgumentParser(
        description="Compare two ConvSAE checkpoints (baseline vs. L_gc) to "
                    "verify Claim C of the Remark after Prop 5.6."
    )
    p.add_argument('--model', type=str, default='resnet50',
                   choices=list(MODEL_CONFIGS.keys()))
    p.add_argument('--target_layer', type=str, default=None)
    p.add_argument('--tau', type=float, default=0.95)
    p.add_argument('--num_images', type=int, default=500)
    p.add_argument('--epsilons', type=float, nargs='+',
                   default=[0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    p.add_argument('--trials_per_eps', type=int, default=3)
    p.add_argument('--seed', type=int, default=42,
                   help='Seed for image subset selection.')
    p.add_argument('--noise_seed', type=int, default=12345,
                   help='Seed for input perturbation noise (shared across '
                        'checkpoints for proper pairing).')
    p.add_argument('--baseline_checkpoint', type=str, required=True,
                   help='Path to ConvSAE trained with lambda_gc = 0.')
    p.add_argument('--gc_checkpoint', type=str, required=True,
                   help='Path to ConvSAE trained with lambda_gc > 0.')
    p.add_argument('--csae_top_k', type=int, default=32)
    p.add_argument('--baseline_log', type=str, default=None,
                   help='Optional: JSON training log for baseline run '
                        '(needs "gamma_K_history" key).')
    p.add_argument('--gc_log', type=str, default=None,
                   help='Optional: JSON training log for L_gc run '
                        '(needs "gamma_K_history" key).')

    args = p.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()