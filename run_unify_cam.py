"""
run_unified_cam.py
==================
Driver + visualization for the unified CAM-style attribution joint solver.

Solves Equation (1) of "A Unified Objective for CAM-Style Attribution" for a
chosen image, at a chosen corner (or the genuinely-new interior) of the
parameter cube, with EITHER solver (block coordinate descent / Riemannian joint
descent), and renders three figures:

  1. CONVERGENCE figure    -- the objective J per iteration. For Option B this
     is the monotone non-increasing sequence Proposition 1 guarantees; for
     'both' it overlays block-then-joint so you can see the joint step squeeze
     J further. The blended-Sigma eigengap (the seed-invariance certificate)
     is annotated.
  2. DECOMPOSITION panel   -- input | true Grad-CAM | rank-D reconstruction |
     residual, then the top-|beta_d| signed component contributions beta_d*z_d.
     Only emitted for phi='id' (the additive identity holds); for a kernel the
     panel degrades to input | Grad-CAM | recon and prints the section-5 scope
     caveat.
  3. FIDELITY curve        -- rank-D Grad-CAM reconstruction cosine/corr over a
     D sweep, with the SOLVED basis (the 'Grad-CAM is low-rank' headline, now
     for whichever corner you solved).

This driver reuses the EXACT image/backbone/Grad-CAM contract of
pca_gradcam_decomp.py and hier_visualize_pca.py: same MODEL_CONFIGS, same
transform, same per-channel 0.99-quantile normalization, same GradCAM module,
same test_metadata.pkl layout. The only difference is the basis: instead of
loading a frozen PCAReconstructor pkl, it SOLVES (P, alpha) jointly.

Usage
-----
  # DCAM corner (id, global, lambda=1, alpha free) -- reproduces the frozen-PCA
  # baseline, but with the basis SOLVED by block coordinate descent.
  python run_unified_cam.py --model resnet50 --class_id 207 \\
      --corner dcam --D 100 --solver block

  # Grad-CAM corner: alpha pinned by the averaging rule, P solved.
  python run_unified_cam.py --model resnet50 --class_id 207 \\
      --corner grad_cam --D 50

  # the NEW interior region: finite-bandwidth local basis, class-tilted.
  python run_unified_cam.py --model resnet50 --class_id 207 \\
      --corner new_interior --lam 0.5 --bandwidth 0.8 --D 60 \\
      --solver both --n_restarts 4

  # KPCA-CAM corner (kernel): solver runs, additive panels disabled.
  python run_unified_cam.py --model resnet50 --class_id 207 \\
      --corner kpca_cam --D 40 --rff_dim 2048

The bank of images the basis is fit on is, by default, the cached test images
of --class_id (so w='global' is a within-class dataset-wide basis and
w='per_image'/'bandwidth' localise around the query within that class). Pass
--bank_classes to widen the bank across classes.
"""

import argparse
import io
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.append('.')
from src.gradcam import GradCAM
from full_classes import IMAGENET2012_CLASSES

from unified_cam_joint import (
    SolveConfig, solve, decompose_query, fidelity_curve,
    config_for_corner, normalize_acts, CORNER_PRESETS,
)


MODEL_CONFIGS = {
    'resnet50': {'model_fn': lambda: models.resnet50(pretrained=True),
                 'default_target_layer': 'layer3',
                 'description': 'ResNet50 (layer3: 1024ch, 14x14)'},
    'resnet18': {'model_fn': lambda: models.resnet18(pretrained=True),
                 'default_target_layer': 'layer3',
                 'description': 'ResNet18 (layer3: 256ch, 14x14)'},
    'vgg16': {'model_fn': lambda: models.vgg16(pretrained=True),
              'default_target_layer': 'features[16]',
              'description': 'VGG16 (features[16]: 256ch, 28x28)'},
    'efficientnet': {'model_fn': lambda: models.efficientnet_b0(pretrained=True),
                     'default_target_layer': 'features[4]',
                     'description': 'EfficientNet-B0 (features[4]: ~80ch)'},
}


# ==========================================================================
# backbone wrapper: captures activations + Grad-CAM alpha   (same contract as
# pca_gradcam_decomp.GradCAMDecomposer, trimmed to what the solver needs)
# ==========================================================================

class CAMBackbone:
    def __init__(self, model_name: str, target_layer_name: str, device='cuda'):
        self.model_name = model_name
        self.target_layer_name = target_layer_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.backbone = MODEL_CONFIGS[model_name]['model_fn']().to(
            self.device).eval()
        self.target_layer = self._get_target_layer()
        self._detect_dims()
        print(f"  Backbone: {model_name} @ {target_layer_name} "
              f"({self.num_channels}ch, {self.spatial_size}x{self.spatial_size})")

        self.layer_activations = None
        self.target_layer.register_forward_hook(self._save_act)
        self.gradcam = GradCAM(self.backbone, self.target_layer)
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def _get_target_layer(self):
        if self.model_name in ['resnet50', 'resnet18']:
            return {'layer1': self.backbone.layer1, 'layer2': self.backbone.layer2,
                    'layer3': self.backbone.layer3, 'layer4': self.backbone.layer4
                    }[self.target_layer_name]
        idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
        return self.backbone.features[idx]

    def _detect_dims(self):
        with torch.no_grad():
            d = torch.randn(1, 3, 224, 224).to(self.device)
            if self.model_name in ['resnet50', 'resnet18']:
                x = self.backbone.conv1(d); x = self.backbone.bn1(x)
                x = self.backbone.relu(x); x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x)
                if 'layer1' not in self.target_layer_name:
                    x = self.backbone.layer2(x)
                    if 'layer2' not in self.target_layer_name:
                        x = self.backbone.layer3(x)
                        if 'layer3' not in self.target_layer_name:
                            x = self.backbone.layer4(x)
            else:
                idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
                x = d
                for i in range(idx + 1):
                    x = self.backbone.features[i](x)
            self.num_channels = x.shape[1]
            self.spatial_size = x.shape[2]

    def _save_act(self, module, inp, out):
        self.layer_activations = out.detach()

    def acts_and_alpha(self, image: Image.Image
                       ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Run one image. Returns (raw activations [1,C,H,W],
        Grad-CAM alpha [C], predicted label)."""
        x = self.transform(image).unsqueeze(0).to(self.device)
        weights, _, pred_label = self.gradcam.forward(x, class_idx=None,
                                                      verbose=False)
        with torch.no_grad():
            A_raw = self.layer_activations.clone()        # [1,C,H,W]
        return A_raw, weights.view(-1).to(self.device), int(pred_label)


# ==========================================================================
# image loading  (same contract as hier_visualize_pca.load_image_for_class)
# ==========================================================================

def load_class_images(test_metadata_path: Path, class_id: int,
                       offset: int, num_images: int
                       ) -> List[Tuple[Image.Image, int]]:
    if not test_metadata_path.exists():
        raise FileNotFoundError(f"Test metadata not found at {test_metadata_path}.")
    meta = joblib.load(test_metadata_path)
    samples = meta['samples']
    class_samples = [(b, lbl) for (b, lbl) in samples if lbl == class_id]
    n = len(class_samples)
    if n == 0:
        raise ValueError(f"No cached test samples for class_id={class_id}")
    out = []
    for k in range(num_images):
        idx = (offset + k) % n
        b, lbl = class_samples[idx]
        out.append((Image.open(io.BytesIO(b)).convert('RGB'), idx))
    return out


# ==========================================================================
# the bank: activations + alpha for every image the basis is fit on
# ==========================================================================

def build_bank(backbone: CAMBackbone, test_meta: Path,
               bank_classes: List[int], per_class: int
               ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """Run every bank image through the backbone, normalize, and stack.

    Returns (acts_norm [N,C,H,W], alpha_bank [N,C], index [(class,idx)]).
    The activations are normalized HERE (per-channel 0.99-quantile, identical
    to export_activation_cache.py) so the bank lives in the same normalized
    space the solver and any PCA baseline operate in.
    """
    raw_list, alpha_list, index = [], [], []
    for cid in bank_classes:
        imgs = load_class_images(test_meta, cid, offset=0,
                                 num_images=per_class)
        for (img, idx) in imgs:
            A_raw, alpha, _ = backbone.acts_and_alpha(img)
            raw_list.append(A_raw.cpu())
            alpha_list.append(alpha.cpu())
            index.append((cid, idx))
    acts_raw = torch.cat(raw_list, 0)                     # [N,C,H,W]
    acts_norm = normalize_acts(acts_raw)                  # same as the cache
    alpha_bank = torch.stack(alpha_list, 0)               # [N,C]
    print(f"  Bank: {acts_norm.shape[0]} images "
          f"({len(bank_classes)} class(es) x {per_class})")
    return acts_norm, alpha_bank, index


# ==========================================================================
# Figure 1: convergence of the objective J
# ==========================================================================

def plot_convergence(result, save_path: str, model_name: str,
                     class_id: int, solver: str):
    diag = result.diagnostics
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))

    block_J = diag.get("block_J_history")
    if solver == "both" and block_J:
        it_b = np.arange(len(block_J))
        ax.plot(it_b, block_J, '-o', color='royalblue',
                label='Option B: block coordinate descent')
        off = len(block_J) - 1
        it_j = np.arange(off, off + len(result.J_history))
        ax.plot(it_j, result.J_history, '-s', color='firebrick',
                label='Option A: Riemannian joint descent (warm-started)')
    else:
        it = np.arange(len(result.J_history))
        label = ('Option B: block coordinate descent'
                 if solver == 'block'
                 else 'Option A: Riemannian joint descent')
        color = 'royalblue' if solver == 'block' else 'firebrick'
        ax.plot(it, result.J_history, '-o', color=color, label=label)

    ax.set_xlabel('iteration')
    ax.set_ylabel(r'objective  $J(P,\alpha;x_0)$')
    ax.set_title(
        f'Joint-solver convergence  --  {model_name}, class {class_id}\n'
        f'corner: {diag["corner"]}', fontsize=10)
    seed_ok = diag["seed_invariant"]
    txt = (f"eigengap $\\lambda_D-\\lambda_{{D+1}}$ = {diag['eigengap']:.3e}\n"
           f"subspace agreement (restarts) = {diag['subspace_agreement']:.4f}\n"
           f"seed-invariant: {'YES' if seed_ok else 'NO (gap/agreement low)'}")
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha='right', va='top',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round',
                      facecolor='honeydew' if seed_ok else 'mistyrose',
                      alpha=0.9))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved convergence figure: {save_path}")


# ==========================================================================
# Figure 2: decomposition panel  (phi='id' only -- additive identity)
# ==========================================================================

def plot_decomposition(dec: Dict, image: Image.Image, true_class: str,
                        pred_class: str, correct: bool, top_components: int,
                        save_path: str, model_name: str, corner: str):
    H, W = dec['H'], dec['W']

    if not dec['additive_exact']:
        # kernel corner: degraded 3-panel + the scope caveat.
        fig, axs = plt.subplots(1, 3, figsize=(11, 4), facecolor='white')
        axs[0].imshow(image); axs[0].set_title('input', fontsize=10)
        axs[1].imshow(dec['gradcam_true'].numpy(), cmap='jet')
        axs[1].set_title('true Grad-CAM', fontsize=10, color='royalblue')
        axs[2].imshow(dec['recon'].numpy(), cmap='jet')
        axs[2].set_title('rank-D recon (projector only)', fontsize=10)
        for a in axs:
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f'{corner}\n{dec.get("note", "")}', fontsize=9)
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved decomposition panel (kernel, degraded): {save_path}")
        return

    recon = dec['recon']
    true_g = dec['gradcam_true']
    residual = true_g - recon
    beta = dec['beta']
    comp = dec['comp']
    order = torch.argsort(beta.abs(), descending=True)[:top_components]

    ncol = max(4, top_components)
    fig, axs = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.6),
                            facecolor='white')

    axs[0, 0].imshow(image); axs[0, 0].set_title('input', fontsize=10)
    axs[0, 1].imshow(true_g.numpy(), cmap='jet')
    axs[0, 1].set_title('true Grad-CAM', fontsize=10, color='royalblue')
    axs[0, 2].imshow(recon.numpy(), cmap='jet')
    axs[0, 2].set_title(f"rank-{dec['D']} recon\ncos={dec['spatial_cos']:.3f}  "
                        f"corr={dec['spatial_corr']:.3f}", fontsize=10)
    vlim = float(residual.abs().max().clamp(min=1e-8))
    axs[0, 3].imshow(residual.numpy(), cmap='bwr', vmin=-vlim, vmax=vlim)
    axs[0, 3].set_title('residual', fontsize=10)
    for j in range(4, ncol):
        axs[0, j].axis('off')
    for j in range(ncol):
        axs[0, j].set_xticks([]); axs[0, j].set_yticks([])

    cmax = float(comp[order].abs().max().clamp(min=1e-8)) if len(order) else 1.0
    for j in range(ncol):
        ax = axs[1, j]
        ax.set_xticks([]); ax.set_yticks([])
        if j < len(order):
            d = int(order[j].item())
            ax.imshow(comp[d].numpy(), cmap='bwr', vmin=-cmax, vmax=cmax)
            ax.set_title(f'v{d}: beta={float(beta[d]):+.2f}', fontsize=9)
        else:
            ax.axis('off')

    status = 'OK' if correct else 'WRONG'
    fig.suptitle(
        f'Unified-CAM decomposition  --  {model_name}  --  '
        f'corner: {corner}\n'
        f'true:{true_class[:28]}  pred:{pred_class[:28]} [{status}]   '
        f'Grad-CAM = ReLU( bias + sum_d beta_d * v_d ),  basis SOLVED jointly',
        fontsize=10)
    fig.text(0.5, 0.48,
             'Top: input | true Grad-CAM | rank-D reconstruction | residual.   '
             'Bottom: signed component contributions beta_d * z_d (blue +, red -).',
             ha='center', fontsize=9, family='monospace')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved decomposition panel: {save_path}")


# ==========================================================================
# Figure 3: rank-D fidelity curve  (phi='id' only)
# ==========================================================================

def plot_fidelity(curve: Dict, save_path: str, model_name: str,
                  class_id: int, corner: str, relu_keep: float):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(curve['D'], curve['cos'], '-o', color='royalblue',
            label='spatial cosine')
    ax.plot(curve['D'], curve['corr'], '-s', color='firebrick',
            label='spatial corr')
    ax.axhline(0.95, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('rank D (number of solved components)')
    ax.set_ylabel('fidelity vs true Grad-CAM')
    ax.set_ylim(0, 1.02)
    ax.set_title(f'Rank-D Grad-CAM reconstruction (SOLVED basis)\n'
                 f'{model_name}, class {class_id}, corner: {corner}  '
                 f'(ReLU keeps {100*relu_keep:.0f}% of pre-ReLU mass)',
                 fontsize=10)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved fidelity curve: {save_path}")


# ==========================================================================
# main
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Joint solver + visualization for the unified CAM '
                    'attribution objective.')
    ap.add_argument('--model', type=str, default='resnet50',
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--target_layer', type=str, default=None)
    ap.add_argument('--class_id', type=int, required=True)
    ap.add_argument('--offset', type=int, default=0,
                    help='Which cached image of the class is the QUERY.')
    ap.add_argument('--corner', type=str, default='dcam',
                    choices=list(CORNER_PRESETS.keys()),
                    help='Which corner of the parameter cube to solve. '
                         'Override individual knobs with --phi/--lam/etc.')
    ap.add_argument('--solver', type=str, default='block',
                    choices=['block', 'joint', 'both'],
                    help="block = Option B (coordinate descent); "
                         "joint = Option A (Riemannian); "
                         "both = block then joint warm-started from it.")
    ap.add_argument('--D', type=int, default=50, help='Subspace rank.')
    # individual knob overrides (override the --corner preset)
    ap.add_argument('--phi', type=str, default=None, choices=['id', 'rbf'])
    ap.add_argument('--locality', type=str, default=None,
                    choices=['global', 'per_image', 'bandwidth'])
    ap.add_argument('--lam', type=float, default=None,
                    help='Supervision dial lambda in [0,1].')
    ap.add_argument('--bandwidth', type=float, default=None,
                    help='h for locality=bandwidth.')
    ap.add_argument('--omega', type=str, default=None,
                    choices=['free', 'gradcam'])
    ap.add_argument('--rff_dim', type=int, default=1024,
                    help='Random-Fourier-feature dimension for phi=rbf.')
    ap.add_argument('--rff_gamma', type=float, default=1.0)
    ap.add_argument('--n_restarts', type=int, default=1,
                    help='>1 runs the multi-start seed-invariance check '
                         '(paper section 4).')
    ap.add_argument('--max_iter', type=int, default=25)
    ap.add_argument('--lr', type=float, default=0.05,
                    help='Step size for Option A (Riemannian descent).')
    ap.add_argument('--bank_classes', type=int, nargs='+', default=None,
                    help='Classes whose cached images form the basis-fitting '
                         'bank. Default: just --class_id.')
    ap.add_argument('--bank_per_class', type=int, default=12,
                    help='Images per bank class.')
    ap.add_argument('--D_sweep', type=int, nargs='+',
                    default=[5, 10, 20, 50, 100])
    ap.add_argument('--top_components', type=int, default=6)
    ap.add_argument('--test_data_dir', type=str,
                    default='/data/imagenet1k_sampletest')
    ap.add_argument('--output_dir', type=str, default='unified_cam_out')
    ap.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cuda', 'cpu'])
    args = ap.parse_args()

    if args.target_layer is None:
        args.target_layer = MODEL_CONFIGS[args.model]['default_target_layer']
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- build the SolveConfig from the corner preset + any overrides ----
    cfg = config_for_corner(args.corner, D=args.D)
    if args.phi is not None:
        cfg.phi = args.phi
    if args.locality is not None:
        cfg.locality = args.locality
    if args.lam is not None:
        cfg.lam = args.lam
    if args.bandwidth is not None:
        cfg.bandwidth = args.bandwidth
    if args.omega is not None:
        cfg.omega = args.omega
    cfg.rff_dim = args.rff_dim
    cfg.rff_gamma = args.rff_gamma
    cfg.n_restarts = args.n_restarts
    cfg.max_iter = args.max_iter
    cfg.lr = args.lr
    cfg.device = args.device

    print("=" * 80)
    print("Unified CAM-style attribution: joint solver")
    print(f"  model={args.model} @ {args.target_layer}  class={args.class_id}")
    print(f"  corner={args.corner}  solver={args.solver}")
    print(f"  knobs: phi={cfg.phi}  w={cfg.locality}  lambda={cfg.lam}  "
          f"omega={cfg.omega}  D={cfg.D}")
    print("=" * 80)

    backbone = CAMBackbone(args.model, args.target_layer,
                           device='cuda' if torch.cuda.is_available()
                           else 'cpu')

    test_meta = Path(args.test_data_dir) / "test_metadata.pkl"
    bank_classes = args.bank_classes or [args.class_id]

    # ---- build the bank the basis is fit on ----
    acts_norm, alpha_bank, index = build_bank(
        backbone, test_meta, bank_classes, args.bank_per_class)

    # ---- locate the query image inside the bank ----
    # the query is image `offset` of `class_id`; find it in the bank index.
    query_imgs = load_class_images(test_meta, args.class_id,
                                   args.offset, num_images=1)
    query_img, query_local_idx = query_imgs[0]
    try:
        query_idx = index.index((args.class_id, query_local_idx))
    except ValueError:
        # query not in the bank (e.g. offset beyond bank_per_class): append it.
        A_raw, alpha_q, _ = backbone.acts_and_alpha(query_img)
        acts_norm = torch.cat([acts_norm, normalize_acts(A_raw.cpu())], 0)
        alpha_bank = torch.cat([alpha_bank, alpha_q.cpu().unsqueeze(0)], 0)
        index.append((args.class_id, query_local_idx))
        query_idx = len(index) - 1
        print(f"  (query not in bank; appended as bank image {query_idx})")

    # query activations + alpha (re-run to get the exact query tensors)
    A_query_raw, alpha_query, pred_label = backbone.acts_and_alpha(query_img)
    A_query_norm = normalize_acts(A_query_raw.cpu())

    # ---- SOLVE the unified objective ----
    print(f"\nSolving (solver={args.solver}, n_restarts={cfg.n_restarts})...")
    result = solve(acts_norm, alpha_bank, alpha_query, query_idx,
                   cfg, solver=args.solver)

    diag = result.diagnostics
    print(f"\n{'='*80}")
    print(f"RESULT")
    print(f"  corner            : {diag['corner']}")
    print(f"  J_final           : {diag['J_final']:.6e}")
    print(f"  eigengap          : {diag['eigengap']:.6e}")
    print(f"  subspace agreement: {diag['subspace_agreement']:.4f}")
    print(f"  seed-invariant    : {diag['seed_invariant']}")
    print(f"  additive identity : {diag['additive_exact']} "
          f"(phi={diag['phi']})")
    if not diag['additive_exact']:
        print(f"  SCOPE CAVEAT      : {diag['scope_caveat']}")
    print(f"{'='*80}")

    # ---- decompose the query with the solved basis ----
    dec = decompose_query(result, A_query_norm, alpha_query)

    true_class = list(IMAGENET2012_CLASSES.values())[args.class_id]
    pred_class = list(IMAGENET2012_CLASSES.values())[pred_label]
    correct = (pred_label == args.class_id)

    tag = f"class{args.class_id}_img{query_local_idx}_{args.model}_{args.corner}"

    # ---- Figure 1: convergence ----
    plot_convergence(result, str(out / f"convergence_{tag}.png"),
                     args.model, args.class_id, args.solver)

    # ---- Figure 2: decomposition panel ----
    plot_decomposition(dec, query_img, true_class, pred_class, correct,
                       args.top_components, str(out / f"decomp_{tag}.png"),
                       args.model, diag['corner'])

    # ---- Figure 3: fidelity curve (phi='id' only) ----
    if diag['additive_exact']:
        D_sweep = [d for d in args.D_sweep if d <= cfg.D]
        curve = fidelity_curve(result, A_query_norm, alpha_query, D_sweep)
        plot_fidelity(curve, str(out / f"fidelity_{tag}.png"),
                      args.model, args.class_id, args.corner,
                      dec['relu_keep_frac'])
        print(f"\nFIDELITY (solved basis): rank D -> cos / corr")
        for D, c, r in zip(curve['D'], curve['cos'], curve['corr']):
            print(f"  D={D:4d}   cos={c:.4f}   corr={r:.4f}")
        knee = next((D for D, c in zip(curve['D'], curve['cos'])
                     if c >= 0.95), None)
        if knee is not None:
            print(f"  cos>=0.95 reached at rank D={knee}.")
        else:
            print(f"  cos>=0.95 not reached within the sweep.")
    else:
        print("\n  (fidelity curve skipped: kernel phi has no additive "
              "identity -- paper section 5.)")

    print("=" * 80)
    print("Done. Figures written to", out)
    print("=" * 80)


if __name__ == "__main__":
    main()