"""
pca_gradcam_decomp.py
=====================
EXACT low-rank decomposition of Grad-CAM into deterministic PCA components.

The story (corrected by the measurements)
-----------------------------------------
The original DCAM hope -- "Grad-CAM = sum of stable, reusable, NAMED parts" --
is not supported: layer-3 activations are a high-rank distributed code (90%
variance needs rank ~687), class signal is ~1.7%/48-dim, and an overcomplete
SAE finds only seed-unstable filler. BUT a deterministic, exactly seed-stable,
low-rank basis DOES exist (top-D PCA of the activations) and is functionally
faithful (rank-200 reconstructs 74.96% top-1 vs 76.68%; the mean alone is
chance). So the decomposition that survives is:

  Grad-CAM decomposes EXACTLY and LINEARLY into deterministic PCA components,
  even though it does NOT decompose into named parts. Class-specificity lives
  entirely in the linear weights, not in the (shared) components.

The algebra (exact)
-------------------
Grad-CAM pre-ReLU:   L~(x) = sum_c alpha_c(x) A_c(x)            [H, W]
PCA of activations:  A(x) ~= mu + sum_{d<=D} z_d(x) v_d,
                     z_d(x) = <A(x) - mu, v_d>  (a SCALAR per cell -> [H,W] map)
Substitute:
  L~(x) = sum_c alpha_c mu_c                      (BIAS map, [H,W] constant
                                                   per cell = scalar offset)
        + sum_{d<=D} ( sum_c alpha_c v_{d,c} ) z_d(x)
        = bias(x) + sum_{d<=D} beta_d(x) * z_d(x)
  beta_d(x) = sum_c alpha_c(x) v_{d,c}   (SCALAR; the class enters HERE)
  Grad-CAM = ReLU( L~ )

So:
  * v_d            : DETERMINISTIC, seed-stable, class-shared spatial basis
                     (the "vocabulary"). Same for every image and class.
  * z_d(x)         : per-image spatial component map [H, W] (coefficient field)
  * beta_d(x)      : scalar weight; the ONLY place class identity enters.
  * The sum is EXACT pre-ReLU. ReLU is the only nonlinearity; we measure how
    much it clips so the "additive" claim stays honest.

What this script outputs
------------------------
1. The rank-D Grad-CAM reconstruction fidelity curve: for D in a sweep, how
   well  ReLU(bias + sum_{d<=D} beta_d z_d)  matches the true Grad-CAM
   (spatial cosine + correlation). This is THE headline figure -- if a small
   D reconstructs Grad-CAM well, Grad-CAM is low-rank.
2. Per-image decomposition panels: true Grad-CAM, rank-D reconstruction, the
   residual, and the top-|beta_d| component maps beta_d * z_d that build it.
3. The ReLU clipping fraction (mean positive-mass kept) -- the additivity
   caveat, reported as a number.

Prerequisite
------------
Build the PCA basis once with csae_pca_baseline.py (it streams the cache and
eigendecomposes the per-cell covariance). This script LOADS that basis (mu, V)
straight out of the saved PCAReconstructor .pkl, so the components are exactly
the ones whose downstream accuracy you already measured.

Usage
-----
  # fidelity curve over a class sample + a few per-image decomposition figures
  python pca_gradcam_decomp.py --model resnet50 \\
      --pca_model pca_baseline_resnet50_D200_model.pkl \\
      --class_id 207 --num_images 8 --D_sweep 5 10 20 50 100 200 \\
      --panel_D 20 --top_components 6

  # single image, offset into the class's cached test images
  python pca_gradcam_decomp.py --model resnet50 \\
      --pca_model pca_baseline_resnet50_D200_model.pkl \\
      --class_id 207 --offset 3 --num_images 1 --panel_D 20
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
#from run_xcsae_full import MultiChannelConvSAE
#from csae_pca_baseline import PCAReconstructor   # noqa: F401 (pickle import)
from src.gradcam import GradCAM
from full_classes import IMAGENET2012_CLASSES


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


# ==========================================
# Image loading (same contract as hier_visualize.load_image_for_class)
# ==========================================

def load_images_for_class(test_metadata_path: Path, class_id: int,
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


# ==========================================
# Decomposer: backbone + Grad-CAM + loaded PCA basis
# ==========================================

class GradCAMDecomposer:
    def __init__(self, model_name: str, target_layer_name: str,
                 pca_model_path: str, device='cuda'):
        self.model_name = model_name
        self.target_layer_name = target_layer_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load the PCA basis straight out of the saved reconstructor.
        print(f"Loading PCA basis from {pca_model_path}...")
        pca = joblib.load(pca_model_path).to(self.device).eval()
        self.mu = pca.pca_mu.detach().to(self.device)             # [C]
        self.V_full = pca.pca_V.detach().to(self.device)          # [C, D_built]
        self.D_built = self.V_full.shape[1]
        self.C = self.mu.shape[0]
        print(f"  PCA basis: C={self.C}, D_built={self.D_built} "
              f"(deterministic, seed-stable)")

        # Backbone
        self.backbone = MODEL_CONFIGS[model_name]['model_fn']().to(self.device).eval()
        self.target_layer = self._get_target_layer()
        self._detect_dims()
        print(f"  Backbone: {model_name} @ {target_layer_name} "
              f"({self.num_channels}ch, {self.spatial_size}x{self.spatial_size})")
        if self.num_channels != self.C:
            raise ValueError(f"PCA basis C={self.C} != backbone C="
                             f"{self.num_channels}. Basis built for another "
                             f"layer/backbone.")

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

    def _normalize(self, acts: torch.Tensor) -> torch.Tensor:
        """Per-channel 0.99-quantile clip+rescale -- the SAME normalization the
        PCA basis was built under (cache was normalize=True). The basis lives
        in normalized space, so we must normalize here before projecting."""
        normalized = acts.clone()
        for c in range(acts.shape[1]):
            ch = acts[0, c]
            if ch.abs().sum() < 1e-8:
                continue
            nz = ch[ch > 1e-8]
            if len(nz) > 0:
                scale = torch.quantile(nz, 0.99)
                if scale > 1e-8:
                    normalized[0, c] = torch.clamp(ch, 0.0, scale) / (scale + 1e-8)
        return normalized

    @torch.no_grad()
    def _project(self, A_norm: torch.Tensor, D: int) -> torch.Tensor:
        """z_d(x) coefficient maps for the top-D components. A_norm: [1,C,H,W].
        Returns z: [D, H, W] where z[d] = <A-mu, v_d> at each cell."""
        _, C, H, W = A_norm.shape
        cells = A_norm[0].permute(1, 2, 0).reshape(-1, C)         # [HW, C]
        centered = cells - self.mu
        Vk = self.V_full[:, :D]                                    # [C, D]
        coeff = centered @ Vk                                      # [HW, D]
        return coeff.T.reshape(D, H, W)                            # [D, H, W]

    def decompose(self, image: Image.Image, label: int,
                  D_panel: int) -> Dict:
        """Run one image. Returns the exact decomposition pieces:
            gradcam_true        ReLU(L~), min-max for display [H,W]
            L_tilde             pre-ReLU sum_c alpha_c A_c    [H,W]  (exact target)
            bias                sum_c alpha_c mu_c            scalar (broadcast)
            beta                [D_built] scalars sum_c alpha_c v_{d,c}
            z                   [D_built, H, W] component maps
            comp                [D_built, H, W] = beta_d * z_d (signed contributions)
            relu_keep_frac      fraction of |L~| mass that survives ReLU
        """
        x = self.transform(image).unsqueeze(0).to(self.device)
        # Grad-CAM weights (alpha_c) + activations captured by the hook
        weights, _, pred_label = self.gradcam.forward(x, class_idx=None,
                                                       verbose=False)
        alpha = weights.view(-1).to(self.device)                  # [C]
        with torch.no_grad():
            A_raw = self.layer_activations.clone()                # [1,C,H,W]
            A_norm = self._normalize(A_raw)
            _, C, H, W = A_norm.shape

            # EXACT pre-ReLU Grad-CAM, computed on the SAME normalized acts the
            # PCA basis uses (so the decomposition identity holds exactly here).
            L_tilde = (alpha.view(-1, 1, 1) * A_norm[0]).sum(dim=0)   # [H,W]
            gradcam_true = F.relu(L_tilde)

            # ReLU clipping diagnostic: positive mass / total |mass|
            pos = L_tilde.clamp(min=0).sum()
            tot = L_tilde.abs().sum().clamp(min=1e-8)
            relu_keep_frac = float((pos / tot).item())

            # Decomposition pieces
            D = min(D_panel, self.D_built)
            z = self._project(A_norm, self.D_built)               # [D_built,H,W]
            beta = (alpha @ self.V_full)                          # [D_built]
            bias = float((alpha @ self.mu).item())                # scalar
            comp = beta.view(-1, 1, 1) * z                        # [D_built,H,W]

        true_class = list(IMAGENET2012_CLASSES.values())[label]
        pred_class = list(IMAGENET2012_CLASSES.values())[pred_label]
        return {
            'image': image, 'label': label, 'pred_label': pred_label,
            'true_class': true_class, 'pred_class': pred_class,
            'correct': pred_label == label,
            'gradcam_true': gradcam_true.cpu(),
            'L_tilde': L_tilde.cpu(),
            'bias': bias, 'beta': beta.cpu(), 'z': z.cpu(),
            'comp': comp.cpu(), 'relu_keep_frac': relu_keep_frac,
            'H': H, 'W': W,
        }

    @staticmethod
    def reconstruct(dec: Dict, D: int) -> torch.Tensor:
        """ReLU(bias + sum_{d<=D} beta_d z_d). Returns [H,W]."""
        comp = dec['comp'][:D]                                     # [D,H,W]
        L_hat = dec['bias'] + comp.sum(dim=0)                     # [H,W]
        return F.relu(L_hat)


# ==========================================
# Fidelity metrics
# ==========================================

def _spatial_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.flatten(), b.flatten()
    return float((af @ bf) / (af.norm().clamp(min=1e-8) * bf.norm().clamp(min=1e-8)))


def _spatial_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten() - a.mean()
    bf = b.flatten() - b.mean()
    return float((af @ bf) / (af.norm().clamp(min=1e-8) * bf.norm().clamp(min=1e-8)))


# ==========================================
# Figure 1: rank-D Grad-CAM reconstruction fidelity curve (the headline)
# ==========================================

def plot_fidelity_curve(per_image_curves: Dict[int, Dict[str, List[float]]],
                        D_sweep: List[int], save_path: str,
                        relu_keep_mean: float, model_name: str, class_id: int):
    cos_mat = np.array([per_image_curves[d]['cos'] for d in D_sweep])   # [nD, nimg]
    corr_mat = np.array([per_image_curves[d]['corr'] for d in D_sweep])
    cos_mean, cos_std = cos_mat.mean(1), cos_mat.std(1)
    corr_mean, corr_std = corr_mat.mean(1), corr_mat.std(1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(D_sweep, cos_mean, '-o', color='royalblue', label='spatial cosine')
    ax.fill_between(D_sweep, cos_mean - cos_std, cos_mean + cos_std,
                    color='royalblue', alpha=0.15)
    ax.plot(D_sweep, corr_mean, '-s', color='firebrick', label='spatial corr')
    ax.fill_between(D_sweep, corr_mean - corr_std, corr_mean + corr_std,
                    color='firebrick', alpha=0.15)
    ax.axhline(0.95, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('rank D (number of PCA components)')
    ax.set_ylabel('fidelity vs true Grad-CAM')
    ax.set_ylim(0, 1.02)
    ax.set_title(f'Grad-CAM is low-rank: rank-D PCA reconstruction\n'
                 f'{model_name}, class {class_id}  '
                 f'(ReLU keeps {100*relu_keep_mean:.0f}% of pre-ReLU mass)',
                 fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved fidelity curve: {save_path}")


# ==========================================
# Figure 2: per-image decomposition panel
# ==========================================

def plot_decomposition_panel(dec: Dict, D_panel: int, top_components: int,
                             save_path: str, model_name: str):
    H, W = dec['H'], dec['W']
    recon = GradCAMDecomposer.reconstruct(dec, D_panel)
    true_g = dec['gradcam_true']
    residual = (true_g - recon)

    # rank the components by |beta_d| (their weight in THIS image's Grad-CAM)
    beta = dec['beta'][:D_panel]
    order = torch.argsort(beta.abs(), descending=True)[:top_components]

    ncol = max(4, top_components)
    fig, axs = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.4),
                            facecolor='white')

    # row 0, col 0..3: input, true Grad-CAM, rank-D reconstruction, residual
    axs[0, 0].imshow(dec['image']); axs[0, 0].set_title('input', fontsize=10)
    axs[0, 1].imshow(true_g.numpy(), cmap='jet')
    axs[0, 1].set_title('true Grad-CAM', fontsize=10, color='royalblue')
    cos = _spatial_cos(recon, true_g)
    axs[0, 2].imshow(recon.numpy(), cmap='jet')
    axs[0, 2].set_title(f'rank-{D_panel} recon\ncos={cos:.3f}', fontsize=10)
    vlim = float(residual.abs().max().clamp(min=1e-8))
    axs[0, 3].imshow(residual.numpy(), cmap='bwr', vmin=-vlim, vmax=vlim)
    axs[0, 3].set_title('residual', fontsize=10)
    for j in range(4, ncol):
        axs[0, j].axis('off')
    for j in range(ncol):
        axs[0, j].set_xticks([]); axs[0, j].set_yticks([])

    # row 1: top-|beta| signed component contributions beta_d * z_d
    comp = dec['comp']
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

    status = 'OK' if dec['correct'] else 'WRONG'
    fig.suptitle(
        f'Exact low-rank Grad-CAM decomposition  --  {model_name}  --  '
        f'true:{dec["true_class"][:30]} pred:{dec["pred_class"][:30]} [{status}]\n'
        f'Grad-CAM = ReLU( bias + sum_d beta_d(x) * v_d ),  '
        f'components v_d deterministic & class-shared; class enters via beta_d',
        fontsize=11)
    fig.text(0.5, 0.485,
             'Top row: input | true Grad-CAM | rank-D reconstruction | residual.   '
             'Bottom row: signed component contributions beta_d * z_d (blue +, red -).',
             ha='center', fontsize=9, family='monospace')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved decomposition panel: {save_path}")


# ==========================================
# Main
# ==========================================

def main():
    ap = argparse.ArgumentParser(
        description='Exact low-rank PCA decomposition of Grad-CAM.')
    ap.add_argument('--model', type=str, default='resnet50',
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--target_layer', type=str, default=None)
    ap.add_argument('--pca_model', type=str, required=True,
                    help='PCAReconstructor .pkl from csae_pca_baseline.py '
                         '(holds mu and V).')
    ap.add_argument('--class_id', type=int, required=True)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--num_images', type=int, default=8,
                    help='Images of the class to average the fidelity curve '
                         'over (and to render panels for).')
    ap.add_argument('--D_sweep', type=int, nargs='+',
                    default=[5, 10, 20, 50, 100, 200],
                    help='Ranks for the fidelity curve.')
    ap.add_argument('--panel_D', type=int, default=20,
                    help='Rank used for the per-image reconstruction panel.')
    ap.add_argument('--top_components', type=int, default=6,
                    help='How many top-|beta| components to show in the panel.')
    ap.add_argument('--max_panels', type=int, default=3,
                    help='Cap on how many per-image panels to render.')
    ap.add_argument('--test_data_dir', type=str,
                    default='/data/imagenet1k_sampletest')
    ap.add_argument('--output_dir', type=str, default='gradcam_decomp')
    args = ap.parse_args()

    if args.target_layer is None:
        args.target_layer = MODEL_CONFIGS[args.model]['default_target_layer']
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Exact low-rank Grad-CAM decomposition")
    print(f"  model={args.model} @ {args.target_layer}  class={args.class_id}")
    print(f"  D_sweep={args.D_sweep}  panel_D={args.panel_D}")
    print("=" * 80)

    dec_engine = GradCAMDecomposer(args.model, args.target_layer,
                                   args.pca_model,
                                   device='cuda' if torch.cuda.is_available()
                                   else 'cpu')
    D_sweep = [d for d in args.D_sweep if d <= dec_engine.D_built]
    if len(D_sweep) < len(args.D_sweep):
        print(f"  (trimmed D_sweep to <= D_built={dec_engine.D_built}: {D_sweep})")

    test_meta = Path(args.test_data_dir) / "test_metadata.pkl"
    images = load_images_for_class(test_meta, args.class_id,
                                   args.offset, args.num_images)
    print(f"  Loaded {len(images)} image(s) for class {args.class_id}")

    # accumulate fidelity curve + render a few panels
    curves = {d: {'cos': [], 'corr': []} for d in D_sweep}
    relu_keeps = []
    panels_done = 0
    for (img, idx) in images:
        dec = dec_engine.decompose(img, args.class_id, D_panel=max(D_sweep))
        relu_keeps.append(dec['relu_keep_frac'])
        for d in D_sweep:
            recon = GradCAMDecomposer.reconstruct(dec, d)
            curves[d]['cos'].append(_spatial_cos(recon, dec['gradcam_true']))
            curves[d]['corr'].append(_spatial_corr(recon, dec['gradcam_true']))
        if panels_done < args.max_panels:
            panel_path = out / (f"decomp_class{args.class_id}_img{idx}"
                                f"_{args.model}_D{args.panel_D}.png")
            plot_decomposition_panel(dec, args.panel_D, args.top_components,
                                     str(panel_path), args.model)
            panels_done += 1

    relu_keep_mean = float(np.mean(relu_keeps))
    curve_path = out / (f"fidelity_class{args.class_id}_{args.model}.png")
    plot_fidelity_curve(curves, D_sweep, str(curve_path),
                        relu_keep_mean, args.model, args.class_id)

    # console summary (the numbers for the paper)
    print("\n" + "=" * 80)
    print("FIDELITY (mean over images): rank D -> spatial cos / corr")
    for d in D_sweep:
        c = np.mean(curves[d]['cos']); r = np.mean(curves[d]['corr'])
        print(f"  D={d:4d}   cos={c:.4f}   corr={r:.4f}")
    print(f"\n  ReLU keeps {100*relu_keep_mean:.1f}% of pre-ReLU |mass| "
          f"(additivity is exact pre-ReLU; this is the clipping caveat).")
    # find the knee: smallest D with cos >= 0.95
    knee = next((d for d in D_sweep if np.mean(curves[d]['cos']) >= 0.95), None)
    if knee is not None:
        print(f"  Grad-CAM reaches cos>=0.95 at rank D={knee}  "
              f"-> Grad-CAM is effectively rank-{knee}.")
    else:
        print(f"  cos>=0.95 not reached within D_sweep; extend the sweep.")
    print("=" * 80)


if __name__ == "__main__":
    main()