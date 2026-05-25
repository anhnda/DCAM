"""
hier_visualize_pca.py
=====================
Hierarchical radial visualization of the EXACT low-rank Grad-CAM decomposition.

This replaces hier_visualize.py's SAE-feature rings (seed-unstable, holistic
filler) with the DETERMINISTIC PCA decomposition that the measurements support:

    Grad-CAM = ReLU( bias + sum_d beta_d(x) * z_d(x) )
      z_d(x) = <A(x) - mu, v_d>      component spatial map [H, W]
      beta_d(x) = sum_c alpha_c v_{d,c}   scalar weight (class enters HERE)
      v_d                              deterministic, seed-stable eigenvector

Layout
------
  CENTER (triangle)
    - Input image
    - True Grad-CAM (ReLU(L~))
    - Rank-D reconstruction  ReLU(bias + sum_{d<=ringK contributors} beta_d z_d)
    Edges: BLUE solid = Grad-CAM <-> rank-D reconstruction (the EXACT pair, with
           spatial cosine label); light gray = image -> Grad-CAM.

  RING 1  (top-|beta_d| components, default 16)
    The PCA components that most build THIS image's Grad-CAM, ranked by
    |beta_d|. Each panel shows the SIGNED contribution map beta_d * z_d
    (blue = pushes Grad-CAM up, red = down), with v{d} and beta_d.

  RING 2  (per-component channel loadings, default 4 "petals")
    For each component v_d, its top-N channels by |v_{d,c}| -- i.e. which
    backbone channels COMPOSE that deterministic direction. Dashed lines:
        green = positive loading (channel adds to the component)
        red   = negative loading (channel subtracts).
    Each petal shows that channel's activation map on this image.

Note vs the SAE version: components are class-SHARED and identical across
seeds; only the beta_d weights are image/class-specific. The ring is therefore
a stable, reproducible decomposition, not a per-run artifact.

Prerequisite: a PCA basis built by csae_pca_baseline.py (holds mu, V).

Usage
-----
  python hier_visualize_pca.py --class_id 207 \\
      --pca_model pca_baseline_resnet50_D200_model.pkl
  python hier_visualize_pca.py --class_id 207 --offset 3 \\
      --pca_model pca_baseline_resnet50_D200_model.pkl \\
      --ring_components 16 --top_channels 4 --recon_D 50
"""

import argparse
import io
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms

sys.path.append('.')
from run_xcsae_full import MultiChannelConvSAE
from csae_pca_baseline import PCAReconstructor   # noqa: F401 (pickle import)
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
# Image loading (verbatim contract from hier_visualize.py)
# ==========================================

def load_image_for_class(test_metadata_path: Path, class_id: int,
                         offset: int = 0) -> Tuple[Image.Image, int, int]:
    if not test_metadata_path.exists():
        raise FileNotFoundError(f"Test metadata not found at {test_metadata_path}.")
    metadata = joblib.load(test_metadata_path)
    samples = metadata['samples']
    class_samples = [(b, lbl) for (b, lbl) in samples if lbl == class_id]
    n = len(class_samples)
    if n == 0:
        raise ValueError(f"No cached test samples for class_id={class_id}")
    idx = offset % n
    if offset != idx:
        print(f"  Note: offset {offset} wrapped to {idx} ({n} cached images).")
    image_bytes, label = class_samples[idx]
    assert label == class_id
    return Image.open(io.BytesIO(image_bytes)).convert('RGB'), n, idx


# ==========================================
# Extractor: backbone + Grad-CAM + loaded PCA basis
# ==========================================

class HierPCAExtractor:
    def __init__(self, model_name: str, target_layer_name: str,
                 pca_model_path: str, device='cuda'):
        self.model_name = model_name
        self.target_layer_name = target_layer_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"Loading PCA basis from {pca_model_path}...")
        pca = joblib.load(pca_model_path).to(self.device).eval()
        self.mu = pca.pca_mu.detach().to(self.device)             # [C]
        self.V_full = pca.pca_V.detach().to(self.device)          # [C, D_built]
        self.D_built = self.V_full.shape[1]
        self.C = self.mu.shape[0]
        print(f"  PCA basis: C={self.C}, D_built={self.D_built} "
              f"(deterministic, seed-stable)")

        self.backbone = MODEL_CONFIGS[model_name]['model_fn']().to(self.device).eval()
        self.target_layer = self._get_target_layer()
        self._detect_dims()
        print(f"  Backbone: {model_name} @ {target_layer_name} "
              f"({self.num_channels}ch, {self.spatial_size}x{self.spatial_size})")
        if self.num_channels != self.C:
            raise ValueError(f"PCA basis C={self.C} != backbone C="
                             f"{self.num_channels}.")

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

    def extract(self, image: Image.Image, label: int,
                ring_components: int = 16, top_channels: int = 4,
                recon_D: int = 50) -> Dict:
        x = self.transform(image).unsqueeze(0).to(self.device)
        weights, _, pred_label = self.gradcam.forward(x, class_idx=None,
                                                      verbose=False)
        alpha = weights.view(-1).to(self.device)                  # [C]

        with torch.no_grad():
            A_raw = self.layer_activations.clone()
            A_norm = self._normalize(A_raw)
            _, C, H, W = A_norm.shape

            L_tilde = (alpha.view(-1, 1, 1) * A_norm[0]).sum(dim=0)
            gradcam_true = F.relu(L_tilde)

            # decomposition
            cells = A_norm[0].permute(1, 2, 0).reshape(-1, C)     # [HW, C]
            centered = cells - self.mu
            z = (centered @ self.V_full).T.reshape(self.D_built, H, W)  # [D,H,W]
            beta = (alpha @ self.V_full)                          # [D_built]
            bias = float((alpha @ self.mu).item())
            comp = beta.view(-1, 1, 1) * z                        # [D,H,W]

            # rank-D reconstruction (for the center pair)
            Dr = min(recon_D, self.D_built)
            recon = F.relu(bias + comp[:Dr].sum(dim=0))           # [H,W]

            af, bf = recon.flatten(), gradcam_true.flatten()
            recon_cos = float((af @ bf) / (af.norm().clamp(min=1e-8)
                                           * bf.norm().clamp(min=1e-8)))

            # ReLU clipping fraction
            pos = L_tilde.clamp(min=0).sum()
            tot = L_tilde.abs().sum().clamp(min=1e-8)
            relu_keep_frac = float((pos / tot).item())

            # Ring 1: top components by |beta_d|
            K = min(ring_components, self.D_built)
            order = torch.argsort(beta.abs(), descending=True)[:K]

            comp_list: List[Dict] = []
            for d_t in order:
                d = int(d_t.item())
                b = float(beta[d].item())
                cmap = comp[d].cpu()                              # signed contribution
                v_d = self.V_full[:, d]                          # [C] eigenvector loadings
                mag = v_d.abs()
                n_in = min(top_channels, mag.numel())
                _, top_ch = torch.topk(mag, k=n_in)
                channels: List[Dict] = []
                for ci in range(n_in):
                    ch_id = int(top_ch[ci].item())
                    loading = float(v_d[ch_id].item())
                    ch_map = A_norm[0, ch_id].cpu()
                    channels.append({'ch_id': ch_id, 'loading': loading,
                                     'map': ch_map})
                comp_list.append({'comp_id': d, 'beta': b,
                                  'contrib_map': cmap, 'channels': channels})

        true_class = list(IMAGENET2012_CLASSES.values())[label]
        pred_class = list(IMAGENET2012_CLASSES.values())[pred_label]
        return {
            'image': image, 'label': label, 'pred_label': pred_label,
            'true_class': true_class, 'pred_class': pred_class,
            'correct': pred_label == label,
            'gradcam_true': gradcam_true.cpu(),
            'recon_map': recon.cpu(), 'recon_cos': recon_cos, 'recon_D': Dr,
            'relu_keep_frac': relu_keep_frac,
            'components': comp_list,
            'num_components_total': self.D_built,
        }


# ==========================================
# Radial figure (geometry reused from hier_visualize.py)
# ==========================================

def _add_inset(fig, cx, cy, w, h):
    ax = fig.add_axes([cx - w / 2.0, cy - h / 2.0, w, h])
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def _set_border(ax, color, lw=2.0):
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(lw); spine.set_visible(True)
    ax.set_frame_on(True)


def _line(fig, p0, p1, color, lw=1.0, linestyle='-', alpha=1.0, zorder=1):
    fig.add_artist(plt.Line2D([p0[0], p1[0]], [p0[1], p1[1]],
                              transform=fig.transFigure, color=color,
                              linewidth=lw, linestyle=linestyle, alpha=alpha,
                              zorder=zorder))


def build_radial_figure(results: Dict, model_name: str, target_layer: str,
                        save_path: str, offset: int = 0, img_idx: int = 0,
                        n_available: int = None):
    image = results['image']
    gradcam_map = results['gradcam_true']
    recon_map = results['recon_map']
    recon_cos = results['recon_cos']
    recon_D = results['recon_D']
    comps = results['components']
    correct = results['correct']

    K = len(comps)
    N = len(comps[0]['channels']) if K > 0 else 0

    fig = plt.figure(figsize=(20, 20), facecolor='white')
    cx, cy = 0.5, 0.5

    tri_r, tri_size = 0.07, 0.11
    tri = {
        'image':   (cx + tri_r * np.cos(np.deg2rad(90)),
                    cy + tri_r * np.sin(np.deg2rad(90))),
        'gradcam': (cx + tri_r * np.cos(np.deg2rad(210)),
                    cy + tri_r * np.sin(np.deg2rad(210))),
        'recon':   (cx + tri_r * np.cos(np.deg2rad(330)),
                    cy + tri_r * np.sin(np.deg2rad(330))),
    }

    R1, F1 = 0.27, 0.08
    feat_pos = []
    for k in range(K):
        ang = 90.0 - (360.0 / K) * k
        feat_pos.append((cx + R1 * np.cos(np.deg2rad(ang)),
                         cy + R1 * np.sin(np.deg2rad(ang)), ang))

    R2, CH = 0.42, 0.055
    arc_half = (360.0 / K) * 0.40 if K > 0 else 0.0
    ch_pos = []
    for (fx, fy, fang) in feat_pos:
        offs = [0.0] if N <= 1 else np.linspace(-arc_half, arc_half, N)
        ch_pos.append([(cx + R2 * np.cos(np.deg2rad(fang + o)),
                        cy + R2 * np.sin(np.deg2rad(fang + o)), fang + o)
                       for o in offs])

    # edges: channel -> component, colored by loading sign
    for k, (fx, fy, _) in enumerate(feat_pos):
        ref = max(abs(comps[k]['channels'][0]['loading']), 1e-8) if N else 1.0
        for cpos, cinfo in zip(ch_pos[k], comps[k]['channels']):
            (xp, yp, _) = cpos
            w = cinfo['loading']
            color = 'forestgreen' if w >= 0 else 'firebrick'
            alpha = 0.5 + 0.5 * min(1.0, abs(w) / ref)
            _line(fig, (xp, yp), (fx, fy), color=color, lw=1.2,
                  linestyle='--', alpha=alpha, zorder=1)

    p_img, p_gc, p_re = tri['image'], tri['gradcam'], tri['recon']
    _line(fig, p_img, p_gc, color='gray', lw=1.0, alpha=0.5, zorder=1)
    _line(fig, p_gc, p_re, color='royalblue', lw=2.5, alpha=0.9, zorder=1)  # EXACT pair
    _line(fig, p_img, p_re, color='gray', lw=1.0, alpha=0.5, zorder=1)

    # center panels
    ax = _add_inset(fig, p_img[0], p_img[1], tri_size, tri_size)
    ax.imshow(image); ax.set_title("Input image", fontsize=9, fontweight='bold')

    ax = _add_inset(fig, p_gc[0], p_gc[1], tri_size, tri_size)
    ax.imshow(gradcam_map.numpy(), cmap='jet', interpolation='bilinear')
    ax.set_title("Grad-CAM", fontsize=9, fontweight='bold', color='royalblue')
    _set_border(ax, 'royalblue', 2.0)

    ax = _add_inset(fig, p_re[0], p_re[1], tri_size, tri_size)
    ax.imshow(recon_map.numpy(), cmap='jet', interpolation='bilinear')
    rc = ('forestgreen' if recon_cos >= 0.8
          else 'darkorange' if recon_cos >= 0.5 else 'firebrick')
    ax.set_title(f"rank-{recon_D} recon\ncos = {recon_cos:.3f}",
                 fontsize=9, fontweight='bold', color=rc)
    _set_border(ax, rc, 2.0)

    # Ring 1: signed component contributions beta_d * z_d (shared diverging scale)
    if K > 0:
        cmax = max(float(c['contrib_map'].abs().max()) for c in comps)
        cmax = cmax if cmax > 1e-8 else 1.0
    else:
        cmax = 1.0
    for k, (fx, fy, _) in enumerate(feat_pos):
        c = comps[k]
        ax = _add_inset(fig, fx, fy, F1, F1)
        ax.imshow(c['contrib_map'].numpy(), cmap='bwr',
                  vmin=-cmax, vmax=cmax, interpolation='bilinear')
        ax.set_title(f"v{c['comp_id']}\nbeta={c['beta']:+.2f}",
                     fontsize=8, fontweight='bold')
        _set_border(ax, 'royalblue' if c['beta'] >= 0 else 'firebrick', 1.5)

    # Ring 2: channel loadings of each eigenvector
    for k, cl in enumerate(ch_pos):
        c = comps[k]
        for (cpos, cinfo) in zip(cl, c['channels']):
            (xp, yp, _) = cpos
            ax = _add_inset(fig, xp, yp, CH, CH)
            ax.imshow(cinfo['map'].numpy(), cmap='viridis', interpolation='bilinear')
            w = cinfo['loading']
            sign = '+' if w >= 0 else '-'
            tc = 'darkgreen' if w >= 0 else 'darkred'
            ax.set_title(f"ch{cinfo['ch_id']}\n{sign}{abs(w):.3f}",
                         fontsize=7, fontweight='bold', color=tc)
            _set_border(ax, tc, 1.0)

    status = "CORRECT" if correct else "WRONG"
    header = (f"Hierarchical PCA Grad-CAM decomposition  --  {model_name} @ "
              f"{target_layer}  --  true: {results['true_class'][:40]}  |  "
              f"pred: {results['pred_class'][:40]}  [{status}]")
    if n_available is not None:
        header += f"  (img {img_idx + 1}/{n_available}, offset={offset})"
    fig.text(0.5, 0.97, header, ha='center', va='top',
             fontsize=13, fontweight='bold')

    legend = [
        f"Grad-CAM = ReLU( bias + sum_d beta_d * v_d ).  Center: input | "
        f"Grad-CAM | rank-{recon_D} reconstruction (EXACT pair, cos={recon_cos:.3f}, "
        f"ReLU keeps {100*results['relu_keep_frac']:.0f}% of pre-ReLU mass).",
        f"Ring 1: top-{K} components by |beta_d| (of {results['num_components_total']}); "
        f"signed contribution beta_d*z_d (blue +, red -). Components are "
        f"DETERMINISTIC and class-shared; only beta_d is image/class-specific.",
        f"Ring 2: top-{N} channel loadings v_(d,c) per component "
        f"(green = positive, red = negative).",
    ]
    fig.text(0.5, 0.03, '\n'.join(legend), ha='center', va='bottom',
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    ap = argparse.ArgumentParser(
        description='Hierarchical radial PCA Grad-CAM decomposition (one image).')
    ap.add_argument('--class_id', type=int, required=True)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--model', type=str, default='resnet50',
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--target_layer', type=str, default=None)
    ap.add_argument('--pca_model', type=str, required=True,
                    help='PCAReconstructor .pkl from csae_pca_baseline.py.')
    ap.add_argument('--test_data_dir', type=str,
                    default='/data/imagenet1k_sampletest')
    ap.add_argument('--ring_components', type=int, default=16,
                    help='How many top-|beta| components in ring 1.')
    ap.add_argument('--top_channels', type=int, default=4,
                    help='Top channel loadings per component in ring 2.')
    ap.add_argument('--recon_D', type=int, default=50,
                    help='Rank used for the center reconstruction panel.')
    ap.add_argument('--output_dir', type=str, default='hier_pca_visualizations')
    args = ap.parse_args()

    if not (0 <= args.class_id < 1000):
        raise ValueError(f"class_id must be in [0, 999], got {args.class_id}")
    if args.target_layer is None:
        args.target_layer = MODEL_CONFIGS[args.model]['default_target_layer']
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Hierarchical PCA Grad-CAM Decomposition")
    cname = list(IMAGENET2012_CLASSES.values())[args.class_id]
    print(f"  Class:      {args.class_id} ({cname[:60]})")
    print(f"  Model:      {args.model} @ {args.target_layer}")
    print(f"  PCA model:  {args.pca_model}")
    print(f"  ring_comp:  {args.ring_components}  top_ch: {args.top_channels}  "
          f"recon_D: {args.recon_D}")
    print("=" * 80)

    test_meta = Path(args.test_data_dir) / "test_metadata.pkl"
    image, n_avail, idx = load_image_for_class(test_meta, args.class_id,
                                               offset=args.offset)
    print(f"  Picked image {idx + 1}/{n_avail} (offset {args.offset} -> {idx})")

    extractor = HierPCAExtractor(
        args.model, args.target_layer, args.pca_model,
        device='cuda' if torch.cuda.is_available() else 'cpu')
    results = extractor.extract(
        image, label=args.class_id, ring_components=args.ring_components,
        top_channels=args.top_channels, recon_D=args.recon_D)

    save_path = out / (f"hierpca_class{args.class_id}_img{idx}_{args.model}.png")
    build_radial_figure(results, args.model, args.target_layer,
                        str(save_path), offset=args.offset, img_idx=idx,
                        n_available=n_avail)
    print("=" * 80 + "\nDone.\n" + "=" * 80)


if __name__ == "__main__":
    main()