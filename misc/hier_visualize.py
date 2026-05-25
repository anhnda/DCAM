"""
Hierarchical Radial Visualization for Multi-Channel ConvSAE on ImageNet-1k.

For a chosen class_id, samples ONE test image and produces a single figure with:

    CENTER  (triangle)
        - Input image
        - Grad-CAM map
        - Sum-of-z map (L1-normalized, with cos vs Grad-CAM label)
        Edges: blue solid edge Grad-CAM <-> Sum-of-z (the constrained pair),
               light edge Image <-> Grad-CAM (input -> attribution).

    RING 1  (top-K features, default 16)
        Top-K SAE features z_d arranged on a circle around the center.
        Each panel shows:
            - feature index F{d}
            - importance (spatial sum)
            - peak activation
            - the feature's H x W activation map.

    RING 2  (per-feature input channels, default 4 per feature, "petals")
        For each z_d in ring 1, its top-N input channels (by |encoder weight|)
        are arranged in a small radial sub-arc OUTWARD from that feature.
        A dashed line connects each input channel back to its parent z_d,
        colored:
            green = positive encoder weight (channel drives z_d up)
            red   = negative encoder weight (channel suppresses z_d)

Usage:
    python hier_visualize.py --class_id 207                  # offset 0 (first cached image)
    python hier_visualize.py --class_id 207 --offset 3       # 4th cached image
    python hier_visualize.py --class_id 207 --top_k_features 12 --top_input_channels 4
    python hier_visualize.py --class_id 207 --model resnet18

    # Sweep all cached images for a class:
    for i in 0 1 2 3 4; do python hier_visualize.py --class_id 207 --offset $i; done

Output:
    hier_class{class_id}_img{idx}_{model}.png    in --output_dir
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
from matplotlib.patches import FancyArrowPatch
from torchvision import transforms

sys.path.append('.')
from run_xcsae_full import MultiChannelConvSAE
from src.gradcam import GradCAM
from full_classes import IMAGENET2012_CLASSES


# ==========================================
# Model configurations (kept in sync with the rest of the pipeline)
# ==========================================

MODEL_CONFIGS = {
    'resnet50': {
        'model_fn': lambda: models.resnet50(pretrained=True),
        'default_target_layer': 'layer3',
        'description': 'ResNet50 (layer3: 1024ch, 14x14)',
    },
    'resnet18': {
        'model_fn': lambda: models.resnet18(pretrained=True),
        'default_target_layer': 'layer3',
        'description': 'ResNet18 (layer3: 256ch, 14x14)',
    },
    'vgg16': {
        'model_fn': lambda: models.vgg16(pretrained=True),
        'default_target_layer': 'features[16]',
        'description': 'VGG16 (features[16]: 256ch, 28x28)',
    },
    'efficientnet': {
        'model_fn': lambda: models.efficientnet_b0(pretrained=True),
        'default_target_layer': 'features[4]',
        'description': 'EfficientNet-B0 (features[4]: ~80ch, 14x14)',
    },
}


# ==========================================
# Image sampling: pull ONE image of a chosen class from the cached test set
# ==========================================

def load_image_for_class(test_metadata_path: Path,
                         class_id: int,
                         offset: int = 0) -> Tuple[Image.Image, int, int]:
    """Pick a single test image from the cached sampled test set by offset.

    The cached test samples are stored in a fixed, deterministic order in
    metadata['samples']. We filter to the rows whose label == class_id (also
    a deterministic, order-preserving operation) and then directly index by
    `offset`. Negative offsets wrap from the end (Python list convention),
    and offsets >= n wrap modulo n.

    Args:
        test_metadata_path: path to test_metadata.pkl produced by the test
            sampler in visualize_testmf_full.py.
        class_id: ImageNet-1k class id in [0, 999].
        offset: which of the cached images for this class to pick. Wrapped
            into [0, n) where n is the number of cached images for the class.

    Returns:
        (image, num_available, chosen_local_idx)
            num_available    : how many cached test images exist for this class
            chosen_local_idx : the wrapped offset actually used (0..num_available-1)
    """
    if not test_metadata_path.exists():
        raise FileNotFoundError(
            f"Test metadata not found at {test_metadata_path}. "
            f"Run visualize_testmf_full.py first to populate the cache."
        )

    metadata = joblib.load(test_metadata_path)
    samples = metadata['samples']

    # Filter by class. Order-preserving, so identical (class_id, offset)
    # always returns the same image.
    class_samples = [(b, lbl) for (b, lbl) in samples if lbl == class_id]
    n = len(class_samples)
    if n == 0:
        raise ValueError(f"No cached test samples for class_id={class_id}")

    idx = offset % n
    if offset != idx:
        print(f"  Note: offset {offset} wrapped to {idx} (only {n} cached "
              f"images for class {class_id})")

    image_bytes, label = class_samples[idx]
    assert label == class_id
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return img, n, idx


# ==========================================
# One-shot feature + input-channel extractor
#
# Single forward pass through the backbone, then for each of the top-K
# features we read off:
#   - the feature's spatial map
#   - importance (sum over space) and peak
#   - the top-N input channels by |encoder weight|, with signed weights and
#     their actual normalized activation maps on THIS image.
# ==========================================

class HierExtractor:
    def __init__(self,
                 model_name: str,
                 target_layer_name: str,
                 csae_model_path: str,
                 device='cuda',
                 cumulative_threshold: float = 0.85):
        self.model_name = model_name
        self.target_layer_name = target_layer_name
        self.cumulative_threshold = cumulative_threshold
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # CSAE
        print(f"Loading ConvSAE from {csae_model_path}...")
        self.csae = joblib.load(csae_model_path).to(self.device)
        self.csae.eval()
        print(f"  CSAE: {self.csae.in_channels} -> {self.csae.hidden_dim} (top_k={self.csae.top_k})")

        # Backbone
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        self.backbone = MODEL_CONFIGS[model_name]['model_fn']().to(self.device)
        self.backbone.eval()
        self.target_layer = self._get_target_layer()
        self._detect_layer_dimensions()
        print(f"  Backbone: {model_name} @ {target_layer_name}  "
              f"({self.num_channels}ch, {self.spatial_size}x{self.spatial_size})")

        # Forward hook to capture layer activations
        self.layer_activations = None
        self.target_layer.register_forward_hook(self._save_layer_activation)

        # Grad-CAM
        self.gradcam = GradCAM(self.backbone, self.target_layer)

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ----- backbone wiring (kept identical to visualize_testmf_full.py) -----

    def _get_target_layer(self):
        if self.model_name in ['resnet50', 'resnet18']:
            if 'layer1' in self.target_layer_name:
                return self.backbone.layer1
            if 'layer2' in self.target_layer_name:
                return self.backbone.layer2
            if 'layer3' in self.target_layer_name:
                return self.backbone.layer3
            if 'layer4' in self.target_layer_name:
                return self.backbone.layer4
            raise ValueError(f"Unknown ResNet layer: {self.target_layer_name}")
        if self.model_name in ['vgg16', 'efficientnet']:
            idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            return self.backbone.features[idx]
        raise ValueError(f"Unknown model: {self.model_name}")

    def _detect_layer_dimensions(self):
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            if self.model_name in ['resnet50', 'resnet18']:
                x = self.backbone.conv1(dummy)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x)
                if 'layer1' not in self.target_layer_name:
                    x = self.backbone.layer2(x)
                    if 'layer2' not in self.target_layer_name:
                        x = self.backbone.layer3(x)
                        if 'layer3' not in self.target_layer_name:
                            x = self.backbone.layer4(x)
            elif self.model_name in ['vgg16', 'efficientnet']:
                idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
                x = dummy
                for i in range(idx + 1):
                    x = self.backbone.features[i](x)
            self.num_channels = x.shape[1]
            self.spatial_size = x.shape[2]

    def _save_layer_activation(self, module, input, output):
        self.layer_activations = output.detach()

    # ----- activation normalization (same as training/viz) -----

    def _normalize_layer_activations(self, acts: torch.Tensor) -> torch.Tensor:
        normalized = acts.clone()
        for c in range(acts.shape[1]):
            ch = acts[0, c, :, :]
            if ch.abs().sum() < 1e-8:
                continue
            nz = ch[ch > 1e-8]
            if len(nz) > 0:
                scale = torch.quantile(nz, 0.99)
                if scale > 1e-8:
                    ch = torch.clamp(ch, min=0.0, max=scale)
                    normalized[0, c, :, :] = ch / (scale + 1e-8)
        return normalized

    # ----- Grad-CAM spatial map (min-max normalized for display) -----

    def _gradcam_spatial_map(self, image: torch.Tensor) -> Tuple[torch.Tensor, int]:
        weights, _, pred_class = self.gradcam.forward(image, class_idx=None, verbose=False)
        with torch.no_grad():
            acts = self.layer_activations[0]                       # [C, H, W]
            weighted = (weights.view(-1, 1, 1) * acts).sum(dim=0)  # [H, W]
            gmap = F.relu(weighted)
            gmin, gmax = gmap.min(), gmap.max()
            if (gmax - gmin) > 1e-8:
                gmap = (gmap - gmin) / (gmax - gmin)
            else:
                gmap = torch.zeros_like(gmap)
        return gmap.cpu(), pred_class

    # ----- main extraction: one pass returns everything we need -----

    def extract(self,
                image: Image.Image,
                label: int,
                top_k_features: int = 16,
                top_input_channels: int = 4,
                input_channel_view: str = 'encoder') -> Dict:

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 1) Forward pass to populate self.layer_activations & get prediction
        with torch.no_grad():
            logits = self.backbone(image_tensor)
            layer_acts = self.layer_activations.clone()              # [1, C, H, W]
            pred_label = logits.argmax(dim=1).item()

        # 2) Grad-CAM map (also re-runs forward, but we then re-capture below)
        gradcam_map, _ = self._gradcam_spatial_map(image_tensor)

        # 3) Normalize the captured activations the same way as training
        layer_acts_norm = self._normalize_layer_activations(layer_acts)

        # 4) Run the SAE
        with torch.no_grad():
            _, sparse_features = self.csae(layer_acts_norm, use_topk=True)
            # sparse_features: [1, D, H, W]

        # 5) Top-K features by spatial-sum importance
        importance = sparse_features.sum(dim=(2, 3)).squeeze(0)       # [D]
        top_k = min(top_k_features, importance.numel())
        top_vals, top_idx = torch.topk(importance, k=top_k)

        # 6) Sum-of-z (L1 normalized) and cosine vs Grad-CAM
        with torch.no_grad():
            z_sum_all = sparse_features[0].sum(dim=0).cpu()          # [H, W]

            def _l1_norm(m):
                s = m.sum()
                return m / s if s.abs() > 1e-8 else m

            z_sum_n = _l1_norm(z_sum_all.clone())
            gc_n = _l1_norm(gradcam_map.clone())
            a = z_sum_n.flatten()
            b = gc_n.flatten()
            cos_sim = float((a @ b) / (a.norm().clamp(min=1e-8)
                                       * b.norm().clamp(min=1e-8)))

        # 7) For each top feature, pull its top input channels (encoder view)
        #    by |encoder weight| magnitude, plus their actual maps on this image.
        with torch.no_grad():
            if input_channel_view == 'encoder':
                W = self.csae.encoder.weight                          # [D, C, kH, kW]
            elif input_channel_view == 'decoder':
                # column d of decoder
                W = self.csae.decoder.weight.permute(1, 0, 2, 3)      # [D, C, kH, kW]
            else:
                raise ValueError(f"input_channel_view must be encoder/decoder")

            top_features: List[Dict] = []
            for d_tensor, imp_tensor in zip(top_idx, top_vals):
                d = int(d_tensor.item())
                imp = float(imp_tensor.item())

                # Feature spatial map (this image)
                fmap = sparse_features[0, d, :, :].cpu()
                peak = float(fmap.max())

                # Top input channels for feature d
                w_d = W[d]                                             # [C, kH, kW]
                w_per_channel = w_d.view(w_d.shape[0], -1).mean(dim=1) # [C]
                mag = w_per_channel.abs()
                n_in = min(top_input_channels, mag.numel())
                top_w_vals, top_ch_idx = torch.topk(mag, k=n_in)

                channels: List[Dict] = []
                for cv_idx in range(n_in):
                    ch_id = int(top_ch_idx[cv_idx].item())
                    w_signed = float(w_per_channel[ch_id].item())
                    ch_map = layer_acts_norm[0, ch_id, :, :].cpu()
                    channels.append({
                        'ch_id': ch_id,
                        'weight': w_signed,
                        'map': ch_map,
                    })

                top_features.append({
                    'feature_id': d,
                    'importance': imp,
                    'peak': peak,
                    'map': fmap,
                    'channels': channels,
                })

        true_class = list(IMAGENET2012_CLASSES.values())[label]
        pred_class = list(IMAGENET2012_CLASSES.values())[pred_label]

        return {
            'image': image,
            'label': label,
            'pred_label': pred_label,
            'true_class': true_class,
            'pred_class': pred_class,
            'correct': pred_label == label,
            'gradcam_map': gradcam_map,
            'z_sum_map': z_sum_n,
            'z_sum_cos_vs_gradcam': cos_sim,
            'top_features': top_features,
            'input_channel_view': input_channel_view,
            'num_input_channels': self.num_channels,
            'num_sae_features': self.csae.hidden_dim,
            'sae_top_k': self.csae.top_k,
        }


# ==========================================
# Radial figure builder
#
# Coordinate system: figure spans [0,1] x [0,1] with the center at (0.5, 0.5).
# Each panel is drawn as an inset axes anchored at a (cx, cy) center point with
# given (w, h) extents. This decouples panel placement from gridspec, which is
# what we need for radial layouts.
# ==========================================

def _add_inset(fig, cx: float, cy: float, w: float, h: float):
    """Add a fresh axes whose box is centered at (cx, cy) with size (w, h)."""
    left = cx - w / 2.0
    bottom = cy - h / 2.0
    ax = fig.add_axes([left, bottom, w, h])
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def _set_border(ax, color: str, lw: float = 2.0):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_visible(True)
    ax.set_frame_on(True)


def _line(fig, p0: Tuple[float, float], p1: Tuple[float, float],
          color: str, lw: float = 1.0, linestyle: str = '-',
          alpha: float = 1.0, zorder: int = 1):
    """Draw a line in figure coordinates."""
    line = plt.Line2D([p0[0], p1[0]], [p0[1], p1[1]],
                      transform=fig.transFigure,
                      color=color, linewidth=lw, linestyle=linestyle,
                      alpha=alpha, zorder=zorder)
    fig.add_artist(line)


def build_radial_figure(results: Dict,
                        model_name: str,
                        target_layer: str,
                        save_path: str,
                        offset: int = 0,
                        img_idx: int = 0,
                        n_available: int = None):
    image = results['image']
    gradcam_map = results['gradcam_map']
    z_sum_map = results['z_sum_map']
    cos_sim = results['z_sum_cos_vs_gradcam']
    top_features = results['top_features']
    correct = results['correct']
    true_class = results['true_class']
    pred_class = results['pred_class']
    view = results['input_channel_view']

    K = len(top_features)
    N = len(top_features[0]['channels']) if K > 0 else 0

    # ---------------- Geometry ----------------
    # Figure is square. Coordinates are normalized [0, 1].
    fig = plt.figure(figsize=(20, 20), facecolor='white')

    cx, cy = 0.5, 0.5

    # Center triangle: three panels at the corners of an equilateral triangle.
    # Triangle "radius" = distance from center to each corner.
    tri_r = 0.07
    tri_size = 0.11
    # Standard orientation: image at top, Grad-CAM bottom-left, sum-of-z bottom-right.
    tri_positions = {
        'image':   (cx + tri_r * np.cos(np.deg2rad(90)),
                    cy + tri_r * np.sin(np.deg2rad(90))),
        'gradcam': (cx + tri_r * np.cos(np.deg2rad(210)),
                    cy + tri_r * np.sin(np.deg2rad(210))),
        'zsum':    (cx + tri_r * np.cos(np.deg2rad(330)),
                    cy + tri_r * np.sin(np.deg2rad(330))),
    }

    # Ring 1 (features): radius R1 from center, K panels evenly spaced.
    R1 = 0.27
    F1_size = 0.08
    # Start at 90 degrees (top) and go clockwise for visual familiarity.
    feat_positions = []
    for k in range(K):
        ang = 90.0 - (360.0 / K) * k
        x = cx + R1 * np.cos(np.deg2rad(ang))
        y = cy + R1 * np.sin(np.deg2rad(ang))
        feat_positions.append((x, y, ang))

    # Ring 2 (channels): each feature's N channels in a small arc OUTWARD.
    # Each channel panel sits at radius R2 from figure center, angularly
    # spread by +/- arc_half_deg around the parent feature's angle.
    R2 = 0.42
    CH_size = 0.055
    arc_half_deg = (360.0 / K) * 0.40   # ~40% of inter-feature angular spacing
    channel_positions = []  # list of lists, one per feature
    for (fx, fy, fang) in feat_positions:
        chs = []
        if N <= 1:
            offsets = [0.0]
        else:
            offsets = np.linspace(-arc_half_deg, arc_half_deg, N)
        for off in offsets:
            ang_c = fang + off
            x = cx + R2 * np.cos(np.deg2rad(ang_c))
            y = cy + R2 * np.sin(np.deg2rad(ang_c))
            chs.append((x, y, ang_c))
        channel_positions.append(chs)

    # ---------------- Draw edges FIRST (so panels sit on top) ----------------

    # Channel -> feature dashed edges, colored by encoder-weight sign
    for k, (fx, fy, _) in enumerate(feat_positions):
        for ch_pos, ch_info in zip(channel_positions[k], top_features[k]['channels']):
            (cxp, cyp, _) = ch_pos
            w = ch_info['weight']
            color = 'forestgreen' if w >= 0 else 'firebrick'
            alpha = 0.5 + 0.5 * min(1.0, abs(w) / max(
                abs(top_features[k]['channels'][0]['weight']), 1e-8))
            _line(fig, (cxp, cyp), (fx, fy),
                  color=color, lw=1.2, linestyle='--', alpha=alpha, zorder=1)

    # Center triangle edges
    p_img = tri_positions['image']
    p_gc = tri_positions['gradcam']
    p_zs = tri_positions['zsum']
    # Image <-> Grad-CAM: light gray (Grad-CAM is computed from the image)
    _line(fig, p_img, p_gc, color='gray', lw=1.0, linestyle='-', alpha=0.5, zorder=1)
    # Grad-CAM <-> Sum-of-z: blue solid, this is the CONSTRAINED pair
    _line(fig, p_gc, p_zs, color='royalblue', lw=2.5, linestyle='-', alpha=0.9, zorder=1)
    # Image <-> Sum-of-z: light gray
    _line(fig, p_img, p_zs, color='gray', lw=1.0, linestyle='-', alpha=0.5, zorder=1)

    # ---------------- Center triangle panels ----------------
    ax_img = _add_inset(fig, p_img[0], p_img[1], tri_size, tri_size)
    ax_img.imshow(image)
    ax_img.set_title("Input image", fontsize=9, fontweight='bold')

    ax_gc = _add_inset(fig, p_gc[0], p_gc[1], tri_size, tri_size)
    ax_gc.imshow(gradcam_map.numpy(), cmap='jet', interpolation='bilinear')
    ax_gc.set_title("Grad-CAM", fontsize=9, fontweight='bold', color='royalblue')
    _set_border(ax_gc, 'royalblue', lw=2.0)

    ax_zs = _add_inset(fig, p_zs[0], p_zs[1], tri_size, tri_size)
    ax_zs.imshow(z_sum_map.numpy(), cmap='jet', interpolation='bilinear')
    if cos_sim >= 0.8:
        sum_color = 'forestgreen'
    elif cos_sim >= 0.5:
        sum_color = 'darkorange'
    else:
        sum_color = 'firebrick'
    ax_zs.set_title(f"Sum of z\ncos = {cos_sim:.3f}",
                    fontsize=9, fontweight='bold', color=sum_color)
    _set_border(ax_zs, sum_color, lw=2.0)

    # ---------------- Ring 1: features ----------------
    # Shared vmax across the top-k features so brightness reflects actual magnitude
    if K > 0:
        shared_vmax = max(float(tf['map'].max()) for tf in top_features)
        if shared_vmax < 1e-8:
            shared_vmax = 1.0
    else:
        shared_vmax = 1.0

    for k, (fx, fy, _) in enumerate(feat_positions):
        tf = top_features[k]
        ax = _add_inset(fig, fx, fy, F1_size, F1_size)
        ax.imshow(tf['map'].numpy(), cmap='hot', interpolation='bilinear',
                  vmin=0.0, vmax=shared_vmax)
        ax.set_title(f"F{tf['feature_id']}\nimp {tf['importance']:.1f}  pk {tf['peak']:.2f}",
                     fontsize=8, fontweight='bold')

    # ---------------- Ring 2: per-feature input channels ----------------
    for k, ch_pos_list in enumerate(channel_positions):
        tf = top_features[k]
        for (ch_pos, ch_info) in zip(ch_pos_list, tf['channels']):
            (xp, yp, _) = ch_pos
            ax = _add_inset(fig, xp, yp, CH_size, CH_size)
            ax.imshow(ch_info['map'].numpy(), cmap='viridis', interpolation='bilinear')
            w = ch_info['weight']
            sign = '+' if w >= 0 else '-'
            title_color = 'darkgreen' if w >= 0 else 'darkred'
            ax.set_title(f"ch{ch_info['ch_id']}\n{sign}{abs(w):.3f}",
                         fontsize=7, fontweight='bold', color=title_color)
            _set_border(ax, title_color, lw=1.0)

    # ---------------- Header / legend ----------------
    status = "CORRECT" if correct else "WRONG"
    header = (f"Hierarchical DCAM view  --  {model_name} @ {target_layer}  --  "
              f"true: {true_class[:40]}  |  pred: {pred_class[:40]}  [{status}]")
    if n_available is not None:
        header += f"  (img {img_idx + 1}/{n_available}, offset={offset})"
    fig.text(0.5, 0.97, header, ha='center', va='top',
             fontsize=13, fontweight='bold')

    legend_lines = [
        f"Center triangle: input | Grad-CAM | Sum_d z_d   "
        f"(constrained pair in blue, cos = {cos_sim:.3f})",
        f"Ring 1: top-{K} SAE features (out of {results['num_sae_features']}, "
        f"SAE top_k={results['sae_top_k']})",
        f"Ring 2: top-{N} input channels per feature ({view} weights; "
        f"green = positive, red = negative)",
    ]
    fig.text(0.5, 0.03, '\n'.join(legend_lines), ha='center', va='bottom',
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Hierarchical radial DCAM visualization for a single image'
    )
    parser.add_argument('--class_id', type=int, required=True,
                        help='ImageNet-1k class id (0..999)')
    parser.add_argument('--offset', type=int, default=0,
                        help='Which of the cached test images for this class '
                             'to pick. Wrapped modulo the number of cached '
                             'images for the class. Default: 0.')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--target_layer', type=str, default=None)
    parser.add_argument('--csae_model', type=str, default=None,
                        help='Path to trained CSAE .pkl (auto-detected if not given)')
    parser.add_argument('--test_data_dir', type=str,
                        default='/data/imagenet1k_sampletest',
                        help='Directory with the cached test sample metadata')
    parser.add_argument('--top_k_features', type=int, default=16)
    parser.add_argument('--top_input_channels', type=int, default=4)
    parser.add_argument('--input_channel_view', type=str, default='encoder',
                        choices=['encoder', 'decoder'])
    parser.add_argument('--output_dir', type=str, default='hier_visualizations')
    args = parser.parse_args()

    if not (0 <= args.class_id < 1000):
        raise ValueError(f"class_id must be in [0, 999], got {args.class_id}")

    if args.target_layer is None:
        args.target_layer = MODEL_CONFIGS[args.model]['default_target_layer']

    if args.csae_model is None:
        args.csae_model = f'imagenet1k_csae_{args.model}_model.pkl'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Hierarchical DCAM Visualization")
    print("=" * 80)
    class_name = list(IMAGENET2012_CLASSES.values())[args.class_id]
    print(f"  Class id:    {args.class_id}  ({class_name[:60]})")
    print(f"  Model:       {args.model} @ {args.target_layer}")
    print(f"  CSAE model:  {args.csae_model}")
    print(f"  top_k feat:  {args.top_k_features}   "
          f"top_in_ch:   {args.top_input_channels}   "
          f"view: {args.input_channel_view}")
    print(f"  Offset:      {args.offset}")
    print("=" * 80)

    # 1) Load test image (deterministic given offset)
    test_meta = Path(args.test_data_dir) / "test_metadata.pkl"
    image, n_avail, chosen_idx = load_image_for_class(
        test_meta, args.class_id, offset=args.offset
    )
    print(f"  Picked image {chosen_idx + 1} / {n_avail} "
          f"for class {args.class_id} (offset {args.offset} -> idx {chosen_idx})")

    # 2) Build extractor and run
    extractor = HierExtractor(
        model_name=args.model,
        target_layer_name=args.target_layer,
        csae_model_path=args.csae_model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    results = extractor.extract(
        image,
        label=args.class_id,
        top_k_features=args.top_k_features,
        top_input_channels=args.top_input_channels,
        input_channel_view=args.input_channel_view,
    )

    # 3) Save figure -- filename uses the resolved index, so iterating
    # --offset 0..N-1 produces N distinct files with no collisions.
    save_path = (output_dir /
                 f"hier_class{args.class_id}"
                 f"_img{chosen_idx}"
                 f"_{args.model}.png")
    build_radial_figure(
        results,
        model_name=args.model,
        target_layer=args.target_layer,
        save_path=str(save_path),
        offset=args.offset,
        img_idx=chosen_idx,
        n_available=n_avail,
    )

    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()