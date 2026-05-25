"""
hier_visualize_unified.py
=========================
Hierarchical radial visualization of the JOINT-SOLVED unified CAM decomposition.

This is the hier_visualize_pca.py figure, but driven by the joint solver
instead of a frozen PCA pkl. hier_visualize_pca.py loads a PCAReconstructor
whose basis V is a fixed eigendecomposition; here the basis V_D is the output
of solving Equation (1) of "A Unified Objective for CAM-Style Attribution" at a
chosen corner of the parameter cube. The radial layout is identical so the two
figures are directly comparable side by side.

What the rings show
-------------------
  CENTER (triangle)
    - Input image
    - True Grad-CAM  ReLU(L~)
    - Rank-D reconstruction  ReLU(bias + sum_d beta_d z_d) with the SOLVED basis
    Edges: BLUE solid = Grad-CAM <-> reconstruction (the exact additive pair,
           labelled with spatial cosine); gray = image -> Grad-CAM.

  RING 1  (top-|beta_d| components, default 16)
    The solved components that most build THIS image's Grad-CAM, ranked by
    |beta_d|. Each panel: the signed contribution beta_d * z_d (blue = pushes
    Grad-CAM up, red = down), labelled v{d}, beta_d.

  RING 2  (per-component channel loadings, default 4 petals)
    For each solved component v_d, its top-N backbone channels by |v_{d,c}|.
    Dashed edges: green = positive loading, red = negative. Each petal shows
    that channel's activation map on this image.

HOW THIS DIFFERS FROM hier_visualize_pca.py (read this -- the figure means
something different)
---------------------------------------------------------------------------
  * The components are NOT deterministic-by-construction. They are SOLVED. At
    the DCAM corner (phi=id, w=global, lambda=1, alpha free) the solve reduces
    to the same eigendecomposition the PCA pkl stores, so the figure matches
    hier_visualize_pca.py. At any other corner -- per-image w, finite
    bandwidth, lambda<1, a kernel -- the components are a DIFFERENT basis,
    fitted to that corner's objective. The header states which corner.
  * Seed-stability is therefore a CHECKED property, not an assumption. Run the
    solver with --n_restarts > 1 and the figure annotates the blended-Sigma
    eigengap and the cross-restart subspace agreement. A green badge means the
    decomposition reproduced; a red badge means this corner's basis is NOT
    seed-stable and the rings should not be read as canonical.
  * Kernel corners (phi=rbf): the exact additive identity breaks (paper
    section 5). Ring 1 / Ring 2 require the additive split, so for a kernel the
    script renders only the center triangle plus the scope-caveat text and
    skips the rings.

Prerequisite
------------
NONE beyond the backbone -- the basis is solved on the fly from a bank of
cached images. (hier_visualize_pca.py needed a prebuilt pkl; this does not.)

Usage
-----
  # DCAM corner: reproduces the hier_visualize_pca.py figure with a solved basis
  python hier_visualize_unified.py --class_id 207 --corner dcam --D 100

  # the NEW interior region: local, class-tilted basis -- a DIFFERENT ring
  python hier_visualize_unified.py --class_id 207 --corner new_interior \\
      --lam 0.5 --bandwidth 0.8 --D 60 --solver both --n_restarts 4 \\
      --ring_components 16 --top_channels 4 --recon_D 50

  # Grad-CAM corner: alpha pinned by the averaging rule, P solved
  python hier_visualize_unified.py --class_id 207 --corner grad_cam --D 50
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
import torchvision.models as models
from PIL import Image
from torchvision import transforms

sys.path.append('.')
from src.gradcam import GradCAM
from full_classes import IMAGENET2012_CLASSES

from unified_cam_joint import (
    SolveConfig, solve, decompose_query, config_for_corner,
    normalize_acts, CORNER_PRESETS,
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
# Image loading  (verbatim contract from hier_visualize_pca.py)
# ==========================================================================

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


def load_class_images(test_metadata_path: Path, class_id: int,
                      offset: int, num_images: int
                      ) -> List[Tuple[Image.Image, int]]:
    metadata = joblib.load(test_metadata_path)
    samples = metadata['samples']
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
# Backbone wrapper  (same contract as hier_visualize_pca.HierPCAExtractor,
# minus the frozen-pkl loading -- the basis is solved, not loaded)
# ==========================================================================

class UnifiedHierExtractor:
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
        """Run one image -> (raw activations [1,C,H,W], Grad-CAM alpha [C],
        predicted label)."""
        x = self.transform(image).unsqueeze(0).to(self.device)
        weights, _, pred_label = self.gradcam.forward(x, class_idx=None,
                                                      verbose=False)
        with torch.no_grad():
            A_raw = self.layer_activations.clone()
        return A_raw, weights.view(-1).to(self.device), int(pred_label)


# ==========================================================================
# Assemble the radial-figure payload from a SolveResult + the query
# ==========================================================================

def build_ring_payload(result, A_query_norm: torch.Tensor,
                        query_alpha: torch.Tensor, image: Image.Image,
                        label: int, pred_label: int,
                        ring_components: int, top_channels: int,
                        recon_D: int) -> Dict:
    """Turn the joint solve into the exact dict the radial figure consumes.

    Uses decompose_query for the additive pieces (bias, beta, z, comp,
    gradcam_true, recon) and reads V_D off the SolveResult for the Ring-2
    channel loadings. Mirrors HierPCAExtractor.extract's return contract so the
    figure code is shared.
    """
    diag = result.diagnostics
    dec = decompose_query(result, A_query_norm, query_alpha)

    true_class = list(IMAGENET2012_CLASSES.values())[label]
    pred_class = list(IMAGENET2012_CLASSES.values())[pred_label]

    payload = {
        'image': image, 'label': label, 'pred_label': pred_label,
        'true_class': true_class, 'pred_class': pred_class,
        'correct': pred_label == label,
        'gradcam_true': dec['gradcam_true'],
        'relu_keep_frac': dec['relu_keep_frac'],
        'additive_exact': dec['additive_exact'],
        'corner': diag['corner'],
        'eigengap': diag['eigengap'],
        'subspace_agreement': diag['subspace_agreement'],
        'seed_invariant': diag['seed_invariant'],
        'n_restarts': diag['n_restarts'],
        'lam': diag['lambda'], 'phi': diag['phi'], 'locality': diag['locality'],
        'J_final': diag['J_final'],
    }

    if not dec['additive_exact']:
        # kernel corner: no additive rings; center triangle only.
        payload.update({
            'recon_map': dec['recon'], 'recon_cos': 1.0,
            'recon_D': 0, 'components': [],
            'num_components_total': result.V_D.shape[1],
            'scope_caveat': dec.get('note', result.diagnostics.get(
                'scope_caveat', '')),
        })
        return payload

    # ---- phi=id: full additive decomposition -> Ring 1 + Ring 2 ----
    C = A_query_norm.shape[1]
    beta = dec['beta']                                    # [D]
    comp = dec['comp']                                    # [D,H,W]
    H, W = dec['H'], dec['W']

    # rank-D reconstruction for the center pair (recon_D, capped at D built)
    Dr = min(recon_D, comp.shape[0])
    recon = torch.relu(dec['bias'] + comp[:Dr].sum(dim=0))   # [H,W]
    gt = dec['gradcam_true']
    af, bf = recon.flatten(), gt.flatten()
    recon_cos = float((af @ bf) / (af.norm().clamp(min=1e-8)
                                   * bf.norm().clamp(min=1e-8)))

    # Ring 1: top components by |beta_d|
    K = min(ring_components, comp.shape[0])
    order = torch.argsort(beta.abs(), descending=True)[:K]

    V_D = result.V_D[:C, :].detach().cpu()                # [C, D] solved basis
    A_norm_cpu = A_query_norm[0].detach().cpu()           # [C,H,W]

    comp_list: List[Dict] = []
    for d_t in order:
        d = int(d_t.item())
        b = float(beta[d].item())
        cmap = comp[d]                                    # signed contribution
        v_d = V_D[:, d]                                   # [C] solved loadings
        mag = v_d.abs()
        n_in = min(top_channels, mag.numel())
        _, top_ch = torch.topk(mag, k=n_in)
        channels: List[Dict] = []
        for ci in range(n_in):
            ch_id = int(top_ch[ci].item())
            loading = float(v_d[ch_id].item())
            ch_map = A_norm_cpu[ch_id]
            channels.append({'ch_id': ch_id, 'loading': loading,
                             'map': ch_map})
        comp_list.append({'comp_id': d, 'beta': b,
                          'contrib_map': cmap, 'channels': channels})

    payload.update({
        'recon_map': recon, 'recon_cos': recon_cos, 'recon_D': Dr,
        'components': comp_list, 'num_components_total': comp.shape[0],
    })
    return payload


# ==========================================================================
# Radial figure  (geometry verbatim from hier_visualize_pca.py)
# ==========================================================================

def _add_inset(fig, cx, cy, w, h):
    ax = fig.add_axes([cx - w / 2.0, cy - h / 2.0, w, h])
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def _set_border(ax, color, lw=2.0):
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(lw)
        spine.set_visible(True)
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
    comps = results['components']
    correct = results['correct']
    additive = results['additive_exact']

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

    # ---- ring geometry (only when there are components to place) ----
    R1, F1 = 0.27, 0.08
    feat_pos = []
    for k in range(K):
        ang = 90.0 - (360.0 / K) * k
        feat_pos.append((cx + R1 * np.cos(np.deg2rad(ang)),
                         cy + R1 * np.sin(np.deg2rad(ang)), ang))

    R2, CH = 0.42, 0.055
    ch_pos = []
    if K > 0:
        arc_half = (360.0 / K) * 0.40
        for (fx, fy, fang) in feat_pos:
            offs = [0.0] if N <= 1 else np.linspace(-arc_half, arc_half, N)
            ch_pos.append([(cx + R2 * np.cos(np.deg2rad(fang + o)),
                            cy + R2 * np.sin(np.deg2rad(fang + o)), fang + o)
                           for o in offs])

    # ---- Ring-2 -> Ring-1 edges, colored by loading sign ----
    for k, (fx, fy, _) in enumerate(feat_pos):
        ref = (max(abs(comps[k]['channels'][0]['loading']), 1e-8)
               if N else 1.0)
        for cpos, cinfo in zip(ch_pos[k], comps[k]['channels']):
            (xp, yp, _) = cpos
            wld = cinfo['loading']
            color = 'forestgreen' if wld >= 0 else 'firebrick'
            a = 0.5 + 0.5 * min(1.0, abs(wld) / ref)
            _line(fig, (xp, yp), (fx, fy), color=color, lw=1.2,
                  linestyle='--', alpha=a, zorder=1)

    # ---- center triangle edges ----
    p_img, p_gc, p_re = tri['image'], tri['gradcam'], tri['recon']
    _line(fig, p_img, p_gc, color='gray', lw=1.0, alpha=0.5, zorder=1)
    _line(fig, p_gc, p_re, color='royalblue', lw=2.5, alpha=0.9, zorder=1)
    _line(fig, p_img, p_re, color='gray', lw=1.0, alpha=0.5, zorder=1)

    # ---- center panels ----
    ax = _add_inset(fig, p_img[0], p_img[1], tri_size, tri_size)
    ax.imshow(image); ax.set_title("Input image", fontsize=9, fontweight='bold')

    ax = _add_inset(fig, p_gc[0], p_gc[1], tri_size, tri_size)
    ax.imshow(gradcam_map.numpy(), cmap='jet', interpolation='bilinear')
    ax.set_title("Grad-CAM", fontsize=9, fontweight='bold', color='royalblue')
    _set_border(ax, 'royalblue', 2.0)

    ax = _add_inset(fig, p_re[0], p_re[1], tri_size, tri_size)
    recon_map = results['recon_map']
    recon_cos = results['recon_cos']
    recon_D = results['recon_D']
    ax.imshow(recon_map.numpy(), cmap='jet', interpolation='bilinear')
    if additive:
        rc = ('forestgreen' if recon_cos >= 0.8
              else 'darkorange' if recon_cos >= 0.5 else 'firebrick')
        ax.set_title(f"rank-{recon_D} recon\ncos = {recon_cos:.3f}",
                     fontsize=9, fontweight='bold', color=rc)
    else:
        rc = 'dimgray'
        ax.set_title("recon (projector only)\nkernel: no additive id.",
                     fontsize=8, fontweight='bold', color=rc)
    _set_border(ax, rc, 2.0)

    # ---- Ring 1: signed component contributions beta_d * z_d ----
    if K > 0:
        cmax = max(float(c['contrib_map'].abs().max()) for c in comps)
        cmax = cmax if cmax > 1e-8 else 1.0
        for k, (fx, fy, _) in enumerate(feat_pos):
            c = comps[k]
            ax = _add_inset(fig, fx, fy, F1, F1)
            ax.imshow(c['contrib_map'].numpy(), cmap='bwr',
                      vmin=-cmax, vmax=cmax, interpolation='bilinear')
            ax.set_title(f"v{c['comp_id']}\nbeta={c['beta']:+.2f}",
                         fontsize=8, fontweight='bold')
            _set_border(ax, 'royalblue' if c['beta'] >= 0 else 'firebrick', 1.5)

    # ---- Ring 2: channel loadings of each solved component ----
    for k, cl in enumerate(ch_pos):
        c = comps[k]
        for (cpos, cinfo) in zip(cl, c['channels']):
            (xp, yp, _) = cpos
            ax = _add_inset(fig, xp, yp, CH, CH)
            ax.imshow(cinfo['map'].numpy(), cmap='viridis',
                      interpolation='bilinear')
            wld = cinfo['loading']
            sign = '+' if wld >= 0 else '-'
            tc = 'darkgreen' if wld >= 0 else 'darkred'
            ax.set_title(f"ch{cinfo['ch_id']}\n{sign}{abs(wld):.3f}",
                         fontsize=7, fontweight='bold', color=tc)
            _set_border(ax, tc, 1.0)

    # ---- header ----
    status = "CORRECT" if correct else "WRONG"
    header = (f"Unified-CAM joint-solved Grad-CAM decomposition  --  "
              f"{model_name} @ {target_layer}  --  "
              f"true: {results['true_class'][:36]}  |  "
              f"pred: {results['pred_class'][:36]}  [{status}]")
    if n_available is not None:
        header += f"  (img {img_idx + 1}/{n_available}, offset={offset})"
    fig.text(0.5, 0.975, header, ha='center', va='top',
             fontsize=13, fontweight='bold')

    # corner + seed-invariance sub-header
    seed_ok = results['seed_invariant']
    sub = (f"corner: {results['corner']}    "
           f"eigengap={results['eigengap']:.2e}    "
           f"subspace agreement={results['subspace_agreement']:.4f} "
           f"(restarts={results['n_restarts']})    "
           f"seed-stable: {'YES' if seed_ok else 'NO'}")
    fig.text(0.5, 0.945, sub, ha='center', va='top', fontsize=10,
             family='monospace',
             bbox=dict(boxstyle='round',
                       facecolor='honeydew' if seed_ok else 'mistyrose',
                       alpha=0.9))

    # ---- legend ----
    if additive:
        legend = [
            f"Grad-CAM = ReLU( bias + sum_d beta_d * v_d ).  Center: input | "
            f"Grad-CAM | rank-{recon_D} reconstruction (exact additive pair, "
            f"cos={recon_cos:.3f}, ReLU keeps "
            f"{100*results['relu_keep_frac']:.0f}% of pre-ReLU mass).",
            f"Ring 1: top-{K} components by |beta_d| (of "
            f"{results['num_components_total']}); signed contribution "
            f"beta_d*z_d (blue +, red -). Basis is SOLVED at the corner above "
            f"-- not deterministic by construction; the badge reports whether "
            f"it reproduced across restarts.",
            f"Ring 2: top-{N} channel loadings v_(d,c) per solved component "
            f"(green = positive, red = negative).",
        ]
    else:
        legend = [
            f"KERNEL corner (phi={results['phi']}): the exact additive "
            f"identity L~ = bias + sum_d beta_d z_d does NOT hold (paper "
            f"section 5).",
            f"Rings 1 and 2 require the additive split and are therefore "
            f"omitted. Only the center triangle is shown; the reconstruction "
            f"is the rank-D projector applied in feature space.",
            results.get('scope_caveat', ''),
        ]
    fig.text(0.5, 0.03, '\n'.join(legend), ha='center', va='bottom',
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ==========================================================================
# Bank builder  (the images the basis is solved on)
# ==========================================================================

def build_bank(extractor: UnifiedHierExtractor, test_meta: Path,
               bank_classes: List[int], per_class: int
               ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    raw_list, alpha_list, index = [], [], []
    for cid in bank_classes:
        imgs = load_class_images(test_meta, cid, offset=0,
                                 num_images=per_class)
        for (img, idx) in imgs:
            A_raw, alpha, _ = extractor.acts_and_alpha(img)
            raw_list.append(A_raw.cpu())
            alpha_list.append(alpha.cpu())
            index.append((cid, idx))
    acts_raw = torch.cat(raw_list, 0)
    acts_norm = normalize_acts(acts_raw)
    alpha_bank = torch.stack(alpha_list, 0)
    print(f"  Bank: {acts_norm.shape[0]} images "
          f"({len(bank_classes)} class(es) x {per_class})")
    return acts_norm, alpha_bank, index


# ==========================================================================
# Main
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Hierarchical radial visualization of the joint-solved '
                    'unified CAM decomposition (one image).')
    ap.add_argument('--class_id', type=int, required=True)
    ap.add_argument('--offset', type=int, default=0,
                    help='Which cached image of the class is the QUERY.')
    ap.add_argument('--model', type=str, default='resnet50',
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--target_layer', type=str, default=None)
    # ---- which corner of the parameter cube to solve ----
    ap.add_argument('--corner', type=str, default='dcam',
                    choices=list(CORNER_PRESETS.keys()),
                    help='Corner preset. dcam reproduces the frozen-PCA '
                         'figure with a solved basis.')
    ap.add_argument('--solver', type=str, default='block',
                    choices=['block', 'joint', 'both'])
    ap.add_argument('--D', type=int, default=100, help='Subspace rank.')
    # ---- individual knob overrides ----
    ap.add_argument('--phi', type=str, default=None, choices=['id', 'rbf'])
    ap.add_argument('--locality', type=str, default=None,
                    choices=['global', 'per_image', 'bandwidth'])
    ap.add_argument('--lam', type=float, default=None)
    ap.add_argument('--bandwidth', type=float, default=None)
    ap.add_argument('--omega', type=str, default=None,
                    choices=['free', 'gradcam'])
    ap.add_argument('--rff_dim', type=int, default=1024)
    ap.add_argument('--rff_gamma', type=float, default=1.0)
    ap.add_argument('--n_restarts', type=int, default=1,
                    help='>1 runs the seed-invariance check and annotates the '
                         'figure with the cross-restart subspace agreement.')
    ap.add_argument('--max_iter', type=int, default=25)
    ap.add_argument('--lr', type=float, default=0.05)
    # ---- bank the basis is solved on ----
    ap.add_argument('--bank_classes', type=int, nargs='+', default=None,
                    help='Classes whose cached images form the basis bank. '
                         'Default: just --class_id.')
    ap.add_argument('--bank_per_class', type=int, default=12)
    # ---- figure knobs (verbatim from hier_visualize_pca.py) ----
    ap.add_argument('--ring_components', type=int, default=16,
                    help='How many top-|beta| components in ring 1.')
    ap.add_argument('--top_channels', type=int, default=4,
                    help='Top channel loadings per component in ring 2.')
    ap.add_argument('--recon_D', type=int, default=50,
                    help='Rank used for the center reconstruction panel.')
    ap.add_argument('--test_data_dir', type=str,
                    default='/data/imagenet1k_sampletest')
    ap.add_argument('--output_dir', type=str,
                    default='hier_unified_visualizations')
    ap.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cuda', 'cpu'])
    args = ap.parse_args()

    if not (0 <= args.class_id < 1000):
        raise ValueError(f"class_id must be in [0, 999], got {args.class_id}")
    if args.target_layer is None:
        args.target_layer = MODEL_CONFIGS[args.model]['default_target_layer']
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- build the SolveConfig ----
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
    print("Hierarchical Unified-CAM Grad-CAM Decomposition (joint-solved)")
    cname = list(IMAGENET2012_CLASSES.values())[args.class_id]
    print(f"  Class:      {args.class_id} ({cname[:60]})")
    print(f"  Model:      {args.model} @ {args.target_layer}")
    print(f"  Corner:     {args.corner}  solver={args.solver}")
    print(f"  Knobs:      phi={cfg.phi}  w={cfg.locality}  lambda={cfg.lam}  "
          f"omega={cfg.omega}  D={cfg.D}")
    print(f"  ring_comp:  {args.ring_components}  top_ch: {args.top_channels}  "
          f"recon_D: {args.recon_D}")
    print("=" * 80)

    extractor = UnifiedHierExtractor(
        args.model, args.target_layer,
        device='cuda' if torch.cuda.is_available() else 'cpu')

    test_meta = Path(args.test_data_dir) / "test_metadata.pkl"
    bank_classes = args.bank_classes or [args.class_id]

    # ---- build the bank the basis is solved on ----
    acts_norm, alpha_bank, index = build_bank(
        extractor, test_meta, bank_classes, args.bank_per_class)

    # ---- pick the query image and locate it in the bank ----
    query_img, n_avail, query_local_idx = load_image_for_class(
        test_meta, args.class_id, offset=args.offset)
    print(f"  Picked image {query_local_idx + 1}/{n_avail} "
          f"(offset {args.offset} -> {query_local_idx})")

    A_query_raw, alpha_query, pred_label = extractor.acts_and_alpha(query_img)
    A_query_norm = normalize_acts(A_query_raw.cpu())

    try:
        query_idx = index.index((args.class_id, query_local_idx))
    except ValueError:
        # query not in the bank -> append it so per_image/bandwidth w can
        # centre on it.
        acts_norm = torch.cat([acts_norm, A_query_norm], 0)
        alpha_bank = torch.cat([alpha_bank, alpha_query.cpu().unsqueeze(0)], 0)
        index.append((args.class_id, query_local_idx))
        query_idx = len(index) - 1
        print(f"  (query appended to bank as image {query_idx})")

    # ---- SOLVE the unified objective ----
    print(f"\nSolving (solver={args.solver}, n_restarts={cfg.n_restarts})...")
    result = solve(acts_norm, alpha_bank, alpha_query, query_idx,
                   cfg, solver=args.solver)
    diag = result.diagnostics
    print(f"  corner={diag['corner']}  J_final={diag['J_final']:.4e}  "
          f"eigengap={diag['eigengap']:.3e}  "
          f"seed_invariant={diag['seed_invariant']}")
    if not diag['additive_exact']:
        print(f"  SCOPE CAVEAT: {diag['scope_caveat']}")

    # ---- assemble the radial payload ----
    payload = build_ring_payload(
        result, A_query_norm, alpha_query, query_img,
        label=args.class_id, pred_label=pred_label,
        ring_components=args.ring_components, top_channels=args.top_channels,
        recon_D=args.recon_D)

    save_path = out / (f"hierunified_class{args.class_id}_"
                       f"img{query_local_idx}_{args.model}_{args.corner}.png")
    build_radial_figure(payload, args.model, args.target_layer,
                        str(save_path), offset=args.offset,
                        img_idx=query_local_idx, n_available=n_avail)
    print("=" * 80 + "\nDone.\n" + "=" * 80)


if __name__ == "__main__":
    main()