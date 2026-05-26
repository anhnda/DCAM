"""
df_hier_visualization.py
========================
Hierarchical radial visualization of the CALIBRATION-FREE (data-free)
Grad-CAM decomposition for a single input image.

This is the data-free counterpart of hier_visualize_pca.py. The radial layout
is identical -- a center triangle plus two component/channel rings -- but the
basis {v_d} and offset mu are NOT estimated from a corpus. They are read from
the network weights by df_decomposition.py:

    Grad-CAM = ReLU( bias + sum_d beta_d(x) * z_d(x) )            (Theorem 1)
      z_d(x)  = <A(x) - mu, v_d>          component spatial map   [H, W]
      beta_d  = <alpha, v_d>              scalar weight (class enters HERE)
      v_d     = top-D left singular vectors of the BN-folded operator W
                (basis='kernel'/'bn'), or eigvecs of a BN-distilled Sigma~
                (basis='distill').  -- all corpus-free, Definition 1 / Sec. 5

Key differences vs hier_visualize_pca.py
----------------------------------------
  * Input basis is a DataFreeReconstructor (df_decomposition.py), built once
    from weights -- no test_metadata corpus is ever read for the basis.
  * The center pair still reports spatial cosine of the rank-D reconstruction
    vs the true Grad-CAM. With basis='kernel' this gap MEASURES the input-patch
    anisotropy (Proposition 1): a non-zero gap is exactly "what the corpus
    would have bought" -- orientation, never completeness.
  * A full-rank completeness check (D = C) is reported: with an orthonormal
    weight-derived basis the residual must be at machine precision (Theorem 1).
  * Header labels the basis kind and the target block kind (projection vs
    identity shortcut, Remark 1).

Layout (verbatim geometry from hier_visualize_pca.py)
-----------------------------------------------------
  CENTER triangle : input image | true Grad-CAM | rank-D reconstruction.
  RING 1          : top-|beta_d| components, signed contribution beta_d * z_d.
  RING 2          : per-component channel loadings v_{d,c} (the channels that
                    COMPOSE each weight-derived direction).

Prerequisite: a basis .pkl built by df_decomposition.py.

Image source -- two mutually exclusive modes
---------------------------------------------
  (a) --image PATH            explain an arbitrary image file on disk.
  (b) --class_id ID [--offset N]
                              pull a cached ImageNet test image for that
                              class from test_metadata.pkl, exactly as
                              hier_visualize_pca.py does. Useful for
                              reproducing the paper's per-class panels.

Usage
-----
  # (a) arbitrary image file
  python df_hier_visualization.py --image cat.jpg \\
      --df_basis df_basis_resnet50_layer3_kernel_D200.pkl

  # (b) cached ImageNet test image, by class id + offset
  python df_hier_visualization.py --class_id 281 --offset 3 \\
      --df_basis df_basis_resnet50_layer3_distill_D200.pkl \\
      --ring_components 10 --top_channels 3 --recon_D 50
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
from PIL import Image
from torchvision import transforms

sys.path.append('.')
from src.gradcam import GradCAM
from full_classes import IMAGENET2012_CLASSES
# DataFreeReconstructor must be importable for joblib to unpickle the basis.
from df_decomposition import (DataFreeReconstructor, MODEL_CONFIGS,  # noqa: F401
                              build_from_weights)


# ======================================================================
# Image loading from the cached test corpus
# (verbatim contract from hier_visualize_pca.py's load_image_for_class)
# ======================================================================

def load_image_for_class(test_metadata_path: Path, class_id: int,
                          offset: int = 0) -> Tuple[Image.Image, int, int]:
    """Pull a cached ImageNet test image for `class_id` from test_metadata.pkl.

    Note: this reads a *test* corpus only to choose an image to EXPLAIN; the
    decomposition basis remains entirely calibration-free. Inference still
    needs the single image -- that is not calibration (Observation 1).
    """
    if not test_metadata_path.exists():
        raise FileNotFoundError(f"Test metadata not found at "
                                f"{test_metadata_path}.")
    metadata = joblib.load(test_metadata_path)
    samples = metadata['samples']
    class_samples = [(b, lbl) for (b, lbl) in samples if lbl == class_id]
    n = len(class_samples)
    if n == 0:
        raise ValueError(f"No cached test samples for class_id={class_id}")
    idx = offset % n
    if offset != idx:
        print(f"  Note: offset {offset} wrapped to {idx} "
              f"({n} cached images).")
    image_bytes, label = class_samples[idx]
    assert label == class_id
    return Image.open(io.BytesIO(image_bytes)).convert('RGB'), n, idx


# ======================================================================
# Extractor: backbone + Grad-CAM + a weight-derived (data-free) basis
# ======================================================================

class DFHierExtractor:
    """Runs Grad-CAM and decomposes it with a calibration-free basis."""

    def __init__(self, df_basis: DataFreeReconstructor, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available()
                                   else 'cpu')
        self.recon = df_basis.to(self.device).eval()
        self.model_name = self.recon.model_name
        self.target_layer_name = self.recon.target_layer
        self.basis_kind = self.recon.basis_kind
        self.block_kind = self.recon.block_kind

        self.mu = self.recon.pca_mu.detach().to(self.device)        # [C]
        self.V_full = self.recon.pca_V.detach().to(self.device)     # [C, D]
        self.D_built = self.V_full.shape[1]
        self.C = self.mu.shape[0]
        print(f"Loaded data-free basis: {self.recon}")
        print(f"  basis='{self.basis_kind}'  C={self.C}  D_built={self.D_built}"
              f"  (weight-derived, corpus-free, seed-stable)")

        self.backbone = MODEL_CONFIGS[self.model_name]['model_fn']() \
            .to(self.device).eval()
        self.target_layer = self._get_target_layer()
        self._detect_dims()
        print(f"  Backbone: {self.model_name} @ {self.target_layer_name} "
              f"({self.num_channels}ch, {self.spatial_size}x"
              f"{self.spatial_size})")
        if self.num_channels != self.C:
            raise ValueError(f"Basis C={self.C} != backbone output channels "
                             f"C={self.num_channels}. The df_basis was built "
                             f"for a different layer/model.")

        self.layer_activations = None
        self.target_layer.register_forward_hook(self._save_act)
        self.gradcam = GradCAM(self.backbone, self.target_layer)
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    # -- backbone plumbing (mirrors hier_visualize_pca.py) --------------

    def _get_target_layer(self):
        if self.model_name in ('resnet50', 'resnet18'):
            return {'layer1': self.backbone.layer1,
                    'layer2': self.backbone.layer2,
                    'layer3': self.backbone.layer3,
                    'layer4': self.backbone.layer4}[self.target_layer_name]
        idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
        return self.backbone.features[idx]

    def _detect_dims(self):
        with torch.no_grad():
            d = torch.randn(1, 3, 224, 224).to(self.device)
            if self.model_name in ('resnet50', 'resnet18'):
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
        """Per-channel 99th-percentile normalization (verbatim contract from
        hier_visualize_pca.py so the two pipelines are directly comparable)."""
        normalized = acts.clone()
        for c in range(acts.shape[1]):
            ch = acts[0, c]
            if ch.abs().sum() < 1e-8:
                continue
            nz = ch[ch > 1e-8]
            if len(nz) > 0:
                scale = torch.quantile(nz, 0.99)
                if scale > 1e-8:
                    normalized[0, c] = torch.clamp(ch, 0.0, scale) \
                        / (scale + 1e-8)
        return normalized

    # -- the decomposition ---------------------------------------------

    def extract(self, image: Image.Image, label: int,
                ring_components: int = 10, top_channels: int = 3,
                recon_D: int = 50) -> Dict:
        x = self.transform(image).unsqueeze(0).to(self.device)
        weights, _, pred_label = self.gradcam.forward(x, class_idx=None,
                                                      verbose=False)
        alpha = weights.view(-1).to(self.device)                  # [C]

        with torch.no_grad():
            A_raw = self.layer_activations.clone()
            A_norm = self._normalize(A_raw)
            _, C, H, W = A_norm.shape

            # true (pre-ReLU) Grad-CAM map and its ReLU
            L_tilde = (alpha.view(-1, 1, 1) * A_norm[0]).sum(dim=0)
            gradcam_true = F.relu(L_tilde)

            # ---- calibration-free additive decomposition (Theorem 1) ----
            #   z_d  = <A - mu, v_d>     beta_d = <alpha, v_d>   b = <alpha, mu>
            cells = A_norm[0].permute(1, 2, 0).reshape(-1, C)     # [HW, C]
            centered = cells - self.mu
            z = (centered @ self.V_full).T.reshape(self.D_built, H, W)
            beta = (alpha @ self.V_full)                          # [D_built]
            bias = float((alpha @ self.mu).item())
            comp = beta.view(-1, 1, 1) * z                        # [D,H,W]

            # rank-D reconstruction for the center pair
            Dr = min(recon_D, self.D_built)
            recon = F.relu(bias + comp[:Dr].sum(dim=0))           # [H,W]
            af, bf = recon.flatten(), gradcam_true.flatten()
            recon_cos = float((af @ bf) / (af.norm().clamp(min=1e-8)
                                           * bf.norm().clamp(min=1e-8)))

            # ---- full-rank completeness check (Theorem 1) ----
            # with an orthonormal weight-derived basis, reconstructing the
            # PRE-ReLU map at D = D_built must match L_tilde to machine
            # precision IF the basis spans the channel space (D_built == C).
            L_recon_full = bias + comp.sum(dim=0)                  # [H,W]
            complete_resid = float((L_recon_full - L_tilde).abs().max())
            spans_full = (self.D_built == C)

            # ReLU clipping fraction (how much pre-ReLU mass survives ReLU)
            pos = L_tilde.clamp(min=0).sum()
            tot = L_tilde.abs().sum().clamp(min=1e-8)
            relu_keep_frac = float((pos / tot).item())

            # ---- beta-energy diagnostics ----
            # beta_d = <alpha, v_d>. If every beta is tiny it is either because
            # alpha itself is small in norm (harmless: recon cosine is scale-
            # invariant) or because the basis is misaligned with alpha so the
            # energy is spread thinly over many components (Proposition 1: this
            # is the input-patch anisotropy the corpus would correct).
            alpha_norm = float(alpha.norm().item())
            beta_norm = float(beta.norm().item())
            beta_sq = beta ** 2
            beta_sq_total = float(beta_sq.sum().clamp(min=1e-12).item())
            K_diag = min(ring_components, self.D_built)
            sorted_sq, _ = torch.sort(beta_sq, descending=True)
            topK_energy_frac = float(
                (sorted_sq[:K_diag].sum().item()) / beta_sq_total)
            # top-1 share: distinguishes "first component dominates, rest
            # flat" from "energy genuinely concentrated in a few".
            top1_share = float(sorted_sq[0].item() / beta_sq_total)
            # participation ratio: effective number of components carrying
            # alpha's energy. PR ~ 1 => concentrated; PR ~ D => spread thin.
            participation_ratio = float(
                ((beta_sq.sum() ** 2)
                 / (beta_sq ** 2).sum().clamp(min=1e-12)).item())
            # concentration ratio: observed top-K energy vs what a RANDOM
            # (isotropic) basis would capture (~ K/D). CR = 1.0 means the
            # basis is no better than random for this image; higher is
            # better. NB: CR alone is misleading when D is large (K/D tiny),
            # so the verdict below also gates on the absolute topK fraction.
            random_baseline = K_diag / self.D_built
            concentration_ratio = topK_energy_frac / max(random_baseline,
                                                         1e-12)

            # ---- Ring 1: top components by |beta_d| ----
            K = min(ring_components, self.D_built)
            order = torch.argsort(beta.abs(), descending=True)[:K]

            comp_list: List[Dict] = []
            for d_t in order:
                d = int(d_t.item())
                b = float(beta[d].item())
                cmap = comp[d].cpu()                              # signed map
                v_d = self.V_full[:, d]                           # [C] loadings
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

        true_class = list(IMAGENET2012_CLASSES.values())[label] \
            if 0 <= label < 1000 else f"class_{label}"
        pred_class = list(IMAGENET2012_CLASSES.values())[pred_label]
        return {
            'image': image, 'label': label, 'pred_label': pred_label,
            'true_class': true_class, 'pred_class': pred_class,
            'correct': pred_label == label,
            'gradcam_true': gradcam_true.cpu(),
            'recon_map': recon.cpu(), 'recon_cos': recon_cos, 'recon_D': Dr,
            'relu_keep_frac': relu_keep_frac,
            'complete_resid': complete_resid, 'spans_full': spans_full,
            'alpha_norm': alpha_norm, 'beta_norm': beta_norm,
            'topK_energy_frac': topK_energy_frac,
            'top1_share': top1_share,
            'concentration_ratio': concentration_ratio,
            'random_baseline': random_baseline,
            'participation_ratio': participation_ratio,
            'components': comp_list,
            'num_components_total': self.D_built,
            'basis_kind': self.basis_kind, 'block_kind': self.block_kind,
        }


# ======================================================================
# beta-energy verdict  (single source of truth, used by console + figure)
# ======================================================================

def grade_beta_energy(results: Dict) -> Dict:
    """Three-tier verdict on whether the basis is well-aligned with alpha.

    Gates on BOTH the absolute top-K energy fraction (tkf) and the
    concentration ratio vs a random basis (CR). CR alone is misleading: with
    D=200 a random basis captures only K/D ~ 5% of the energy, so even a
    mediocre basis scores a large CR. tkf is the honest absolute gate.

      CONCENTRATED : CR >= 8  and tkf >= 0.80
                     basis well-aligned; the corpus would buy little here.
      PARTIAL      : CR >= 3  and tkf >= 0.45
                     mild spread; some Proposition-1 anisotropy -- worth
                     comparing the 'bn' / 'distill' bases.
      SPREAD       : otherwise
                     basis misaligned with alpha; energy spread thin over
                     many components -- this is exactly the input-patch
                     anisotropy the corpus corrects (Proposition 1).

    Returns a dict with the tier, a one-line message, and a colour.
    """
    tkf = results['topK_energy_frac']
    cr = results['concentration_ratio']
    s1 = results['top1_share']
    pr = results['participation_ratio']
    basis = results['basis_kind']

    if cr >= 8.0 and tkf >= 0.80:
        tier = 'CONCENTRATED'
        color = 'darkgreen'
        verdict = (f"energy concentrated (basis well-aligned with alpha); "
                   f"the corpus would buy little for this image.")
    elif cr >= 3.0 and tkf >= 0.45:
        tier = 'PARTIAL'
        color = 'darkorange'
        verdict = (f"mild spread -- some Proposition-1 anisotropy; "
                   f"compare the 'bn' / 'distill' bases to see beta "
                   f"concentrate.")
    else:
        tier = 'SPREAD'
        color = 'firebrick'
        verdict = (f"energy SPREAD over many components: the '{basis}' basis "
                   f"is misaligned with alpha (Proposition 1 anisotropy) -- "
                   f"try 'bn' / 'distill'.")

    msg = (f"||alpha||={results['alpha_norm']:.2e}  "
           f"||beta||={results['beta_norm']:.2e}  |  "
           f"top-K energy={100 * tkf:.0f}% (random~{100 * results['random_baseline']:.0f}%, "
           f"CR={cr:.0f}x)  top-1={100 * s1:.0f}%  PR={pr:.0f}")
    return {'tier': tier, 'color': color, 'verdict': verdict, 'msg': msg}


# ======================================================================
# Radial figure (geometry reused verbatim from hier_visualize_pca.py)
# ======================================================================

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
                        save_path: str, offset: int = 0,
                        img_idx: int = None, n_available: int = None):
    image = results['image']
    gradcam_map = results['gradcam_true']
    recon_map = results['recon_map']
    recon_cos = results['recon_cos']
    recon_D = results['recon_D']
    comps = results['components']
    correct = results['correct']
    basis_kind = results['basis_kind']
    block_kind = results['block_kind']

    K = len(comps)
    N = len(comps[0]['channels']) if K > 0 else 0

    fig = plt.figure(figsize=(20, 20), facecolor='white')
    cx, cy = 0.5, 0.5

    # center triangle
    tri_r, tri_size = 0.07, 0.11
    tri = {
        'image':   (cx + tri_r * np.cos(np.deg2rad(90)),
                    cy + tri_r * np.sin(np.deg2rad(90))),
        'gradcam': (cx + tri_r * np.cos(np.deg2rad(210)),
                    cy + tri_r * np.sin(np.deg2rad(210))),
        'recon':   (cx + tri_r * np.cos(np.deg2rad(330)),
                    cy + tri_r * np.sin(np.deg2rad(330))),
    }

    # ring 1 positions
    R1, F1 = 0.240, 0.086
    feat_pos = []
    for k in range(K):
        ang = 90.0 - (360.0 / K) * k
        feat_pos.append((cx + R1 * np.cos(np.deg2rad(ang)),
                         cy + R1 * np.sin(np.deg2rad(ang)), ang))

    # ring 2 positions. Geometry is sized for the default 10 components x 3
    # channels (30 panels). R2 is kept modest so the top panel clears the
    # 3-line status banner. If the user requests substantially more panels,
    # they may crowd -- pass smaller --ring_components / --top_channels.
    R2, CH = 0.370, 0.060
    arc_half = (360.0 / K) * 0.36 if K > 0 else 0.0
    ch_pos = []
    for (fx, fy, fang) in feat_pos:
        offs = [0.0] if N <= 1 else np.linspace(-arc_half, arc_half, N)
        ch_pos.append([(cx + R2 * np.cos(np.deg2rad(fang + o)),
                        cy + R2 * np.sin(np.deg2rad(fang + o)), fang + o)
                       for o in offs])

    # edges: channel -> component, coloured by loading sign
    for k, (fx, fy, _) in enumerate(feat_pos):
        ref = (max(abs(comps[k]['channels'][0]['loading']), 1e-8)
               if N else 1.0)
        for cpos, cinfo in zip(ch_pos[k], comps[k]['channels']):
            (xp, yp, _) = cpos
            w = cinfo['loading']
            color = 'forestgreen' if w >= 0 else 'firebrick'
            alpha = 0.5 + 0.5 * min(1.0, abs(w) / ref)
            _line(fig, (xp, yp), (fx, fy), color=color, lw=1.2,
                  linestyle='--', alpha=alpha, zorder=1)

    # triangle edges
    p_img, p_gc, p_re = tri['image'], tri['gradcam'], tri['recon']
    _line(fig, p_img, p_gc, color='gray', lw=1.0, alpha=0.5, zorder=1)
    _line(fig, p_gc, p_re, color='royalblue', lw=2.5, alpha=0.9, zorder=1)
    _line(fig, p_img, p_re, color='gray', lw=1.0, alpha=0.5, zorder=1)

    # center panels
    ax = _add_inset(fig, p_img[0], p_img[1], tri_size, tri_size)
    ax.imshow(image); ax.set_title("Input image", fontsize=9,
                                   fontweight='bold')

    ax = _add_inset(fig, p_gc[0], p_gc[1], tri_size, tri_size)
    ax.imshow(gradcam_map.numpy(), cmap='jet', interpolation='bilinear')
    ax.set_title("Grad-CAM", fontsize=9, fontweight='bold',
                 color='royalblue')
    _set_border(ax, 'royalblue', 2.0)

    ax = _add_inset(fig, p_re[0], p_re[1], tri_size, tri_size)
    ax.imshow(recon_map.numpy(), cmap='jet', interpolation='bilinear')
    rc = ('forestgreen' if recon_cos >= 0.8
          else 'darkorange' if recon_cos >= 0.5 else 'firebrick')
    ax.set_title(f"rank-{recon_D} recon\ncos = {recon_cos:.3f}",
                 fontsize=9, fontweight='bold', color=rc)
    _set_border(ax, rc, 2.0)

    # ring 1: signed component contributions beta_d * z_d.
    # Each panel is normalized to its OWN symmetric range: beta_d values are
    # often tiny and vary by orders of magnitude across components, so a single
    # shared scale would render all but the largest panel as blank white. The
    # raw beta_d is reported in the title; the map shows the SHAPE of the
    # contribution, the title carries its MAGNITUDE.
    for k, (fx, fy, _) in enumerate(feat_pos):
        c = comps[k]
        cmap_arr = c['contrib_map'].numpy()
        local_max = float(np.abs(cmap_arr).max())
        local_max = local_max if local_max > 1e-12 else 1.0
        ax = _add_inset(fig, fx, fy, F1, F1)
        ax.imshow(cmap_arr, cmap='bwr', vmin=-local_max, vmax=local_max,
                  interpolation='bilinear')
        ax.set_title(f"v{c['comp_id']}\nbeta={c['beta']:+.3f}",
                     fontsize=9, fontweight='bold')
        _set_border(ax, 'royalblue' if c['beta'] >= 0 else 'firebrick', 1.5)

    # ring 2: channel loadings of each weight-derived eigenvector
    for k, cl in enumerate(ch_pos):
        c = comps[k]
        for (cpos, cinfo) in zip(cl, c['channels']):
            (xp, yp, _) = cpos
            ax = _add_inset(fig, xp, yp, CH, CH)
            ax.imshow(cinfo['map'].numpy(), cmap='viridis',
                      interpolation='bilinear')
            w = cinfo['loading']
            sign = '+' if w >= 0 else '-'
            tc = 'darkgreen' if w >= 0 else 'darkred'
            ax.set_title(f"ch{cinfo['ch_id']}\n{sign}{abs(w):.3f}",
                         fontsize=8, fontweight='bold', color=tc)
            _set_border(ax, tc, 1.0)

    # header
    status = "CORRECT" if correct else "WRONG"
    basis_label = {'kernel': 'weight-derived (WW^T, white surrogate)',
                   'bn': 'weight-derived (BN-tilted)',
                   'distill': 'BN-distilled (ZeroQ-style synthesis)'
                   }.get(basis_kind, basis_kind)
    header = (f"Calibration-Free Grad-CAM decomposition  --  {model_name} @ "
              f"{target_layer}  [{basis_label}]\n"
              f"true: {results['true_class'][:40]}  |  "
              f"pred: {results['pred_class'][:40]}  [{status}]")
    if n_available is not None and img_idx is not None:
        header += f"  (img {img_idx + 1}/{n_available}, offset={offset})"
    fig.text(0.5, 0.992, header, ha='center', va='top',
             fontsize=13, fontweight='bold')

    # Status banner -- two lines in one box, placed in the clear strip between
    # the header and the topmost ring panel (which reaches y~0.915).
    # Line 1: completeness (the headline data-free claim, Theorem 1).
    # Line 2: beta-energy diagnostics with the 3-tier verdict (Proposition 1).
    cr = results['complete_resid']
    if results['spans_full']:
        comp_txt = (f"Completeness (D=C, Theorem 1): pre-ReLU residual "
                    f"max-err = {cr:.2e} -> exact (machine precision).")
    else:
        comp_txt = (f"Completeness: basis truncated "
                    f"(D={results['num_components_total']} < C); rank-D "
                    f"residual is the truncation tail sum_(d>D) beta_d z_d, "
                    f"NOT an identity error.")

    grade = grade_beta_energy(results)
    diag_txt = (f"beta diag [{grade['tier']}]: {grade['msg']}\n"
                f"-> {grade['verdict']}")

    fig.text(0.5, 0.937, comp_txt + '\n' + diag_txt, ha='center',
             va='center', fontsize=8.4, family='monospace', color='black',
             linespacing=1.5,
             bbox=dict(boxstyle='round', facecolor='white',
                       edgecolor=grade['color'], alpha=0.95, linewidth=1.8))

    # footer legend
    legend = [
        f"Grad-CAM = ReLU( bias + sum_d beta_d * v_d ).  Center: input | "
        f"Grad-CAM | rank-{recon_D} reconstruction (cos={recon_cos:.3f}, "
        f"ReLU keeps {100 * results['relu_keep_frac']:.0f}% of pre-ReLU mass).",
        f"Basis is CALIBRATION-FREE: {basis_label} -- read from network "
        f"weights, no corpus. Target block kind: {block_kind} "
        f"(Remark 1: identity-shortcut blocks carry upstream skip directions).",
        f"Ring 1: top-{K} components by |beta_d| (of "
        f"{results['num_components_total']}); signed contribution beta_d*z_d "
        f"(blue +, red -). Components are deterministic; only beta_d is "
        f"image/class-specific.",
        f"Ring 2: top-{N} channel loadings v_(d,c) per component "
        f"(green = positive, red = negative).  For basis='kernel' the "
        f"recon-cos gap measures input-patch anisotropy (Proposition 1).",
    ]
    fig.text(0.5, 0.03, '\n'.join(legend), ha='center', va='bottom',
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ======================================================================
# CLI
# ======================================================================

def _load_basis(args, device: str) -> DataFreeReconstructor:
    """Load a prebuilt basis .pkl, or build one on the fly from weights."""
    if args.df_basis is not None:
        path = Path(args.df_basis)
        if not path.exists():
            raise FileNotFoundError(f"df_basis not found: {path}")
        print(f"Loading data-free basis from {path}...")
        return joblib.load(path)
    # no prebuilt basis -> construct it now (still corpus-free)
    print(f"No --df_basis given; building '{args.basis}' basis from "
          f"{args.model} weights on the fly...")
    return build_from_weights(
        args.model, args.target_layer, args.D, basis=args.basis,
        distill_batches=args.distill_batches, distill_bs=args.distill_bs,
        distill_iters=args.distill_iters, device=device)


def main():
    ap = argparse.ArgumentParser(
        description='Hierarchical radial visualization of the calibration-'
                    'free (data-free) Grad-CAM decomposition for one image.')
    ap.add_argument('--image', type=str, default=None,
                    help='Path to an input image file (jpg/png). Mutually '
                         'exclusive with --class_id; one of the two is '
                         'required.')
    ap.add_argument('--class_id', type=int, default=None,
                    help='ImageNet class id. If given (instead of --image), '
                         'a cached test image for this class is loaded from '
                         'test_metadata.pkl, as in hier_visualize_pca.py. '
                         'Also used as the true label for the header.')
    ap.add_argument('--offset', type=int, default=0,
                    help='Which cached image to pick for --class_id '
                         '(wraps modulo the number available).')
    ap.add_argument('--test_data_dir', type=str,
                    default='/data/imagenet1k_sampletest',
                    help='Directory holding test_metadata.pkl (used only '
                         'in --class_id mode).')
    # basis: either load a prebuilt one, or build from weights
    ap.add_argument('--df_basis', type=str, default=None,
                    help='Prebuilt DataFreeReconstructor .pkl from '
                         'df_decomposition.py. If omitted, one is built '
                         'on the fly from --model/--target_layer/--basis.')
    ap.add_argument('--model', type=str, default='resnet50',
                    choices=list(MODEL_CONFIGS.keys()),
                    help='Used only when --df_basis is omitted.')
    ap.add_argument('--target_layer', type=str, default=None,
                    help='Used only when --df_basis is omitted.')
    ap.add_argument('--basis', type=str, default='kernel',
                    choices=['kernel', 'bn', 'distill'],
                    help='Used only when --df_basis is omitted.')
    ap.add_argument('--D', type=int, default=200,
                    help='Basis size when building on the fly.')
    ap.add_argument('--distill_batches', type=int, default=8)
    ap.add_argument('--distill_bs', type=int, default=16)
    ap.add_argument('--distill_iters', type=int, default=400)
    # visualization knobs (parity with hier_visualize_pca.py)
    ap.add_argument('--ring_components', type=int, default=10,
                    help='How many top-|beta| components in ring 1.')
    ap.add_argument('--top_channels', type=int, default=3,
                    help='Top channel loadings per component in ring 2.')
    ap.add_argument('--recon_D', type=int, default=50,
                    help='Rank used for the center reconstruction panel.')
    ap.add_argument('--output_dir', type=str,
                    default='df_hier_visualizations')
    args = ap.parse_args()

    if args.target_layer is None:
        args.target_layer = MODEL_CONFIGS[args.model]['default_target_layer']
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- resolve the image source: --image XOR --class_id ----
    if (args.image is None) == (args.class_id is None):
        raise ValueError("Provide exactly one of --image PATH or "
                         "--class_id ID (the latter loads a cached test "
                         "image; --offset selects which one).")

    if args.class_id is not None and not (0 <= args.class_id < 1000):
        raise ValueError(f"class_id must be in [0, 999], got {args.class_id}")

    if args.image is not None:
        # mode (a): arbitrary image file on disk
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Input image not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        stem = img_path.stem
        img_idx, n_avail = None, None
        source_desc = str(img_path)
    else:
        # mode (b): cached ImageNet test image, by class id + offset
        test_meta = Path(args.test_data_dir) / "test_metadata.pkl"
        image, n_avail, img_idx = load_image_for_class(
            test_meta, args.class_id, offset=args.offset)
        cname = list(IMAGENET2012_CLASSES.values())[args.class_id]
        stem = f"class{args.class_id}_img{img_idx}"
        source_desc = (f"cached test img {img_idx + 1}/{n_avail} for "
                       f"class {args.class_id} ({cname[:40]}), "
                       f"offset={args.offset}")

    print("=" * 80)
    print("Calibration-Free Hierarchical Grad-CAM Decomposition")
    print(f"  Image:      {source_desc}")
    print(f"  Device:     {device}")
    print("=" * 80)

    df_basis = _load_basis(args, device)
    extractor = DFHierExtractor(df_basis, device=device)

    # ---- resolve the true label for the header ----
    if args.class_id is not None:
        # both modes: class_id is the ground-truth label when supplied
        label = args.class_id
    else:
        # --image mode without a class id: label by the model's own
        # prediction (a quick forward pass, purely for the header).
        tmp = extractor.transform(image).unsqueeze(0).to(extractor.device)
        with torch.no_grad():
            logits = extractor.backbone(tmp)
        label = int(logits.argmax(dim=1).item())
        print(f"  No --class_id given; using predicted class {label} "
              f"as the label.")

    results = extractor.extract(
        image, label=label, ring_components=args.ring_components,
        top_channels=args.top_channels, recon_D=args.recon_D)

    print(f"  Prediction:        {results['pred_class'][:50]} "
          f"({'correct' if results['correct'] else 'wrong'})")
    print(f"  rank-{results['recon_D']} recon cosine: "
          f"{results['recon_cos']:.4f}")
    print(f"  full-rank completeness residual: "
          f"{results['complete_resid']:.3e} "
          f"({'exact' if results['spans_full'] else 'truncated basis'})")
    print(f"  --- beta diagnostics ---")
    print(f"  ||alpha|| = {results['alpha_norm']:.4e}   "
          f"||beta|| = {results['beta_norm']:.4e}")
    print(f"  top-{args.ring_components} components capture "
          f"{100 * results['topK_energy_frac']:.1f}% of ||beta||^2 energy "
          f"(random basis ~ {100 * results['random_baseline']:.1f}%)")
    print(f"  top-1 share = {100 * results['top1_share']:.1f}%   "
          f"concentration ratio = {results['concentration_ratio']:.1f}x   "
          f"participation ratio = {results['participation_ratio']:.1f} "
          f"(of {results['num_components_total']})")
    grade = grade_beta_energy(results)
    print(f"  verdict [{grade['tier']}]: {grade['verdict']}")

    save_path = out / (f"dfhier_{stem}_{extractor.model_name}_"
                       f"{extractor.basis_kind}.png")
    build_radial_figure(results, extractor.model_name,
                        extractor.target_layer_name, str(save_path),
                        offset=args.offset, img_idx=img_idx,
                        n_available=n_avail)
    print("=" * 80 + "\nDone.\n" + "=" * 80)


if __name__ == '__main__':
    main()