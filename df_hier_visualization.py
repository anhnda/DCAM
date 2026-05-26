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

PER-IMAGE PCA MODE (--pca)
--------------------------
When --pca is set, the basis and offset are estimated by PCA on the activation
cells of the SINGLE IMAGE being explained:

    cells   = A_norm[0].permute(1,2,0).reshape(HW, C)
    mu_img  = cells.mean(0)                       (per-image channel mean)
    Sigma~  = (cells - mu_img)^T (cells - mu_img) / HW
    v_d     = top-D eigenvectors of Sigma~

This is a fourth rung on the H_in ladder: kernel (white) -> bn (diagonal) ->
distill (full, synthesised) -> pca (full, the image itself). It is per-image,
not corpus-derived, so it is NOT "calibration-free" in the dataset-shared
sense -- but it requires nothing the explanation pass doesn't already need
(just one image), and gives the BEST possible basis FOR THIS IMAGE under
Proposition 1: the data-optimal basis of an image-conditional Sigma~ at zero
extra cost. The rank of Sigma~ is at most min(HW, C) - 1; D is silently capped.

Theorem 1 still holds (completeness is basis- and mean-agnostic). At full rank
(D = rank(Sigma~)) the reconstruction is exact in range(Sigma~); the truncation
deficit on the FULL pre-ReLU map is the projection error onto that range, which
is zero for any image's own activations because every cell lives in
range(cells - mu_img) by construction.

PER-LATENT CHANNEL SUPPORT (unchanged)
--------------------------------------
Each latent map z_d = <A - mu, v_d> is a sum over ALL C output channels. The
ring-2 panels only show the top-`top_channels` (default 3) loadings, which is
FAR too few to actually rebuild z_d -- you cannot see "the sum" from 3 panels.
So each ring-1 panel reports, in its title:

    Nz=<m> ch -> 95%     m = smallest number of channels (added most-significant
                             first by |v_{d,c}|) whose partial sum reconstructs
                             the full z_d MAP to >= 95% spatial cosine. This is
                             IMAGE-conditional (depends on A(x)).
    Nv=<m'>              m' = channels needed for the LOADING vector v_d to reach
                             95% of its OWN energy (||v_d^(m')|| / ||v_d|| >= 0.95).
                             This is INTRINSIC to the basis (image-independent).

Layout (verbatim geometry from hier_visualize_pca.py)
-----------------------------------------------------
  CENTER triangle : input image | true Grad-CAM | rank-D reconstruction.
  RING 1          : top-|beta_d| components, signed contribution beta_d * z_d.
  RING 2          : per-component channel loadings v_{d,c} (the channels that
                    COMPOSE each weight-derived direction).

Image source -- two mutually exclusive modes
---------------------------------------------
  (a) --image PATH            explain an arbitrary image file on disk.
  (b) --class_id ID [--offset N]
                              pull a cached ImageNet test image for that
                              class from test_metadata.pkl.

Usage
-----
  python df_hier_visualization.py --image cat.jpg \\
      --df_basis df_basis_resnet50_layer3_kernel_D200.pkl

  # per-image PCA (no .pkl needed; basis is built from the image's activations)
  python df_hier_visualization.py --image cat.jpg --pca \\
      --model resnet50 --target_layer layer3 --D 100

  python df_hier_visualization.py --class_id 281 --offset 3 \\
      --df_basis df_basis_resnet50_layer3_distill_D200.pkl \\
      --ring_components 10 --top_channels 3 --recon_D 50
"""

import argparse
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# Cosine threshold for the per-latent channel-support count.
COS_TARGET = 0.95


# ======================================================================
# Image loading from the cached test corpus
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
# Per-image PCA: build {v_d}, mu from the activations of one image
# ======================================================================

def build_image_pca_basis(cells: torch.Tensor, D: int
                          ) -> Tuple[torch.Tensor, torch.Tensor,
                                     torch.Tensor, int]:
    """PCA of ONE image's per-cell activations.

    cells : [HW, C] activation cells from the single image being explained.
    D     : requested number of components; capped at rank(Sigma~).

    Returns
    -------
    mu_img : [C]      per-image channel mean (the proper data offset for THIS
                     image's covariance; gives genuinely zero-mean cells).
    V      : [C, D']  top-D' eigenvectors of Sigma~ = (cells - mu)^T(cells - mu)/HW.
                     Orthonormal in R^C; columns sign-fixed (largest |entry| > 0)
                     for full determinism, matching Theorem 2's symmetry rule.
    sv     : [D']     corresponding eigenvalues (nonneg), descending.
    D_eff  : int      effective rank kept = min(D, rank(Sigma~)).

    Rank of Sigma~ is at most min(HW, C) - 1 (one DoF removed by centering).
    For ResNet-50 layer3 (HW=196, C=1024) that ceiling is 195. The truncation
    deficit (Proposition 1) collapses to zero within range(cells - mu) because
    every cell is exactly representable there by construction; outside that
    range the basis is silent, but Theorem 1's completeness still holds on
    the per-image span.
    """
    HW, C = cells.shape
    mu_img = cells.mean(dim=0)                                  # [C]
    Xc = cells - mu_img.view(1, -1)                              # [HW, C]

    # SVD of the centred cell matrix gives eigenvectors of Xc^T Xc (== Sigma~ up
    # to the 1/HW factor that does not affect eigenvectors). Use SVD rather
    # than forming the C x C product when HW < C: it's cheaper and numerically
    # cleaner. Returns V_h with rows = right singular vectors of Xc, which are
    # the eigenvectors of Sigma~ in C-space.
    # Xc = U S V^T  =>  Sigma~ ~ V S^2 V^T  =>  eigvecs = columns of V.
    _, s, Vh = torch.linalg.svd(Xc, full_matrices=False)         # s:[k], Vh:[k,C]
    # rank cap: numerically nonzero singular values
    tol = max(HW, C) * s.max().clamp(min=1e-12) * torch.finfo(s.dtype).eps
    rank = int((s > tol).sum().item())
    D_eff = min(D, rank)
    V = Vh[:D_eff].T.contiguous()                                # [C, D_eff]
    sv = (s[:D_eff] ** 2) / max(HW, 1)                           # eigvals of Sigma~

    # sign convention (matches df_decomposition._topD_left_singular)
    for d in range(D_eff):
        col = V[:, d]
        if col[torch.argmax(col.abs())] < 0:
            V[:, d] = -col

    return mu_img, V, sv, D_eff


# ======================================================================
# Extractor: backbone + Grad-CAM + a (data-free or per-image) basis
# ======================================================================

class DFHierExtractor:
    """Runs Grad-CAM and decomposes it with a calibration-free OR per-image
    basis. The per-image PCA path overrides `self.mu` / `self.V_full` inside
    `extract()` AFTER the forward pass has produced this image's activations.
    """

    def __init__(self, df_basis: DataFreeReconstructor, device='cuda',
                 image_pca: bool = False, image_pca_D: int = 200):
        self.device = torch.device(device if torch.cuda.is_available()
                                   else 'cpu')
        self.recon = df_basis.to(self.device).eval()
        self.model_name = self.recon.model_name
        self.target_layer_name = self.recon.target_layer
        self.image_pca = image_pca
        self.image_pca_D = image_pca_D
        # basis_kind/block_kind get reset per image when image_pca=True.
        self.basis_kind = ('pca_image' if image_pca else self.recon.basis_kind)
        self.block_kind = self.recon.block_kind

        # mu / V_full are only the placeholder when image_pca=True. They are
        # OVERWRITTEN inside extract() once we have the image's activations.
        self.mu = self.recon.pca_mu.detach().to(self.device)        # [C]
        self.V_full = self.recon.pca_V.detach().to(self.device)     # [C, D]
        self.D_built = self.V_full.shape[1]
        self.C = self.mu.shape[0]

        if self.image_pca:
            print(f"Per-image PCA mode: basis will be built from the image's "
                  f"own activations (D requested = {self.image_pca_D}).")
            print(f"  Placeholder df_basis (for backbone/layer/shapes only): "
                  f"{self.recon}")
        else:
            print(f"Loaded data-free basis: {self.recon}")
            print(f"  basis='{self.basis_kind}'  C={self.C}  "
                  f"D_built={self.D_built}  "
                  f"(weight-derived, corpus-free, seed-stable)")

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
            cells = A_norm[0].permute(1, 2, 0).reshape(-1, C)     # [HW, C]

            # =========================================================
            # PER-IMAGE PCA: build the basis NOW, from this image's cells.
            # Override mu / V_full / D_built for the rest of extract().
            # =========================================================
            if self.image_pca:
                mu_img, V_img, sv_img, D_eff = build_image_pca_basis(
                    cells, self.image_pca_D)
                self.mu = mu_img
                self.V_full = V_img
                self.D_built = D_eff
                print(f"  [pca] per-image basis built: D_eff={D_eff} "
                      f"(rank-capped from D_req={self.image_pca_D}; "
                      f"HW={H * W}, C={C}, ceiling={min(H * W, C) - 1})")
                print(f"  [pca] top-5 eigvals of Sigma~: "
                      f"{sv_img[:5].cpu().numpy().round(4).tolist()}")

            # true (pre-ReLU) Grad-CAM map and its ReLU
            L_tilde = (alpha.view(-1, 1, 1) * A_norm[0]).sum(dim=0)
            gradcam_true = F.relu(L_tilde)

            # ---- additive decomposition (Theorem 1) ----
            #   z_d  = <A - mu, v_d>     beta_d = <alpha, v_d>   b = <alpha, mu>
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
            # In image-PCA mode, range(V_full) is exactly the per-image span
            # of centred cells, so the residual is ~0 even when D_built < C.
            L_recon_full = bias + comp.sum(dim=0)                  # [H,W]
            complete_resid = float((L_recon_full - L_tilde).abs().max())
            spans_full = (self.D_built == C) or self.image_pca

            # ReLU clipping fraction
            pos = L_tilde.clamp(min=0).sum()
            tot = L_tilde.abs().sum().clamp(min=1e-8)
            relu_keep_frac = float((pos / tot).item())

            # ---- beta-energy diagnostics ----
            alpha_norm = float(alpha.norm().item())
            beta_norm = float(beta.norm().item())
            beta_sq = beta ** 2
            beta_sq_total = float(beta_sq.sum().clamp(min=1e-12).item())
            K_diag = min(ring_components, self.D_built)
            sorted_sq, _ = torch.sort(beta_sq, descending=True)
            topK_energy_frac = float(
                (sorted_sq[:K_diag].sum().item()) / beta_sq_total)
            top1_share = float(sorted_sq[0].item() / beta_sq_total)
            participation_ratio = float(
                ((beta_sq.sum() ** 2)
                 / (beta_sq ** 2).sum().clamp(min=1e-12)).item())
            random_baseline = K_diag / self.D_built
            concentration_ratio = topK_energy_frac / max(random_baseline,
                                                         1e-12)

            # ---- Ring 1: top components by |beta_d| ----
            K = min(ring_components, self.D_built)
            order = torch.argsort(beta.abs(), descending=True)[:K]

            nz_list, nv_list = [], []
            comp_list: List[Dict] = []
            for d_t in order:
                d = int(d_t.item())
                b = float(beta[d].item())
                cmap = comp[d].cpu()                              # signed map
                v_d = self.V_full[:, d]                           # [C] loadings

                # ============================================================
                # PER-LATENT CHANNEL SUPPORT (Nz, Nv)
                # ============================================================
                order_ch = torch.argsort(v_d.abs(), descending=True)  # [C]

                z_d_full = z[d].reshape(-1)                       # [HW]
                z_norm = z_d_full.norm().clamp(min=1e-12)
                terms = centered[:, order_ch] * v_d[order_ch].view(1, -1)
                z_cum = torch.cumsum(terms, dim=1)                # [HW,C]
                cos_cum = (z_cum * z_d_full.view(-1, 1)).sum(dim=0) \
                          / (z_cum.norm(dim=0).clamp(min=1e-12) * z_norm)
                hit_z = (cos_cum >= COS_TARGET).nonzero()
                n_ch_z = (int(hit_z[0].item()) + 1) if hit_z.numel() > 0 else C

                v_sq_sorted = (v_d[order_ch] ** 2)
                v_cum = torch.cumsum(v_sq_sorted, dim=0)
                v_total = v_cum[-1].clamp(min=1e-12)
                hit_v = ((v_cum / v_total) >= COS_TARGET).nonzero()
                n_ch_v = (int(hit_v[0].item()) + 1) if hit_v.numel() > 0 else C

                nz_list.append(n_ch_z)
                nv_list.append(n_ch_v)

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
                                  'contrib_map': cmap, 'channels': channels,
                                  'n_ch_z': n_ch_z, 'n_ch_v': n_ch_v})

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
            'C_channels': self.C,
            'nz_median': int(np.median(nz_list)) if nz_list else 0,
            'nv_median': int(np.median(nv_list)) if nv_list else 0,
            'nz_range': (min(nz_list), max(nz_list)) if nz_list else (0, 0),
            'nv_range': (min(nv_list), max(nv_list)) if nv_list else (0, 0),
        }


# ======================================================================
# beta-energy verdict
# ======================================================================

def grade_beta_energy(results: Dict) -> Dict:
    """Three-tier verdict on whether the basis is well-aligned with alpha."""
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
                   f"compare other bases (bn / distill / pca) to see beta "
                   f"concentrate.")
    else:
        tier = 'SPREAD'
        color = 'firebrick'
        verdict = (f"energy SPREAD over many components: the '{basis}' basis "
                   f"is misaligned with alpha (Proposition 1 anisotropy) -- "
                   f"try 'bn' / 'distill' / 'pca'.")
    msg = (f"||alpha||={results['alpha_norm']:.2e}  "
           f"||beta||={results['beta_norm']:.2e}  |  "
           f"top-K energy={100 * tkf:.0f}% (random~{100 * results['random_baseline']:.0f}%, "
           f"CR={cr:.0f}x)  top-1={100 * s1:.0f}%  PR={pr:.0f}")
    return {'tier': tier, 'color': color, 'verdict': verdict, 'msg': msg}


# ======================================================================
# Radial figure
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
    C_channels = results['C_channels']

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

    # ring 2 positions
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
    for k, (fx, fy, _) in enumerate(feat_pos):
        c = comps[k]
        cmap_arr = c['contrib_map'].numpy()
        local_max = float(np.abs(cmap_arr).max())
        local_max = local_max if local_max > 1e-12 else 1.0
        ax = _add_inset(fig, fx, fy, F1, F1)
        ax.imshow(cmap_arr, cmap='bwr', vmin=-local_max, vmax=local_max,
                  interpolation='bilinear')
        ax.set_title(f"v{c['comp_id']}  beta={c['beta']:+.3f}\n"
                     f"Nz={c['n_ch_z']}ch->95%  (Nv={c['n_ch_v']})",
                     fontsize=8, fontweight='bold')
        _set_border(ax, 'royalblue' if c['beta'] >= 0 else 'firebrick', 1.5)

    # ring 2: channel loadings of each eigenvector
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
                   'distill': 'BN-distilled (ZeroQ-style synthesis)',
                   'pca_image': 'per-image PCA (Sigma~ from THIS image\'s cells)',
                   }.get(basis_kind, basis_kind)
    header = (f"Calibration-Free Grad-CAM decomposition  --  {model_name} @ "
              f"{target_layer}  [{basis_label}]\n"
              f"true: {results['true_class'][:40]}  |  "
              f"pred: {results['pred_class'][:40]}  [{status}]")
    if n_available is not None and img_idx is not None:
        header += f"  (img {img_idx + 1}/{n_available}, offset={offset})"
    fig.text(0.5, 0.992, header, ha='center', va='top',
             fontsize=13, fontweight='bold')

    # status banner
    cr = results['complete_resid']
    if basis_kind == 'pca_image':
        comp_txt = (f"Completeness (per-image PCA): pre-ReLU residual "
                    f"max-err = {cr:.2e} -> exact on the per-image span "
                    f"(every cell lies in range(Sigma~) by construction).")
    elif results['spans_full']:
        comp_txt = (f"Completeness (D=C, Theorem 1): pre-ReLU residual "
                    f"max-err = {cr:.2e} -> exact (machine precision).")
    else:
        comp_txt = (f"Completeness: basis truncated "
                    f"(D={results['num_components_total']} < C); rank-D "
                    f"residual is the truncation tail sum_(d>D) beta_d z_d, "
                    f"NOT an identity error.")

    grade = grade_beta_energy(results)
    chan_txt = (f"per-latent channel support (of C={C_channels}): "
                f"Nz median={results['nz_median']}ch "
                f"range={results['nz_range']} (to rebuild z_d map >=95% cos)  |  "
                f"Nv median={results['nv_median']}ch "
                f"range={results['nv_range']} (loading energy)")
    diag_txt = (f"beta diag [{grade['tier']}]: {grade['msg']}\n"
                f"{chan_txt}\n"
                f"-> {grade['verdict']}")

    fig.text(0.5, 0.934, comp_txt + '\n' + diag_txt, ha='center',
             va='center', fontsize=8.2, family='monospace', color='black',
             linespacing=1.5,
             bbox=dict(boxstyle='round', facecolor='white',
                       edgecolor=grade['color'], alpha=0.95, linewidth=1.8))

    # footer legend
    if basis_kind == 'pca_image':
        basis_blurb = (f"Basis is PER-IMAGE PCA: {basis_label} -- NOT "
                       f"corpus-derived, but NOT dataset-shared either. The "
                       f"image-conditional optimum under Proposition 1.")
    else:
        basis_blurb = (f"Basis is CALIBRATION-FREE: {basis_label} -- read "
                       f"from network weights, no corpus. Target block kind: "
                       f"{block_kind} (Remark 1: identity-shortcut blocks "
                       f"carry upstream skip directions).")
    legend = [
        f"Grad-CAM = ReLU( bias + sum_d beta_d * v_d ).  Center: input | "
        f"Grad-CAM | rank-{recon_D} reconstruction (cos={recon_cos:.3f}, "
        f"ReLU keeps {100 * results['relu_keep_frac']:.0f}% of pre-ReLU mass).",
        basis_blurb,
        f"Ring 1: top-{K} components by |beta_d| (of "
        f"{results['num_components_total']}). Title Nz = # channels to rebuild "
        f"that z_d MAP to >=95% cos (the 'sum' the {N}-panel ring 2 can't show); "
        f"Nv = # channels for the loading's 95% energy.",
        f"Ring 2: top-{N} channel loadings v_(d,c) per component "
        f"(green = positive, red = negative). These {N} are a tiny subset of "
        f"the Nz channels that actually compose z_d.",
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
    """Load a prebuilt basis .pkl, or build one on the fly from weights.

    When --pca is set the returned basis is only a PLACEHOLDER for backbone /
    target-layer / shape info; the real per-image basis is constructed inside
    extract(). To keep that placeholder cheap we always build a small 'kernel'
    basis (no synthesis, no I/O) when --pca is on and no .pkl is given.
    """
    if args.df_basis is not None:
        path = Path(args.df_basis)
        if not path.exists():
            raise FileNotFoundError(f"df_basis not found: {path}")
        print(f"Loading basis (used as placeholder in --pca mode) from "
              f"{path}..." if args.pca
              else f"Loading data-free basis from {path}...")
        return joblib.load(path)
    if args.pca:
        print(f"--pca mode: building a small 'kernel' placeholder basis from "
              f"{args.model} weights (the real basis is per-image)...")
        return build_from_weights(args.model, args.target_layer,
                                  D=max(args.D, 2), basis='kernel',
                                  device=device)
    print(f"No --df_basis given; building '{args.basis}' basis from "
          f"{args.model} weights on the fly...")
    return build_from_weights(
        args.model, args.target_layer, args.D, basis=args.basis,
        distill_batches=args.distill_batches, distill_bs=args.distill_bs,
        distill_iters=args.distill_iters, device=device)


def main():
    ap = argparse.ArgumentParser(
        description='Hierarchical radial visualization of the calibration-'
                    'free (data-free) Grad-CAM decomposition for one image. '
                    'Pass --pca to estimate the basis from the explained '
                    'image\'s own activations instead of from weights.')
    ap.add_argument('--image', type=str, default=None,
                    help='Path to an input image file (jpg/png). Mutually '
                         'exclusive with --class_id.')
    ap.add_argument('--class_id', type=int, default=None,
                    help='ImageNet class id; loads a cached test image.')
    ap.add_argument('--offset', type=int, default=0,
                    help='Which cached image to pick for --class_id.')
    ap.add_argument('--test_data_dir', type=str,
                    default='/data/imagenet1k_sampletest',
                    help='Directory holding test_metadata.pkl.')
    ap.add_argument('--df_basis', type=str, default=None,
                    help='Prebuilt DataFreeReconstructor .pkl. Optional in '
                         '--pca mode (only used as a placeholder).')
    ap.add_argument('--model', type=str, default='resnet50',
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--target_layer', type=str, default=None)
    ap.add_argument('--basis', type=str, default='kernel',
                    choices=['kernel', 'bn', 'distill'],
                    help='Weight-derived basis to build if --df_basis is not '
                         'given. Ignored when --pca is set.')
    ap.add_argument('--pca', action='store_true',
                    help='Per-image PCA mode: estimate the basis {v_d} and '
                         'offset mu from the activation cells of the SINGLE '
                         'image being explained (no corpus, no synthesis). '
                         'D is rank-capped at min(HW, C) - 1.')
    ap.add_argument('--D', type=int, default=200,
                    help='Number of components requested. For --pca, '
                         'silently capped at rank(Sigma~).')
    ap.add_argument('--distill_batches', type=int, default=8)
    ap.add_argument('--distill_bs', type=int, default=16)
    ap.add_argument('--distill_iters', type=int, default=400)
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

    if (args.image is None) == (args.class_id is None):
        raise ValueError("Provide exactly one of --image PATH or "
                         "--class_id ID.")

    if args.class_id is not None and not (0 <= args.class_id < 1000):
        raise ValueError(f"class_id must be in [0, 999], got {args.class_id}")

    if args.image is not None:
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Input image not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        stem = img_path.stem
        img_idx, n_avail = None, None
        source_desc = str(img_path)
    else:
        test_meta = Path(args.test_data_dir) / "test_metadata.pkl"
        image, n_avail, img_idx = load_image_for_class(
            test_meta, args.class_id, offset=args.offset)
        cname = list(IMAGENET2012_CLASSES.values())[args.class_id]
        stem = f"class{args.class_id}_img{img_idx}"
        source_desc = (f"cached test img {img_idx + 1}/{n_avail} for "
                       f"class {args.class_id} ({cname[:40]}), "
                       f"offset={args.offset}")

    print("=" * 80)
    if args.pca:
        print("PER-IMAGE PCA Hierarchical Grad-CAM Decomposition")
    else:
        print("Calibration-Free Hierarchical Grad-CAM Decomposition")
    print(f"  Image:      {source_desc}")
    print(f"  Device:     {device}")
    print("=" * 80)

    df_basis = _load_basis(args, device)
    extractor = DFHierExtractor(df_basis, device=device,
                                image_pca=args.pca, image_pca_D=args.D)

    if args.class_id is not None:
        label = args.class_id
    else:
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
          f"({'exact / per-image span' if results['spans_full'] else 'truncated basis'})")
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
    print(f"  --- per-latent channel support (of C={results['C_channels']} "
          f"channels), shown components ---")
    print(f"  Nz (rebuild z_d map >=95% cos): median="
          f"{results['nz_median']}ch  range={results['nz_range']}")
    print(f"  Nv (loading 95% energy):        median="
          f"{results['nv_median']}ch  range={results['nv_range']}")
    for c in results['components']:
        print(f"    v{c['comp_id']:<4d} beta={c['beta']:+.4f}  "
              f"Nz={c['n_ch_z']:>4d}ch -> 95% cos   Nv={c['n_ch_v']:>4d}ch")
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