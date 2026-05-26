"""
df_decomposition.py
===================
Calibration-free (data-free) decomposition of Grad-CAM.

Implements the construction of "Calibration-Free Decomposition of Grad-CAM:
Global Bases from Network Weights". The data-PCA pipeline of
csae_pca_baseline.py / hier_visualize_pca.py estimates the basis {v_d} and
mean mu from a corpus of cached activations. This module removes the corpus:
the basis is read directly from the network weights.

The additive Grad-CAM identity (exact, Theorem 1) is

    L_tilde(x) = b(x) + sum_d beta_d(x) * z_d(x)
      z_d(x)   = <A(x) - mu, v_d>     (channel contraction)  -> spatial map [H,W]
      beta_d   = <alpha, v_d>                                -> scalar
      b(x)     = <alpha, mu>                                 -> scalar bias
    Grad-CAM(x) = ReLU(L_tilde(x))

What changes vs the data-PCA baseline is ONLY where {v_d}, mu come from:

  data-PCA baseline      v_d = eigvecs of  Sigma = E[(A-mu)(A-mu)^T]   (corpus)
                         mu  = empirical channel mean over corpus

  calibration-free       v_d = top-D LEFT singular vectors of W        (weights)
  (this module)              W = BatchNorm-folded effective operator
                                 producing the target layer's output
                         mu  = BatchNorm running mean (BN proxy)       (Sec. B)

Three weight-derived bases are provided (Definition 1 + Section 5):

  'kernel'   top-D left singular subspace of W            -> WW^T
             the white-input surrogate H_in = I  (Observation 2).
  'bn'       top-D eigenspace of  W diag(h_in) W^T         -> BN-tilted
             h_in = UPSTREAM BatchNorm running variances on the INPUT-patch
             axis (the BN feeding conv3 -- bn2 in a bottleneck). This is the
             diagonal proxy for H_in in Sigma = W H_in W^T (Definition 1).
  'distill'  top-D eigenspace of Sigma~ estimated from a BatchNorm-matched
             synthetic batch run through the true nonlinear forward pass
             (Section 5, ZeroQ-style). Zero real images.

  The three form a ladder on the assumed input covariance:
      kernel  : H_in = I              (white)
      bn      : H_in = diag(h_in)     (diagonal input anisotropy, FREE)
      distill : H_in = full           (off-diagonals too, via synthesis)

Theorem 2: 'kernel' and 'bn' are deterministic functions of the checkpoint --
seed-, ordering-, and corpus-invariant. 'distill' depends only on the synthesis
seed, never on real data.

This file is import-compatible with hier_visualize_pca.py's expectations: it
exposes a `DataFreeReconstructor` carrying `pca_mu` and `pca_V` attributes
(named for drop-in parity with PCAReconstructor), plus a from-weights builder.

Usage
-----
  # build the weight-derived basis and cache it (one-off, no corpus)
  python df_decomposition.py --model resnet50 --target_layer layer3 \\
      --basis kernel --D 200 --output df_basis_resnet50_layer3_kernel.pkl

  # BN-tilted variant (input-axis upstream-BN tilt; still zero data)
  python df_decomposition.py --model resnet50 --target_layer layer3 \\
      --basis bn --D 200 --output df_basis_resnet50_layer3_bn.pkl

  # BN-distilled variant (synthesises images, still zero real data)
  python df_decomposition.py --model resnet50 --target_layer layer3 \\
      --basis distill --D 200 --distill_batches 8 --distill_iters 500 \\
      --output df_basis_resnet50_layer3_distill.pkl

  # then visualize a single image with df_hier_visualization.py

CHANGELOG (vs prior revision)
-----------------------------
  * build_bn_basis now tilts the INPUT axis (columns of W) by the UPSTREAM
    BN variance, matching Definition 1 ("upstream BatchNorm running variances
    feeding the branch"). The prior revision tilted the OUTPUT axis by bn3's
    variance, then undid the scaling and re-QR'd -- a near-identity no-op that
    left the 'bn' basis indistinguishable from 'kernel'. Eigenvectors of the
    symmetric PSD  W diag(h_in) W^T  are already orthonormal, so no undo / QR
    is needed.
  * Added get_upstream_bn_var() to locate the BN feeding the folded operator.
"""

import argparse
import sys
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ----------------------------------------------------------------------
# Backbone configuration (mirrors hier_visualize_pca.py)
# ----------------------------------------------------------------------

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
}


# ======================================================================
# BatchNorm folding: conv (+ following BN) -> single linear operator
# ======================================================================

def fold_conv_bn(conv: nn.Conv2d,
                 bn: Optional[nn.BatchNorm2d]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold a Conv2d followed by a BatchNorm2d into one affine operator.

    At inference BN is the affine map  x -> gamma (x - mu) / sqrt(var + eps) + beta
    (Section B). Composing it with the preceding convolution gives a single
    linear operator on patches:

        W_eff[o, :] = scale_o * W_conv[o, :]          (scale_o = gamma_o / sqrt(var_o + eps))
        b_eff[o]    = scale_o * (b_conv[o] - mu_o) + beta_o

    Returns
    -------
    W_eff : [out_ch, in_ch * kh * kw]   row-major over the conv kernel
    b_eff : [out_ch]
    """
    Wc = conv.weight.detach().clone()                       # [O, I, kh, kw]
    O = Wc.shape[0]
    # IMPORTANT: the fallback bias must live on the SAME device/dtype as the
    # conv weight. ResNet conv layers have no bias (BN follows), so this
    # branch is the common case -- a CPU-default zeros() here would clash
    # with a CUDA backbone.
    bc = (conv.bias.detach().clone() if conv.bias is not None
          else torch.zeros(O, dtype=Wc.dtype, device=Wc.device))

    if bn is None:
        return Wc.reshape(O, -1), bc

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps
    scale = gamma / torch.sqrt(var + eps)                   # [O]

    W_eff = Wc * scale.view(O, 1, 1, 1)
    b_eff = scale * (bc - mean) + beta
    return W_eff.reshape(O, -1), b_eff


# ======================================================================
# Effective output operator W of the target layer  (Lemma 1 / Section B)
# ======================================================================

def get_effective_operator(backbone: nn.Module, model_name: str,
                            target_layer_name: str
                            ) -> Tuple[torch.Tensor, str, Dict]:
    """Read the BatchNorm-folded effective operator W that produces the
    target layer's output channels.

    For a ResNet bottleneck stage (`layerN`), the layer output is

        out = ReLU( F(u) + S(u) )

    where the *last* operation of the residual branch F is conv3 -> bn3, a
    linear expand map W_F (Lemma 1: F(u) in range(W_F)). The pre-ReLU output
    therefore lies in  C = range(W_F) + range(S). We take W as the folded
    conv3+bn3 of the LAST block in the stage: its left singular subspace is
    the freshly-injected part of C, and dominates the layer's channel
    representation in practice. (At an identity-shortcut block the carried
    skip directions are not local; that is the honest limitation removed by
    the 'distill' basis -- see Section 5 / Remark 1.)

    For VGG (`features[idx]`) the target layer is itself a conv; if it is
    immediately followed by a BN we fold it, otherwise W is the raw conv.

    Returns
    -------
    W            : [C_out, in_dim]   effective linear operator
    block_kind   : 'projection' | 'identity' | 'plain'  (Remark 1)
    info         : dict of diagnostic dimensions
    """
    info: Dict = {}

    if model_name in ('resnet50', 'resnet18'):
        stage = {'layer1': backbone.layer1, 'layer2': backbone.layer2,
                 'layer3': backbone.layer3, 'layer4': backbone.layer4
                 }[target_layer_name]
        last_block = stage[-1]

        # Bottleneck (resnet50): conv3+bn3 is the expand map W_F.
        # BasicBlock (resnet18): conv2+bn2 is the last residual conv.
        if hasattr(last_block, 'conv3'):
            conv_f, bn_f = last_block.conv3, last_block.bn3
        else:
            conv_f, bn_f = last_block.conv2, last_block.bn2

        W_F, _ = fold_conv_bn(conv_f, bn_f)                 # [C_out, in*1*1] usually

        # The last block of a stage has an identity shortcut (stage-entry
        # block carries the projection). Remark 1: identity => C not locally
        # capped. We still expose W_F as the readable injected part.
        downsample = getattr(last_block, 'downsample', None)
        block_kind = 'identity' if downsample is None else 'projection'

        info['stage_blocks'] = len(stage)
        info['residual_conv'] = ('conv3' if hasattr(last_block, 'conv3')
                                 else 'conv2')
        info['expand_kernel'] = tuple(conv_f.weight.shape)
        W = W_F

    elif model_name == 'vgg16':
        idx = int(target_layer_name.split('[')[1].rstrip(']'))
        conv = backbone.features[idx]
        if not isinstance(conv, nn.Conv2d):
            raise ValueError(f"features[{idx}] is {type(conv).__name__}, "
                             f"not a Conv2d.")
        bn = None
        if idx + 1 < len(backbone.features) and isinstance(
                backbone.features[idx + 1], nn.BatchNorm2d):
            bn = backbone.features[idx + 1]
        W, _ = fold_conv_bn(conv, bn)
        block_kind = 'plain'
        info['conv_kernel'] = tuple(conv.weight.shape)
    else:
        raise ValueError(f"Unsupported model '{model_name}'.")

    info['W_shape'] = tuple(W.shape)
    info['block_kind'] = block_kind
    return W, block_kind, info


# ======================================================================
# BatchNorm mean read-off  (offset mu)
# ======================================================================

def get_bn_stats(backbone: nn.Module, model_name: str,
                 target_layer_name: str, C_out: int
                 ) -> torch.Tensor:
    """Read the BatchNorm running MEAN that describes the target layer's
    OUTPUT channels -- used as the offset mu (Section B, a proxy for the
    post-activation channel mean). Theorem 1 holds for any mu; the proxy
    only repartitions mass between offset and components.

    (The OUTPUT-channel running VARIANCE is intentionally NOT returned here:
    the previous revision used it as the 'bn' tilt, on the wrong axis. The
    'bn' tilt now uses the UPSTREAM/input variance -- see get_upstream_bn_var.)
    """
    dev = next(backbone.parameters()).device

    if model_name in ('resnet50', 'resnet18'):
        stage = {'layer1': backbone.layer1, 'layer2': backbone.layer2,
                 'layer3': backbone.layer3, 'layer4': backbone.layer4
                 }[target_layer_name]
        last_block = stage[-1]
        bn_f = last_block.bn3 if hasattr(last_block, 'bn3') else last_block.bn2
        mu = bn_f.running_mean.detach().clone()
    elif model_name == 'vgg16':
        idx = int(target_layer_name.split('[')[1].rstrip(']'))
        if idx + 1 < len(backbone.features) and isinstance(
                backbone.features[idx + 1], nn.BatchNorm2d):
            mu = backbone.features[idx + 1].running_mean.detach().clone()
        else:
            mu = torch.zeros(C_out, device=dev)
    else:
        raise ValueError(f"Unsupported model '{model_name}'.")

    if mu.numel() != C_out:
        mu = torch.zeros(C_out, device=dev)
    return mu


def get_upstream_bn_var(backbone: nn.Module, model_name: str,
                        target_layer_name: str, in_dim: int) -> torch.Tensor:
    """Running variance of the BN feeding the folded operator's INPUT.

    This is Definition 1's "upstream BatchNorm running variances feeding the
    branch" -- the per-input-channel scale on the INPUT-patch axis, used as a
    diagonal proxy for H_in in  Sigma = W H_in W^T.

    ResNet-50 bottleneck:  conv1->bn1->conv2->bn2->conv3->bn3.
        The folded operator is conv3+bn3; conv3's input is bn2's output, so
        the upstream variance is last_block.bn2.running_var  (length C_in of
        conv3, i.e. the bottleneck width).
    ResNet-18 BasicBlock:  conv1->bn1->conv2->bn2.
        The folded operator is conv2+bn2; conv2's input is bn1's output, so
        the upstream variance is last_block.bn1.running_var.
    VGG / plain:  no reliable upstream BN on the conv input -> unit (white),
        i.e. this collapses gracefully back to the 'kernel' surrogate.

    The returned vector is broadcast to length `in_dim`. For a 1x1 conv3
    (the ResNet-50 expand) in_dim == C_in exactly. For a kxk conv (VGG,
    ResNet-18 conv2) in_dim == C_in * kh * kw and we tile the per-channel
    variance across the kh*kw spatial taps (the BN variance is per channel,
    shared across taps -- the honest diagonal proxy).
    """
    dev = next(backbone.parameters()).device

    if model_name in ('resnet50', 'resnet18'):
        stage = {'layer1': backbone.layer1, 'layer2': backbone.layer2,
                 'layer3': backbone.layer3, 'layer4': backbone.layer4
                 }[target_layer_name]
        last_block = stage[-1]
        if hasattr(last_block, 'conv3'):           # bottleneck: conv3 <- bn2
            h_in_ch = last_block.bn2.running_var.detach().clone()
        else:                                      # basicblock: conv2 <- bn1
            h_in_ch = last_block.bn1.running_var.detach().clone()
    else:
        # VGG plain conv: no folded-in upstream BN on the conv input.
        # White fallback => 'bn' reduces to 'kernel'.
        return torch.ones(in_dim, device=dev)

    C_in = h_in_ch.numel()
    if in_dim == C_in:
        return h_in_ch                              # 1x1 conv: exact
    if in_dim % C_in == 0:
        taps = in_dim // C_in                        # kxk conv: tile over taps
        return h_in_ch.repeat_interleave(taps)
    # shape mismatch we can't reconcile -> white fallback
    return torch.ones(in_dim, device=dev)


# ======================================================================
# Basis builders
# ======================================================================

def _topD_left_singular(M: torch.Tensor, D: int) -> Tuple[torch.Tensor,
                                                          torch.Tensor]:
    """Top-D left singular vectors of M and the corresponding singular values.

    Returns U_D : [rows(M), D]   (columns are orthonormal),  s_D : [D].
    Sign convention: each column is fixed so its largest-magnitude entry is
    positive -- making the basis fully deterministic (Theorem 2 symmetry).
    """
    # left singular vectors of M == eigenvectors of M M^T; SVD is numerically
    # cleaner and works for non-square / tall M.
    U, s, _ = torch.linalg.svd(M, full_matrices=False)
    D = min(D, U.shape[1])
    U_D = U[:, :D].contiguous()
    s_D = s[:D].contiguous()
    for d in range(D):
        col = U_D[:, d]
        if col[torch.argmax(col.abs())] < 0:
            U_D[:, d] = -col
    return U_D, s_D


def build_kernel_basis(W: torch.Tensor, D: int) -> Tuple[torch.Tensor,
                                                         torch.Tensor]:
    """White-input surrogate (Observation 2, H_in = I): basis = top-D left
    singular vectors of W, i.e. top-D eigenvectors of WW^T."""
    return _topD_left_singular(W, D)


def build_bn_basis(W: torch.Tensor, h_in: torch.Tensor,
                   D: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """BN-tilted basis (Definition 1, corrected): top-D eigenspace of
    W diag(h_in) W^T, with h_in the UPSTREAM BN running variances on the
    INPUT-patch axis -- the diagonal proxy for H_in in  Sigma = W H_in W^T.

    The left singular vectors of  M = W diag(sqrt h_in)  are exactly the
    eigenvectors of  M M^T = W diag(h_in) W^T, and are ALREADY orthonormal in
    R^{C_out}. So -- unlike the previous (output-axis) revision -- there is no
    undo-scaling step and no re-QR: tilting the input axis does not move the
    frame off the channel space, it only re-orients within it.

    Parameters
    ----------
    W     : [C_out, in_dim]   effective operator (columns index input patch).
    h_in  : [in_dim]          upstream BN variances on the input axis.
    """
    if h_in.shape[0] != W.shape[1]:
        raise ValueError(f"h_in length {h_in.shape[0]} != W input dim "
                         f"{W.shape[1]}; check get_upstream_bn_var.")
    s_in = torch.sqrt(h_in.clamp(min=1e-12)).to(W.device)   # [in_dim]
    M = W * s_in.view(1, -1)                                 # scale COLUMNS
    U_D, sv = _topD_left_singular(M, D)
    return U_D, sv


def build_distill_basis(backbone: nn.Module, model_name: str,
                        target_layer_name: str, C_out: int, D: int,
                        n_batches: int = 2, batch_size: int = 32,
                        iters: int = 400, lr: float = 0.1,
                        device: torch.device = torch.device('cpu'),
                        seed: int = 0,
                        bn_stride: int = 1,
                        use_amp: bool = True,
                        harvest_passes: int = 8,
                        harvest_bs: int = 64) -> Tuple[torch.Tensor,
                                                       torch.Tensor, Dict]:
    """ZeroQ-style BN-distilled basis, decoupled synthesis vs harvest.

    Key insight: synthesis is the expensive part (forward+backward+optimizer);
    cell harvest is just forwards. Past revisions tied them 1:1 -- every
    optimizer round produced exactly one harvest pass. That made the only
    knob for "more cells" (which is what Sigma~ actually needs) also a knob
    for "more synthesis" (which it doesn't).

    Now:
      * n_batches synthesis rounds produce n_batches distinct image sets
      * each set is harvested `harvest_passes` times with random crops + flips
        through the forward path (no backward, no optimizer) at `harvest_bs`
      * total cells per round = harvest_passes * harvest_bs * H * W
        e.g. 8 * 64 * 14 * 14 = 100k cells from ONE synthesis round
      * compare to old code's 16 * 14 * 14 = 3136 cells per round

    Sensible budgets:
      * fast, decent Sigma~ : n_batches=1, iters=300, harvest_passes=4
      * thorough            : n_batches=2, iters=400, harvest_passes=8

    Other speed changes carried over from the previous patch:
      * freeze backbone params (no autograd graph for ~25M weights)
      * AMP synthesis with fp32 BN-stat reduction
      * cosine LR
      * pre-allocated hook slots (no dict churn)
      * fp64 covariance accumulators
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # ---- BN match setup ----
    all_bns = [m for m in backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_layers = all_bns[::bn_stride] if bn_stride > 1 else all_bns
    n_bn = len(bn_layers)
    bn_mean_targets = [bn.running_mean.detach().to(device) for bn in bn_layers]
    bn_var_targets  = [bn.running_var.detach().to(device)  for bn in bn_layers]

    bn_feats_mean: list = [None] * n_bn
    bn_feats_var:  list = [None] * n_bn

    def make_hook(i):
        def hook(module, inp, out):
            xf = inp[0].float()
            bn_feats_mean[i] = xf.mean(dim=(0, 2, 3))
            bn_feats_var[i]  = xf.var(dim=(0, 2, 3), unbiased=False)
        return hook

    handles = [bn.register_forward_hook(make_hook(i))
               for i, bn in enumerate(bn_layers)]

    # ---- target-layer hook ----
    if model_name in ('resnet50', 'resnet18'):
        target_module = {'layer1': backbone.layer1, 'layer2': backbone.layer2,
                          'layer3': backbone.layer3, 'layer4': backbone.layer4
                          }[target_layer_name]
    else:
        idx = int(target_layer_name.split('[')[1].rstrip(']'))
        target_module = backbone.features[idx]

    collected = {'act': None}
    def tgt_hook(module, inp, out):
        collected['act'] = out.detach()
    tgt_handle = target_module.register_forward_hook(tgt_hook)

    cell_sum = torch.zeros(C_out, device=device, dtype=torch.float64)
    cell_cov = torch.zeros(C_out, C_out, device=device, dtype=torch.float64)
    n_cells = 0

    amp_enabled = use_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    import time
    t_synth_total = 0.0
    t_harvest_total = 0.0

    for batch in range(n_batches):
        # =============== SYNTHESIS ===============
        t0 = time.time()
        x = torch.randn(batch_size, 3, 224, 224, device=device,
                        requires_grad=True)
        opt = torch.optim.Adam([x], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)

        for it in range(iters):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                _ = backbone(x)
                loss = x.new_zeros((), dtype=torch.float32)
                for i in range(n_bn):
                    if bn_feats_mean[i] is None:
                        continue
                    loss = loss + F.mse_loss(bn_feats_mean[i], bn_mean_targets[i]) \
                                + F.mse_loss(bn_feats_var[i],  bn_var_targets[i])
                loss = loss + 1e-4 * (x.float() ** 2).mean()

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_synth = time.time() - t0
        t_synth_total += t_synth

        # =============== HARVEST ===============
        # Many forward passes through the synthesised images, with light
        # augmentation (flip + random crop within the 224 frame), to extract
        # many cells without paying for more optimisation.
        t0 = time.time()
        x_pool = x.detach()
        with torch.no_grad():
            for hp in range(harvest_passes):
                # sample harvest_bs images (with replacement if needed) and
                # apply hflip
                if harvest_bs <= batch_size:
                    idx_pool = torch.randperm(batch_size, device=device)[:harvest_bs]
                else:
                    idx_pool = torch.randint(0, batch_size, (harvest_bs,),
                                             device=device)
                xh = x_pool[idx_pool]
                if torch.rand(()) < 0.5:
                    xh = torch.flip(xh, dims=[3])
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    _ = backbone(xh)
                act = collected['act'].float()
                cells = act.permute(0, 2, 3, 1).reshape(-1, C_out).double()
                cell_sum += cells.sum(dim=0)
                cell_cov += cells.T @ cells
                n_cells += cells.shape[0]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_harvest = time.time() - t0
        t_harvest_total += t_harvest

        print(f"  [distill] batch {batch + 1}/{n_batches}  "
              f"synth={t_synth:.1f}s  harvest={t_harvest:.1f}s  "
              f"cells={n_cells}  final bn-loss={float(loss):.4f}")

    for h in handles:
        h.remove()
    tgt_handle.remove()

    print(f"  [distill] totals: synth={t_synth_total:.1f}s  "
          f"harvest={t_harvest_total:.1f}s  cells={n_cells}")

    mu_distill = (cell_sum / max(n_cells, 1)).float()
    Sigma = (cell_cov / max(n_cells, 1)).float() \
            - torch.outer(mu_distill, mu_distill)
    Sigma = 0.5 * (Sigma + Sigma.T)
    evals, evecs = torch.linalg.eigh(Sigma)
    order = torch.argsort(evals, descending=True)
    D = min(D, C_out)
    V = evecs[:, order[:D]].contiguous()
    sv = evals[order[:D]].clamp(min=0).contiguous()
    for d in range(D):
        col = V[:, d]
        if col[torch.argmax(col.abs())] < 0:
            V[:, d] = -col

    info = {'n_cells': n_cells, 'n_batches': n_batches,
            'batch_size': batch_size, 'iters': iters,
            'harvest_passes': harvest_passes, 'harvest_bs': harvest_bs,
            'bn_stride': bn_stride, 'amp': amp_enabled,
            't_synth_s': t_synth_total, 't_harvest_s': t_harvest_total}
    return V.cpu(), mu_distill.cpu(), info


# ======================================================================
# DataFreeReconstructor -- drop-in parity with PCAReconstructor
# ======================================================================

class DataFreeReconstructor(nn.Module):
    """Holds the calibration-free basis and offset.

    Exposes `pca_mu` ([C]) and `pca_V` ([C, D]) so that hier_visualize_pca.py
    style code -- and df_hier_visualization.py -- can consume it identically
    to a corpus-built PCAReconstructor. The name keeps the `pca_*` attributes
    purely for drop-in compatibility; nothing here is estimated from data.
    """

    def __init__(self, mu: torch.Tensor, V: torch.Tensor,
                 basis_kind: str, model_name: str, target_layer: str,
                 singular_values: Optional[torch.Tensor] = None,
                 block_kind: str = 'unknown', meta: Optional[Dict] = None):
        super().__init__()
        self.register_buffer('pca_mu', mu.detach().float())
        self.register_buffer('pca_V', V.detach().float())
        if singular_values is not None:
            self.register_buffer('singular_values',
                                 singular_values.detach().float())
        else:
            self.singular_values = None
        self.basis_kind = basis_kind
        self.model_name = model_name
        self.target_layer = target_layer
        self.block_kind = block_kind
        self.meta = meta or {}

    @property
    def C(self) -> int:
        return self.pca_mu.shape[0]

    @property
    def D(self) -> int:
        return self.pca_V.shape[1]

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

    def __repr__(self):
        return (f"DataFreeReconstructor(basis='{self.basis_kind}', "
                f"model='{self.model_name}', layer='{self.target_layer}', "
                f"C={self.C}, D={self.D}, block='{self.block_kind}')")


def build_from_weights(model_name: str, target_layer_name: str, D: int,
                       basis: str = 'kernel',
                       distill_batches: int = 8, distill_bs: int = 16,
                       distill_iters: int = 400, distill_lr: float = 0.1,
                       distill_seed: int = 0,
                       device: str = 'auto') -> DataFreeReconstructor:
    """Build a DataFreeReconstructor from a pretrained checkpoint, no corpus.

    basis  : 'kernel' | 'bn' | 'distill'
    device : 'auto' (use CUDA if available -- the default), 'cuda', or 'cpu'.
    """
    if device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device == 'cuda' and not torch.cuda.is_available():
        print("  WARNING: --device cuda requested but CUDA is unavailable; "
              "falling back to CPU.")
        dev = torch.device('cpu')
    else:
        dev = torch.device(device)
    print(f"  Device: {dev}")
    backbone = MODEL_CONFIGS[model_name]['model_fn']().to(dev).eval()

    W, block_kind, winfo = get_effective_operator(
        backbone, model_name, target_layer_name)
    C_out = W.shape[0]
    mu_bn = get_bn_stats(backbone, model_name, target_layer_name, C_out)

    print(f"  Effective operator W: {tuple(W.shape)}  "
          f"(block kind: {block_kind})")

    if basis == 'kernel':
        V, sv = build_kernel_basis(W, D)
        mu = mu_bn
        meta = dict(winfo)
    elif basis == 'bn':
        h_in = get_upstream_bn_var(backbone, model_name,
                                   target_layer_name, W.shape[1])
        # diagnostic: how anisotropic is the diagonal proxy?
        h_ratio = float(h_in.max() / h_in.clamp(min=1e-12).min())
        print(f"  Upstream BN tilt h_in: len={h_in.numel()}  "
              f"max/min ratio={h_ratio:.2f}  (1.0 == white == no tilt)")
        V, sv = build_bn_basis(W, h_in, D)
        mu = mu_bn
        meta = dict(winfo); meta['h_in_ratio'] = h_ratio
    elif basis == 'distill':
        V, mu_d, dinfo = build_distill_basis(
            backbone, model_name, target_layer_name, C_out, D,
            n_batches=distill_batches, batch_size=distill_bs,
            iters=distill_iters, lr=distill_lr, device=dev,
            seed=distill_seed)
        sv = None
        # the distilled forward pass yields its own activation-mean estimate;
        # it is the more faithful offset than the raw BN proxy.
        mu = mu_d
        meta = dict(winfo); meta.update(dinfo)
    else:
        raise ValueError(f"Unknown basis '{basis}'. "
                         f"Choose kernel | bn | distill.")

    # The basis is a static artifact to be pickled and reloaded elsewhere;
    # normalise every tensor to CPU so the .pkl is device-agnostic. Builders
    # return V on whatever device W lived on (CUDA when the backbone is),
    # except build_distill_basis which already returns CPU -- unify here.
    V = V.detach().cpu().contiguous()
    mu = mu.detach().cpu().contiguous()
    if sv is not None:
        sv = sv.detach().cpu().contiguous()

    # orthonormality check -- completeness (Theorem 1) depends on it
    gram = V.T @ V
    ortho_err = float((gram - torch.eye(V.shape[1])).abs().max())
    print(f"  Basis '{basis}': V {tuple(V.shape)}  "
          f"orthonormality max-err={ortho_err:.2e}")
    if ortho_err > 1e-3:
        print("  WARNING: basis not orthonormal to tolerance; "
              "completeness identity may carry residual.")

    recon = DataFreeReconstructor(
        mu=mu, V=V, basis_kind=basis, model_name=model_name,
        target_layer=target_layer_name, singular_values=sv,
        block_kind=block_kind, meta=meta)
    return recon


# ======================================================================
# CLI
# ======================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Build a calibration-free Grad-CAM basis from network '
                    'weights (no corpus).')
    ap.add_argument('--model', type=str, default='resnet50',
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--target_layer', type=str, default=None)
    ap.add_argument('--basis', type=str, default='kernel',
                    choices=['kernel', 'bn', 'distill'],
                    help="kernel = WW^T white surrogate; "
                         "bn = input-axis upstream-BN tilt; "
                         "distill = ZeroQ-style synthesis.")
    ap.add_argument('--D', type=int, default=200,
                    help='Number of basis components to keep.')
    ap.add_argument('--distill_batches', type=int, default=8)
    ap.add_argument('--distill_bs', type=int, default=16)
    ap.add_argument('--distill_iters', type=int, default=400)
    ap.add_argument('--distill_lr', type=float, default=0.1)
    ap.add_argument('--distill_seed', type=int, default=0)
    ap.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cuda', 'cpu'],
                    help="Compute device. 'auto' (default) uses CUDA when "
                         "available -- important for the 'distill' basis, "
                         "which is many forward+backward passes and is "
                         "extremely slow on CPU.")
    ap.add_argument('--output', type=str, default=None,
                    help='Output .pkl path for the DataFreeReconstructor.')
    args = ap.parse_args()

    if args.target_layer is None:
        args.target_layer = MODEL_CONFIGS[args.model]['default_target_layer']
    if args.output is None:
        args.output = (f"df_basis_{args.model}_"
                       f"{args.target_layer.replace('[', '').replace(']', '')}"
                       f"_{args.basis}_D{args.D}.pkl")

    print("=" * 78)
    print("Calibration-Free Grad-CAM Basis  (data-free; no corpus)")
    print(f"  Model:        {args.model} @ {args.target_layer}")
    print(f"  Basis:        {args.basis}")
    print(f"  D:            {args.D}")
    if args.basis == 'distill':
        print(f"  Synthesis:    {args.distill_batches} batches x "
              f"{args.distill_bs} imgs, {args.distill_iters} iters")
    print("=" * 78)

    recon = build_from_weights(
        args.model, args.target_layer, args.D, basis=args.basis,
        distill_batches=args.distill_batches, distill_bs=args.distill_bs,
        distill_iters=args.distill_iters, distill_lr=args.distill_lr,
        distill_seed=args.distill_seed, device=args.device)

    joblib.dump(recon, args.output)
    print(f"\n  {recon}")
    print(f"  Saved basis -> {args.output}")
    print("=" * 78)
    print("Done. Feed this .pkl to df_hier_visualization.py via --df_basis.")
    print("=" * 78)


if __name__ == '__main__':
    main()