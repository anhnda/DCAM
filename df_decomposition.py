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
  'bn'       top-D eigenspace of  W diag(h_hat) W^T        -> BN-tilted
             h_hat = upstream BatchNorm running variances feeding the branch.
  'distill'  top-D eigenspace of Sigma~ estimated from a BatchNorm-matched
             synthetic batch run through the true nonlinear forward pass
             (Section 5, ZeroQ-style). Zero real images.

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

  # BN-distilled variant (synthesises images, still zero real data)
  python df_decomposition.py --model resnet50 --target_layer layer3 \\
      --basis distill --D 200 --distill_batches 8 --distill_iters 500 \\
      --output df_basis_resnet50_layer3_distill.pkl

  # then visualize a single image with df_hier_visualization.py
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
# BatchNorm mean / variance read-off  (offset mu; BN-tilt for 'bn' basis)
# ======================================================================

def get_bn_stats(backbone: nn.Module, model_name: str,
                 target_layer_name: str, C_out: int
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read the BatchNorm running statistics that describe the target layer's
    OUTPUT channels.

    mu (offset)       : BN running mean of the operator we folded (Section B,
                        a proxy for the post-activation channel mean).
    h_hat (BN tilt)   : BN running variance of the same channels, used by the
                        'bn' basis to weight WW^T by per-channel scale
                        (Definition 1, BN-tilted variant).
    """
    # device of the backbone -- all fallback tensors must match it.
    dev = next(backbone.parameters()).device

    if model_name in ('resnet50', 'resnet18'):
        stage = {'layer1': backbone.layer1, 'layer2': backbone.layer2,
                 'layer3': backbone.layer3, 'layer4': backbone.layer4
                 }[target_layer_name]
        last_block = stage[-1]
        bn_f = last_block.bn3 if hasattr(last_block, 'bn3') else last_block.bn2
        mu = bn_f.running_mean.detach().clone()
        h_hat = bn_f.running_var.detach().clone()
    elif model_name == 'vgg16':
        idx = int(target_layer_name.split('[')[1].rstrip(']'))
        if idx + 1 < len(backbone.features) and isinstance(
                backbone.features[idx + 1], nn.BatchNorm2d):
            bn = backbone.features[idx + 1]
            mu = bn.running_mean.detach().clone()
            h_hat = bn.running_var.detach().clone()
        else:
            # plain VGG (no BN): BN proxy unavailable -> zero offset, unit tilt.
            mu = torch.zeros(C_out, device=dev)
            h_hat = torch.ones(C_out, device=dev)
    else:
        raise ValueError(f"Unsupported model '{model_name}'.")

    if mu.numel() != C_out:
        # defensive: fall back to zero offset of the right width
        mu = torch.zeros(C_out, device=dev)
        h_hat = torch.ones(C_out, device=dev)
    return mu, h_hat


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


def build_bn_basis(W: torch.Tensor, h_hat: torch.Tensor,
                   D: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """BN-tilted basis (Definition 1): top-D eigenspace of W diag(h_hat) W^T.

    h_hat are the BatchNorm running variances of the operator's OUTPUT
    channels; tilting WW^T by them is a cheap, corpus-free proxy for the
    input-patch anisotropy H_in. Equivalently the top-D left singular vectors
    of  diag(sqrt(h_hat)) ... -- we form it on the output side since h_hat
    indexes output channels here.
    """
    # Symmetric PSD matrix M = W W^T tilted by output-channel scale.
    # We weight each *output* channel by its BN std: A_tilt = diag(sqrt h) W.
    s = torch.sqrt(h_hat.clamp(min=1e-12))
    M = s.view(-1, 1) * W                                   # [C_out, in_dim]
    U_D, sv = _topD_left_singular(M, D)
    # Undo the output-side scaling so vectors live in the un-tilted channel
    # space, then re-orthonormalise (Gram-Schmidt via QR) to keep the frame
    # orthonormal -- completeness (Theorem 1) requires orthonormality.
    U_D = U_D / s.clamp(min=1e-12).view(-1, 1)
    Q, _ = torch.linalg.qr(U_D)
    for d in range(Q.shape[1]):
        col = Q[:, d]
        if col[torch.argmax(col.abs())] < 0:
            Q[:, d] = -col
    return Q.contiguous(), sv


def build_distill_basis(backbone: nn.Module, model_name: str,
                        target_layer_name: str, C_out: int, D: int,
                        n_batches: int = 8, batch_size: int = 16,
                        iters: int = 400, lr: float = 0.1,
                        device: torch.device = torch.device('cpu'),
                        seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor,
                                                Dict]:
    """ZeroQ-style BN-distilled basis (Section 5). Zero real images.

    Synthesise batches x~ by minimising the BatchNorm-matching objective

        L(x~) = sum_l ||mean_l(x~) - mu_BN_l||^2 + ||var_l(x~) - var_BN_l||^2

    then run the true nonlinear residual forward pass and estimate Sigma~ from
    the resulting per-cell activations at the target layer. The basis is the
    top-D eigenspace of Sigma~ -- it captures residual-skip directions and the
    cross-channel orientation unreadable from a local SVD (Remark 1).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    backbone = backbone.to(device).eval()

    # ---- hook every BN layer to read its batch statistics ----
    bn_layers = [m for m in backbone.modules()
                 if isinstance(m, nn.BatchNorm2d)]
    bn_feats: Dict[int, torch.Tensor] = {}

    def make_hook(i):
        def hook(module, inp, out):
            x = inp[0]
            bn_feats[i] = (x.mean(dim=(0, 2, 3)),
                           x.var(dim=(0, 2, 3), unbiased=False))
        return hook

    handles = [bn.register_forward_hook(make_hook(i))
               for i, bn in enumerate(bn_layers)]

    bn_target_mean = [bn.running_mean.detach().to(device) for bn in bn_layers]
    bn_target_var = [bn.running_var.detach().to(device) for bn in bn_layers]

    # ---- hook the target layer to collect synthetic activations ----
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

    # accumulate Sigma~ as a running second moment of centred cells
    cell_sum = torch.zeros(C_out, device=device)
    cell_cov = torch.zeros(C_out, C_out, device=device)
    n_cells = 0

    for batch in range(n_batches):
        x = torch.randn(batch_size, 3, 224, 224, device=device,
                        requires_grad=True)
        opt = torch.optim.Adam([x], lr=lr)
        for it in range(iters):
            opt.zero_grad()
            bn_feats.clear()
            _ = backbone(x)
            loss = x.new_zeros(())
            for i in range(len(bn_layers)):
                if i not in bn_feats:
                    continue
                m, v = bn_feats[i]
                loss = loss + F.mse_loss(m, bn_target_mean[i]) \
                            + F.mse_loss(v, bn_target_var[i])
            # mild input prior keeps the synthesis well-conditioned
            loss = loss + 1e-4 * (x ** 2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            bn_feats.clear()
            _ = backbone(x.detach())
            act = collected['act']                          # [B, C, H, W]
            cells = act.permute(0, 2, 3, 1).reshape(-1, C_out)
            cell_sum += cells.sum(dim=0)
            cell_cov += cells.T @ cells
            n_cells += cells.shape[0]
        print(f"  [distill] batch {batch + 1}/{n_batches}  "
              f"bn-match loss={float(loss):.4f}  cells={n_cells}")

    for h in handles:
        h.remove()
    tgt_handle.remove()

    mu_distill = cell_sum / max(n_cells, 1)
    # Sigma~ = E[c c^T] - mu mu^T
    Sigma = cell_cov / max(n_cells, 1) - torch.outer(mu_distill, mu_distill)
    Sigma = 0.5 * (Sigma + Sigma.T)                         # symmetrise
    evals, evecs = torch.linalg.eigh(Sigma)                 # ascending
    order = torch.argsort(evals, descending=True)
    D = min(D, C_out)
    V = evecs[:, order[:D]].contiguous()
    sv = evals[order[:D]].clamp(min=0).contiguous()
    for d in range(D):
        col = V[:, d]
        if col[torch.argmax(col.abs())] < 0:
            V[:, d] = -col

    info = {'n_cells': n_cells, 'n_batches': n_batches,
            'batch_size': batch_size, 'iters': iters}
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
    mu_bn, h_hat = get_bn_stats(backbone, model_name, target_layer_name, C_out)

    print(f"  Effective operator W: {tuple(W.shape)}  "
          f"(block kind: {block_kind})")

    if basis == 'kernel':
        V, sv = build_kernel_basis(W, D)
        mu = mu_bn
        meta = dict(winfo)
    elif basis == 'bn':
        V, sv = build_bn_basis(W, h_hat, D)
        mu = mu_bn
        meta = dict(winfo)
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
                         "bn = BN-tilted; distill = ZeroQ-style synthesis.")
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