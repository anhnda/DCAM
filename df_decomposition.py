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
             axis (the BN feeding the folded operator). This is the diagonal
             proxy for H_in in Sigma = W H_in W^T (Definition 1).
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

Supported backbones
-------------------
The backbone-specific logic -- "find the last linear operator that produces
the target layer's output", "find the BN whose running stats describe that
operator's output and input" -- is dispatched through a `BackboneAdapter`
registry, so new architectures plug in without editing the basis builders.

Registered out of the box:

  ResNet family (BasicBlock / Bottleneck):
      resnet18, resnet34, resnet50, resnet101, resnet152
      wide_resnet50_2, wide_resnet101_2
      resnext50_32x4d, resnext101_32x8d

  VGG family (plain conv + optional BN):
      vgg11, vgg13, vgg16, vgg19
      vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

  DenseNet family (concatenation; last DenseLayer's conv2 is the injected map):
      densenet121, densenet169, densenet201

  MobileNet family (inverted residual; project 1x1 conv+BN is the folded op):
      mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small

  EfficientNet family (MBConv; project 1x1 conv+BN is the folded op):
      efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3

  ConvNeXt family (LayerNorm, no BN; the 'bn' basis gracefully reduces to
  'kernel' as it does for plain VGG):
      convnext_tiny, convnext_small, convnext_base

For any non-residual architecture (VGG, ConvNeXt) the "block kind" reported by
the operator finder is 'plain', and the basis spans only the freshly-emitted
channel directions, exactly as Lemma 1 prescribes. For DenseNet -- where the
layer output is a CONCATENATION of all earlier DenseLayer outputs with the
new DenseLayer's growth_rate channels -- the folded operator only spans the
growth_rate part (block kind 'concat'); this is the same honest limitation as
the ResNet identity-shortcut case (Remark 1), and is exactly what 'distill'
removes.

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

  # new backbones use the same CLI:
  python df_decomposition.py --model mobilenet_v2 --target_layer features[14] \\
      --basis bn --D 96 --output df_basis_mbv2_f14_bn.pkl
  python df_decomposition.py --model densenet121 --target_layer denseblock3 \\
      --basis kernel --D 64 --output df_basis_dn121_db3_kernel.pkl

  # then visualize a single image with df_hier_visualization.py

CHANGELOG
---------
  * Generalised backbone support via a `BackboneAdapter` registry. The
    architecture-specific code that used to live inside three if/elif chains
    (get_effective_operator, get_bn_stats, get_upstream_bn_var) now lives in
    one adapter per family. Adding a new architecture is one class.
  * Added DenseNet, MobileNetV2/V3, EfficientNet, ConvNeXt adapters; widened
    ResNet adapter to cover 34/101/152/wide/ResNeXt; widened VGG adapter to
    cover 11/13/19 with and without BN. VGG-BN now uses the PREVIOUS BN in
    `features` as the upstream-axis tilt instead of falling back to white.
  * Backward-compatible: existing call sites and `.pkl` files keep working;
    MODEL_CONFIGS entries for resnet50, resnet18 are unchanged.
  * BUGFIX: vgg16 default target updated features[16] -> features[14]. The
    original default pointed at a MaxPool2d (no Conv2d there); the original
    description ("256ch, 28x28") matched features[14], the last 256-channel
    conv before the next pool. Anyone passing --target_layer features[14]
    explicitly is unaffected; default callers now succeed where they would
    have raised "features[16] is MaxPool2d, not a Conv2d".
  * (Prior revision) build_bn_basis now tilts the INPUT axis (columns of W)
    by the UPSTREAM BN variance, matching Definition 1. Eigenvectors of the
    symmetric PSD W diag(h_in) W^T are already orthonormal in C_out, so no
    undo-scaling and no re-QR.
"""

import argparse
import sys
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ----------------------------------------------------------------------
# torchvision compat: `pretrained=True` is deprecated in newer torchvision
# in favor of `weights='DEFAULT'`. _load wraps both so MODEL_CONFIGS reads
# the same regardless of which torchvision version is installed.
# ----------------------------------------------------------------------

def _load(name: str):
    """Return a pretrained model by name across torchvision versions."""
    fn = getattr(models, name)
    try:
        return fn(weights='DEFAULT')
    except TypeError:                                         # very old API
        return fn(pretrained=True)


# ======================================================================
# Backbone registry
# ======================================================================
#
# Each adapter answers three architecture-specific questions; the basis
# builders are architecture-agnostic and read from these.
#
#   find_operator(backbone, target_layer_name) -> (conv, bn_or_None,
#                                                  block_kind, info_dict)
#       The conv whose output (after the optional bn) IS the layer's output
#       (for plain stacks like VGG) or is the freshly-injected residual /
#       concatenated part of the layer's output (for ResNet, DenseNet,
#       MobileNet, EfficientNet, ConvNeXt). The conv may be a Conv2d, or a
#       Linear masquerading as a 1x1 conv (ConvNeXt). bn_or_None is the BN
#       to FOLD with it (BN that immediately follows in inference), or None.
#       `block_kind` is a human-readable label echoed back in info / .pkl.
#
#   find_output_bn(backbone, target_layer_name) -> bn_or_None
#       The BN whose RUNNING MEAN describes the OUTPUT channels of the folded
#       operator. By construction this is the same BN as in find_operator's
#       second return when present; this hook exists separately so adapters
#       that don't fold a BN can still expose one to read mu from.
#
#   find_upstream_bn(backbone, target_layer_name) -> bn_or_None
#       The BN whose RUNNING VAR describes the INPUT channels feeding the
#       folded operator (the "h_in" tilt in Definition 1). Returns None
#       when no BN feeds the operator's input (VGG plain, ConvNeXt): the
#       'bn' basis then degenerates to 'kernel' (white H_in).
#
# Adapters are looked up by FAMILY KEY (resnet / vgg / densenet / mobilenet
# / efficientnet / convnext); MODEL_CONFIGS maps each individual model name
# to a family key.
# ======================================================================


class BackboneAdapter:
    """Base class. Subclasses override the three locator methods."""

    def find_operator(self, backbone: nn.Module, target: str
                       ) -> Tuple[nn.Module, Optional[nn.BatchNorm2d],
                                   str, Dict]:
        raise NotImplementedError

    def find_output_bn(self, backbone: nn.Module, target: str
                        ) -> Optional[nn.BatchNorm2d]:
        raise NotImplementedError

    def find_upstream_bn(self, backbone: nn.Module, target: str
                          ) -> Optional[nn.BatchNorm2d]:
        raise NotImplementedError

    @staticmethod
    def get_target_module(backbone: nn.Module, target: str) -> nn.Module:
        """For the distill basis: the live submodule whose output to hook."""
        raise NotImplementedError


def _resolve_features_idx(backbone: nn.Module, target: str) -> int:
    """Parse `features[<idx>]` style target into an int index."""
    if '[' not in target or ']' not in target:
        raise ValueError(f"Expected 'features[<idx>]', got '{target}'.")
    return int(target.split('[')[1].rstrip(']'))


# ---------------------------- ResNet family ---------------------------

class ResNetAdapter(BackboneAdapter):
    """ResNet18/34/50/101/152, WideResNet, ResNeXt -- all use BasicBlock or
    Bottleneck and expose stages as layer1..layer4."""

    @staticmethod
    def _stage(backbone: nn.Module, target: str) -> nn.Sequential:
        stages = {'layer1': backbone.layer1, 'layer2': backbone.layer2,
                   'layer3': backbone.layer3, 'layer4': backbone.layer4}
        if target not in stages:
            raise ValueError(f"ResNet target must be one of {list(stages)};"
                              f" got '{target}'.")
        return stages[target]

    @staticmethod
    def _last_residual_conv_bn(block: nn.Module
                                ) -> Tuple[nn.Conv2d, nn.BatchNorm2d, str]:
        # Bottleneck has conv3/bn3 (the expand 1x1); BasicBlock has conv2/bn2.
        if hasattr(block, 'conv3'):
            return block.conv3, block.bn3, 'conv3'
        return block.conv2, block.bn2, 'conv2'

    def find_operator(self, backbone, target):
        stage = self._stage(backbone, target)
        last_block = stage[-1]
        conv_f, bn_f, conv_name = self._last_residual_conv_bn(last_block)
        block_kind = ('identity' if getattr(last_block, 'downsample', None)
                                     is None else 'projection')
        info = {'stage_blocks': len(stage),
                'residual_conv': conv_name,
                'expand_kernel': tuple(conv_f.weight.shape)}
        return conv_f, bn_f, block_kind, info

    def find_output_bn(self, backbone, target):
        _, bn_f, _, _ = self.find_operator(backbone, target)
        return bn_f

    def find_upstream_bn(self, backbone, target):
        stage = self._stage(backbone, target)
        last_block = stage[-1]
        # Bottleneck (resnet50/101/152/wide/ResNeXt): conv3 <- bn2.
        # BasicBlock (resnet18/34): conv2 <- bn1.
        if hasattr(last_block, 'conv3'):
            return last_block.bn2
        return last_block.bn1

    @staticmethod
    def get_target_module(backbone, target):
        return ResNetAdapter._stage(backbone, target)


# ---------------------------- VGG family ------------------------------

class VGGAdapter(BackboneAdapter):
    """VGG11/13/16/19, with and without BN. Target form: 'features[<idx>]'."""

    def find_operator(self, backbone, target):
        idx = _resolve_features_idx(backbone, target)
        conv = backbone.features[idx]
        if not isinstance(conv, nn.Conv2d):
            raise ValueError(f"features[{idx}] is {type(conv).__name__}, "
                              f"not a Conv2d.")
        bn = None
        if (idx + 1 < len(backbone.features) and
                isinstance(backbone.features[idx + 1], nn.BatchNorm2d)):
            bn = backbone.features[idx + 1]
        info = {'conv_kernel': tuple(conv.weight.shape)}
        return conv, bn, 'plain', info

    def find_output_bn(self, backbone, target):
        _, bn, _, _ = self.find_operator(backbone, target)
        return bn

    def find_upstream_bn(self, backbone, target):
        """Previous BN in features, walking back from features[idx].
        For VGG-BN this returns a real BN; for plain VGG (no BN) it returns
        None, and the 'bn' basis falls back to 'kernel'."""
        idx = _resolve_features_idx(backbone, target)
        for j in range(idx - 1, -1, -1):
            m = backbone.features[j]
            if isinstance(m, nn.BatchNorm2d):
                return m
            if isinstance(m, nn.Conv2d):
                # crossed an unnormalised conv -> no upstream BN we can trust
                return None
        return None

    @staticmethod
    def get_target_module(backbone, target):
        idx = _resolve_features_idx(backbone, target)
        return backbone.features[idx]


# --------------------------- DenseNet family --------------------------

class DenseNetAdapter(BackboneAdapter):
    """DenseNet121/169/201. Stages live under .features.denseblock{1..4}.
    A DenseLayer's tail is `conv2` (3x3, growth_rate channels) -- no BN
    follows (the next BN is at the START of the next DenseLayer), so we
    fold conv2 alone. Block kind is 'concat': the DenseBlock output
    concatenates all earlier DenseLayer outputs with this new growth_rate
    contribution, so W spans only the new growth_rate channels.
    """

    @staticmethod
    def _block(backbone: nn.Module, target: str) -> nn.Module:
        if not hasattr(backbone.features, target):
            valid = [n for n in dir(backbone.features)
                     if n.startswith('denseblock')]
            raise ValueError(f"DenseNet target must be one of {valid}; "
                              f"got '{target}'.")
        return getattr(backbone.features, target)

    @staticmethod
    def _last_layer(block: nn.Module) -> nn.Module:
        # _DenseLayer instances are named denselayer1..denselayerN
        names = [n for n in dir(block) if n.startswith('denselayer')]
        # numeric sort
        names = sorted(names, key=lambda s: int(s.replace('denselayer', '')))
        return getattr(block, names[-1])

    def find_operator(self, backbone, target):
        block = self._block(backbone, target)
        last = self._last_layer(block)
        conv2 = last.conv2                                       # 3x3 growth conv
        info = {'block_layers': sum(1 for n in dir(block)
                                    if n.startswith('denselayer')),
                'growth_kernel': tuple(conv2.weight.shape)}
        return conv2, None, 'concat', info

    def find_output_bn(self, backbone, target):
        # No BN follows conv2 inside the layer; the next BN belongs to the
        # NEXT DenseLayer's norm1. The cleanest "channel mean for the
        # growth_rate output" we can read is None -> mu defaults to zeros.
        return None

    def find_upstream_bn(self, backbone, target):
        # conv2 is preceded by norm2 -> relu2; norm2 IS the upstream BN.
        last = self._last_layer(self._block(backbone, target))
        return last.norm2

    @staticmethod
    def get_target_module(backbone, target):
        return DenseNetAdapter._block(backbone, target)


# -------------------- MobileNet / EfficientNet families ---------------

class InvertedResidualAdapter(BackboneAdapter):
    """Shared adapter for MobileNetV2, MobileNetV3, EfficientNet (MBConv).

    Each block ends with a project Conv2dNormActivation (or in MV2 a raw
    Conv2d + BatchNorm2d pair). The folded operator is that project conv +
    project BN. The upstream BN is the depthwise BN.

    For MobileNetV3 / EfficientNet there is an SE module between the
    depthwise BN and the project conv; SE multiplies its input by a learned
    per-channel scale post-BN. We still use the depthwise BN's variance as
    the upstream tilt -- it is the closest principled diagonal proxy
    available without running the SE branch on real data.

    Target form: 'features[<idx>]', where features[idx] is a single block
    (Conv2dNormActivation, InvertedResidual, MBConv, or a Sequential of
    such blocks -- in which case we take its last element).
    """

    @staticmethod
    def _block(backbone: nn.Module, target: str) -> nn.Module:
        idx = _resolve_features_idx(backbone, target)
        m = backbone.features[idx]
        # EfficientNet's `features[k]` for k in [1..7] is a Sequential of
        # MBConv blocks; take the last block of the stage.
        if isinstance(m, nn.Sequential):
            return m[-1]
        return m

    @staticmethod
    def _peel(block: nn.Module) -> nn.Sequential:
        """Return the Sequential of sub-blocks inside an inverted residual.

        Layout:
          MV2 InvertedResidual : .conv          (Sequential)
          MV3 InvertedResidual : .block         (Sequential)
          EfficientNet MBConv  : .block         (Sequential)
          Initial / final Conv2dNormActivation  : itself a Sequential
        """
        if isinstance(block, nn.Sequential):
            return block
        for attr in ('block', 'conv'):
            sub = getattr(block, attr, None)
            if isinstance(sub, nn.Sequential):
                return sub
        raise ValueError(f"Cannot find Sequential inside "
                          f"{type(block).__name__}.")

    @staticmethod
    def _project_conv_bn(block: nn.Module
                          ) -> Tuple[nn.Conv2d, Optional[nn.BatchNorm2d]]:
        """Last Conv2d + optional following BN inside the block."""
        seq = InvertedResidualAdapter._peel(block)
        # Walk children in order; the last Conv2d is the project. Its BN is
        # the next BatchNorm2d after it in the same flat traversal.
        flat: list = []
        def collect(m):
            for c in m.children():
                if list(c.children()) and not isinstance(
                        c, (nn.Conv2d, nn.BatchNorm2d)):
                    collect(c)
                else:
                    flat.append(c)
        collect(seq)
        last_conv_idx = None
        for i, m in enumerate(flat):
            if isinstance(m, nn.Conv2d):
                last_conv_idx = i
        if last_conv_idx is None:
            raise ValueError("No Conv2d found in block.")
        conv = flat[last_conv_idx]
        bn = None
        for m in flat[last_conv_idx + 1:]:
            if isinstance(m, nn.BatchNorm2d):
                bn = m
                break
            if isinstance(m, nn.Conv2d):
                break
        return conv, bn

    @staticmethod
    def _depthwise_bn(block: nn.Module) -> Optional[nn.BatchNorm2d]:
        """Find the depthwise conv's BN. Heuristic: among Conv2d's in the
        block, the depthwise one has groups == in_channels == out_channels
        (groups == out_channels actually -- channel-wise). We return the
        BN immediately following the LAST such depthwise conv (the project
        conv comes after it in inverted residuals)."""
        seq = InvertedResidualAdapter._peel(block)
        flat: list = []
        def collect(m):
            for c in m.children():
                if list(c.children()) and not isinstance(
                        c, (nn.Conv2d, nn.BatchNorm2d)):
                    collect(c)
                else:
                    flat.append(c)
        collect(seq)
        last_dw_idx = None
        for i, m in enumerate(flat):
            if isinstance(m, nn.Conv2d) and m.groups == m.in_channels \
                    and m.groups > 1:
                last_dw_idx = i
        if last_dw_idx is None:
            return None
        for m in flat[last_dw_idx + 1:]:
            if isinstance(m, nn.BatchNorm2d):
                return m
            if isinstance(m, nn.Conv2d):
                break
        return None

    def find_operator(self, backbone, target):
        block = self._block(backbone, target)
        conv, bn = self._project_conv_bn(block)
        # use_res_connect is True for inverted residuals with a (skip) path.
        has_skip = bool(getattr(block, 'use_res_connect', False))
        block_kind = ('identity' if has_skip else 'projection')
        info = {'project_kernel': tuple(conv.weight.shape),
                'block_type': type(block).__name__}
        return conv, bn, block_kind, info

    def find_output_bn(self, backbone, target):
        _, bn, _, _ = self.find_operator(backbone, target)
        return bn

    def find_upstream_bn(self, backbone, target):
        return self._depthwise_bn(self._block(backbone, target))

    @staticmethod
    def get_target_module(backbone, target):
        idx = _resolve_features_idx(backbone, target)
        return backbone.features[idx]


# --------------------------- ConvNeXt family --------------------------

class ConvNeXtAdapter(BackboneAdapter):
    """ConvNeXt tiny/small/base/large. A CNBlock's tail is the second
    Linear (1xchannel pointwise via channels-last) -- structurally a 1x1
    conv mapping `4*C -> C`. There is no BatchNorm anywhere in a CNBlock;
    only LayerNorm. We therefore:
      * fold the second Linear with no BN (W = Linear.weight),
      * read mu = 0 (no BN running mean),
      * return None for upstream BN -> 'bn' basis falls back to 'kernel'.

    This is the same graceful degeneration as VGG (plain, no BN).

    Target form: 'features[<idx>]', where idx in {1,3,5,7} for the four
    ConvNeXt stages.
    """

    @staticmethod
    def _stage(backbone: nn.Module, target: str) -> nn.Sequential:
        idx = _resolve_features_idx(backbone, target)
        m = backbone.features[idx]
        if not isinstance(m, nn.Sequential):
            raise ValueError(f"convnext features[{idx}] is "
                              f"{type(m).__name__}, expected a Sequential "
                              f"of CNBlocks (use idx in {{1,3,5,7}}).")
        return m

    @staticmethod
    def _last_block(stage: nn.Sequential) -> nn.Module:
        return stage[-1]

    @staticmethod
    def _project_linear(block: nn.Module) -> nn.Linear:
        seq = getattr(block, 'block', None)
        if not isinstance(seq, nn.Sequential):
            raise ValueError("CNBlock has no .block Sequential.")
        last_linear = None
        for m in seq:
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise ValueError("CNBlock has no Linear layer.")
        return last_linear

    def find_operator(self, backbone, target):
        stage = self._stage(backbone, target)
        block = self._last_block(stage)
        lin = self._project_linear(block)
        # Wrap the Linear in a minimal Conv2d-look-alike so fold_conv_bn
        # can treat it identically. Linear.weight has shape [C_out, C_in];
        # equivalent to a 1x1 conv with kernel [C_out, C_in, 1, 1].
        conv = _LinearAsConv1x1(lin)
        info = {'stage_blocks': len(stage),
                'project_linear': tuple(lin.weight.shape),
                'block_type': type(block).__name__}
        return conv, None, 'identity', info

    def find_output_bn(self, backbone, target):
        return None                                              # LayerNorm only

    def find_upstream_bn(self, backbone, target):
        return None                                              # LayerNorm only

    @staticmethod
    def get_target_module(backbone, target):
        return ConvNeXtAdapter._stage(backbone, target)


class _LinearAsConv1x1(nn.Module):
    """Thin shim presenting a Linear's weight as a 1x1 Conv2d to fold_conv_bn.
    Carries `.weight` and `.bias` attributes with the conv-flavored shape."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        # [C_out, C_in] -> [C_out, C_in, 1, 1]
        self.weight = nn.Parameter(lin.weight.detach().clone()
                                   .unsqueeze(-1).unsqueeze(-1),
                                   requires_grad=False)
        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.detach().clone(),
                                     requires_grad=False)
        else:
            self.bias = None


# ---------------------- Adapter registry and config -------------------

_ADAPTERS: Dict[str, BackboneAdapter] = {
    'resnet': ResNetAdapter(),
    'vgg': VGGAdapter(),
    'densenet': DenseNetAdapter(),
    'inverted_residual': InvertedResidualAdapter(),
    'convnext': ConvNeXtAdapter(),
}


def _cfg(family: str, default_target: str, description: str,
         loader_name: Optional[str] = None) -> Dict:
    """Build a MODEL_CONFIGS entry. loader_name defaults to the dict key."""
    return {'family': family,
            'loader_name': loader_name,                          # filled below
            'default_target_layer': default_target,
            'description': description}


MODEL_CONFIGS: Dict[str, Dict] = {
    # ---- ResNet family ----
    'resnet18':           _cfg('resnet', 'layer3',
                                'ResNet18 (BasicBlock; layer3: 256ch)'),
    'resnet34':           _cfg('resnet', 'layer3',
                                'ResNet34 (BasicBlock; layer3: 256ch)'),
    'resnet50':           _cfg('resnet', 'layer3',
                                'ResNet50 (Bottleneck; layer3: 1024ch, 14x14)'),
    'resnet101':          _cfg('resnet', 'layer3',
                                'ResNet101 (Bottleneck; layer3: 1024ch)'),
    'resnet152':          _cfg('resnet', 'layer3',
                                'ResNet152 (Bottleneck; layer3: 1024ch)'),
    'wide_resnet50_2':    _cfg('resnet', 'layer3',
                                'Wide ResNet50_2 (Bottleneck; layer3: 1024ch)'),
    'wide_resnet101_2':   _cfg('resnet', 'layer3',
                                'Wide ResNet101_2 (Bottleneck; layer3: 1024ch)'),
    'resnext50_32x4d':    _cfg('resnet', 'layer3',
                                'ResNeXt50_32x4d (Bottleneck; layer3: 1024ch)'),
    'resnext101_32x8d':   _cfg('resnet', 'layer3',
                                'ResNeXt101_32x8d (Bottleneck; layer3: 1024ch)'),

    # ---- VGG family ----
    'vgg11':              _cfg('vgg', 'features[16]',
                                'VGG11 plain (no BN)'),
    'vgg13':              _cfg('vgg', 'features[20]',
                                'VGG13 plain (no BN)'),
    'vgg16':              _cfg('vgg', 'features[14]',
                                'VGG16 (features[14]: 256ch, 28x28)'),
    'vgg19':              _cfg('vgg', 'features[28]',
                                'VGG19 plain (no BN)'),
    'vgg11_bn':           _cfg('vgg', 'features[22]',
                                'VGG11 with BN'),
    'vgg13_bn':           _cfg('vgg', 'features[28]',
                                'VGG13 with BN'),
    'vgg16_bn':           _cfg('vgg', 'features[34]',
                                'VGG16 with BN (deep stage)'),
    'vgg19_bn':           _cfg('vgg', 'features[40]',
                                'VGG19 with BN'),

    # ---- DenseNet family ----
    'densenet121':        _cfg('densenet', 'denseblock3',
                                'DenseNet121 (denseblock3 growth)'),
    'densenet169':        _cfg('densenet', 'denseblock3',
                                'DenseNet169 (denseblock3 growth)'),
    'densenet201':        _cfg('densenet', 'denseblock3',
                                'DenseNet201 (denseblock3 growth)'),

    # ---- MobileNet family ----
    'mobilenet_v2':       _cfg('inverted_residual', 'features[14]',
                                'MobileNetV2 (last 160ch IR block)'),
    'mobilenet_v3_large': _cfg('inverted_residual', 'features[12]',
                                'MobileNetV3 Large (mid 112ch IR block)'),
    'mobilenet_v3_small': _cfg('inverted_residual', 'features[8]',
                                'MobileNetV3 Small (mid 48ch IR block)'),

    # ---- EfficientNet family ----
    'efficientnet_b0':    _cfg('inverted_residual', 'features[5]',
                                'EfficientNet-B0 (stage 5 MBConv)'),
    'efficientnet_b1':    _cfg('inverted_residual', 'features[5]',
                                'EfficientNet-B1 (stage 5 MBConv)'),
    'efficientnet_b2':    _cfg('inverted_residual', 'features[5]',
                                'EfficientNet-B2 (stage 5 MBConv)'),
    'efficientnet_b3':    _cfg('inverted_residual', 'features[5]',
                                'EfficientNet-B3 (stage 5 MBConv)'),

    # ---- ConvNeXt family (LayerNorm; 'bn' degenerates to 'kernel') ----
    'convnext_tiny':      _cfg('convnext', 'features[5]',
                                'ConvNeXt Tiny (stage 3, 384ch)'),
    'convnext_small':     _cfg('convnext', 'features[5]',
                                'ConvNeXt Small (stage 3, 384ch)'),
    'convnext_base':      _cfg('convnext', 'features[5]',
                                'ConvNeXt Base (stage 3, 512ch)'),
}

# fill loader_name (default == registry key)
for _k, _v in MODEL_CONFIGS.items():
    if _v['loader_name'] is None:
        _v['loader_name'] = _k


def get_adapter(model_name: str) -> BackboneAdapter:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model '{model_name}'. "
                          f"Registered: {sorted(MODEL_CONFIGS.keys())}.")
    family = MODEL_CONFIGS[model_name]['family']
    return _ADAPTERS[family]


def load_backbone(model_name: str) -> nn.Module:
    return _load(MODEL_CONFIGS[model_name]['loader_name'])


# ======================================================================
# BatchNorm folding: conv (+ following BN) -> single linear operator
# ======================================================================

def fold_conv_bn(conv: nn.Module,
                 bn: Optional[nn.BatchNorm2d]) -> Tuple[torch.Tensor,
                                                         torch.Tensor]:
    """Fold a Conv2d (or 1x1 Linear surrogate) followed by a BatchNorm2d
    into one affine operator.

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
    bc = (conv.bias.detach().clone() if getattr(conv, 'bias', None) is not None
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

    For non-residual / non-VGG architectures the adapter returns the
    appropriate "last linear map producing the layer's output" (DenseNet:
    the last DenseLayer's conv2; MobileNet/EfficientNet: the project conv
    + project BN; ConvNeXt: the project Linear).

    Returns
    -------
    W            : [C_out, in_dim]   effective linear operator
    block_kind   : 'projection' | 'identity' | 'plain' | 'concat'  (Remark 1)
    info         : dict of diagnostic dimensions
    """
    adapter = get_adapter(model_name)
    conv, bn, block_kind, info = adapter.find_operator(
        backbone, target_layer_name)
    W, _ = fold_conv_bn(conv, bn)
    info = dict(info)
    info['W_shape'] = tuple(W.shape)
    info['block_kind'] = block_kind
    info['has_folded_bn'] = bn is not None
    return W, block_kind, info


# ======================================================================
# BatchNorm mean read-off  (offset mu)
# ======================================================================

def get_bn_stats(backbone: nn.Module, model_name: str,
                 target_layer_name: str, C_out: int) -> torch.Tensor:
    """Read the BatchNorm running MEAN that describes the target layer's
    OUTPUT channels -- used as the offset mu (Section B, a proxy for the
    post-activation channel mean). Theorem 1 holds for any mu; the proxy
    only repartitions mass between offset and components.

    Falls back to zeros (i.e. mu = 0) when the architecture has no BN on
    the output channel (DenseNet's growth_rate conv, ConvNeXt's project
    Linear). This is still a valid choice in Theorem 1; the components
    simply absorb the missing channel-mean mass.
    """
    dev = next(backbone.parameters()).device
    bn = get_adapter(model_name).find_output_bn(backbone, target_layer_name)
    if bn is None:
        return torch.zeros(C_out, device=dev)
    mu = bn.running_mean.detach().clone().to(dev)
    if mu.numel() != C_out:
        return torch.zeros(C_out, device=dev)
    return mu


def get_upstream_bn_var(backbone: nn.Module, model_name: str,
                         target_layer_name: str, in_dim: int) -> torch.Tensor:
    """Running variance of the BN feeding the folded operator's INPUT.

    This is Definition 1's "upstream BatchNorm running variances feeding the
    branch" -- the per-input-channel scale on the INPUT-patch axis, used as a
    diagonal proxy for H_in in  Sigma = W H_in W^T.

    Per-architecture mapping (see adapters):
      ResNet50/101/152/wide/ResNeXt (Bottleneck):  last_block.bn2  (length = bottleneck width)
      ResNet18/34 (BasicBlock)                  :  last_block.bn1
      VGG-BN                                    :  previous BN in features (walking back)
      VGG plain                                 :  none -> white fallback
      DenseNet (DenseLayer tail)                :  last_layer.norm2
      MobileNetV2/V3, EfficientNet              :  depthwise BN of the last block
      ConvNeXt                                  :  none (LayerNorm only) -> white fallback

    The returned vector is broadcast to length `in_dim`. For a 1x1 conv
    in_dim == C_in exactly. For a kxk conv in_dim == C_in * kh * kw and we
    tile the per-channel variance across the kh*kw spatial taps (the BN
    variance is per channel, shared across taps -- the honest diagonal
    proxy).

    'White fallback': when no upstream BN is available, returns ones, so
    'bn' degenerates gracefully to 'kernel'.
    """
    dev = next(backbone.parameters()).device
    bn = get_adapter(model_name).find_upstream_bn(backbone, target_layer_name)
    if bn is None:
        return torch.ones(in_dim, device=dev)

    h_in_ch = bn.running_var.detach().clone().to(dev)
    C_in = h_in_ch.numel()
    if in_dim == C_in:
        return h_in_ch
    if in_dim % C_in == 0:
        taps = in_dim // C_in
        return h_in_ch.repeat_interleave(taps)
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
    """BN-tilted basis (Definition 1): top-D eigenspace of
    W diag(h_in) W^T, with h_in the UPSTREAM BN running variances on the
    INPUT-patch axis -- the diagonal proxy for H_in in  Sigma = W H_in W^T.

    The left singular vectors of  M = W diag(sqrt h_in)  are exactly the
    eigenvectors of  M M^T = W diag(h_in) W^T, and are ALREADY orthonormal
    in R^{C_out}. So tilting the input axis does not move the frame off the
    channel space, it only re-orients within it.
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

    Other speed considerations:
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
    if n_bn == 0:
        # ConvNeXt and any other LN-only architecture: nothing to match.
        # We fall back to plain Gaussian images run through the network.
        print("  [distill] WARNING: no BatchNorm2d found in backbone; "
              "synthesis reduces to passing white-noise images through the "
              "forward path. Consider whether 'kernel' is preferable.")
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
    target_module = get_adapter(model_name).get_target_module(
        backbone, target_layer_name)

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
        # augmentation (flip + random subset), to extract many cells without
        # paying for more optimisation.
        t0 = time.time()
        x_pool = x.detach()
        with torch.no_grad():
            for hp in range(harvest_passes):
                if harvest_bs <= batch_size:
                    idx_pool = torch.randperm(batch_size,
                                              device=device)[:harvest_bs]
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
                       distill_batches: int = 2, distill_bs: int = 32,
                       distill_iters: int = 400, distill_lr: float = 0.1,
                       distill_seed: int = 0,
                       distill_harvest_passes: int = 8,
                       distill_harvest_bs: int = 64,
                       distill_bn_stride: int = 1,
                       distill_use_amp: bool = True,
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
    backbone = load_backbone(model_name).to(dev).eval()

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
        if h_ratio < 1.0 + 1e-6:
            print(f"  Upstream BN tilt h_in: len={h_in.numel()}  "
                  f"max/min ratio={h_ratio:.2f}  -> WHITE (no real BN "
                  f"upstream for this architecture); 'bn' will match 'kernel'.")
        else:
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
            seed=distill_seed,
            harvest_passes=distill_harvest_passes,
            harvest_bs=distill_harvest_bs,
            bn_stride=distill_bn_stride,
            use_amp=distill_use_amp)
        sv = None
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
    ap.add_argument('--distill_harvest_passes', type=int, default=8,
                    help='Forward-only harvest passes per synthesised batch.')
    ap.add_argument('--distill_harvest_bs', type=int, default=64,
                    help='Batch size for each harvest pass.')
    ap.add_argument('--distill_bn_stride', type=int, default=1,
                    help='Match every k-th BN layer during synthesis.')
    ap.add_argument('--no_amp', action='store_true',
                    help='Disable mixed-precision synthesis (debug only).')
    ap.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--output', type=str, default=None,
                    help='Output .pkl path for the DataFreeReconstructor.')
    ap.add_argument('--list_models', action='store_true',
                    help='Print all registered models and exit.')
    args = ap.parse_args()

    if args.list_models:
        print("Registered backbones (model -> default target layer  | description):")
        for k in sorted(MODEL_CONFIGS.keys()):
            v = MODEL_CONFIGS[k]
            print(f"  {k:24s} -> {v['default_target_layer']:20s} | "
                  f"{v['description']}")
        return

    if args.target_layer is None:
        args.target_layer = MODEL_CONFIGS[args.model]['default_target_layer']
    if args.output is None:
        safe_target = (args.target_layer
                       .replace('[', '').replace(']', '')
                       .replace('.', '_'))
        args.output = (f"df_basis_{args.model}_{safe_target}"
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
        distill_seed=args.distill_seed,
        distill_harvest_passes=args.distill_harvest_passes,
        distill_harvest_bs=args.distill_harvest_bs,
        distill_bn_stride=args.distill_bn_stride,
        distill_use_amp=not args.no_amp,
        device=args.device)

    joblib.dump(recon, args.output)
    print(f"\n  {recon}")
    print(f"  Saved basis -> {args.output}")
    print("=" * 78)
    print("Done. Feed this .pkl to df_hier_visualization.py via --df_basis.")
    print("=" * 78)


if __name__ == '__main__':
    main()