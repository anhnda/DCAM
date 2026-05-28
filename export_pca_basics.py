"""
export_pca_basics.py
====================
Given (model, target_layer, D), compute the per-cell CHANNEL COVARIANCE
Sigma of the target layer and its PCA decomposition -- WITHOUT ever writing
the full activation cache to disk.

This fuses the two halves of the old pipeline:

    export_activation_cache.py   (forward pass, per-channel 0.99 normalize)
            +
    csae_pca_baseline.py         (per-cell mean/cov -> eigh -> PCAReconstructor)

...into one streaming pass over ImageNet. The old flow materialized ~50k
activation maps [N, C, H, W] on disk, then read them all back to form a C x C
covariance. That round-trip is pure overhead: covariance is a STREAMING
statistic. We accumulate it online and throw every batch away.

WHY THIS IS FASTER / LIGHTER
----------------------------
Per-cell channel covariance needs only three running accumulators, no matter
how many images you feed it:

    n  : scalar          number of spatial cells seen   (= images * H * W)
    s  : R^C             sum of cell vectors  sum_i x_i
    S  : R^{C x C}       sum of outer products  sum_i x_i x_i^T

Then
    mu    = s / n
    Sigma = S / n - mu mu^T            (population per-cell covariance)

Memory is O(C^2) TOTAL (resnet50 layer3: 1024^2 float64 ~= 8 MB), independent
of N. No activation cache, no chunk files, no re-read. GradCAM is never
computed -- it's irrelevant to the covariance -- so we run the backbone in
full batches (not batch-of-1), which is the other big speedup.

THE ONE GLOBAL STATISTIC: per-channel 0.99 normalization
--------------------------------------------------------
The original cache normalized each channel by its 99th percentile over all
non-zero activations, then clamped to [0, scale] and divided. A percentile is
NOT a streaming sum, so to MATCH the original normalization we do two passes:

  PASS 1  build a per-channel histogram of non-zero activations (bounded
          memory: C x n_bins counts) and read off the 0.99 quantile per
          channel. This reproduces torch.quantile(., 0.99) to histogram
          resolution -- "exact-ish".
  PASS 2  re-run the forward pass; for each batch apply the SAME clamp+divide
          the cache used, then fold the normalized cells into (n, s, S).

Two forward passes over the data, zero activation bytes on disk. If you don't
care about exact parity, --fast_norm estimates the scales from a warm-up batch
in a single pass (see flag help). --no_normalize skips it entirely (raw Sigma).

OUTPUTS (mirrors csae_pca_baseline.py, plus Sigma)
--------------------------------------------------
  --save        PCAReconstructor .pkl   (drop-in for check_drop_csae_fixed.py;
                                          identical class/interface to the old
                                          csae_pca_baseline output)
  --save_cov    Sigma .npz in a SEPARATE file, containing:
                    cov          [C, C]  per-cell channel covariance
                    mean         [C]     per-cell channel mean (mu)
                    eigvals      [C]     eigenvalues, descending
                    eigvecs      [C, C]  eigenvectors (columns), descending
                    channel_scale[C]     the 0.99 normalization scale used
                    meta         dict    model/layer/D/N/evr/...

The .pkl is a run_xcsae_full.MultiChannelConvSAE subclass instance, so it must
be loadable: keep this file importable as `export_pca_basics` (joblib stores
the qualified class name export_pca_basics.PCAReconstructor) in the directory
you run check_drop_csae_fixed.py from.

USAGE
-----
  # resnet50 layer3, rank-200 PCA, streamed (no cache), exact-ish normalize
  python export_pca_basics.py \
      --model resnet50 --target_layer layer3 --D 200 \
      --images_per_class 50 --device cuda \
      --save pca_baseline_resnet50_D200_model.pkl \
      --save_cov cov_resnet50_layer3_D200.npz

  # then evaluate exactly like the old CSAE / PCA baseline
  python check_drop_csae_fixed.py --model resnet50 \
      --csae_model pca_baseline_resnet50_D200_model.pkl \
      --norm_mode per_image --debug_batches 3

  # sweep D from a SINGLE pass: --D_list reuses one streamed Sigma for all D
  python export_pca_basics.py --model resnet50 --target_layer layer3 \
      --D_list 50,100,200,400 --device cuda \
      --save_prefix pca_baseline_resnet50

  # D=0 is the MEAN-ONLY control (every cell -> mu, no projection)
  python export_pca_basics.py --model resnet50 --D 0 \
      --save pca_meanonly_resnet50_model.pkl
"""

import argparse
import io
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import joblib
from tqdm import tqdm
import pandas as pd

# Must import the same base class so the saved .pkl is loadable by
# check_drop_csae_fixed.py (joblib stores the parent's module path too).
sys.path.append('.')
from run_xcsae_full import MultiChannelConvSAE
from full_classes import IMAGENET2012_CLASSES


# ==========================================
# Configuration (matches export_activation_cache.py defaults)
# ==========================================

IMAGENET_RAW_DIR = Path("/data/imagenet_raw/data")
IMAGENET_SAMPLED_DIR = Path("/data/imagenet1k_sampled")

IMAGES_PER_CLASS = 50
NUM_CLASSES = 1000

BATCH_SIZE_COLLECTION = 32
DEFAULT_DATA_SEED = 42

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
# Reproducibility (verbatim from export_activation_cache.py)
# ==========================================

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"  [seed] RNGs seeded with {seed} "
          f"(cudnn deterministic={'on' if deterministic else 'off'})")


# ==========================================
# ImageNet-1k Dataset Sampler
# (verbatim from export_activation_cache.py)
# ==========================================

class ImageNet1kSampledDataset(Dataset):
    """Loads sampled ImageNet-1k images from parquet files."""

    def __init__(self, raw_dir: Path, sampled_dir: Path,
                 images_per_class: int = 50, transform=None,
                 force_resample: bool = False):
        self.raw_dir = raw_dir
        self.sampled_dir = sampled_dir
        self.images_per_class = images_per_class
        self.transform = transform

        self.sampled_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.sampled_dir / "metadata.pkl"

        if self.metadata_path.exists() and not force_resample:
            print(f"\n{'='*80}")
            print(f"Loading cached sampled dataset from {self.sampled_dir}")
            print(f"{'='*80}")
            self.load_cached_dataset()
        else:
            print(f"\n{'='*80}")
            print(f"Creating new sampled dataset...")
            print(f"{'='*80}")
            self.create_sampled_dataset()

    def create_sampled_dataset(self):
        print(f"Sampling {self.images_per_class} images per class from "
              f"{NUM_CLASSES} classes...")
        print(f"Total target images: "
              f"{self.images_per_class * NUM_CLASSES}")

        self.wnid_to_idx = {wnid: idx for idx, wnid in
                            enumerate(IMAGENET2012_CLASSES.keys())}
        self.idx_to_wnid = {idx: wnid for wnid, idx in
                            self.wnid_to_idx.items()}

        class_samples = defaultdict(list)
        train_parquet_files = sorted(self.raw_dir.glob("train-*.parquet"))

        if len(train_parquet_files) == 0:
            raise FileNotFoundError(
                f"No train parquet files found in {self.raw_dir}")

        print(f"Found {len(train_parquet_files)} train parquet files")

        for parquet_file in tqdm(train_parquet_files,
                                 desc="Reading parquet files"):
            df = pd.read_parquet(parquet_file)
            for idx, row in df.iterrows():
                label = row['label']
                if len(class_samples[label]) < self.images_per_class:
                    image_bytes = row['image']['bytes']
                    class_samples[label].append((image_bytes, label))

            min_samples = min(len(s) for s in class_samples.values())
            if (min_samples >= self.images_per_class and
                    len(class_samples) == NUM_CLASSES):
                print(f"\nCollected {self.images_per_class} samples for "
                      f"all {NUM_CLASSES} classes!")
                break

        self.samples = []
        for class_idx in range(NUM_CLASSES):
            if len(class_samples[class_idx]) >= self.images_per_class:
                sampled = random.sample(class_samples[class_idx],
                                        self.images_per_class)
                self.samples.extend(sampled)
            else:
                print(f"WARNING: Class {class_idx} has only "
                      f"{len(class_samples[class_idx])} samples")
                self.samples.extend(class_samples[class_idx])

        print(f"\nTotal sampled images: {len(self.samples)}")
        print(f"Saving sampled dataset to {self.sampled_dir}...")
        joblib.dump({
            'samples': self.samples,
            'images_per_class': self.images_per_class,
            'num_classes': NUM_CLASSES,
            'wnid_to_idx': self.wnid_to_idx,
            'idx_to_wnid': self.idx_to_wnid,
        }, self.metadata_path)
        print(f"Sampled dataset cached!")

    def load_cached_dataset(self):
        metadata = joblib.load(self.metadata_path)
        self.samples = metadata['samples']
        self.wnid_to_idx = metadata['wnid_to_idx']
        self.idx_to_wnid = metadata['idx_to_wnid']
        print(f"Loaded {len(self.samples)} images from cache")
        print(f"  Images per class: {metadata['images_per_class']}")
        print(f"  Number of classes: {metadata['num_classes']}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_bytes, label = self.samples[idx]
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ==========================================
# PCAReconstructor -- IDENTICAL semantics to csae_pca_baseline.py
# (kept verbatim so the saved .pkl behaves the same in the eval)
# ==========================================

class PCAReconstructor(MultiChannelConvSAE):
    """Deterministic rank-D PCA reconstruction wearing the CSAE interface.

    forward(x) returns (x_hat, z) where
        z      = V^T (x - mu)         per-cell PCA coefficients [B, D, H, W]
        x_hat  = mu + V z             per-cell reconstruction   [B, C, H, W]
    z is a DENSE projection (not sparse); the eval's "active units" stat is
    meaningless here -- read avg_relerr_normalized.
    """

    def __init__(self, in_channels: int, D: int,
                 mu: torch.Tensor, V: torch.Tensor):
        D = int(D)
        hd = max(D, 1)
        super().__init__(in_channels=in_channels, hidden_dim=hd,
                         kernel_size=1, top_k=hd)
        self.pca_rank = D
        self.register_buffer('pca_mu', mu.view(in_channels).contiguous())

        if D > 0:
            self.register_buffer('pca_V', V.contiguous())        # [C, D]
            with torch.no_grad():
                self.encoder.weight.copy_(V.T.view(D, in_channels, 1, 1))
                if self.encoder.bias is not None:
                    self.encoder.bias.zero_()
                self.decoder.weight.copy_(V.view(in_channels, D, 1, 1))
        else:
            # mean-only control: no basis.
            self.register_buffer('pca_V', torch.zeros(in_channels, 0))
            with torch.no_grad():
                self.encoder.weight.zero_()
                if self.encoder.bias is not None:
                    self.encoder.bias.zero_()
                self.decoder.weight.zero_()

    def forward(self, x: torch.Tensor, use_topk: bool = True):
        B, C, H, W = x.shape
        if self.pca_rank > 0:
            xc = x.permute(0, 2, 3, 1).reshape(-1, C)        # [N, C]
            centered = xc - self.pca_mu                      # [N, C]
            coeff = centered @ self.pca_V                    # [N, D]
            recon = coeff @ self.pca_V.T + self.pca_mu       # [N, C]
            recon = recon.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z = coeff.view(B, H, W, self.pca_rank).permute(
                0, 3, 1, 2).contiguous()
        else:
            recon = self.pca_mu.view(1, C, 1, 1).expand(
                B, C, H, W).contiguous()
            z = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        return recon, z


# ==========================================
# Backbone forward-to-layer (verbatim logic from
# export_activation_cache.py:ActivationExtractor)
# ==========================================

class LayerActivationStreamer:
    """Loads a backbone, hooks the target layer, and yields normalized
    activation batches [B, C, H, W] one DataLoader batch at a time. Never
    stores them. GradCAM is never built.
    """

    def __init__(self, model_name='resnet50', target_layer=None,
                 device='cuda'):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. "
                             f"Choose from {list(MODEL_CONFIGS.keys())}")
        config = MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self.device = device
        self.target_layer_name = (target_layer if target_layer
                                  else config['default_target_layer'])

        print(f"\n{'='*80}")
        print(f"Initializing {model_name.upper()} layer streamer "
              f"(activations only, no GradCAM, no cache)")
        print(f"{'='*80}")
        print(f"Model: {config['description']}")
        print(f"Target layer: {self.target_layer_name}")

        self.model = config['model_fn']().to(device)
        self.model.eval()
        self.target_layer = self._get_layer_by_name(self.target_layer_name)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            out = self._forward_to_target_layer(dummy)
            self.num_channels = out.shape[1]
            self.spatial_size = out.shape[2]

        print(f"  Output channels: {self.num_channels}")
        print(f"  Spatial resolution: "
              f"{self.spatial_size}x{self.spatial_size}")

        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        print(f"Streamer ready!")

    def _get_layer_by_name(self, layer_name: str):
        if '[' in layer_name:
            parts = layer_name.split('[')
            attr_name = parts[0]
            index = int(parts[1].rstrip(']'))
            return getattr(self.model, attr_name)[index]
        return getattr(self.model, layer_name)

    def _forward_to_target_layer(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name in ['resnet50', 'resnet18']:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            if 'layer1' in self.target_layer_name:
                return x
            x = self.model.layer2(x)
            if 'layer2' in self.target_layer_name:
                return x
            x = self.model.layer3(x)
            if 'layer3' in self.target_layer_name:
                return x
            x = self.model.layer4(x)
            return x
        elif self.model_name == 'vgg16':
            target_idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            for i in range(target_idx + 1):
                x = self.model.features[i](x)
            return x
        elif self.model_name == 'efficientnet':
            target_idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            for i in range(target_idx + 1):
                x = self.model.features[i](x)
            return x
        return x

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    @torch.no_grad()
    def iter_batches(self, data_loader: DataLoader, desc="forward"):
        """Yield raw (un-normalized) activation batches [B, C, H, W] on GPU."""
        for images, _labels in tqdm(data_loader, desc=desc):
            batch = images.to(self.device)
            _ = self.model(batch)
            yield self.activations           # [B, C, H, W], on device


# ==========================================
# Streaming per-channel 0.99 normalization scale (PASS 1)
# ==========================================

class PerChannelQuantile:
    """Histogram-based per-channel quantile estimator.

    Reproduces the cache's normalization scale: torch.quantile(non_zero, 0.99)
    per channel, where non_zero = activations > 1e-8. A percentile can't be
    streamed as a running sum, so we bin non-zero activations per channel into
    a fixed-range histogram and read the 0.99 cut from the cumulative counts.

    Range is auto-calibrated from a warm-up batch (per-channel running max), so
    you don't need to know the activation scale in advance. With enough bins
    (default 2048) the estimate matches torch.quantile to bin resolution.
    """

    def __init__(self, C: int, n_bins: int = 2048, device='cpu'):
        self.C = C
        self.n_bins = n_bins
        self.device = device
        self.vmax = torch.zeros(C, device=device)            # per-channel max
        self.counts = torch.zeros(C, n_bins, dtype=torch.float64,
                                  device=device)
        self._calibrated = False

    def calibrate(self, vmax: torch.Tensor):
        # add 1% headroom so the true max lands inside the last bin
        self.vmax = vmax.clamp_min(1e-8) * 1.01
        self._calibrated = True

    @torch.no_grad()
    def update(self, acts: torch.Tensor):
        # acts: [B, C, H, W]
        C = acts.shape[1]
        x = acts.permute(1, 0, 2, 3).reshape(C, -1)          # [C, M]
        for c in range(C):
            col = x[c]
            col = col[col > 1e-8]
            if col.numel() == 0:
                continue
            edges = self.vmax[c] / self.n_bins
            idx = torch.clamp((col / edges).long(), 0, self.n_bins - 1)
            self.counts[c].index_add_(
                0, idx, torch.ones_like(idx, dtype=torch.float64))

    def quantile(self, q: float = 0.99) -> torch.Tensor:
        """Return per-channel q-quantile [C]. Channels with no non-zero
        activations get scale 0 (handled as 'leave unchanged' downstream)."""
        scales = torch.zeros(self.C, device=self.device)
        bin_w = self.vmax / self.n_bins                      # [C]
        total = self.counts.sum(dim=1)                       # [C]
        cum = torch.cumsum(self.counts, dim=1)               # [C, n_bins]
        for c in range(self.C):
            if total[c] <= 0:
                continue
            target = q * total[c]
            b = int(torch.searchsorted(cum[c], target).item())
            b = min(b, self.n_bins - 1)
            # upper edge of the bin that crosses the quantile
            scales[c] = (b + 1) * bin_w[c]
        return scales


def compute_channel_scales(streamer, data_loader, n_bins=2048,
                           device='cpu') -> torch.Tensor:
    """PASS 1: estimate per-channel 0.99 normalization scales by streaming."""
    C = streamer.num_channels
    pcq = PerChannelQuantile(C, n_bins=n_bins, device=device)

    # warm-up: one batch to calibrate histogram range (per-channel max)
    vmax = torch.zeros(C, device=device)
    calibrated = False
    for acts in streamer.iter_batches(data_loader, desc="pass1/calibrate"):
        a = acts.detach().to(device)
        cur = a.permute(1, 0, 2, 3).reshape(C, -1).amax(dim=1)
        vmax = torch.maximum(vmax, cur)
        if not calibrated:
            pcq.calibrate(vmax)
            calibrated = True
        pcq.update(a)
    scales = pcq.quantile(0.99)
    print(f"  [pass1] per-channel 0.99 scales: "
          f"min={scales[scales>0].min().item():.4g} "
          f"max={scales.max().item():.4g} "
          f"({int((scales>0).sum().item())}/{C} channels non-degenerate)")
    return scales


def normalize_batch(acts: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Apply the SAME clamp+divide the cache used, per channel:
        x = clamp(x, 0, scale) / (scale + 1e-8)
    Channels with scale<=1e-8 are left unchanged (matches the cache's guard).
    """
    C = acts.shape[1]
    out = acts.clone()
    sc = scales.view(1, C, 1, 1)
    valid = (scales > 1e-8).view(1, C, 1, 1)
    clamped = torch.clamp(acts, min=0.0)
    clamped = torch.minimum(clamped, sc.clamp_min(1e-8))
    normed = clamped / (sc + 1e-8)
    out = torch.where(valid, normed, acts)
    return out


# ==========================================
# Streaming covariance (PASS 2)
# ==========================================

class StreamingCovariance:
    """Online per-cell channel covariance via running (n, s, S).

        n : cells seen
        s : sum_i x_i            [C]
        S : sum_i x_i x_i^T      [C, C]
        mu    = s / n
        Sigma = S / n - mu mu^T
    O(C^2) memory total. Accumulated in float64 for numerical stability.
    """

    def __init__(self, C: int, device='cpu', dtype=torch.float64):
        self.C = C
        self.device = device
        self.dtype = dtype
        self.n = 0
        self.s = torch.zeros(C, dtype=dtype, device=device)
        self.S = torch.zeros(C, C, dtype=dtype, device=device)

    @torch.no_grad()
    def update(self, acts: torch.Tensor):
        # acts: [B, C, H, W] -> cells [M, C]
        C = acts.shape[1]
        x = acts.permute(0, 2, 3, 1).reshape(-1, C).to(self.dtype)
        self.n += x.shape[0]
        self.s += x.sum(dim=0)
        self.S += x.T @ x

    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n == 0:
            raise RuntimeError("StreamingCovariance saw no cells.")
        mu = self.s / self.n                                 # [C]
        cov = self.S / self.n - torch.outer(mu, mu)          # [C, C]
        cov = 0.5 * (cov + cov.T)                            # symmetrize
        return mu, cov


# ==========================================
# Build mu, Sigma, eigendecomposition (streamed)
# ==========================================

def stream_mean_and_cov(streamer, data_loader, scales: Optional[torch.Tensor],
                        device='cpu', dtype=torch.float64,
                        fast_norm_warmup: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """PASS 2: stream the (optionally normalized) activations into mu, Sigma."""
    C = streamer.num_channels
    cov_acc = StreamingCovariance(C, device=device, dtype=dtype)
    for acts in streamer.iter_batches(data_loader, desc="pass2/covariance"):
        a = acts.detach().to(device)
        if scales is not None:
            a = normalize_batch(a, scales.to(device))
        cov_acc.update(a)
    mu, cov = cov_acc.finalize()
    return mu, cov


def eigendecompose(cov: torch.Tensor, D: int):
    """Symmetric eigendecomposition, descending. Returns (w_desc, V_desc, evr@D,
    cumulative-variance report)."""
    C = cov.shape[0]
    w, V = torch.linalg.eigh(cov)            # ascending
    w = w.flip(0).clamp_min(0.0)             # descending eigenvalues
    V = V.flip(1)                            # columns = eigenvectors, descending
    total_var = float(w.sum().item())
    D = max(0, min(int(D), C))
    evr = (float(w[:D].sum().item()) / max(total_var, 1e-12)) if D > 0 else 0.0
    return w, V, total_var, evr


def report_variance(w: torch.Tensor, total_var: float, C: int, D: int,
                    evr: float):
    print(f"\n{'='*64}")
    if D == 0:
        print(f"  MEAN-ONLY baseline (D=0): every cell reconstructed as mu, "
              f"no projection.  C={C}")
        print(f"  CONTROL: high accuracy here => the result is just the mean "
              f"pattern classifying (eval artifact); near-chance => the "
              f"projection does the real work.")
    else:
        print(f"  PCA basis: C={C}, D={D}")
        print(f"  eigval range (top-D) [{w[D-1]:.4e}, {w[0]:.4e}]")
        print(f"  EXPLAINED VARIANCE at D={D}: {evr:.4f} "
              f"({100*evr:.1f}% of per-cell variance)")
    cum = torch.cumsum(w, dim=0) / max(total_var, 1e-12)
    for thr in (0.90, 0.95, 0.99):
        k = int(torch.searchsorted(
            cum, torch.tensor(thr, dtype=cum.dtype, device=cum.device)
        ).item()) + 1
        print(f"    {int(thr*100)}% variance reached at rank {k}")
    print(f"{'='*64}")


def make_reconstructor(C: int, D: int, mu: torch.Tensor,
                       V: torch.Tensor) -> PCAReconstructor:
    D = max(0, min(int(D), C))
    Vk = V[:, :D].to(torch.float32).cpu() if D > 0 else torch.zeros(C, 0)
    mu32 = mu.to(torch.float32).cpu()
    return PCAReconstructor(in_channels=C, D=D, mu=mu32, V=Vk).cpu().eval()


def save_covariance_npz(path: str, cov: torch.Tensor, mu: torch.Tensor,
                        w: torch.Tensor, V: torch.Tensor,
                        scales: Optional[torch.Tensor], meta: Dict):
    np.savez_compressed(
        path,
        cov=cov.double().cpu().numpy(),
        mean=mu.double().cpu().numpy(),
        eigvals=w.double().cpu().numpy(),
        eigvecs=V.double().cpu().numpy(),
        channel_scale=(scales.double().cpu().numpy()
                       if scales is not None else np.array([])),
        meta=np.array([meta], dtype=object),
    )


# ==========================================
# CLI
# ==========================================

def main():
    ap = argparse.ArgumentParser(
        description="Streamed per-cell channel covariance Sigma + PCA "
                    "decomposition for a backbone target layer, with NO "
                    "activation cache on disk.")
    ap.add_argument('--model', type=str, default='resnet50',
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--target_layer', type=str, default=None,
                    help="Defaults to the model's default target layer.")
    ap.add_argument('--D', type=int, default=200,
                    help="PCA rank for the saved reconstructor. D=0 is the "
                         "MEAN-ONLY control.")
    ap.add_argument('--D_list', type=str, default=None,
                    help="Comma-separated ranks (e.g. 50,100,200,400). When "
                         "set, ONE streamed Sigma is reused to emit a "
                         "reconstructor per D. Overrides --D for the .pkl "
                         "outputs; uses --save_prefix for naming.")
    ap.add_argument('--images_per_class', type=int, default=IMAGES_PER_CLASS)
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE_COLLECTION)
    ap.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--accum_device', type=str, default='cpu',
                    choices=['cpu', 'cuda'],
                    help="Where to accumulate the C x C covariance. 'cpu' "
                         "keeps GPU memory free; 'cuda' is faster if C is "
                         "large and VRAM allows.")
    ap.add_argument('--dtype', type=str, default='float64',
                    choices=['float64', 'float32'])
    ap.add_argument('--n_bins', type=int, default=2048,
                    help="Histogram bins for the per-channel 0.99 quantile "
                         "(pass 1). More bins => closer to torch.quantile.")

    norm_group = ap.add_mutually_exclusive_group()
    norm_group.add_argument(
        '--normalize', dest='norm_mode', action='store_const', const='exact',
        help="Two-pass exact-ish per-channel 0.99 normalization (DEFAULT, "
             "matches the old cache).")
    norm_group.add_argument(
        '--fast_norm', dest='norm_mode', action='store_const', const='fast',
        help="One-pass approx: estimate 0.99 scales from the FIRST batch only, "
             "then stream Sigma in the same pass. Faster, slightly off parity.")
    norm_group.add_argument(
        '--no_normalize', dest='norm_mode', action='store_const', const='none',
        help="Skip normalization; raw-activation covariance.")
    ap.set_defaults(norm_mode='exact')

    ap.add_argument('--force_resample', action='store_true')
    ap.add_argument('--data_seed', type=int, default=DEFAULT_DATA_SEED)

    ap.add_argument('--save', type=str, default='pca_basics_model.pkl',
                    help="PCAReconstructor .pkl output (single --D mode).")
    ap.add_argument('--save_prefix', type=str, default='pca_basics',
                    help="Prefix for per-D .pkl files in --D_list mode.")
    ap.add_argument('--save_cov', type=str, default='pca_basics_cov.npz',
                    help="SEPARATE Sigma file (.npz): cov, mean, eigvals, "
                         "eigvecs, channel_scale, meta.")
    args = ap.parse_args()

    if args.device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(args.device)
    accum_dev = torch.device(args.accum_device
                             if torch.cuda.is_available() or
                             args.accum_device == 'cpu' else 'cpu')
    dt = torch.float64 if args.dtype == 'float64' else torch.float32

    print("=" * 80)
    print("export_pca_basics: streamed Sigma + PCA (no activation cache)")
    print(f"Backbone: {args.model.upper()}  device={dev}  "
          f"accum_device={accum_dev}  norm={args.norm_mode}")
    print("=" * 80)

    set_seed(args.data_seed)

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageNet1kSampledDataset(
        raw_dir=IMAGENET_RAW_DIR,
        sampled_dir=IMAGENET_SAMPLED_DIR,
        images_per_class=args.images_per_class,
        transform=data_transform,
        force_resample=args.force_resample,
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    streamer = LayerActivationStreamer(
        model_name=args.model, target_layer=args.target_layer, device=dev)
    C = streamer.num_channels

    # ---- normalization scales ----
    scales = None
    if args.norm_mode == 'exact':
        print(f"\n[pass 1/2] estimating per-channel 0.99 scales "
              f"(histogram, {args.n_bins} bins)...")
        scales = compute_channel_scales(
            streamer, data_loader, n_bins=args.n_bins, device=accum_dev)
    elif args.norm_mode == 'fast':
        print(f"\n[fast_norm] estimating 0.99 scales from the first batch...")
        first = next(streamer.iter_batches(data_loader, desc="warmup"))
        a0 = first.detach().to(accum_dev)
        xc = a0.permute(1, 0, 2, 3).reshape(C, -1)
        sc = torch.zeros(C, device=accum_dev)
        for c in range(C):
            col = xc[c]
            col = col[col > 1e-8]
            if col.numel() > 0:
                sc[c] = torch.quantile(col, 0.99)
        scales = sc
        print(f"  [fast_norm] scales from {a0.shape[0]} images "
              f"({int((sc>0).sum().item())}/{C} non-degenerate)")

    # ---- stream Sigma (pass 2, or the only pass for fast/none) ----
    print(f"\n[pass 2/2] streaming per-cell covariance...")
    mu, cov = stream_mean_and_cov(
        streamer, data_loader, scales=scales,
        device=accum_dev, dtype=dt)
    print(f"  Sigma: {tuple(cov.shape)}  cells seen via streaming "
          f"(mu norm={mu.norm().item():.4g})")

    # ---- eigendecomposition ----
    w, V, total_var, evr_main = eigendecompose(cov, args.D)

    # ---- determine D set ----
    if args.D_list:
        d_set = [int(x) for x in args.D_list.split(',') if x.strip() != '']
    else:
        d_set = [args.D]

    # ---- save Sigma (separate file) ----
    meta = {
        'model': args.model,
        'target_layer': streamer.target_layer_name,
        'C': C,
        'spatial_size': streamer.spatial_size,
        'images_per_class': args.images_per_class,
        'total_variance': total_var,
        'norm_mode': args.norm_mode,
        'n_bins': args.n_bins if args.norm_mode == 'exact' else None,
        'D_main': args.D,
        'D_list': d_set,
        'explained_variance_ratio_D_main': evr_main,
    }
    save_covariance_npz(args.save_cov, cov, mu, w, V, scales, meta)
    print(f"\nSaved covariance + eigendecomposition to {args.save_cov}")
    print(f"  keys: cov[{C},{C}], mean[{C}], eigvals[{C}], eigvecs[{C},{C}], "
          f"channel_scale[{C if scales is not None else 0}], meta")

    # ---- save reconstructor(s) ----
    saved = []
    for D in d_set:
        _, _, _, evr = eigendecompose(cov, D)  # cheap report value
        report_variance(w, total_var, C, D, evr)
        module = make_reconstructor(C, D, mu, V)
        if args.D_list:
            out = f"{args.save_prefix}_{args.model}_D{D}_model.pkl"
        else:
            out = args.save
        joblib.dump(module, out)
        saved.append((D, out, evr))
        print(f"Saved PCAReconstructor (D={D}, evr={evr:.4f}) -> {out}")

    print(f"\n{'='*80}")
    print(f"Done. Evaluate any reconstructor with the SAME fixed eval:")
    for D, out, evr in saved:
        print(f"  python check_drop_csae_fixed.py --model {args.model} \\")
        print(f"      --csae_model {out} --norm_mode per_image")
    print(f"\nCompare acc_reconstructed / avg_relerr_normalized against the "
          f"8192-atom CSAE. If PCA-D matches, overcompleteness was wasted "
          f"capacity. Keep export_pca_basics.py importable so the .pkl loads.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()