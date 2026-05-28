"""
export_fft_basics.py
====================
Per-channel |FFT| (translation-invariant magnitude-spectrum) DESCRIPTOR for a
backbone target layer, plus channel clustering into "basics" -- streamed over
ImageNet, NO activation cache on disk. Sibling of export_pca_basics.py.

WHY |FFT|  (the invariance this buys, and the one it does NOT)
--------------------------------------------------------------
A channel's 14x14 map is the SAME spatial pattern shifted/rotated across images
(object moves around the frame). Averaging the maps in pixel space SMEARS that
pattern into mush. The 2D-FFT magnitude fixes the SHIFT problem EXACTLY: by the
shift theorem a translated map has an identical |FFT| (translation only changes
phase, not magnitude). So we average |FFT| per channel instead of averaging the
map, and translation no longer destroys the signal.

  * |FFT| magnitude  : translation-INVARIANT (exact), rotation-EQUIVARIANT
                       (a rotated map gives a rotated spectrum), shape-lossy.
  * radial profile of : also rotation-INVARIANT (angle integrated out), but
    |FFT|               even more shape-lossy (a vertical vs horizontal edge
                        collapse to the same profile).

We emit BOTH descriptors so you can choose downstream:
  --descriptor full   [C, H*W]   keep orientation, drop translation  (DEFAULT)
  --descriptor radial [C, n_rbin] drop orientation too

NOTE on rotation: full |FFT| is NOT rotation-invariant. If you need that, use
--descriptor radial (cheap, lossy) or escalate to Zernike/GW later. We keep
'full' as the default deliberately: translation is the nuisance you certainly
want gone; orientation usually carries real meaning and you can always merge
orientation groups post-hoc, but you can't un-merge.

THE STREAMING STATISTIC
-----------------------
The per-channel mean |FFT| is a STREAMING sum, exactly like the covariance in
export_pca_basics.py. We never store maps:

    n_c : scalar             #(image,cell-block) contributions for channel c
    P_c : R^{H x W}          running sum of |FFT(map)| for channel c
  -> Pbar_c = P_c / n_c      per-channel mean magnitude spectrum

(Every image contributes ONE |FFT| per channel: we FFT the channel's full HxW
map, not per-cell. n_c is just the image count for that channel.)

Normalization: the SAME per-channel 0.99 clamp+divide as the cache/PCA path,
applied to the MAP before the FFT (ported verbatim from export_pca_basics.py,
two-pass exact-ish; --fast_norm / --no_normalize available).

CLUSTERING  (--k, default 128)
------------------------------
We cluster the C channels on their FFT descriptor into k "basics". Two modes:
  --cluster kmeans   hard k-means on L2-normalized descriptors (DEFAULT;
                     deterministic given --cluster_seed, no overlap).
  --cluster nmf      NMF of the [C, n_feat] descriptor -> overlapping soft
                     membership (a channel can join several basics). Uses
                     init='nndsvda' for determinism.

OUTPUTS  (mirrors export_pca_basics.py so hier_visualize_pca.py can load it)
----------------------------------------------------------------------------
  --save        FFTBasisReconstructor .pkl  -- a MultiChannelConvSAE subclass
                with pca_mu [C] and pca_V [C, k], DROP-IN for
                hier_visualize_pca.py. Each "component" d is one CLUSTER:
                column V[:,d] is that cluster's (L2-normalized) channel
                membership vector, so the viz's z_d = <centered cells, V_d>
                renders "the spatial map of channel-group d" and Ring 2 shows
                its member channels by |V_{d,c}|. pca_mu = per-channel activation
                mean (so 'centered' has the same meaning as the PCA path).

  --save_fft    .npz with the RAW per-channel descriptor + clustering:
                  fft_mean      [C, H, W]   per-channel mean |FFT| magnitude
                  descriptor    [C, n_feat] the clustered feature matrix
                                            (full=[C,H*W] or radial=[C,n_rbin])
                  labels        [C]         hard cluster id per channel (kmeans;
                                            for nmf = argmax of membership)
                  membership    [C, k]      soft membership (nmf) or one-hot
                                            (kmeans)
                  centers       [k, n_feat] cluster centers / NMF components H
                  channel_mean  [C]         per-channel activation mean (mu)
                  channel_scale [C]         the 0.99 normalization scale used
                  meta          dict        model/layer/k/descriptor/...

USAGE
-----
  # resnet50 layer3, k=128, full |FFT| descriptor, hard k-means
  python export_fft_basics.py \
      --model resnet50 --target_layer layer3 --k 128 \
      --images_per_class 50 --device cuda \
      --save fft_basics_resnet50_k128_model.pkl \
      --save_fft fft_resnet50_layer3_k128.npz

  # overlapping basics via NMF, rotation-invariant radial descriptor
  python export_fft_basics.py --model resnet50 --k 64 \
      --cluster nmf --descriptor radial \
      --save fft_basics_resnet50_k64_nmf_model.pkl

  # then visualize with the PCA radial code (UNCHANGED), e.g.
  python hier_visualize_pca.py --class_id 108 \
      --pca_model fft_basics_resnet50_k128_model.pkl \
      --ring_components 12 --top_channels 4 --recon_D 64
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
# hier_visualize_pca.py (joblib stores the parent's module path too). The
# visualizer pickle-imports PCAReconstructor / SparsePCAReconstructor; our
# class only needs to expose pca_mu and pca_V with the same semantics.
sys.path.append('.')
from run_xcsae_full import MultiChannelConvSAE
from full_classes import IMAGENET2012_CLASSES


# ==========================================
# Configuration (matches export_pca_basics.py defaults)
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
# Reproducibility (verbatim from export_pca_basics.py)
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
# (verbatim from export_pca_basics.py)
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
# FFTBasisReconstructor -- wears the PCAReconstructor interface so the
# existing hier_visualize_pca.py loads and renders it UNCHANGED.
# ==========================================

class FFTBasisReconstructor(MultiChannelConvSAE):
    """Cluster "basics" wearing the deterministic-PCA reconstructor interface.

    Same contract as export_pca_basics.PCAReconstructor:
        forward(x) -> (x_hat, z), with
            z      = V^T (x - mu)       per-cell coefficients [B, k, H, W]
            x_hat  = mu + V z           per-cell reconstruction [B, C, H, W]

    The ONLY semantic difference from PCA: the columns of V are NOT eigenvectors
    of a covariance. Each column V[:, d] is cluster d's channel-membership
    vector (L2-normalized), derived from FFT-descriptor clustering. So in the
    visualizer, component d's map z_d = <centered cells, V_d> reads as "the
    activation pattern carried by channel-group d", and Ring 2's top channels by
    |V_{d,c}| are exactly that group's member channels. beta_d (class-specific)
    still works identically.

    V is NOT orthonormal across columns (clusters can share channels under NMF),
    so x_hat is an oblique reconstruction, not an orthogonal projection. That is
    fine for the visualizer (it only uses mu, V, and the z_d / beta_d maps); do
    NOT interpret its recon cosine as a PCA explained-variance.
    """

    def __init__(self, in_channels: int, k: int,
                 mu: torch.Tensor, V: torch.Tensor):
        k = int(k)
        hd = max(k, 1)
        super().__init__(in_channels=in_channels, hidden_dim=hd,
                         kernel_size=1, top_k=hd)
        # Expose the SAME attribute names the visualizer reads off PCA models.
        self.pca_rank = k
        self.register_buffer('pca_mu', mu.view(in_channels).contiguous())

        if k > 0:
            self.register_buffer('pca_V', V.contiguous())        # [C, k]
            with torch.no_grad():
                self.encoder.weight.copy_(V.T.view(k, in_channels, 1, 1))
                if self.encoder.bias is not None:
                    self.encoder.bias.zero_()
                self.decoder.weight.copy_(V.view(in_channels, k, 1, 1))
        else:
            self.register_buffer('pca_V', torch.zeros(in_channels, 0))
            with torch.no_grad():
                self.encoder.weight.zero_()
                if self.encoder.bias is not None:
                    self.encoder.bias.zero_()
                self.decoder.weight.zero_()

    def forward(self, x: torch.Tensor, use_topk: bool = True):
        B, C, H, W = x.shape
        if self.pca_rank > 0:
            xc = x.permute(0, 2, 3, 1).reshape(-1, C)            # [N, C]
            centered = xc - self.pca_mu                          # [N, C]
            coeff = centered @ self.pca_V                        # [N, k]
            recon = coeff @ self.pca_V.T + self.pca_mu           # [N, C]
            recon = recon.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z = coeff.view(B, H, W, self.pca_rank).permute(
                0, 3, 1, 2).contiguous()
        else:
            recon = self.pca_mu.view(1, C, 1, 1).expand(
                B, C, H, W).contiguous()
            z = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        return recon, z


# ==========================================
# Backbone forward-to-layer (verbatim from export_pca_basics.py)
# ==========================================

class LayerActivationStreamer:
    """Loads a backbone, hooks the target layer, yields activation batches
    [B, C, H, W] one DataLoader batch at a time. Never stores them. No GradCAM.
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
# Per-channel 0.99 normalization (PASS 1)  -- verbatim from export_pca_basics.py
# ==========================================

class PerChannelQuantile:
    """Histogram-based per-channel 0.99 quantile estimator (matches the cache's
    torch.quantile(non_zero, 0.99) per channel to bin resolution)."""

    def __init__(self, C: int, n_bins: int = 2048, device='cpu'):
        self.C = C
        self.n_bins = n_bins
        self.device = device
        self.vmax = torch.zeros(C, device=device)
        self.counts = torch.zeros(C, n_bins, dtype=torch.float64, device=device)
        self._calibrated = False

    def calibrate(self, vmax: torch.Tensor):
        self.vmax = vmax.clamp_min(1e-8) * 1.01
        self._calibrated = True

    @torch.no_grad()
    def update(self, acts: torch.Tensor):
        C = acts.shape[1]
        x = acts.permute(1, 0, 2, 3).reshape(C, -1)
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
        scales = torch.zeros(self.C, device=self.device)
        bin_w = self.vmax / self.n_bins
        total = self.counts.sum(dim=1)
        cum = torch.cumsum(self.counts, dim=1)
        for c in range(self.C):
            if total[c] <= 0:
                continue
            target = q * total[c]
            b = int(torch.searchsorted(cum[c], target).item())
            b = min(b, self.n_bins - 1)
            scales[c] = (b + 1) * bin_w[c]
        return scales


def compute_channel_scales(streamer, data_loader, n_bins=2048,
                           device='cpu') -> torch.Tensor:
    """PASS 1: estimate per-channel 0.99 normalization scales by streaming."""
    C = streamer.num_channels
    pcq = PerChannelQuantile(C, n_bins=n_bins, device=device)
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
    nz = scales[scales > 0]
    print(f"  [pass1] per-channel 0.99 scales: "
          f"min={nz.min().item() if nz.numel() else 0:.4g} "
          f"max={scales.max().item():.4g} "
          f"({int((scales>0).sum().item())}/{C} channels non-degenerate)")
    return scales


def normalize_batch(acts: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """SAME clamp+divide the cache/PCA path used, per channel:
        x = clamp(x, 0, scale) / (scale + 1e-8)
    Channels with scale<=1e-8 left unchanged."""
    C = acts.shape[1]
    sc = scales.view(1, C, 1, 1)
    valid = (scales > 1e-8).view(1, C, 1, 1)
    clamped = torch.clamp(acts, min=0.0)
    clamped = torch.minimum(clamped, sc.clamp_min(1e-8))
    normed = clamped / (sc + 1e-8)
    return torch.where(valid, normed, acts)


# ==========================================
# Streaming per-channel mean |FFT| magnitude (PASS 2)
# ==========================================

class StreamingChannelFFT:
    """Online per-channel mean magnitude spectrum.

        n   : R^C            #images contributing to each channel
        P   : R^{C x H x W}  running sum of |FFT2(map)| per channel
      -> Pbar = P / n        per-channel mean |FFT| magnitude  [C, H, W]

    We FFT each channel's full HxW map per image (translation-invariant
    magnitude), fftshift so the DC term is centered (nice for radial binning and
    for viewing), and accumulate. O(C*H*W) memory total, independent of N.
    """

    def __init__(self, C: int, H: int, W: int,
                 device='cpu', dtype=torch.float64):
        self.C, self.H, self.W = C, H, W
        self.device = device
        self.dtype = dtype
        self.n = torch.zeros(C, dtype=dtype, device=device)
        self.P = torch.zeros(C, H, W, dtype=dtype, device=device)

    @torch.no_grad()
    def update(self, acts: torch.Tensor):
        # acts: [B, C, H, W] (already normalized). FFT over the last two dims.
        B = acts.shape[0]
        a = acts.to(torch.float32)
        spec = torch.fft.fft2(a, dim=(-2, -1))                   # [B,C,H,W] cplx
        mag = torch.fft.fftshift(spec.abs(), dim=(-2, -1))       # center DC
        mag = mag.to(self.dtype).to(self.device)
        self.P += mag.sum(dim=0)                                 # [C,H,W]
        self.n += float(B)

    def finalize(self) -> torch.Tensor:
        n = self.n.clamp_min(1.0).view(self.C, 1, 1)
        return (self.P / n)                                      # [C, H, W]


def stream_channel_fft(streamer, data_loader, scales: Optional[torch.Tensor],
                       device='cpu', dtype=torch.float64) -> torch.Tensor:
    """PASS 2: stream the (optionally normalized) maps into per-channel mean
    |FFT| magnitude  [C, H, W]."""
    C = streamer.num_channels
    H = W = streamer.spatial_size
    acc = StreamingChannelFFT(C, H, W, device=device, dtype=dtype)
    for acts in streamer.iter_batches(data_loader, desc="pass2/fft"):
        a = acts.detach().to(device)
        if scales is not None:
            a = normalize_batch(a, scales.to(device))
        acc.update(a)
    return acc.finalize()


# ==========================================
# Per-channel activation mean (mu) -- same semantics as PCA's mu, streamed in
# the SAME pass 2 (so 'centered' in the visualizer matches).
# ==========================================

class StreamingChannelMean:
    """Per-channel mean activation (over all cells), matching PCA's per-cell
    channel mean mu so the visualizer's centering is identical."""

    def __init__(self, C: int, device='cpu', dtype=torch.float64):
        self.C = C
        self.device = device
        self.dtype = dtype
        self.n = 0
        self.s = torch.zeros(C, dtype=dtype, device=device)

    @torch.no_grad()
    def update(self, acts: torch.Tensor):
        C = acts.shape[1]
        x = acts.permute(0, 2, 3, 1).reshape(-1, C).to(self.dtype)
        self.n += x.shape[0]
        self.s += x.sum(dim=0)

    def finalize(self) -> torch.Tensor:
        if self.n == 0:
            raise RuntimeError("StreamingChannelMean saw no cells.")
        return self.s / self.n


def stream_fft_and_mean(streamer, data_loader, scales, device, dtype):
    """Single pass-2 that fills BOTH the per-channel mean |FFT| and the
    per-channel activation mean mu."""
    C = streamer.num_channels
    H = W = streamer.spatial_size
    fft_acc = StreamingChannelFFT(C, H, W, device=device, dtype=dtype)
    mean_acc = StreamingChannelMean(C, device=device, dtype=dtype)
    for acts in streamer.iter_batches(data_loader, desc="pass2/fft+mean"):
        a = acts.detach().to(device)
        if scales is not None:
            a = normalize_batch(a, scales.to(device))
        fft_acc.update(a)
        mean_acc.update(a)
    return fft_acc.finalize(), mean_acc.finalize()


# ==========================================
# Descriptor construction (full magnitude  OR  radial profile)
# ==========================================

def radial_profile_bins(H: int, W: int, n_rbin: int, device='cpu'):
    """Precompute, for an fftshifted HxW grid, the radial bin index of each
    cell (distance from center) -> [H*W] long, plus bin counts [n_rbin]."""
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij')
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)             # [H,W]
    rmax = r.max().clamp_min(1e-8)
    idx = torch.clamp((r / rmax * (n_rbin - 1)).round().long(), 0, n_rbin - 1)
    idx_flat = idx.reshape(-1)                                  # [H*W]
    counts = torch.zeros(n_rbin, device=device)
    counts.index_add_(0, idx_flat, torch.ones_like(idx_flat,
                                                    dtype=torch.float32))
    return idx_flat, counts.clamp_min(1.0)


def build_descriptor(fft_mean: torch.Tensor, mode: str,
                     n_rbin: int) -> Tuple[torch.Tensor, str]:
    """fft_mean: [C, H, W] mean magnitude spectrum (fftshifted).
    Returns (descriptor [C, n_feat], human-readable feature description)."""
    C, H, W = fft_mean.shape
    if mode == 'full':
        desc = fft_mean.reshape(C, -1)                          # [C, H*W]
        return desc, f"full |FFT| magnitude flattened ({H}x{W}={H*W} feats)"
    elif mode == 'radial':
        idx_flat, counts = radial_profile_bins(H, W, n_rbin,
                                               device=fft_mean.device)
        flat = fft_mean.reshape(C, -1)                          # [C, H*W]
        prof = torch.zeros(C, n_rbin, dtype=flat.dtype, device=flat.device)
        prof.index_add_(1, idx_flat, flat)                      # sum per ring
        prof = prof / counts.view(1, -1)                        # mean per ring
        return prof, f"rotation-invariant radial profile ({n_rbin} bins)"
    else:
        raise ValueError(f"Unknown descriptor mode: {mode}")


# ==========================================
# Clustering -> (membership [C,k], centers [k,n_feat], labels [C])
# ==========================================

def l2norm_rows(X: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp_min(eps)


def cluster_kmeans(desc: torch.Tensor, k: int, seed: int,
                   n_iter: int = 100) -> Tuple[torch.Tensor, torch.Tensor,
                                               torch.Tensor]:
    """Hard k-means (cosine via L2-normalized rows) -> one-hot membership.
    Deterministic given seed (k-means++ init seeded). Pure torch, CPU/GPU."""
    Xn = l2norm_rows(desc.to(torch.float32))
    C, F_ = Xn.shape
    g = torch.Generator(device='cpu').manual_seed(seed)

    # k-means++ init (on CPU for deterministic generator use)
    Xn_cpu = Xn.cpu()
    centers = torch.empty(k, F_)
    first = int(torch.randint(0, C, (1,), generator=g).item())
    centers[0] = Xn_cpu[first]
    d2 = ((Xn_cpu - centers[0]) ** 2).sum(dim=1)
    for j in range(1, k):
        probs = (d2 / d2.sum().clamp_min(1e-12))
        cdf = torch.cumsum(probs, dim=0)
        r = torch.rand(1, generator=g).item()
        nxt = int(torch.searchsorted(cdf, torch.tensor(r)).clamp(0, C - 1).item())
        centers[j] = Xn_cpu[nxt]
        d2 = torch.minimum(d2, ((Xn_cpu - centers[j]) ** 2).sum(dim=1))

    centers = centers.to(Xn.device)
    labels = torch.zeros(C, dtype=torch.long, device=Xn.device)
    for _ in range(n_iter):
        # assign by max cosine == min sq-dist on unit sphere
        sims = Xn @ centers.T                                   # [C, k]
        new_labels = sims.argmax(dim=1)
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for j in range(k):
            sel = (labels == j)
            if sel.any():
                centers[j] = l2norm_rows(Xn[sel].mean(dim=0, keepdim=True))[0]
            # empty cluster: leave center as-is (stable, reproducible)
    membership = F.one_hot(labels, num_classes=k).to(torch.float32)
    return membership, centers, labels


def cluster_nmf(desc: torch.Tensor, k: int, seed: int,
                max_iter: int = 500) -> Tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    """Overlapping soft clustering via NMF of the [C, n_feat] descriptor:
        desc ~= W @ H,  W [C,k] (>=0) membership,  H [k,n_feat] (>=0) centers.
    Uses sklearn NMF with init='nndsvda' for determinism. labels = argmax W."""
    from sklearn.decomposition import NMF
    X = desc.clamp_min(0.0).to(torch.float32).cpu().numpy()
    model = NMF(n_components=k, init='nndsvda', max_iter=max_iter,
                random_state=seed)
    W = model.fit_transform(X)                                  # [C, k]
    Hc = model.components_                                       # [k, n_feat]
    membership = torch.from_numpy(W).to(desc.device)
    centers = torch.from_numpy(Hc).to(desc.device)
    labels = membership.argmax(dim=1)
    return membership, centers, labels


def membership_to_V(membership: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """Turn [C,k] membership into a [C,k] basis whose COLUMNS are L2-normalized
    (so each behaves like a unit 'eigenvector' for the visualizer's z_d / beta_d
    machinery). A channel shared across clusters keeps nonzero entries in
    multiple columns (overlap preserved)."""
    V = membership.to(torch.float32).clone()
    col_norm = V.norm(dim=0, keepdim=True).clamp_min(eps)        # [1,k]
    return V / col_norm


# ==========================================
# Save helpers
# ==========================================

def make_reconstructor(C: int, k: int, mu: torch.Tensor,
                       V: torch.Tensor) -> FFTBasisReconstructor:
    Vk = V.to(torch.float32).cpu() if k > 0 else torch.zeros(C, 0)
    mu32 = mu.to(torch.float32).cpu()
    return FFTBasisReconstructor(in_channels=C, k=k, mu=mu32,
                                 V=Vk).cpu().eval()


def save_fft_npz(path: str, fft_mean: torch.Tensor, descriptor: torch.Tensor,
                 labels: torch.Tensor, membership: torch.Tensor,
                 centers: torch.Tensor, channel_mean: torch.Tensor,
                 scales: Optional[torch.Tensor], meta: Dict):
    np.savez_compressed(
        path,
        fft_mean=fft_mean.double().cpu().numpy(),
        descriptor=descriptor.double().cpu().numpy(),
        labels=labels.long().cpu().numpy(),
        membership=membership.double().cpu().numpy(),
        centers=centers.double().cpu().numpy(),
        channel_mean=channel_mean.double().cpu().numpy(),
        channel_scale=(scales.double().cpu().numpy()
                       if scales is not None else np.array([])),
        meta=np.array([meta], dtype=object),
    )


def report_clusters(labels: torch.Tensor, k: int):
    sizes = torch.bincount(labels.cpu(), minlength=k)
    nonempty = int((sizes > 0).sum().item())
    print(f"\n{'='*64}")
    print(f"  Clustering: k={k}, non-empty clusters={nonempty}")
    print(f"  cluster size: min={int(sizes[sizes>0].min().item()) if nonempty else 0} "
          f"max={int(sizes.max().item())} "
          f"mean={float(sizes[sizes>0].float().mean().item()) if nonempty else 0:.1f}")
    # show the largest few
    top = torch.argsort(sizes, descending=True)[:min(8, k)]
    for j in top.tolist():
        print(f"    cluster {j:>4d}: {int(sizes[j].item())} channels")
    print(f"{'='*64}")


# ==========================================
# CLI
# ==========================================

def main():
    ap = argparse.ArgumentParser(
        description="Streamed per-channel |FFT| descriptor + channel clustering "
                    "into 'basics', PCA-output-compatible (no activation cache).")
    ap.add_argument('--model', type=str, default='resnet50',
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--target_layer', type=str, default=None,
                    help="Defaults to the model's default target layer.")
    ap.add_argument('--k', type=int, default=128,
                    help="Number of channel clusters ('basics'). Default 128.")
    ap.add_argument('--descriptor', type=str, default='full',
                    choices=['full', 'radial'],
                    help="full = |FFT| magnitude (translation-invariant, "
                         "orientation-KEPT; DEFAULT). radial = radial profile "
                         "(also rotation-invariant, more lossy).")
    ap.add_argument('--n_rbin', type=int, default=8,
                    help="Radial bins for --descriptor radial (14x14 -> ~8).")
    ap.add_argument('--cluster', type=str, default='kmeans',
                    choices=['kmeans', 'nmf'],
                    help="kmeans = hard, deterministic, no overlap (DEFAULT). "
                         "nmf = overlapping soft membership (channel can join "
                         "several basics).")
    ap.add_argument('--cluster_seed', type=int, default=0,
                    help="Seed for the clustering init (kmeans++ / NMF). "
                         "Sweep this to check seed stability.")
    ap.add_argument('--images_per_class', type=int, default=IMAGES_PER_CLASS)
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE_COLLECTION)
    ap.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--accum_device', type=str, default='cpu',
                    choices=['cpu', 'cuda'],
                    help="Where to accumulate the [C,H,W] FFT sum. 'cpu' keeps "
                         "GPU memory free.")
    ap.add_argument('--dtype', type=str, default='float64',
                    choices=['float64', 'float32'])
    ap.add_argument('--n_bins', type=int, default=2048,
                    help="Histogram bins for the per-channel 0.99 quantile "
                         "(pass 1).")

    norm_group = ap.add_mutually_exclusive_group()
    norm_group.add_argument(
        '--normalize', dest='norm_mode', action='store_const', const='exact',
        help="Two-pass exact-ish per-channel 0.99 normalization (DEFAULT, "
             "matches the cache/PCA path).")
    norm_group.add_argument(
        '--fast_norm', dest='norm_mode', action='store_const', const='fast',
        help="One-pass approx: 0.99 scales from the FIRST batch only.")
    norm_group.add_argument(
        '--no_normalize', dest='norm_mode', action='store_const', const='none',
        help="Skip normalization; raw-activation |FFT|.")
    ap.set_defaults(norm_mode='exact')

    ap.add_argument('--force_resample', action='store_true')
    ap.add_argument('--data_seed', type=int, default=DEFAULT_DATA_SEED)

    ap.add_argument('--save', type=str, default='fft_basics_model.pkl',
                    help="FFTBasisReconstructor .pkl (drop-in for "
                         "hier_visualize_pca.py).")
    ap.add_argument('--save_fft', type=str, default='fft_basics.npz',
                    help="SEPARATE .npz: fft_mean, descriptor, labels, "
                         "membership, centers, channel_mean, channel_scale, "
                         "meta.")
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
    print("export_fft_basics: streamed per-channel |FFT| + clustering "
          "(no activation cache)")
    print(f"Backbone: {args.model.upper()}  device={dev}  "
          f"accum_device={accum_dev}  norm={args.norm_mode}")
    print(f"descriptor={args.descriptor}  cluster={args.cluster}  k={args.k}  "
          f"cluster_seed={args.cluster_seed}")
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
    H = W = streamer.spatial_size

    # ---- normalization scales (pass 1) ----
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

    # ---- stream mean |FFT| + per-channel mean mu (pass 2) ----
    print(f"\n[pass 2/2] streaming per-channel mean |FFT| + activation mean...")
    fft_mean, mu = stream_fft_and_mean(
        streamer, data_loader, scales=scales, device=accum_dev, dtype=dt)
    print(f"  fft_mean: {tuple(fft_mean.shape)}  "
          f"mu: {tuple(mu.shape)}  (mu norm={mu.norm().item():.4g})")

    # ---- build descriptor ----
    descriptor, desc_info = build_descriptor(
        fft_mean, mode=args.descriptor, n_rbin=args.n_rbin)
    print(f"  descriptor: {tuple(descriptor.shape)}  [{desc_info}]")

    # ---- cluster ----
    k = max(0, min(int(args.k), C))
    if k != args.k:
        print(f"  NOTE: clamped k from {args.k} to {k} (must be <= C={C}).")
    print(f"\nClustering channels into {k} basics ({args.cluster})...")
    if args.cluster == 'kmeans':
        membership, centers, labels = cluster_kmeans(
            descriptor, k, seed=args.cluster_seed)
    else:
        membership, centers, labels = cluster_nmf(
            descriptor, k, seed=args.cluster_seed)
    report_clusters(labels, k)

    # ---- basis V from membership (PCA-compatible) ----
    V = membership_to_V(membership)                             # [C, k]

    # ---- save .npz (raw descriptor + clustering) ----
    meta = {
        'model': args.model,
        'target_layer': streamer.target_layer_name,
        'C': C,
        'spatial_size': H,
        'images_per_class': args.images_per_class,
        'descriptor_mode': args.descriptor,
        'descriptor_info': desc_info,
        'n_rbin': args.n_rbin if args.descriptor == 'radial' else None,
        'cluster': args.cluster,
        'k': k,
        'cluster_seed': args.cluster_seed,
        'norm_mode': args.norm_mode,
        'n_bins': args.n_bins if args.norm_mode == 'exact' else None,
        'invariance': ('translation (exact); orientation KEPT'
                       if args.descriptor == 'full'
                       else 'translation + rotation'),
    }
    save_fft_npz(args.save_fft, fft_mean, descriptor, labels, membership,
                 centers, mu, scales, meta)
    print(f"\nSaved per-channel |FFT| + clustering to {args.save_fft}")
    print(f"  keys: fft_mean[{C},{H},{W}], descriptor[{C},{descriptor.shape[1]}], "
          f"labels[{C}], membership[{C},{k}], centers[{k},{descriptor.shape[1]}], "
          f"channel_mean[{C}], channel_scale[{C if scales is not None else 0}], meta")

    # ---- save reconstructor (.pkl, drop-in for hier_visualize_pca.py) ----
    module = make_reconstructor(C, k, mu, V)
    joblib.dump(module, args.save)
    print(f"Saved FFTBasisReconstructor (k={k}) -> {args.save}")

    print(f"\n{'='*80}")
    print(f"Done. Visualize with the EXISTING PCA radial code (unchanged):")
    print(f"  python hier_visualize_pca.py --class_id 108 \\")
    print(f"      --pca_model {args.save} \\")
    print(f"      --ring_components 12 --top_channels 4 --recon_D {k}")
    print(f"\nEach 'component' in that figure is a CHANNEL-GROUP (cluster), not a")
    print(f"PCA direction; Ring 2 shows the group's member channels. Keep")
    print(f"export_fft_basics.py importable so the .pkl loads (joblib stores the")
    print(f"FFTBasisReconstructor class path).")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()