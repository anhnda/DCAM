"""
Multi-Model ConvSAE Training Script on Full ImageNet-1k
(Masked Loss + Grad-CAM Decomposition Constraint Variant)

NEW IN THIS VERSION:
====================
Adds a Grad-CAM Decomposition Constraint so that the spatial sum of the sparse
latent features z equals (approximately) the Grad-CAM heatmap:

    sum_d z_d(x)  ≈  L_GradCAM(x)        (spatially, H x W)

This converts each active feature z_d into an interpretable, additive
sub-pattern: the Grad-CAM heatmap literally decomposes as a sum of meaningful
spatial patterns z_1 + z_2 + ... + z_D, exactly matching the framing of DCAM
(Decomposed Class Activation Maps).

Concretely, we:
  1. During activation extraction, also compute and cache the spatial Grad-CAM
     map L_GradCAM = ReLU(sum_k alpha_k * A_k)  of shape (H, W) per image,
     normalized so its sum equals 1 (probability-like) -- gives a stable target
     regardless of class/image scale.
  2. Add a new loss term
            L_gradcam = MSE( sum_d z_d ,  L_GradCAM_normalized )
     where sum_d z_d is also rescaled to sum to 1 per sample so the constraint
     is scale-invariant.
  3. Add LAMBDA_GRADCAM (default 1.0) to control the strength of this term.

Cache compatibility: this version uses a new cache key suffix ("gcmap1") so old
caches will be ignored automatically.

Supports multiple backbone architectures:
- ResNet50 (default): layer3, 1024 channels, 14x14 resolution
- ResNet18: layer3, 256 channels, 14x14 resolution
- VGG16: features[16], 256 channels, 28x28 resolution
- EfficientNet-B0: features[4], ~80 channels, 14x14 resolution

Usage:
    python run_xcsae_full.py
    python run_xcsae_full.py --model resnet18
    python run_xcsae_full.py --lambda_gradcam 1.0
    python run_xcsae_full.py --force_reextract     # re-extract to populate gradcam maps
"""

import torch
torch.cuda.init()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision.models as models
from torchvision import transforms
import joblib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import sys
import os
import hashlib
from pathlib import Path
from PIL import Image
import io
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict
import random
import argparse

sys.path.append('.')
from src.gradcam import GradCAM
from full_classes import IMAGENET2012_CLASSES

# ==========================================
# Configuration
# ==========================================

IMAGENET_RAW_DIR = Path("/data/imagenet_raw/data")
IMAGENET_SAMPLED_DIR = Path("/data/imagenet1k_sampled")
ACTIVATION_CACHE_DIR = Path("cache_activations")

IMAGES_PER_CLASS = 50
NUM_CLASSES = 1000

ACTIVATION_BATCH_SIZE = 500
ACTIVATION_CHUNK_SIZE = 100

BATCH_SIZE_COLLECTION = 32
BATCH_SIZE_TRAIN = 32

MODEL_CONFIGS = {
    'resnet50': {
        'model_fn': lambda: models.resnet50(pretrained=True),
        'default_target_layer': 'layer3',
        'valid_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'description': 'ResNet50 (layer3: 1024ch, 14x14)'
    },
    'resnet18': {
        'model_fn': lambda: models.resnet18(pretrained=True),
        'default_target_layer': 'layer3',
        'valid_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'description': 'ResNet18 (layer3: 256ch, 14x14)'
    },
    'vgg16': {
        'model_fn': lambda: models.vgg16(pretrained=True),
        'default_target_layer': 'features[16]',
        'valid_layers': ['features[10]', 'features[16]', 'features[23]', 'features[30]'],
        'description': 'VGG16 (features[16]: 256ch, 28x28)'
    },
    'efficientnet': {
        'model_fn': lambda: models.efficientnet_b0(pretrained=True),
        'default_target_layer': 'features[4]',
        'valid_layers': ['features[2]', 'features[3]', 'features[4]', 'features[5]'],
        'description': 'EfficientNet-B0 (features[4]: ~80ch, 14x14)'
    }
}


# ==========================================
# Multi-Channel ConvSAE Architecture
# ==========================================

class MultiChannelConvSAE(nn.Module):
    """Convolutional Sparse Autoencoder for multi-channel input with Two-Level Sparsity."""

    def __init__(self, in_channels: int = 256, hidden_dim: int = 2048,
                 kernel_size: int = 1, top_k: int = 10):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.top_k = top_k

        self.encoder = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True
        )

        self.decoder = nn.Conv2d(
            hidden_dim,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        nn.init.kaiming_normal_(self.encoder.weight, mode='fan_out', nonlinearity='relu')
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_normal_(self.decoder.weight, mode='fan_in')

    def topk_activation(self, x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """Apply Top-K channel selection."""
        B, C, H, W = x.shape
        channel_importance = x.sum(dim=[2, 3])
        topk_vals, topk_indices = torch.topk(channel_importance, k=self.top_k, dim=1)

        if threshold > 0:
            threshold_mask = topk_vals > threshold
        else:
            threshold_mask = None

        channel_mask = torch.zeros(B, C, device=x.device, dtype=torch.bool)
        channel_mask.scatter_(1, topk_indices, True)

        if threshold_mask is not None:
            for b in range(B):
                valid_channels = topk_indices[b][threshold_mask[b]]
                temp_mask = torch.zeros(C, device=x.device, dtype=torch.bool)
                temp_mask[valid_channels] = True
                channel_mask[b] = temp_mask

        channel_mask_4d = channel_mask.unsqueeze(2).unsqueeze(3)
        result = x * channel_mask_4d.float()

        return result

    def forward(self, x: torch.Tensor, use_topk: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder."""
        features = self.encoder(x)
        features = F.relu(features)

        if use_topk:
            sparse_features = self.topk_activation(features)
        else:
            sparse_features = features

        reconstruction = self.decoder(sparse_features)
        return reconstruction, sparse_features

    def normalize_decoder_weights(self):
        """Normalize decoder weights to have unit norm per feature."""
        with torch.no_grad():
            weight = self.decoder.weight.data
            norm = weight.norm(p=2, dim=(0, 2, 3), keepdim=True).clamp(min=1e-8)
            self.decoder.weight.data = weight / norm


class LateralInhibitionLoss(nn.Module):
    """Penalizes neighboring features from activating together."""

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        feat_center = features[:, :, 1:-1, 1:-1]
        feat_left = features[:, :, 1:-1, :-2]
        feat_right = features[:, :, 1:-1, 2:]
        feat_up = features[:, :, :-2, 1:-1]
        feat_down = features[:, :, 2:, 1:-1]

        corr = (
            (feat_center * feat_left).mean() +
            (feat_center * feat_right).mean() +
            (feat_center * feat_up).mean() +
            (feat_center * feat_down).mean()
        ) / 4.0

        return corr


class SpatialCompactnessLoss(nn.Module):
    """Spatial Compactness Regularization using Total Variation."""

    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        diff_h = torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])
        diff_w = torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])
        tv_loss = diff_h.mean() + diff_w.mean()
        return tv_loss


class FeatureChannelSparsityLoss(nn.Module):
    """Feature-Channel Sparsity Loss."""

    def __init__(self):
        super().__init__()

    def forward(self, encoder_weight: torch.Tensor) -> torch.Tensor:
        weight = encoder_weight.squeeze()
        feature_channel_usage = weight.abs().sum(dim=1)
        sparsity_loss = feature_channel_usage.mean()
        return sparsity_loss


# ==========================================
# NEW: Grad-CAM Decomposition Loss
# ==========================================

class GradCAMDecompositionLoss(nn.Module):
    """Constrains the spatial sum of latent features z to match the Grad-CAM map.

    We want: sum_d z_d(x) approx L_GradCAM(x)   (both H x W).

    To make this scale-invariant, we normalize BOTH sides to sum to 1 (or have
    unit L1 norm) per sample. This means each z_d becomes an additive,
    proportional sub-component of the overall Grad-CAM attention -- so the
    Grad-CAM heatmap really IS the sum of the discovered sub-patterns.

    Args:
        eps: small constant for numerical stability when normalizing.
        normalize: if True, normalize both maps to sum to 1 per-sample before
                   computing MSE (scale-invariant). If False, use raw MSE
                   (requires the Grad-CAM target to already be on a comparable
                   scale to z's activations).
    """

    def __init__(self, eps: float = 1e-8, normalize: bool = True):
        super().__init__()
        self.eps = eps
        self.normalize = normalize

    def forward(self, sparse_features: torch.Tensor,
                gradcam_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sparse_features: [B, D, H, W] latent feature maps (post Top-K, post ReLU)
            gradcam_map:     [B, H, W]    target Grad-CAM map (already non-negative)
        Returns:
            scalar MSE loss between sum_d z_d and gradcam_map.
        """
        # Sum over the feature-dimension D --> [B, H, W]
        z_sum = sparse_features.sum(dim=1)

        if self.normalize:
            # Per-sample L1 normalization (so both maps sum to 1)
            B = z_sum.shape[0]
            z_sum_flat = z_sum.view(B, -1)
            gc_flat = gradcam_map.view(B, -1)

            z_norm = z_sum_flat.sum(dim=1, keepdim=True).clamp(min=self.eps)
            gc_norm = gc_flat.sum(dim=1, keepdim=True).clamp(min=self.eps)

            z_sum_n = z_sum_flat / z_norm
            gc_n = gc_flat / gc_norm

            # Sum squared error over spatial cells, mean over batch.
            # This is the correct scaling for "MSE between two probability
            # distributions over H*W cells": dividing by H*W (i.e. using .mean())
            # makes the loss vanish as resolution grows, even when the two
            # distributions are entirely different. With .sum(dim=1).mean()
            # the loss is O(1e-3) and comparable to the reconstruction loss,
            # giving the optimizer a real signal.
            loss = ((z_sum_n - gc_n) ** 2).sum(dim=1).mean()
        else:
            # Raw MSE (use only if you know z_sum and gradcam_map are already
            # on a comparable scale).
            loss = ((z_sum - gradcam_map) ** 2).mean()

        return loss


# ==========================================
# ImageNet-1k Dataset Sampler
# ==========================================

class ImageNet1kSampledDataset(Dataset):
    """Dataset that loads sampled ImageNet-1k images from parquet files."""

    def __init__(self, raw_dir: Path, sampled_dir: Path,
                 images_per_class: int = 50, transform=None, force_resample: bool = False):
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
        """Sample images from parquet files and save to disk."""
        print(f"Sampling {self.images_per_class} images per class from {NUM_CLASSES} classes...")
        print(f"Total target images: {self.images_per_class * NUM_CLASSES}")

        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(IMAGENET2012_CLASSES.keys())}
        self.idx_to_wnid = {idx: wnid for wnid, idx in self.wnid_to_idx.items()}

        class_samples = defaultdict(list)

        train_parquet_files = sorted(self.raw_dir.glob("train-*.parquet"))

        if len(train_parquet_files) == 0:
            raise FileNotFoundError(f"No train parquet files found in {self.raw_dir}")

        print(f"Found {len(train_parquet_files)} train parquet files")

        for parquet_file in tqdm(train_parquet_files, desc="Reading parquet files"):
            df = pd.read_parquet(parquet_file)

            for idx, row in df.iterrows():
                label = row['label']

                if len(class_samples[label]) < self.images_per_class:
                    image_bytes = row['image']['bytes']
                    class_samples[label].append((image_bytes, label))

            min_samples = min(len(samples) for samples in class_samples.values())
            if min_samples >= self.images_per_class and len(class_samples) == NUM_CLASSES:
                print(f"\nCollected {self.images_per_class} samples for all {NUM_CLASSES} classes!")
                break

        print(f"\nSampling complete. Samples per class:")
        for class_idx in range(min(10, NUM_CLASSES)):
            print(f"  Class {class_idx}: {len(class_samples[class_idx])} images")
        print(f"  ...")

        self.samples = []
        for class_idx in range(NUM_CLASSES):
            if len(class_samples[class_idx]) >= self.images_per_class:
                sampled = random.sample(class_samples[class_idx], self.images_per_class)
                self.samples.extend(sampled)
            else:
                print(f"WARNING: Class {class_idx} has only {len(class_samples[class_idx])} samples")
                self.samples.extend(class_samples[class_idx])

        print(f"\nTotal sampled images: {len(self.samples)}")

        print(f"Saving sampled dataset to {self.sampled_dir}...")
        joblib.dump({
            'samples': self.samples,
            'images_per_class': self.images_per_class,
            'num_classes': NUM_CLASSES,
            'wnid_to_idx': self.wnid_to_idx,
            'idx_to_wnid': self.idx_to_wnid
        }, self.metadata_path)
        print(f"Sampled dataset cached!")

    def load_cached_dataset(self):
        """Load cached sampled dataset."""
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
# Multi-Model Activation Extractor (with Grad-CAM map caching)
# ==========================================

class MultiModelActivationExtractor:
    """Extracts activation channels AND the spatial Grad-CAM map per image."""

    def __init__(self, model_name: str = 'resnet18', target_layer: str = None,
                 device='cuda', cumulative_threshold=0.85, cache_dir: Path = None):
        self.device = device
        self.cumulative_threshold = cumulative_threshold
        self.model_name = model_name
        self.cache_dir = cache_dir or ACTIVATION_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[model_name]
        self.target_layer_name = target_layer if target_layer else config['default_target_layer']

        print(f"\n{'='*80}")
        print(f"Initializing {model_name.upper()} Activation Extractor (with Grad-CAM map caching)")
        print(f"{'='*80}")
        print(f"Model: {config['description']}")
        print(f"Target layer: {self.target_layer_name}")
        print(f"Cache directory: {self.cache_dir}")

        self.model = config['model_fn']().to(device)
        self.model.eval()

        self.target_layer = self._get_layer_by_name(self.target_layer_name)

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_output = self._forward_to_target_layer(dummy_input)
            self.num_channels = dummy_output.shape[1]
            self.spatial_size = dummy_output.shape[2]

        print(f"  Output channels: {self.num_channels}")
        print(f"  Spatial resolution: {self.spatial_size}x{self.spatial_size}")

        self.gradcam = GradCAM(self.model, self.target_layer)

        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

        print(f"Extractor ready!")

    def _get_layer_by_name(self, layer_name: str):
        if '[' in layer_name:
            parts = layer_name.split('[')
            attr_name = parts[0]
            index = int(parts[1].rstrip(']'))
            return getattr(self.model, attr_name)[index]
        else:
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

    def _select_channels_and_gradcam_map(self, image: torch.Tensor,
                                          class_idx: int = None
                                          ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Use GradCAM to:
          (1) select important channels (binary mask), and
          (2) compute the spatial Grad-CAM heatmap L_GradCAM = ReLU(sum_k alpha_k A_k)
              of shape (H, W), normalized to sum to 1 per image.

        Returns:
            channel_mask:   [C] bool tensor of selected channels
            num_selected:   int, number of channels selected
            gradcam_map:    [H, W] float tensor, non-negative, summing to 1
        """
        weights, _, pred_class = self.gradcam.forward(image, class_idx=class_idx, verbose=False)
        # weights: [C], the alpha_k values from Grad-CAM
        # self.activations: [1, C, H, W] (captured by the forward hook during gradcam.forward)

        # --- Channel mask selection (unchanged logic) ---
        sorted_indices = torch.argsort(weights, descending=True)
        sorted_weights = weights[sorted_indices]

        total_score = sorted_weights.sum()
        if total_score > 0:
            cumsum = torch.cumsum(sorted_weights / total_score, dim=0)
            num_selected = (cumsum < self.cumulative_threshold).sum().item() + 1
            num_selected = min(num_selected, len(sorted_indices))
        else:
            num_selected = max(1, int(0.1 * len(sorted_indices)))

        channel_mask = torch.zeros(self.num_channels, dtype=torch.bool, device=self.device)
        selected_channels = sorted_indices[:num_selected]
        channel_mask[selected_channels] = True

        # --- Build the spatial Grad-CAM map ---
        # L_GradCAM = ReLU( sum_k alpha_k * A_k )    shape: [H, W]
        # self.activations is [1, C, H, W]; weights is [C]
        with torch.no_grad():
            acts = self.activations[0]                       # [C, H, W]
            # Weighted sum over channels with the GradCAM alphas
            weighted = (weights.view(-1, 1, 1) * acts).sum(dim=0)  # [H, W]
            gradcam_map = F.relu(weighted)                   # non-negative

            # Per-image L1 normalization so the map sums to 1
            s = gradcam_map.sum()
            if s > 1e-8:
                gradcam_map = gradcam_map / s
            else:
                # degenerate case: uniform map
                gradcam_map = torch.full_like(gradcam_map,
                                              1.0 / gradcam_map.numel())

        return channel_mask, num_selected, gradcam_map

    def _generate_cache_key(self, num_samples: int, chunk_size: int) -> str:
        """Cache key. The 'gcmap1' suffix marks this version (with Grad-CAM maps)
        so existing caches without gradcam maps are not reused."""
        config_str = (
            f"{self.model_name}_"
            f"{self.target_layer_name}_"
            f"thresh{self.cumulative_threshold}_"
            f"samples{num_samples}_"
            f"chunk{chunk_size}_"
            f"gcmap1"
        )
        config_str = config_str.replace('[', '_').replace(']', '').replace('.', 'p')
        return config_str

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"activations_{cache_key}.pkl"

    def _save_chunk_part(self, cache_key: str, part_idx: int,
                        activation_chunk: torch.Tensor,
                        mask_chunk: torch.Tensor,
                        label_chunk: torch.Tensor,
                        gradcam_chunk: torch.Tensor):
        """Save a single chunk part to disk (incremental saving)."""
        cache_dir = self.cache_dir / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True)

        part_path = cache_dir / f"part_{part_idx:04d}.pkl"

        part_data = {
            'activation': activation_chunk,
            'mask': mask_chunk,
            'label': label_chunk,
            'gradcam_map': gradcam_chunk,
        }

        joblib.dump(part_data, part_path, compress=3)

    def _save_activations_to_cache(self, cache_key: str, metadata: Dict):
        cache_dir = self.cache_dir / cache_key
        metadata_path = cache_dir / "metadata.pkl"

        print(f"\nSaving cache metadata...")
        print(f"  Cache directory: {cache_dir}")

        joblib.dump(metadata, metadata_path, compress=3)

        total_size = sum(f.stat().st_size for f in cache_dir.glob("*.pkl"))
        cache_size_mb = total_size / (1024 * 1024)
        print(f"  Total cache size: {cache_size_mb:.1f} MB")
        print(f"  Number of parts: {metadata['num_chunks']}")
        print(f"Activations + Grad-CAM maps cached!")

    def _load_activations_from_cache(self, cache_key: str):
        cache_dir = self.cache_dir / cache_key
        metadata_path = cache_dir / "metadata.pkl"

        print(f"\n{'='*80}")
        print(f"Loading cached activations + Grad-CAM maps...")
        print(f"{'='*80}")
        print(f"  Cache directory: {cache_dir}")

        metadata = joblib.load(metadata_path)

        activation_chunks = []
        mask_chunks = []
        label_chunks = []
        gradcam_chunks = []

        num_parts = metadata['num_chunks']
        print(f"  Loading {num_parts} chunks incrementally...")

        for part_idx in tqdm(range(num_parts), desc="Loading cache parts"):
            part_path = cache_dir / f"part_{part_idx:04d}.pkl"

            if not part_path.exists():
                raise FileNotFoundError(f"Cache part missing: {part_path}")

            part_data = joblib.load(part_path)

            activation_chunks.append(part_data['activation'])
            mask_chunks.append(part_data['mask'])
            label_chunks.append(part_data['label'])
            # Backward-compat: if old cache part lacks gradcam_map, fail loudly
            if 'gradcam_map' not in part_data:
                raise KeyError(
                    f"Cache part {part_path} has no 'gradcam_map'. "
                    "This cache predates the Grad-CAM decomposition constraint. "
                    "Run with --force_reextract to rebuild the cache."
                )
            gradcam_chunks.append(part_data['gradcam_map'])

            if (part_idx + 1) % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        total_samples = sum(chunk.shape[0] for chunk in activation_chunks)
        total_size = sum(f.stat().st_size for f in cache_dir.glob("*.pkl"))
        cache_size_mb = total_size / (1024 * 1024)

        print(f"\n  Total samples: {total_samples}")
        print(f"  Number of chunks: {len(activation_chunks)}")
        print(f"  Cache size: {cache_size_mb:.1f} MB")
        print(f"Cached activations + Grad-CAM maps loaded!")

        return activation_chunks, mask_chunks, label_chunks, gradcam_chunks, metadata

    def _check_cache_exists(self, cache_key: str) -> bool:
        cache_dir = self.cache_dir / cache_key
        metadata_path = cache_dir / "metadata.pkl"
        return metadata_path.exists()

    def collect_activation_maps_chunked(
        self,
        data_loader: DataLoader,
        normalize: bool = True,
        chunk_size: int = 100,
        use_cache: bool = True
    ):
        """Collect activations + Grad-CAM maps in memory-efficient chunks.

        Returns:
            (activation_chunks, mask_chunks, label_chunks, gradcam_chunks)
        """
        num_samples = len(data_loader.dataset)
        cache_key = self._generate_cache_key(num_samples, chunk_size)

        if use_cache and self._check_cache_exists(cache_key):
            (activation_chunks, mask_chunks, label_chunks,
             gradcam_chunks, metadata) = self._load_activations_from_cache(cache_key)

            if (metadata.get('normalized') == normalize and
                metadata.get('cumulative_threshold') == self.cumulative_threshold):
                print(f"  Metadata validated - cache is compatible!")
                return activation_chunks, mask_chunks, label_chunks, gradcam_chunks
            else:
                print(f"  WARNING: Cache metadata mismatch, re-extracting...")

        all_activations = []
        all_masks = []
        all_labels = []
        all_gradcams = []
        chunk_activations = []
        chunk_masks = []
        chunk_labels = []
        chunk_gradcams = []
        channel_selection_stats = []
        total_processed = 0
        chunk_idx = 0

        print(f"\nCollecting activation maps + Grad-CAM maps (chunked)...")
        print(f"  Chunk size: {chunk_size} images")
        print(f"  GradCAM threshold: {self.cumulative_threshold * 100:.0f}%")
        print(f"  Caching: {'Enabled (incremental)' if use_cache else 'Disabled'}")

        for images, labels in tqdm(data_loader, desc="Extracting"):
            for i in range(images.size(0)):
                image = images[i:i+1].to(self.device)
                label = labels[i:i+1]

                with torch.no_grad():
                    _ = self.model(image)
                    activations = self.activations.clone()

                # NEW: also compute Grad-CAM spatial map
                channel_mask, num_selected, gradcam_map = \
                    self._select_channels_and_gradcam_map(image)
                channel_selection_stats.append(num_selected)

                chunk_activations.append(activations.cpu())
                chunk_masks.append(channel_mask.cpu())
                chunk_labels.append(label)
                chunk_gradcams.append(gradcam_map.cpu().unsqueeze(0))  # [1, H, W]
                total_processed += 1

                if len(chunk_activations) >= chunk_size:
                    act_chunk = torch.cat(chunk_activations, dim=0)
                    mask_chunk = torch.stack(chunk_masks, dim=0)
                    label_chunk = torch.cat(chunk_labels, dim=0)
                    gradcam_chunk = torch.cat(chunk_gradcams, dim=0)  # [chunk, H, W]

                    if normalize:
                        for c in range(act_chunk.shape[1]):
                            channel_data = act_chunk[:, c, :, :]
                            flat = channel_data.flatten()
                            non_zero_flat = flat[flat > 1e-8]
                            if len(non_zero_flat) > 0:
                                scale_factor = torch.quantile(non_zero_flat, 0.99)
                                if scale_factor > 1e-8:
                                    channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                                    act_chunk[:, c, :, :] = channel_data / (scale_factor + 1e-8)

                    if use_cache:
                        self._save_chunk_part(cache_key, chunk_idx,
                                              act_chunk, mask_chunk,
                                              label_chunk, gradcam_chunk)

                    all_activations.append(act_chunk)
                    all_masks.append(mask_chunk)
                    all_labels.append(label_chunk)
                    all_gradcams.append(gradcam_chunk)

                    chunk_idx += 1
                    chunk_activations = []
                    chunk_masks = []
                    chunk_labels = []
                    chunk_gradcams = []

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Handle last incomplete chunk
        if len(chunk_activations) > 0:
            act_chunk = torch.cat(chunk_activations, dim=0)
            mask_chunk = torch.stack(chunk_masks, dim=0)
            label_chunk = torch.cat(chunk_labels, dim=0)
            gradcam_chunk = torch.cat(chunk_gradcams, dim=0)

            if normalize:
                for c in range(act_chunk.shape[1]):
                    channel_data = act_chunk[:, c, :, :]
                    flat = channel_data.flatten()
                    non_zero_flat = flat[flat > 1e-8]
                    if len(non_zero_flat) > 0:
                        scale_factor = torch.quantile(non_zero_flat, 0.99)
                        if scale_factor > 1e-8:
                            channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                            act_chunk[:, c, :, :] = channel_data / (scale_factor + 1e-8)

            if use_cache:
                self._save_chunk_part(cache_key, chunk_idx,
                                      act_chunk, mask_chunk,
                                      label_chunk, gradcam_chunk)

            all_activations.append(act_chunk)
            all_masks.append(mask_chunk)
            all_labels.append(label_chunk)
            all_gradcams.append(gradcam_chunk)

        total_samples = sum(chunk.shape[0] for chunk in all_activations)
        avg_selected = np.mean(channel_selection_stats)
        std_selected = np.std(channel_selection_stats)

        print(f"\nCollection complete:")
        print(f"  Total samples: {total_samples}")
        print(f"  Number of chunks: {len(all_activations)}")
        print(f"  Average channels selected: {avg_selected:.1f} +/- {std_selected:.1f} (out of {self.num_channels})")
        if normalize and len(all_activations) > 0:
            print(f"  Activation range (chunk 0): [{all_activations[0].min():.4f}, {all_activations[0].max():.4f}]")
            print(f"  Grad-CAM map sum (chunk 0): mean={all_gradcams[0].sum(dim=(1,2)).mean():.4f} (target=1.0)")

        if use_cache:
            metadata = {
                'model_name': self.model_name,
                'target_layer': self.target_layer_name,
                'cumulative_threshold': self.cumulative_threshold,
                'normalized': normalize,
                'num_channels': self.num_channels,
                'spatial_size': self.spatial_size,
                'avg_channels_selected': avg_selected,
                'std_channels_selected': std_selected,
                'total_samples': total_samples,
                'num_chunks': len(all_activations),
                'has_gradcam_map': True,
            }

            self._save_activations_to_cache(cache_key=cache_key, metadata=metadata)

        return all_activations, all_masks, all_labels, all_gradcams


# ==========================================
# Masked Reconstruction Loss
# ==========================================

def masked_reconstruction_loss(reconstruction: torch.Tensor,
                               target: torch.Tensor,
                               masks: torch.Tensor) -> torch.Tensor:
    """Compute MSE reconstruction loss only on GradCAM-selected channels."""
    masks_4d = masks.unsqueeze(2).unsqueeze(3).float()
    squared_error = (reconstruction - target) ** 2
    masked_squared_error = squared_error * masks_4d

    num_selected = masks.sum(dim=1, keepdim=True).float().clamp(min=1.0)
    loss_per_sample = masked_squared_error.sum(dim=(1, 2, 3)) / (
        num_selected.squeeze() * reconstruction.shape[2] * reconstruction.shape[3]
    )
    loss = loss_per_sample.mean()

    return loss


# ==========================================
# Chunked Dataset (now also carries Grad-CAM maps)
# ==========================================

class ChunkedActivationDataset(Dataset):
    """Memory-efficient dataset that works with chunked activation data.

    Now also returns the per-sample Grad-CAM spatial map alongside activations,
    masks, and labels.
    """

    def __init__(self,
                 activation_chunks: List[torch.Tensor],
                 mask_chunks: List[torch.Tensor],
                 label_chunks: List[torch.Tensor],
                 gradcam_chunks: List[torch.Tensor]):
        self.activation_chunks = activation_chunks
        self.mask_chunks = mask_chunks
        self.label_chunks = label_chunks
        self.gradcam_chunks = gradcam_chunks

        self.index_map = []
        self.class_to_indices = defaultdict(list)

        global_idx = 0
        for chunk_idx, label_chunk in enumerate(label_chunks):
            for sample_idx in range(len(label_chunk)):
                self.index_map.append((chunk_idx, sample_idx))
                label = label_chunk[sample_idx].item()
                self.class_to_indices[label].append(global_idx)
                global_idx += 1

        self.total_samples = len(self.index_map)

        print(f"\nChunkedActivationDataset initialized:")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Number of chunks: {len(activation_chunks)}")
        print(f"  Number of classes: {len(self.class_to_indices)}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        chunk_idx, sample_idx = self.index_map[idx]

        activation = self.activation_chunks[chunk_idx][sample_idx]
        mask = self.mask_chunks[chunk_idx][sample_idx]
        label = self.label_chunks[chunk_idx][sample_idx]
        gradcam_map = self.gradcam_chunks[chunk_idx][sample_idx]

        return activation, mask, label, gradcam_map


class ClassBalancedBatchSampler(Sampler):
    """Samples batches ensuring all classes are represented in each batch."""

    def __init__(self, class_to_indices: Dict[int, List[int]],
                 batch_size: int,
                 drop_last: bool = True):
        self.class_to_indices = class_to_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_classes = len(class_to_indices)

        self.samples_per_class = max(1, batch_size // self.num_classes)
        self.actual_batch_size = self.samples_per_class * self.num_classes

        print(f"\nClassBalancedBatchSampler:")
        print(f"  Batch size: {batch_size} -> {self.actual_batch_size} (balanced)")
        print(f"  Samples per class per batch: {self.samples_per_class}")
        print(f"  Number of classes: {self.num_classes}")

    def __iter__(self):
        class_indices = {
            cls: np.random.permutation(indices).tolist()
            for cls, indices in self.class_to_indices.items()
        }

        min_samples = min(len(indices) for indices in class_indices.values())
        num_batches = min_samples // self.samples_per_class

        for batch_idx in range(num_batches):
            batch = []
            for cls in sorted(class_indices.keys()):
                start = batch_idx * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(class_indices[cls][start:end])

            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        min_samples = min(len(indices) for indices in self.class_to_indices.values())
        return min_samples // self.samples_per_class


# ==========================================
# Visualization
# ==========================================

def plot_training_logs(logs: Dict[str, List], model_name: str, save_path: str):
    """Plot training metrics."""
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'Multi-Channel ConvSAE Training ({model_name.upper()} - ImageNet-1k) - with Grad-CAM Sum Constraint',
                 fontsize=14, fontweight='bold')

    axs[0, 0].plot(logs["recon_loss"], color='blue', linewidth=1.5)
    axs[0, 0].set_title("Masked Reconstruction Loss")
    axs[0, 0].set_ylabel("MSE (masked)")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(logs["l1_loss"], color='green', linewidth=1.5)
    axs[0, 1].set_title("L1 Sparsity Loss")
    axs[0, 1].grid(True, alpha=0.3)

    axs[0, 2].plot(logs["gradcam_loss"], color='magenta', linewidth=1.5)
    axs[0, 2].set_title("Grad-CAM Sum Loss (sum_d z_d vs L_GradCAM)")
    axs[0, 2].grid(True, alpha=0.3)

    axs[1, 0].plot(logs["lateral_loss"], color='orange', linewidth=1.5)
    axs[1, 0].set_title("Lateral Inhibition Loss")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(logs["compact_loss"], color='red', linewidth=1.5)
    axs[1, 1].set_title("Spatial Compactness Loss")
    axs[1, 1].grid(True, alpha=0.3)

    axs[1, 2].plot(logs["active_pct"], color='teal', linewidth=1.5)
    axs[1, 2].set_title("Active Channels %")
    axs[1, 2].set_ylim(0, 10)
    axs[1, 2].grid(True, alpha=0.3)

    axs[2, 0].plot(logs["total_loss"], color='black', linewidth=2)
    axs[2, 0].set_title("Total Loss")
    axs[2, 0].grid(True, alpha=0.3)

    axs[2, 1].plot(logs["recon_loss"], label='Recon', alpha=0.7)
    axs[2, 1].plot(logs["l1_loss"], label='L1', alpha=0.7)
    axs[2, 1].plot(logs["gradcam_loss"], label='GradCAM', alpha=0.7)
    axs[2, 1].set_title("Loss Components (Log)")
    axs[2, 1].set_yscale('log')
    axs[2, 1].legend(fontsize=7)
    axs[2, 1].grid(True, alpha=0.3)

    axs[2, 2].plot(logs["channel_sparsity_loss"], color='purple', linewidth=1.5)
    axs[2, 2].set_title("Channel Sparsity Loss")
    axs[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model ConvSAE Training on Full ImageNet-1k (with Grad-CAM sum constraint)'
    )
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--target_layer', type=str, default=None)
    parser.add_argument('--force_resample', action='store_true')
    parser.add_argument('--force_reextract', action='store_true',
                       help='Force re-extraction of activations (needed when upgrading from a cache without Grad-CAM maps)')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    # NEW: hyperparameter for the Grad-CAM decomposition constraint
    parser.add_argument('--lambda_gradcam', type=float, default=1.0,
                       help='Weight for the Grad-CAM sum constraint loss '
                            '(sum_d z_d ~ L_GradCAM). Default: 1.0')
    parser.add_argument('--gradcam_no_normalize', action='store_true',
                       help='If set, do NOT L1-normalize z_sum and gradcam_map '
                            'before comparing (raw MSE). Default: normalize.')
    parser.add_argument('--cumulative_threshold', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=32)

    args = parser.parse_args()

    if args.batch_size is None:
        if args.model == 'resnet50':
            args.batch_size = 16
        elif args.model == 'vgg16':
            args.batch_size = 16
        else:
            args.batch_size = 32

    print("="*80)
    print(f"Multi-Channel ConvSAE Training on Full ImageNet-1k")
    print(f"Backbone: {args.model.upper()}")
    print(f"Grad-CAM Decomposition Constraint: ENABLED  (lambda_gradcam={args.lambda_gradcam})")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Data
    print(f"\n{'='*80}")
    print("Setting up dataset...")
    print(f"{'='*80}")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageNet1kSampledDataset(
        raw_dir=IMAGENET_RAW_DIR,
        sampled_dir=IMAGENET_SAMPLED_DIR,
        images_per_class=IMAGES_PER_CLASS,
        transform=data_transform,
        force_resample=args.force_resample
    )

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE_COLLECTION,
                             shuffle=False, num_workers=4)

    # Extract activations + Grad-CAM maps
    extractor = MultiModelActivationExtractor(
        model_name=args.model,
        target_layer=args.target_layer,
        device=device, 
        cumulative_threshold=args.cumulative_threshold,
    )

    (activation_chunks, mask_chunks,
     label_chunks, gradcam_chunks) = extractor.collect_activation_maps_chunked(
        data_loader,
        normalize=True,
        chunk_size=ACTIVATION_CHUNK_SIZE,
        use_cache=not args.force_reextract
    )

    # Setup training
    INPUT_CHANNELS = extractor.num_channels
    HIDDEN_DIM = INPUT_CHANNELS * 8
    KERNEL_SIZE = 1
    TOP_K = args.top_k # int(HIDDEN_DIM * 0.05)

    LAMBDA_L1 = 0.3
    LAMBDA_LAT = 0.01
    LAMBDA_COMPACT = 0.01
    LAMBDA_CHANNEL_SPARSITY = 0.0
    LAMBDA_GRADCAM = args.lambda_gradcam   # NEW

    effective_batch_size = args.batch_size * args.accumulation_steps

    print(f"\nTraining Configuration:")
    print(f"  Model: {args.model.upper()}")
    print(f"  Target Layer: {extractor.target_layer_name}")
    print(f"  Input Channels: {INPUT_CHANNELS}")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Top-K: {TOP_K}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size} (GPU)")
    print(f"  Lambda GradCAM: {LAMBDA_GRADCAM}")
    print(f"  GradCAM normalization: {'OFF (raw MSE)' if args.gradcam_no_normalize else 'ON (per-sample L1 norm)'}")
    if args.accumulation_steps > 1:
        print(f"  Accumulation Steps: {args.accumulation_steps}")
        print(f"  Effective Batch Size: {effective_batch_size}")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Memory: {gpu_mem:.1f} GB")

    csae_model = MultiChannelConvSAE(
        in_channels=INPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        kernel_size=KERNEL_SIZE,
        top_k=TOP_K,
    ).to(device)

    optimizer = optim.Adam(csae_model.parameters(), lr=args.lr, weight_decay=1e-5)
    lat_inhib_loss = LateralInhibitionLoss().to(device)
    compact_loss_fn = SpatialCompactnessLoss().to(device)
    channel_sparsity_loss_fn = FeatureChannelSparsityLoss().to(device)
    gradcam_decomp_loss_fn = GradCAMDecompositionLoss(
        normalize=not args.gradcam_no_normalize
    ).to(device)

    train_dataset = ChunkedActivationDataset(
        activation_chunks=activation_chunks,
        mask_chunks=mask_chunks,
        label_chunks=label_chunks,
        gradcam_chunks=gradcam_chunks,
    )

    num_classes = len(train_dataset.class_to_indices)

    if num_classes <= 100:
        print(f"\nUsing class-balanced batch sampling ({num_classes} classes)")
        batch_sampler = ClassBalancedBatchSampler(
            class_to_indices=train_dataset.class_to_indices,
            batch_size=args.batch_size,
            drop_last=True
        )
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    else:
        print(f"\nUsing random sampling ({num_classes} classes, too many for class balancing)")
        print(f"  Actual batch size: {args.batch_size}")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True
        )

    logs = {
        "total_loss": [], "recon_loss": [], "l1_loss": [],
        "lateral_loss": [], "compact_loss": [], "channel_sparsity_loss": [],
        "gradcam_loss": [],          # NEW
        "active_pct": []
    }

    print(f"\n{'='*80}")
    print("Starting Training...")
    print(f"{'='*80}")

    for epoch in range(args.epochs):
        epoch_metrics = {k: 0 for k in logs.keys()}
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch_acts, batch_masks, batch_labels, batch_gradcam = batch
            batch_acts = batch_acts.to(device)
            batch_masks = batch_masks.to(device)
            batch_gradcam = batch_gradcam.to(device)   # [B, H, W]

            # Forward pass
            reconstruction, sparse_features = csae_model(batch_acts, use_topk=True)

            # Compute losses
            loss_recon = masked_reconstruction_loss(reconstruction, batch_acts, batch_masks)
            loss_l1 = sparse_features.abs().mean()
            loss_lateral = lat_inhib_loss(sparse_features)
            loss_compact = compact_loss_fn(sparse_features)
            loss_channel_sparsity = channel_sparsity_loss_fn(csae_model.encoder.weight)
            # NEW: enforce sum_d z_d ~ Grad-CAM map
            loss_gradcam = gradcam_decomp_loss_fn(sparse_features, batch_gradcam)

            loss = (loss_recon +
                   LAMBDA_L1 * loss_l1 +
                   LAMBDA_LAT * loss_lateral +
                   LAMBDA_COMPACT * loss_compact +
                   LAMBDA_CHANNEL_SPARSITY * loss_channel_sparsity +
                   LAMBDA_GRADCAM * loss_gradcam)

            loss = loss / args.accumulation_steps
            loss.backward()

            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(csae_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                csae_model.normalize_decoder_weights()

                if torch.cuda.is_available() and (batch_idx + 1) % (args.accumulation_steps * 10) == 0:
                    torch.cuda.empty_cache()

            with torch.no_grad():
                active_pct = (sparse_features > 0).float().mean().item() * 100
                displayed_loss = loss.item() * args.accumulation_steps

                logs["total_loss"].append(displayed_loss)
                logs["recon_loss"].append(loss_recon.item())
                logs["l1_loss"].append(loss_l1.item())
                logs["lateral_loss"].append(loss_lateral.item())
                logs["compact_loss"].append(loss_compact.item())
                logs["channel_sparsity_loss"].append(loss_channel_sparsity.item())
                logs["gradcam_loss"].append(loss_gradcam.item())
                logs["active_pct"].append(active_pct)

                for k in epoch_metrics.keys():
                    epoch_metrics[k] += logs[k][-1]
                n_batches += 1

            if batch_idx % 20 == 0:
                displayed_loss = loss.item() * args.accumulation_steps
                print(f"\rEpoch {epoch+1}/{args.epochs} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {displayed_loss:.4f} | Recon: {loss_recon.item():.4f} | "
                      f"GradCAM: {loss_gradcam.item():.6f} | "
                      f"Active: {active_pct:.1f}%", end="")

        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        print(f"\n[Epoch {epoch+1}/{args.epochs}] Summary:")
        print(f"  Total Loss:     {avg_metrics['total_loss']:.4f}")
        print(f"  Reconstruction: {avg_metrics['recon_loss']:.4f}")
        print(f"  GradCAM Sum:    {avg_metrics['gradcam_loss']:.6f}")
        print(f"  Active Channels: {avg_metrics['active_pct']:.2f}%")
        print("-" * 80)

    # Save
    output_prefix = f"imagenet1k_csae_{args.model}_gcsum"
    if args.target_layer:
        layer_suffix = args.target_layer.replace('[', '_').replace(']', '')
        output_prefix += f"_{layer_suffix}"

    torch.save(csae_model.state_dict(), f'{output_prefix}_model.pth')
    joblib.dump(csae_model.cpu(), f'{output_prefix}_model.pkl')

    training_info = {
        'config': {
            'model': args.model,
            'target_layer': extractor.target_layer_name,
            'input_channels': INPUT_CHANNELS,
            'hidden_dim': HIDDEN_DIM,
            'top_k': TOP_K,
            'epochs': args.epochs,
            'lr': args.lr,
            'lambda_gradcam': LAMBDA_GRADCAM,
            'gradcam_normalize': not args.gradcam_no_normalize,
        },
        'logs': logs,
        'final_metrics': avg_metrics
    }
    joblib.dump(training_info, f'{output_prefix}_training_info.pkl')

    plot_training_logs(logs, args.model, f'{output_prefix}_logs.png')

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"  Model: {output_prefix}_model.pkl")
    print(f"  Logs: {output_prefix}_logs.png")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()