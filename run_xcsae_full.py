"""
Multi-Model ConvSAE Training Script on Full ImageNet-1k
(Masked Loss Variant with Multiple Backbone Support)

Supports multiple backbone architectures:
- ResNet50 (default): layer3, 1024 channels, 14×14 resolution
- ResNet18: layer3, 256 channels, 14×14 resolution
- VGG16: features[16], 256 channels, 28×28 resolution
- EfficientNet-B0: features[4], ~80 channels, 14×14 resolution

Key Design:
- Input: ALL activation channels (no pre-masking)
- CSAE: Processes all channels with two-level sparsity
- Loss: Reconstruction error computed ONLY on top 85% GradCAM-selected channels

Dataset:
- Full ImageNet-1k (1000 classes)
- Samples 50 images per class from training set (50,000 total)
- Caches sampled dataset to /data/imagenet1k_sampled for reuse

Usage:
    # ResNet50 (default, auto-adjusted batch_size=16)
    python run_xcsae_full.py

    # ResNet18 (auto-adjusted batch_size=32)
    python run_xcsae_full.py --model resnet18

    # VGG16
    python run_xcsae_full.py --model vgg16

    # EfficientNet-B0
    python run_xcsae_full.py --model efficientnet

    # Custom target layer for ResNet50
    python run_xcsae_full.py --model resnet50 --target_layer layer2

    # Gradient accumulation (batch_size=8, accumulation=4, effective=32)
    python run_xcsae_full.py --model resnet50 --batch_size 8 --accumulation_steps 4

    # Force resample dataset
    python run_xcsae_full.py --force_resample

    # Force re-extract activations (ignore cache)
    python run_xcsae_full.py --force_reextract

Memory Management:
    - Batch size is auto-adjusted based on model size
    - Use --accumulation_steps to train with larger effective batch sizes
    - ResNet50: default batch_size=16 (fits on 16GB GPU)
    - For OOM errors, reduce batch_size or increase accumulation_steps
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

# Paths
IMAGENET_RAW_DIR = Path("/data/imagenet_raw/data")
IMAGENET_SAMPLED_DIR = Path("/data/imagenet1k_sampled")
ACTIVATION_CACHE_DIR = Path("cache_activations")

# Sampling parameters
IMAGES_PER_CLASS = 50  # 50 images × 1000 classes = 50,000 images
NUM_CLASSES = 1000

# Memory management
ACTIVATION_BATCH_SIZE = 500
ACTIVATION_CHUNK_SIZE = 100

# Training parameters
BATCH_SIZE_COLLECTION = 32
BATCH_SIZE_TRAIN = 32

# Model configurations
MODEL_CONFIGS = {
    'resnet50': {
        'model_fn': lambda: models.resnet50(pretrained=True),
        'default_target_layer': 'layer3',
        'valid_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'description': 'ResNet50 (layer3: 1024ch, 14×14)'
    },
    'resnet18': {
        'model_fn': lambda: models.resnet18(pretrained=True),
        'default_target_layer': 'layer3',
        'valid_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'description': 'ResNet18 (layer3: 256ch, 14×14)'
    },
    'vgg16': {
        'model_fn': lambda: models.vgg16(pretrained=True),
        'default_target_layer': 'features[16]',
        'valid_layers': ['features[10]', 'features[16]', 'features[23]', 'features[30]'],
        'description': 'VGG16 (features[16]: 256ch, 28×28)'
    },
    'efficientnet': {
        'model_fn': lambda: models.efficientnet_b0(pretrained=True),
        'default_target_layer': 'features[4]',
        'valid_layers': ['features[2]', 'features[3]', 'features[4]', 'features[5]'],
        'description': 'EfficientNet-B0 (features[4]: ~80ch, 14×14)'
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
        print(f"✓ Sampled dataset cached!")

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
# Multi-Model Activation Extractor
# ==========================================

class MultiModelActivationExtractor:
    """Extracts activation channels from various backbone models WITHOUT masking."""

    def __init__(self, model_name: str = 'resnet18', target_layer: str = None,
                 device='cuda', cumulative_threshold=0.85, cache_dir: Path = None):
        self.device = device
        self.cumulative_threshold = cumulative_threshold
        self.model_name = model_name
        self.cache_dir = cache_dir or ACTIVATION_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Get model configuration
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[model_name]
        self.target_layer_name = target_layer if target_layer else config['default_target_layer']

        print(f"\n{'='*80}")
        print(f"Initializing {model_name.upper()} Activation Extractor")
        print(f"{'='*80}")
        print(f"Model: {config['description']}")
        print(f"Target layer: {self.target_layer_name}")
        print(f"Cache directory: {self.cache_dir}")

        # Load model
        self.model = config['model_fn']().to(device)
        self.model.eval()

        # Get target layer
        self.target_layer = self._get_layer_by_name(self.target_layer_name)

        # Get dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_output = self._forward_to_target_layer(dummy_input)
            self.num_channels = dummy_output.shape[1]
            self.spatial_size = dummy_output.shape[2]

        print(f"  Output channels: {self.num_channels}")
        print(f"  Spatial resolution: {self.spatial_size}×{self.spatial_size}")

        # GradCAM
        self.gradcam = GradCAM(self.model, self.target_layer)

        # Hook
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

        print(f"✓ Extractor ready!")

    def _get_layer_by_name(self, layer_name: str):
        """Get layer by name (e.g., 'layer3' or 'features[16]')."""
        if '[' in layer_name:  # features[16] style
            parts = layer_name.split('[')
            attr_name = parts[0]
            index = int(parts[1].rstrip(']'))
            return getattr(self.model, attr_name)[index]
        else:  # layer3 style
            return getattr(self.model, layer_name)

    def _forward_to_target_layer(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to target layer."""
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
            # Parse features[X]
            target_idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            for i in range(target_idx + 1):
                x = self.model.features[i](x)
            return x

        elif self.model_name == 'efficientnet':
            # Parse features[X]
            target_idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            for i in range(target_idx + 1):
                x = self.model.features[i](x)
            return x

        return x

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def _select_channels_with_gradcam(self, image: torch.Tensor, class_idx: int = None) -> Tuple[torch.Tensor, int]:
        """Use GradCAM to select important channels and create a binary mask."""
        weights, _, pred_class = self.gradcam.forward(image, class_idx=class_idx, verbose=False)

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

        return channel_mask, num_selected

    def _generate_cache_key(self, num_samples: int, chunk_size: int) -> str:
        """Generate a unique cache key based on extraction configuration."""
        # Create a string with all relevant parameters
        config_str = (
            f"{self.model_name}_"
            f"{self.target_layer_name}_"
            f"thresh{self.cumulative_threshold}_"
            f"samples{num_samples}_"
            f"chunk{chunk_size}"
        )

        # Clean up special characters for filename
        config_str = config_str.replace('[', '_').replace(']', '').replace('.', 'p')

        return config_str

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given cache key."""
        return self.cache_dir / f"activations_{cache_key}.pkl"

    def _save_chunk_part(self, cache_key: str, part_idx: int,
                        activation_chunk: torch.Tensor,
                        mask_chunk: torch.Tensor,
                        label_chunk: torch.Tensor):
        """Save a single chunk part to disk (incremental saving)."""
        cache_dir = self.cache_dir / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True)

        part_path = cache_dir / f"part_{part_idx:04d}.pkl"

        part_data = {
            'activation': activation_chunk,
            'mask': mask_chunk,
            'label': label_chunk
        }

        joblib.dump(part_data, part_path, compress=3)

    def _save_activations_to_cache(self, cache_key: str, metadata: Dict):
        """Save metadata for cached activations (chunks already saved incrementally)."""
        cache_dir = self.cache_dir / cache_key
        metadata_path = cache_dir / "metadata.pkl"

        print(f"\nSaving cache metadata...")
        print(f"  Cache directory: {cache_dir}")

        joblib.dump(metadata, metadata_path, compress=3)

        # Calculate total cache size
        total_size = sum(f.stat().st_size for f in cache_dir.glob("*.pkl"))
        cache_size_mb = total_size / (1024 * 1024)
        print(f"  Total cache size: {cache_size_mb:.1f} MB")
        print(f"  Number of parts: {metadata['num_chunks']}")
        print(f"✓ Activations cached!")

    def _load_activations_from_cache(self, cache_key: str) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict]:
        """Load cached activations incrementally from multi-file cache."""
        cache_dir = self.cache_dir / cache_key
        metadata_path = cache_dir / "metadata.pkl"

        print(f"\n{'='*80}")
        print(f"Loading cached activations...")
        print(f"{'='*80}")
        print(f"  Cache directory: {cache_dir}")

        # Load metadata
        metadata = joblib.load(metadata_path)

        # Load chunks incrementally
        activation_chunks = []
        mask_chunks = []
        label_chunks = []

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

            # Clear memory periodically
            if (part_idx + 1) % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        total_samples = sum(chunk.shape[0] for chunk in activation_chunks)
        total_size = sum(f.stat().st_size for f in cache_dir.glob("*.pkl"))
        cache_size_mb = total_size / (1024 * 1024)

        print(f"\n  Total samples: {total_samples}")
        print(f"  Number of chunks: {len(activation_chunks)}")
        print(f"  Cache size: {cache_size_mb:.1f} MB")
        print(f"✓ Cached activations loaded!")

        return activation_chunks, mask_chunks, label_chunks, metadata

    def _check_cache_exists(self, cache_key: str) -> bool:
        """Check if cache exists for the given cache key."""
        cache_dir = self.cache_dir / cache_key
        metadata_path = cache_dir / "metadata.pkl"
        return metadata_path.exists()

    def collect_activation_maps_chunked(
        self,
        data_loader: DataLoader,
        normalize: bool = True,
        chunk_size: int = 100,
        use_cache: bool = True
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Collect activation maps in memory-efficient chunks with caching support.

        Args:
            data_loader: DataLoader for the dataset
            normalize: Whether to apply robust normalization
            chunk_size: Number of samples per chunk
            use_cache: If True, load from cache if available, save to cache after extraction

        Returns:
            Tuple of (activation_chunks, mask_chunks, label_chunks)
            Each is a list of tensors to avoid memory explosion from concatenation.
        """
        # Calculate total samples for cache key
        num_samples = len(data_loader.dataset)
        cache_key = self._generate_cache_key(num_samples, chunk_size)

        # Check if cache exists and use_cache is True
        if use_cache and self._check_cache_exists(cache_key):
            activation_chunks, mask_chunks, label_chunks, metadata = self._load_activations_from_cache(cache_key)

            # Verify metadata matches current configuration
            if (metadata.get('normalized') == normalize and
                metadata.get('cumulative_threshold') == self.cumulative_threshold):
                print(f"  Metadata validated - cache is compatible!")
                return activation_chunks, mask_chunks, label_chunks
            else:
                print(f"  WARNING: Cache metadata mismatch, re-extracting...")
                print(f"    Cached normalized={metadata.get('normalized')}, current={normalize}")
                print(f"    Cached threshold={metadata.get('cumulative_threshold')}, current={self.cumulative_threshold}")

        # Cache miss or invalid - extract activations
        all_activations = []
        all_masks = []
        all_labels = []
        chunk_activations = []
        chunk_masks = []
        chunk_labels = []
        channel_selection_stats = []
        total_processed = 0
        chunk_idx = 0

        print(f"\nCollecting activation maps (chunked processing)...")
        print(f"  Chunk size: {chunk_size} images")
        print(f"  GradCAM threshold: {self.cumulative_threshold * 100:.0f}%")
        print(f"  Caching: {'Enabled (incremental)' if use_cache else 'Disabled'}")

        for images, labels in tqdm(data_loader, desc="Extracting activations"):
            for i in range(images.size(0)):
                image = images[i:i+1].to(self.device)
                label = labels[i:i+1]

                with torch.no_grad():
                    _ = self.model(image)
                    activations = self.activations.clone()

                channel_mask, num_selected = self._select_channels_with_gradcam(image)
                channel_selection_stats.append(num_selected)

                chunk_activations.append(activations.cpu())
                chunk_masks.append(channel_mask.cpu())
                chunk_labels.append(label)
                total_processed += 1

                if len(chunk_activations) >= chunk_size:
                    # Concatenate chunk
                    act_chunk = torch.cat(chunk_activations, dim=0)
                    mask_chunk = torch.stack(chunk_masks, dim=0)
                    label_chunk = torch.cat(chunk_labels, dim=0)

                    # Normalize chunk if needed (before saving to ensure cache has normalized data)
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

                    # Save to cache immediately (incremental saving)
                    if use_cache:
                        self._save_chunk_part(cache_key, chunk_idx, act_chunk, mask_chunk, label_chunk)

                    # Add to list for return
                    all_activations.append(act_chunk)
                    all_masks.append(mask_chunk)
                    all_labels.append(label_chunk)

                    chunk_idx += 1
                    chunk_activations = []
                    chunk_masks = []
                    chunk_labels = []

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Handle last incomplete chunk
        if len(chunk_activations) > 0:
            act_chunk = torch.cat(chunk_activations, dim=0)
            mask_chunk = torch.stack(chunk_masks, dim=0)
            label_chunk = torch.cat(chunk_labels, dim=0)

            # Normalize chunk if needed
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

            # Save to cache immediately
            if use_cache:
                self._save_chunk_part(cache_key, chunk_idx, act_chunk, mask_chunk, label_chunk)

            all_activations.append(act_chunk)
            all_masks.append(mask_chunk)
            all_labels.append(label_chunk)

        # Calculate statistics WITHOUT concatenating all data
        total_samples = sum(chunk.shape[0] for chunk in all_activations)
        avg_selected = np.mean(channel_selection_stats)
        std_selected = np.std(channel_selection_stats)

        print(f"\nCollection complete (memory-efficient chunked format):")
        print(f"  Total samples: {total_samples}")
        print(f"  Number of chunks: {len(all_activations)}")
        print(f"  Average channels selected: {avg_selected:.1f} ± {std_selected:.1f} (out of {self.num_channels})")
        if normalize and len(all_activations) > 0:
            print(f"  Normalized range (sample from chunk 0): [{all_activations[0].min():.4f}, {all_activations[0].max():.4f}]")

        # Save metadata to cache if enabled (chunks already saved incrementally)
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
                'num_chunks': len(all_activations)
            }

            self._save_activations_to_cache(
                cache_key=cache_key,
                metadata=metadata
            )

        return all_activations, all_masks, all_labels


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
    loss_per_sample = masked_squared_error.sum(dim=(1, 2, 3)) / (num_selected.squeeze() * reconstruction.shape[2] * reconstruction.shape[3])
    loss = loss_per_sample.mean()

    return loss


# ==========================================
# Chunked Dataset (Memory-Efficient)
# ==========================================

class ChunkedActivationDataset(Dataset):
    """Memory-efficient dataset that works with chunked activation data.

    Avoids concatenating all chunks into a single tensor to prevent memory explosion.
    Instead, keeps data in chunks and dynamically retrieves samples.
    """

    def __init__(self, activation_chunks: List[torch.Tensor],
                 mask_chunks: List[torch.Tensor],
                 label_chunks: List[torch.Tensor]):
        """
        Args:
            activation_chunks: List of activation tensors [chunk_size, C, H, W]
            mask_chunks: List of mask tensors [chunk_size, C]
            label_chunks: List of label tensors [chunk_size]
        """
        self.activation_chunks = activation_chunks
        self.mask_chunks = mask_chunks
        self.label_chunks = label_chunks

        # Build index mapping: (chunk_idx, sample_idx_in_chunk)
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
        """Get activation, mask, and label by global index."""
        chunk_idx, sample_idx = self.index_map[idx]

        activation = self.activation_chunks[chunk_idx][sample_idx]
        mask = self.mask_chunks[chunk_idx][sample_idx]
        label = self.label_chunks[chunk_idx][sample_idx]

        return activation, mask, label


class ClassBalancedBatchSampler(Sampler):
    """Samples batches ensuring all classes are represented in each batch.

    This ensures that each mini-batch contains samples from diverse classes,
    which helps with learning discriminative features.
    """

    def __init__(self, class_to_indices: Dict[int, List[int]],
                 batch_size: int,
                 drop_last: bool = True):
        self.class_to_indices = class_to_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_classes = len(class_to_indices)

        # Calculate samples per class per batch
        self.samples_per_class = max(1, batch_size // self.num_classes)
        self.actual_batch_size = self.samples_per_class * self.num_classes

        print(f"\nClassBalancedBatchSampler:")
        print(f"  Batch size: {batch_size} -> {self.actual_batch_size} (balanced)")
        print(f"  Samples per class per batch: {self.samples_per_class}")
        print(f"  Number of classes: {self.num_classes}")

    def __iter__(self):
        # Shuffle indices within each class
        class_indices = {
            cls: np.random.permutation(indices).tolist()
            for cls, indices in self.class_to_indices.items()
        }

        # Determine number of batches
        min_samples = min(len(indices) for indices in class_indices.values())
        num_batches = min_samples // self.samples_per_class

        for batch_idx in range(num_batches):
            batch = []
            for cls in sorted(class_indices.keys()):
                start = batch_idx * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(class_indices[cls][start:end])

            # Shuffle within batch
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        min_samples = min(len(indices) for indices in self.class_to_indices.values())
        return min_samples // self.samples_per_class


# ==========================================
# Visualization Functions
# ==========================================

def plot_training_logs(logs: Dict[str, List], model_name: str, save_path: str):
    """Plot training metrics."""
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'Multi-Channel ConvSAE Training ({model_name.upper()} - ImageNet-1k)',
                 fontsize=14, fontweight='bold')

    axs[0, 0].plot(logs["recon_loss"], color='blue', linewidth=1.5)
    axs[0, 0].set_title("Masked Reconstruction Loss")
    axs[0, 0].set_ylabel("MSE (masked)")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(logs["l1_loss"], color='green', linewidth=1.5)
    axs[0, 1].set_title("L1 Sparsity Loss")
    axs[0, 1].grid(True, alpha=0.3)

    axs[0, 2].plot(logs["channel_sparsity_loss"], color='purple', linewidth=1.5)
    axs[0, 2].set_title("Channel Sparsity Loss")
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
    axs[2, 1].set_title("Loss Components (Log)")
    axs[2, 1].set_yscale('log')
    axs[2, 1].legend(fontsize=7)
    axs[2, 1].grid(True, alpha=0.3)

    axs[2, 2].scatter(logs["channel_sparsity_loss"], logs["recon_loss"],
                     c=range(len(logs["recon_loss"])), cmap='viridis', alpha=0.5, s=5)
    axs[2, 2].set_title("Recon vs Channel Sparsity")
    axs[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


# ==========================================
# Main Training Script
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model ConvSAE Training on Full ImageNet-1k'
    )
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Backbone model (default: resnet50)')
    parser.add_argument('--target_layer', type=str, default=None,
                       help='Target layer name (default: model-specific default)')
    parser.add_argument('--force_resample', action='store_true',
                       help='Force resampling of dataset')
    parser.add_argument('--force_reextract', action='store_true',
                       help='Force re-extraction of activations (ignore cache)')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Training batch size (auto-adjusted if not specified)')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps (default: 1, no accumulation)')

    args = parser.parse_args()

    # Auto-adjust batch size based on model if not specified
    if args.batch_size is None:
        if args.model == 'resnet50':
            args.batch_size = 16  # Large model, small batches
        elif args.model == 'vgg16':
            args.batch_size = 16  # Large model
        else:
            args.batch_size = 32  # ResNet18, EfficientNet

    print("="*80)
    print(f"Multi-Channel ConvSAE Training on Full ImageNet-1k")
    print(f"Backbone: {args.model.upper()}")
    print("="*80)

    # Setup
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

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE_COLLECTION, shuffle=False, num_workers=4)

    # Extract activations
    extractor = MultiModelActivationExtractor(
        model_name=args.model,
        target_layer=args.target_layer,
        device=device
    )

    activation_chunks, mask_chunks, label_chunks = extractor.collect_activation_maps_chunked(
        data_loader,
        normalize=True,
        chunk_size=ACTIVATION_CHUNK_SIZE,
        use_cache=not args.force_reextract  # Cache enabled unless --force_reextract is set
    )

    # Setup training
    INPUT_CHANNELS = extractor.num_channels
    HIDDEN_DIM = INPUT_CHANNELS * 8
    KERNEL_SIZE = 1
    TOP_K = int(HIDDEN_DIM * 0.4)

    #LAMBDA_L1 = 0.3
    LAMBDA_L1 = 0.3
    LAMBDA_LAT = 0.01
    LAMBDA_COMPACT = 0.01
    LAMBDA_CHANNEL_SPARSITY = 0.0

    # Calculate effective batch size
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
    if args.accumulation_steps > 1:
        print(f"  Accumulation Steps: {args.accumulation_steps}")
        print(f"  Effective Batch Size: {effective_batch_size}")

    # Memory info
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Memory: {gpu_mem:.1f} GB")

    csae_model = MultiChannelConvSAE(
        in_channels=INPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        kernel_size=KERNEL_SIZE,
        top_k=TOP_K
    ).to(device)

    optimizer = optim.Adam(csae_model.parameters(), lr=args.lr, weight_decay=1e-5)
    lat_inhib_loss = LateralInhibitionLoss().to(device)
    compact_loss_fn = SpatialCompactnessLoss().to(device)
    channel_sparsity_loss_fn = FeatureChannelSparsityLoss().to(device)

    # Create memory-efficient chunked dataset
    train_dataset = ChunkedActivationDataset(
        activation_chunks=activation_chunks,
        mask_chunks=mask_chunks,
        label_chunks=label_chunks
    )

    # Use class-balanced sampling only for small datasets (< 100 classes)
    # For ImageNet-1k (1000 classes), use regular random sampling
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
        "active_pct": []
    }

    # Training loop
    print(f"\n{'='*80}")
    print("Starting Training...")
    print(f"{'='*80}")

    for epoch in range(args.epochs):
        epoch_metrics = {k: 0 for k in logs.keys()}
        n_batches = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch

        for batch_idx, (batch_acts, batch_masks, batch_labels) in enumerate(train_loader):
            batch_acts = batch_acts.to(device)
            batch_masks = batch_masks.to(device)
            # batch_labels available but not used in unsupervised training

            # Forward pass
            reconstruction, sparse_features = csae_model(batch_acts, use_topk=True)

            # Compute losses
            loss_recon = masked_reconstruction_loss(reconstruction, batch_acts, batch_masks)
            loss_l1 = sparse_features.abs().mean()
            loss_lateral = lat_inhib_loss(sparse_features)
            loss_compact = compact_loss_fn(sparse_features)
            loss_channel_sparsity = channel_sparsity_loss_fn(csae_model.encoder.weight)

            loss = (loss_recon +
                   LAMBDA_L1 * loss_l1 +
                   LAMBDA_LAT * loss_lateral +
                   LAMBDA_COMPACT * loss_compact +
                   LAMBDA_CHANNEL_SPARSITY * loss_channel_sparsity)

            # Scale loss for gradient accumulation
            loss = loss / args.accumulation_steps
            loss.backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(csae_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                csae_model.normalize_decoder_weights()

                # Clear GPU cache periodically
                if torch.cuda.is_available() and (batch_idx + 1) % (args.accumulation_steps * 10) == 0:
                    torch.cuda.empty_cache()

            # Logging (scale loss back for display)
            with torch.no_grad():
                active_pct = (sparse_features > 0).float().mean().item() * 100
                displayed_loss = loss.item() * args.accumulation_steps  # Scale back for display

                logs["total_loss"].append(displayed_loss)
                logs["recon_loss"].append(loss_recon.item())
                logs["l1_loss"].append(loss_l1.item())
                logs["lateral_loss"].append(loss_lateral.item())
                logs["compact_loss"].append(loss_compact.item())
                logs["channel_sparsity_loss"].append(loss_channel_sparsity.item())
                logs["active_pct"].append(active_pct)

                for k in epoch_metrics.keys():
                    epoch_metrics[k] += logs[k][-1]
                n_batches += 1

            if batch_idx % 20 == 0:
                displayed_loss = loss.item() * args.accumulation_steps
                print(f"\rEpoch {epoch+1}/{args.epochs} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {displayed_loss:.4f} | Recon: {loss_recon.item():.4f} | "
                      f"Active: {active_pct:.1f}%", end="")

        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        print(f"\n[Epoch {epoch+1}/{args.epochs}] Summary:")
        print(f"  Total Loss: {avg_metrics['total_loss']:.4f}")
        print(f"  Reconstruction: {avg_metrics['recon_loss']:.4f}")
        print(f"  Active Channels: {avg_metrics['active_pct']:.2f}%")
        print("-" * 80)

    # Save
    output_prefix = f"imagenet1k_csae_{args.model}"
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
        },
        'logs': logs,
        'final_metrics': avg_metrics
    }
    joblib.dump(training_info, f'{output_prefix}_training_info.pkl')

    plot_training_logs(logs, args.model, f'{output_prefix}_logs.png')

    print(f"\n{'='*80}")
    print("✓ Training Complete!")
    print(f"  Model: {output_prefix}_model.pkl")
    print(f"  Logs: {output_prefix}_logs.png")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()