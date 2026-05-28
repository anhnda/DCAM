"""
export_activation_cache.py
==========================
STANDALONE cache extractor. Does ONLY the extraction phase of
run_xcsae_full.py: loads the backbone, samples ImageNet-1k from parquet,
runs forward + GradCAM per image, normalizes per-channel by the 99th
percentile, and writes the gcmap1 cache.

No training. No PCA. No SAE. Just the cache.

A --skip_grad mode (ON by default) skips GradCAM entirely: it only runs the
forward pass to capture activations, and fills the mask/gradcam_map fields
with neutral placeholders (mask = all True, gradcam_map = uniform). The cache
keeps the SAME interface (identical keys, dtypes, and shapes) so the downstream
pipeline keeps working unchanged. Pass --no_skip_grad to compute real GradCAM
masks and maps.

Output (same layout as run_xcsae_full.py)
-----------------------------------------
    cache_activations/activations_<cache_key>/
        metadata.pkl
        part_0000.pkl
        part_0001.pkl
        ...

cache_key is
    {model}_{layer}_thresh{thr_with_p_for_dot}_samples{N}_chunk{C}_gcmap1
e.g.
    resnet50_layer3_thresh0p95_samples50000_chunk100_gcmap1

When --skip_grad is active the key gets a trailing _nograd marker so the two
cache flavors never collide on disk:
    resnet50_layer3_thresh0p95_samples50000_chunk100_gcmap1_nograd

Each part_*.pkl is a dict with keys: 'activation' [n, C, H, W],
'mask' [n, C] bool, 'label' [n], 'gradcam_map' [n, H, W] (sums to 1
per image). The cache is BYTE-COMPATIBLE with the cache run_xcsae_full.py
produces -- either script will accept the other's cache.

External modules used (same strategy as run_xcsae_full.py):
  - full_classes.IMAGENET2012_CLASSES   (1000-entry wnid->idx mapping)
  - src.gradcam.GradCAM                 (the project's GradCAM module)

USAGE
-----
  # default: resnet50 layer3, threshold 0.95, 50 imgs/class -> 50000 samples
  # (GradCAM SKIPPED by default -- activations only)
  python export_activation_cache.py

  # explicit, activations only (default behavior)
  python export_activation_cache.py \
      --model resnet50 --target_layer layer3 \
      --images_per_class 50

  # compute real GradCAM masks + maps (original behavior)
  python export_activation_cache.py --no_skip_grad \
      --cumulative_threshold 0.95

  # other backbones
  python export_activation_cache.py --model resnet18
  python export_activation_cache.py --model vgg16
  python export_activation_cache.py --model efficientnet

  # force re-extraction even if cache exists
  python export_activation_cache.py --force_reextract

  # force re-sample the dataset (re-run parquet sampling)
  python export_activation_cache.py --force_resample
"""

import argparse
import io
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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

# External project modules (same as run_xcsae_full.py)
sys.path.append('.')
from src.gradcam import GradCAM
from full_classes import IMAGENET2012_CLASSES


# ==========================================
# Configuration (matches run_xcsae_full.py defaults)
# ==========================================

IMAGENET_RAW_DIR = Path("/data/imagenet_raw/data")
IMAGENET_SAMPLED_DIR = Path("/data/imagenet1k_sampled")
ACTIVATION_CACHE_DIR = Path("cache_activations")

IMAGES_PER_CLASS = 50
NUM_CLASSES = 1000

BATCH_SIZE_COLLECTION = 32
ACTIVATION_CHUNK_SIZE = 100

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
# Reproducibility
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
# (verbatim from run_xcsae_full.py)
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
# Activation Extractor (with optional GradCAM map caching)
# (verbatim logic from run_xcsae_full.py:
#  MultiModelActivationExtractor)
# ==========================================

class ActivationExtractor:
    """Extracts activation channels AND (optionally) the spatial Grad-CAM map
    per image, writes byte-compatible gcmap1 cache.

    When skip_grad=True, GradCAM is not computed at all. The mask is filled
    with all-True (every channel "selected") and the gradcam_map is filled with
    a uniform distribution that still sums to 1 per image. The on-disk schema
    (keys/dtypes/shapes) is identical, so downstream consumers are unaffected.
    """

    def __init__(self, model_name: str = 'resnet50',
                 target_layer: str = None,
                 device='cuda', cumulative_threshold=0.95,
                 cache_dir: Path = None,
                 skip_grad: bool = True):
        self.device = device
        self.cumulative_threshold = cumulative_threshold
        self.model_name = model_name
        self.skip_grad = skip_grad
        self.cache_dir = cache_dir or ACTIVATION_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Choose from {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[model_name]
        self.target_layer_name = (target_layer if target_layer
                                  else config['default_target_layer'])

        print(f"\n{'='*80}")
        print(f"Initializing {model_name.upper()} Activation Extractor "
              f"({'NO GradCAM -- activations only' if skip_grad else 'with Grad-CAM map caching'})")
        print(f"{'='*80}")
        print(f"Model: {config['description']}")
        print(f"Target layer: {self.target_layer_name}")
        print(f"Skip GradCAM: {self.skip_grad}")
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
        print(f"  Spatial resolution: "
              f"{self.spatial_size}x{self.spatial_size}")

        # Only build the GradCAM helper when we actually need it.
        self.gradcam = (None if self.skip_grad
                        else GradCAM(self.model, self.target_layer))

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

    def _neutral_mask_and_map(self) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Placeholder channel mask + gradcam map used when skip_grad=True.

        mask        : all-True [C] bool  (every channel "selected")
        num_selected: C
        gradcam_map : uniform [H, W] float summing to 1
        """
        channel_mask = torch.ones(
            self.num_channels, dtype=torch.bool, device=self.device)
        h = w = self.spatial_size
        gradcam_map = torch.full(
            (h, w), 1.0 / (h * w), device=self.device)
        return channel_mask, self.num_channels, gradcam_map

    def _select_channels_and_gradcam_map(
            self, image: torch.Tensor, class_idx: int = None
            ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Returns (channel_mask [C] bool, num_selected int,
        gradcam_map [H, W] float summing to 1)."""
        weights, _, pred_class = self.gradcam.forward(
            image, class_idx=class_idx, verbose=False)

        sorted_indices = torch.argsort(weights, descending=True)
        sorted_weights = weights[sorted_indices]
        total_score = sorted_weights.sum()
        if total_score > 0:
            cumsum = torch.cumsum(sorted_weights / total_score, dim=0)
            num_selected = (cumsum < self.cumulative_threshold).sum().item() + 1
            num_selected = min(num_selected, len(sorted_indices))
        else:
            num_selected = max(1, int(0.1 * len(sorted_indices)))

        channel_mask = torch.zeros(
            self.num_channels, dtype=torch.bool, device=self.device)
        selected_channels = sorted_indices[:num_selected]
        channel_mask[selected_channels] = True

        with torch.no_grad():
            acts = self.activations[0]                       # [C, H, W]
            weighted = (weights.view(-1, 1, 1) * acts).sum(dim=0)  # [H, W]
            gradcam_map = F.relu(weighted)
            s = gradcam_map.sum()
            if s > 1e-8:
                gradcam_map = gradcam_map / s
            else:
                gradcam_map = torch.full_like(
                    gradcam_map, 1.0 / gradcam_map.numel())

        return channel_mask, num_selected, gradcam_map

    def _generate_cache_key(self, num_samples: int, chunk_size: int) -> str:
        config_str = (
            f"{self.model_name}_"
            f"{self.target_layer_name}_"
            f"thresh{self.cumulative_threshold}_"
            f"samples{num_samples}_"
            f"chunk{chunk_size}_"
            f"gcmap1"
        )
        config_str = config_str.replace('[', '_').replace(']', '').replace('.', 'p')
        # Keep skip-grad caches on a separate key so the two flavors never
        # overwrite each other. The schema is identical either way.
        if self.skip_grad:
            config_str += "_nograd"
        return config_str

    def _save_chunk_part(self, cache_key: str, part_idx: int,
                         activation_chunk, mask_chunk, label_chunk,
                         gradcam_chunk):
        cache_dir = self.cache_dir / f"activations_{cache_key}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        part_path = cache_dir / f"part_{part_idx:04d}.pkl"
        joblib.dump({
            'activation': activation_chunk,
            'mask': mask_chunk,
            'label': label_chunk,
            'gradcam_map': gradcam_chunk,
        }, part_path, compress=3)

    def _save_metadata(self, cache_key: str, metadata: Dict):
        cache_dir = self.cache_dir / f"activations_{cache_key}"
        metadata_path = cache_dir / "metadata.pkl"
        print(f"\nSaving cache metadata...")
        print(f"  Cache directory: {cache_dir}")
        joblib.dump(metadata, metadata_path, compress=3)
        total_size = sum(f.stat().st_size for f in cache_dir.glob("*.pkl"))
        cache_size_mb = total_size / (1024 * 1024)
        print(f"  Total cache size: {cache_size_mb:.1f} MB")
        print(f"  Number of parts: {metadata['num_chunks']}")
        print(f"Activations{'' if self.skip_grad else ' + Grad-CAM maps'} cached!")

    def _check_cache_exists(self, cache_key: str) -> bool:
        cache_dir = self.cache_dir / f"activations_{cache_key}"
        return (cache_dir / "metadata.pkl").exists()

    def extract(self, data_loader: DataLoader,
                normalize: bool = True,
                chunk_size: int = ACTIVATION_CHUNK_SIZE,
                force_reextract: bool = False):
        """Run the extraction. Writes chunks incrementally to disk."""
        num_samples = len(data_loader.dataset)
        cache_key = self._generate_cache_key(num_samples, chunk_size)
        cache_dir = self.cache_dir / f"activations_{cache_key}"

        if self._check_cache_exists(cache_key) and not force_reextract:
            print(f"\n{'='*80}")
            print(f"Cache already exists -- skipping extraction.")
            print(f"  {cache_dir}")
            print(f"  (pass --force_reextract to rebuild)")
            print(f"{'='*80}")
            return cache_dir, cache_key

        chunk_activations = []
        chunk_masks = []
        chunk_labels = []
        chunk_gradcams = []
        channel_selection_stats = []
        total_processed = 0
        chunk_idx = 0

        print(f"\nCollecting activation maps"
              f"{'' if self.skip_grad else ' + Grad-CAM maps'} "
              f"(streaming to disk)...")
        print(f"  Chunk size: {chunk_size} images")
        if self.skip_grad:
            print(f"  GradCAM: SKIPPED (mask=all-True, "
                  f"gradcam_map=uniform placeholder)")
        else:
            print(f"  GradCAM threshold: "
                  f"{self.cumulative_threshold * 100:.0f}%")
        print(f"  Output: {cache_dir}")

        for images, labels in tqdm(data_loader, desc="Extracting"):
            if self.skip_grad:
                # Fast path: run the whole batch through the model in one shot.
                # The forward hook fills self.activations for the full batch;
                # we slice per image afterward. Numerically identical to the
                # batch-of-1 path -- the hook captures the same tensor either
                # way -- but with ~1/batch_size the GPU launch overhead.
                batch = images.to(self.device)
                with torch.no_grad():
                    _ = self.model(batch)
                    batch_acts = self.activations.cpu()      # [B, C, H, W]

                for i in range(batch.size(0)):
                    activations = batch_acts[i:i+1]          # [1, C, H, W]
                    label = labels[i:i+1]

                    channel_mask, num_selected, gradcam_map = \
                        self._neutral_mask_and_map()
                    channel_selection_stats.append(num_selected)

                    chunk_activations.append(activations)
                    chunk_masks.append(channel_mask.cpu())
                    chunk_labels.append(label)
                    chunk_gradcams.append(gradcam_map.cpu().unsqueeze(0))
                    total_processed += 1

                    if len(chunk_activations) >= chunk_size:
                        chunk_idx = self._flush_chunk(
                            cache_key, chunk_idx,
                            chunk_activations, chunk_masks,
                            chunk_labels, chunk_gradcams,
                            normalize)
                        chunk_activations, chunk_masks = [], []
                        chunk_labels, chunk_gradcams = [], []
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            else:
                # GradCAM path: needs per-image backprop, so process one at a
                # time exactly as before.
                for i in range(images.size(0)):
                    image = images[i:i+1].to(self.device)
                    label = labels[i:i+1]

                    with torch.no_grad():
                        _ = self.model(image)
                        activations = self.activations.clone()

                    channel_mask, num_selected, gradcam_map = \
                        self._select_channels_and_gradcam_map(image)
                    channel_selection_stats.append(num_selected)

                    chunk_activations.append(activations.cpu())
                    chunk_masks.append(channel_mask.cpu())
                    chunk_labels.append(label)
                    chunk_gradcams.append(gradcam_map.cpu().unsqueeze(0))
                    total_processed += 1

                    if len(chunk_activations) >= chunk_size:
                        chunk_idx = self._flush_chunk(
                            cache_key, chunk_idx,
                            chunk_activations, chunk_masks,
                            chunk_labels, chunk_gradcams,
                            normalize)
                        chunk_activations, chunk_masks = [], []
                        chunk_labels, chunk_gradcams = [], []
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        if len(chunk_activations) > 0:
            chunk_idx = self._flush_chunk(
                cache_key, chunk_idx,
                chunk_activations, chunk_masks,
                chunk_labels, chunk_gradcams,
                normalize)

        avg_selected = float(np.mean(channel_selection_stats))
        std_selected = float(np.std(channel_selection_stats))

        print(f"\nCollection complete:")
        print(f"  Total samples: {total_processed}")
        print(f"  Number of chunks: {chunk_idx}")
        print(f"  Avg channels selected: {avg_selected:.1f} +/- "
              f"{std_selected:.1f} (out of {self.num_channels})")

        metadata = {
            'model_name': self.model_name,
            'target_layer': self.target_layer_name,
            'cumulative_threshold': self.cumulative_threshold,
            'normalized': normalize,
            'num_channels': self.num_channels,
            'spatial_size': self.spatial_size,
            'avg_channels_selected': avg_selected,
            'std_channels_selected': std_selected,
            'total_samples': total_processed,
            'num_chunks': chunk_idx,
            'has_gradcam_map': True,
            'skip_grad': self.skip_grad,
        }
        self._save_metadata(cache_key, metadata)
        return cache_dir, cache_key

    def _flush_chunk(self, cache_key, chunk_idx,
                     chunk_activations, chunk_masks,
                     chunk_labels, chunk_gradcams, normalize):
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
                        channel_data = torch.clamp(
                            channel_data, min=0.0, max=scale_factor)
                        act_chunk[:, c, :, :] = (
                            channel_data / (scale_factor + 1e-8))

        self._save_chunk_part(cache_key, chunk_idx,
                              act_chunk, mask_chunk,
                              label_chunk, gradcam_chunk)
        return chunk_idx + 1


# ==========================================
# CLI
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone activation cache extractor "
                    "(byte-compatible with run_xcsae_full.py).")
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--target_layer', type=str, default=None,
                        help="Defaults to the model's default target layer "
                             "(see MODEL_CONFIGS).")
    parser.add_argument('--cumulative_threshold', type=float, default=0.95,
                        help="GradCAM cumulative-weight threshold for "
                             "channel selection (ignored when --skip_grad).")
    parser.add_argument('--images_per_class', type=int,
                        default=IMAGES_PER_CLASS)
    parser.add_argument('--chunk_size', type=int,
                        default=ACTIVATION_CHUNK_SIZE)
    parser.add_argument('--force_resample', action='store_true',
                        help="Re-run parquet sampling even if "
                             "/data/imagenet1k_sampled/metadata.pkl exists.")
    parser.add_argument('--force_reextract', action='store_true',
                        help="Re-extract activations even if the gcmap1 "
                             "cache exists.")
    parser.add_argument('--data_seed', type=int, default=DEFAULT_DATA_SEED,
                        help="Seed for parquet sampling (controls "
                             "random.sample in create_sampled_dataset).")
    parser.add_argument('--cache_dir', type=str,
                        default=str(ACTIVATION_CACHE_DIR))

    # --skip_grad is ON by default. Use --no_skip_grad to compute real GradCAM.
    grad_group = parser.add_mutually_exclusive_group()
    grad_group.add_argument(
        '--skip_grad', dest='skip_grad', action='store_true',
        help="Only cache activations; skip GradCAM/scoring entirely. "
             "mask is filled all-True and gradcam_map uniform so the cache "
             "keeps the same interface. (DEFAULT)")
    grad_group.add_argument(
        '--no_skip_grad', dest='skip_grad', action='store_false',
        help="Compute real GradCAM channel masks and spatial maps "
             "(original behavior).")
    parser.set_defaults(skip_grad=True)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*80)
    print("Activation Cache Extractor (standalone)")
    print(f"Backbone: {args.model.upper()}")
    print(f"Device: {device}")
    print(f"Mode: {'ACTIVATIONS ONLY (skip_grad)' if args.skip_grad else 'ACTIVATIONS + GRADCAM'}")
    print("="*80)

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
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE_COLLECTION,
                             shuffle=False, num_workers=4)

    extractor = ActivationExtractor(
        model_name=args.model,
        target_layer=args.target_layer,
        device=device,
        cumulative_threshold=args.cumulative_threshold,
        cache_dir=Path(args.cache_dir),
        skip_grad=args.skip_grad,
    )

    cache_dir, cache_key = extractor.extract(
        data_loader,
        normalize=True,
        chunk_size=args.chunk_size,
        force_reextract=args.force_reextract,
    )

    print(f"\n{'='*80}")
    print(f"Done.")
    print(f"  Cache dir : {cache_dir}")
    print(f"  Cache key : {cache_key}")
    print(f"\nUse with pca_extract.py:")
    print(f"  python pca_extract.py --cache_key {cache_key} \\")
    print(f"      --D 200 --device cuda \\")
    print(f"      --save pca_baseline_{args.model}_D200_model.pkl")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()