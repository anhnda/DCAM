"""
Multi-Channel ConvSAE Visualization on ImageNet-1k Test Set (Multi-Model Support)

This script has TWO modes:

1. RANDOM MODE: Visualize random individual test images
2. CONSISTENCY MODE: Analyze feature consistency across multiple images per class (3 images per class)

Supports multiple backbone architectures:
- ResNet50 (default): layer3, 1024 channels, 14×14 resolution
- ResNet18: layer3, 256 channels, 14×14 resolution
- VGG16: features[16], 256 channels, 28×28 resolution
- EfficientNet-B0: features[4], ~80 channels, 14×14 resolution

The script:
1. Samples test images from ImageNet-1k validation set (from parquet files)
2. Caches sampled test images to /data/imagenet1k_sampletest for reuse
3. Visualizes ConvSAE features on test images using trained model
4. In consistency mode: Shows which features are common across images of the same class

Usage:

    # CONSISTENCY MODE: Analyze 10 classes with ResNet50 (default)
    python visualize_testmf_full.py --num_classes 10 --top_k_features 12

    # CONSISTENCY MODE: Custom number of images per class
    python visualize_testmf_full.py --num_classes 5 --images_per_class_viz 3 --top_k_features 16

    # RANDOM MODE: Visualize 10 random individual images
    python visualize_testmf_full.py --num_samples 10 --top_k_features 16

    # Use VGG16 backbone
    python visualize_testmf_full.py --model vgg16 --num_classes 5

    # Use EfficientNet backbone
    python visualize_testmf_full.py --model efficientnet --num_samples 10

    # Use ResNet18 backbone
    python visualize_testmf_full.py --model resnet18 --num_classes 5

    # Force re-sampling with more test images per class
    python visualize_testmf_full.py --force_resample --test_images_per_class 10 --num_classes 5

Output:
    - Consistency mode: class_{label}_consistency.png showing 3 images with common features highlighted
    - Random mode: test_sample_{n}_label{label}.png for individual images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import joblib
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import io
import pandas as pd
import random
from collections import defaultdict
from tqdm import tqdm

# Import our model class
sys.path.append('.')
from run_xcsae_full import MultiChannelConvSAE
from src.gradcam import GradCAM
from full_classes import IMAGENET2012_CLASSES


# ==========================================
# Model Configurations
# ==========================================

MODEL_CONFIGS = {
    'resnet50': {
        'model_fn': lambda: models.resnet50(pretrained=True),
        'default_target_layer': 'layer3',
        'description': 'ResNet50 (layer3: 1024ch, 14×14)'
    },
    'resnet18': {
        'model_fn': lambda: models.resnet18(pretrained=True),
        'default_target_layer': 'layer3',
        'description': 'ResNet18 (layer3: 256ch, 14×14)'
    },
    'vgg16': {
        'model_fn': lambda: models.vgg16(pretrained=True),
        'default_target_layer': 'features[16]',
        'description': 'VGG16 (features[16]: 256ch, 28×28)'
    },
    'efficientnet': {
        'model_fn': lambda: models.efficientnet_b0(pretrained=True),
        'default_target_layer': 'features[4]',
        'description': 'EfficientNet-B0 (features[4]: ~80ch, 14×14)'
    }
}


# ==========================================
# Test Image Sampler
# ==========================================

class ImageNet1kTestSampler:
    """
    Samples test images from ImageNet-1k validation parquet files.
    Caches sampled images to disk for reuse.
    """

    def __init__(self,
                 raw_dir: Path,
                 test_dir: Path,
                 images_per_class: int = 5,
                 force_resample: bool = False):
        """
        Args:
            raw_dir: Path to raw ImageNet parquet files
            test_dir: Path to save sampled test images
            images_per_class: Number of test images to sample per class
            force_resample: Force resampling even if cache exists
        """
        self.raw_dir = raw_dir
        self.test_dir = test_dir
        self.images_per_class = images_per_class

        # Create test directory
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.test_dir / "test_metadata.pkl"

        # Check cache
        if self.metadata_path.exists() and not force_resample:
            print(f"\n{'='*80}")
            print(f"Loading cached test samples from {self.test_dir}")
            print(f"{'='*80}")
            self.load_cached_samples()
        else:
            print(f"\n{'='*80}")
            print(f"Creating new test sample dataset...")
            print(f"{'='*80}")
            self.create_test_samples()

    def create_test_samples(self):
        """Sample test images from validation parquet files."""
        print(f"Sampling {self.images_per_class} test images per class...")
        print(f"Total target: {self.images_per_class * 1000} images")

        # Class mappings
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(IMAGENET2012_CLASSES.keys())}

        # Storage
        class_samples = defaultdict(list)

        # Find validation parquet files
        val_parquet_files = sorted(self.raw_dir.glob("validation-*.parquet"))

        if len(val_parquet_files) == 0:
            raise FileNotFoundError(f"No validation parquet files found in {self.raw_dir}")

        print(f"Found {len(val_parquet_files)} validation parquet files")

        # Read and sample
        for parquet_file in tqdm(val_parquet_files, desc="Reading validation parquet files"):
            df = pd.read_parquet(parquet_file)

            for idx, row in df.iterrows():
                label = row['label']

                if len(class_samples[label]) < self.images_per_class:
                    image_bytes = row['image']['bytes']
                    class_samples[label].append((image_bytes, label))

            # Check if done
            min_samples = min(len(samples) for samples in class_samples.values()) if class_samples else 0
            if min_samples >= self.images_per_class and len(class_samples) == 1000:
                print(f"\nCollected {self.images_per_class} samples for all 1000 classes!")
                break

        # Build final sample list
        self.samples = []
        for class_idx in range(1000):
            if len(class_samples[class_idx]) >= self.images_per_class:
                sampled = random.sample(class_samples[class_idx], self.images_per_class)
                self.samples.extend(sampled)
            else:
                print(f"WARNING: Class {class_idx} has only {len(class_samples[class_idx])} test samples")
                self.samples.extend(class_samples[class_idx])

        print(f"\nTotal sampled test images: {len(self.samples)}")

        # Save
        print(f"Saving test samples to {self.test_dir}...")
        joblib.dump({
            'samples': self.samples,
            'images_per_class': self.images_per_class,
            'num_classes': 1000,
            'wnid_to_idx': self.wnid_to_idx
        }, self.metadata_path)
        print(f"✓ Test samples cached!")

    def load_cached_samples(self):
        """Load cached test samples."""
        metadata = joblib.load(self.metadata_path)
        self.samples = metadata['samples']
        self.wnid_to_idx = metadata['wnid_to_idx']

        print(f"Loaded {len(self.samples)} test images from cache")
        print(f"  Images per class: {metadata['images_per_class']}")
        print(f"  Number of classes: {metadata['num_classes']}")

    def get_random_samples(self, n: int) -> List[Tuple[Image.Image, int]]:
        """
        Get n random test samples.

        Returns:
            List of (PIL Image, label) tuples
        """
        sampled_indices = random.sample(range(len(self.samples)), min(n, len(self.samples)))

        result = []
        for idx in sampled_indices:
            image_bytes, label = self.samples[idx]
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            result.append((image, label))

        return result

    def get_samples_by_class(self, num_classes: int, images_per_class: int = 3) -> Dict[int, List[Image.Image]]:
        """
        Get samples grouped by class to analyze feature consistency.

        Args:
            num_classes: Number of classes to sample
            images_per_class: Number of images per class (default: 3)

        Returns:
            Dictionary mapping class_idx -> list of PIL Images
        """
        # Group samples by class
        class_to_samples = defaultdict(list)
        for image_bytes, label in self.samples:
            class_to_samples[label].append(image_bytes)

        # Select random classes that have enough samples
        available_classes = [cls for cls, samples in class_to_samples.items()
                           if len(samples) >= images_per_class]

        if len(available_classes) < num_classes:
            print(f"Warning: Only {len(available_classes)} classes have {images_per_class}+ samples")
            num_classes = len(available_classes)

        selected_classes = random.sample(available_classes, num_classes)

        # Sample images from each class
        result = {}
        for cls in selected_classes:
            sampled_bytes = random.sample(class_to_samples[cls], images_per_class)
            images = [Image.open(io.BytesIO(img_bytes)).convert('RGB') for img_bytes in sampled_bytes]
            result[cls] = images

        return result


# ==========================================
# Multi-Model Multi-Channel SAE Visualizer
# ==========================================

class MultiModelSAEVisualizerTestMF:
    """
    Visualizer for Multi-Channel ConvSAE on ImageNet-1k test set.
    Supports multiple backbone architectures.
    """

    def __init__(self,
                 model_name: str,
                 target_layer_name: str,
                 csae_model_path: str,
                 device='cuda',
                 cumulative_threshold=0.85):
        """
        Args:
            model_name: Backbone model name (resnet50, resnet18, vgg16, efficientnet)
            target_layer_name: Target layer name for activation extraction
            csae_model_path: Path to trained Multi-Channel ConvSAE
            device: Device to run on
            cumulative_threshold: GradCAM threshold for visualization
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.target_layer_name = target_layer_name
        self.cumulative_threshold = cumulative_threshold

        # Load ConvSAE
        print(f"Loading Multi-Channel ConvSAE from {csae_model_path}...")
        self.csae_model = joblib.load(csae_model_path).to(self.device)
        self.csae_model.eval()
        print(f"  ✓ Model loaded: {self.csae_model.in_channels}→{self.csae_model.hidden_dim}, top_k={self.csae_model.top_k}")

        # Load backbone model
        print(f"Loading {model_name} backbone...")
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

        self.backbone = MODEL_CONFIGS[model_name]['model_fn']().to(self.device)
        self.backbone.eval()

        # Get target layer
        self.target_layer = self._get_target_layer()

        # Get dimensions
        self._detect_layer_dimensions()

        print(f"  ✓ Target layer: {target_layer_name}, {self.num_channels} channels, {self.spatial_size}×{self.spatial_size}")

        # Hook for activations
        self.layer_activations = None
        self.target_layer.register_forward_hook(self._save_layer_activation)

        # GradCAM
        self.gradcam = GradCAM(self.backbone, self.target_layer)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("✓ Visualizer ready!\n")

    def _get_target_layer(self):
        """Get target layer from model."""
        if self.model_name in ['resnet50', 'resnet18']:
            if 'layer1' in self.target_layer_name:
                return self.backbone.layer1
            elif 'layer2' in self.target_layer_name:
                return self.backbone.layer2
            elif 'layer3' in self.target_layer_name:
                return self.backbone.layer3
            elif 'layer4' in self.target_layer_name:
                return self.backbone.layer4
            else:
                raise ValueError(f"Unknown ResNet layer: {self.target_layer_name}")

        elif self.model_name == 'vgg16':
            target_idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            return self.backbone.features[target_idx]

        elif self.model_name == 'efficientnet':
            target_idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            return self.backbone.features[target_idx]

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _detect_layer_dimensions(self):
        """Auto-detect layer dimensions."""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

            if self.model_name in ['resnet50', 'resnet18']:
                x = self.backbone.conv1(dummy_input)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x)

                if 'layer1' in self.target_layer_name:
                    pass
                elif 'layer2' in self.target_layer_name:
                    x = self.backbone.layer2(x)
                elif 'layer3' in self.target_layer_name:
                    x = self.backbone.layer2(x)
                    x = self.backbone.layer3(x)
                elif 'layer4' in self.target_layer_name:
                    x = self.backbone.layer2(x)
                    x = self.backbone.layer3(x)
                    x = self.backbone.layer4(x)

            elif self.model_name == 'vgg16':
                target_idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
                for i in range(target_idx + 1):
                    x = self.backbone.features[i](dummy_input if i == 0 else x)

            elif self.model_name == 'efficientnet':
                target_idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
                for i in range(target_idx + 1):
                    x = self.backbone.features[i](dummy_input if i == 0 else x)

            self.num_channels = x.shape[1]
            self.spatial_size = x.shape[2]

    def _save_layer_activation(self, module, input, output):
        """Hook to save layer activations."""
        self.layer_activations = output.detach()

    def _normalize_layer_activations(self, acts: torch.Tensor) -> torch.Tensor:
        """Normalize activations using per-channel 99th percentile."""
        normalized = acts.clone()

        for c in range(acts.shape[1]):
            channel_data = acts[0, c, :, :]

            if channel_data.abs().sum() < 1e-8:
                continue

            non_zero_vals = channel_data[channel_data > 1e-8]
            if len(non_zero_vals) > 0:
                scale_factor = torch.quantile(non_zero_vals, 0.99)

                if scale_factor > 1e-8:
                    channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                    normalized[0, c, :, :] = channel_data / (scale_factor + 1e-8)

        return normalized

    def _select_channels_with_gradcam(self, image: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor, int]:
        """Use GradCAM to identify top channels."""
        weights, _, pred_class = self.gradcam.forward(image, class_idx=None, verbose=False)

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

        return channel_mask, num_selected, weights, pred_class

    def extract_features(self, image: Image.Image, label: int, top_k: int = 16) -> Dict:
        """
        Extract top-k activated features for a test image.

        Args:
            image: PIL Image
            label: Ground truth label
            top_k: Number of top features to extract

        Returns:
            Dictionary with visualization data
        """
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract activations
        with torch.no_grad():
            logits = self.backbone(image_tensor)
            layer_acts = self.layer_activations.clone()
            pred_label = logits.argmax(dim=1).item()

        # GradCAM
        channel_mask, num_selected, channel_weights, gradcam_pred = self._select_channels_with_gradcam(image_tensor)

        # Normalize
        layer_acts_norm = self._normalize_layer_activations(layer_acts)

        # CSAE
        with torch.no_grad():
            _, sparse_features = self.csae_model(layer_acts_norm, use_topk=True)

        # Feature importance
        feature_importance = sparse_features.sum(dim=(2, 3)).squeeze()
        top_k_values, top_k_indices = torch.topk(feature_importance, k=min(top_k, len(feature_importance)))

        # Top features
        top_features = []
        for idx, importance in zip(top_k_indices, top_k_values):
            activation_map = sparse_features[0, idx, :, :].cpu()
            top_features.append((idx.item(), importance.item(), activation_map))

        # Get class names
        idx_to_wnid = {idx: wnid for wnid, idx in enumerate(IMAGENET2012_CLASSES.keys())}
        true_class = list(IMAGENET2012_CLASSES.values())[label]
        pred_class = list(IMAGENET2012_CLASSES.values())[pred_label]

        results = {
            'image': image,
            'label': label,
            'pred_label': pred_label,
            'true_class': true_class,
            'pred_class': pred_class,
            'correct': (label == pred_label),
            'num_selected_channels': num_selected,
            'channel_weights': channel_weights.cpu(),
            'top_features': top_features,
            'feature_importance': feature_importance.cpu()
        }

        return results

    def visualize_features(self, image: Image.Image, label: int,
                          top_k: int = 16, save_path: str = None):
        """
        Visualize top-k CSAE features for a test image.

        Args:
            image: PIL Image
            label: Ground truth label
            top_k: Number of features to visualize
            save_path: Path to save visualization
        """
        print(f"Processing test image (label={label})...")

        # Extract features
        results = self.extract_features(image, label, top_k=top_k)

        # Visualize
        print("Generating visualization...")
        self._plot_feature_grid(results, save_path)

        print(f"✓ Visualization complete!")
        if save_path:
            print(f"  Saved to: {save_path}")

    def _plot_feature_grid(self, results: Dict, save_path: str = None):
        """Plot grid of CSAE features for test image."""
        image = results['image']
        top_features = results['top_features']
        true_class = results['true_class']
        pred_class = results['pred_class']
        correct = results['correct']
        num_selected = results['num_selected_channels']

        n_features = len(top_features)
        n_cols = 8
        n_rows = 1 + (n_features + 3) // 4

        fig = plt.figure(figsize=(24, 3.5 * n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

        # ===== Row 0: Overview =====
        # Input image
        ax_img = fig.add_subplot(gs[0, 0:2])
        ax_img.imshow(image)
        ax_img.set_title("Test Image", fontsize=12, fontweight='bold')
        ax_img.axis('off')

        # Prediction info
        ax_info = fig.add_subplot(gs[0, 2:4])
        ax_info.axis('off')

        status = "✓ CORRECT" if correct else "✗ WRONG"
        status_color = "green" if correct else "red"

        info_text = f"Prediction: {status}\n"
        info_text += f"  True: {true_class[:40]}...\n" if len(true_class) > 40 else f"  True: {true_class}\n"
        info_text += f"  Pred: {pred_class[:40]}...\n\n" if len(pred_class) > 40 else f"  Pred: {pred_class}\n\n"
        info_text += f"Model: ImageNet-1k Full (1000 classes)\n"
        info_text += f"  • Backbone: {self.model_name}\n"
        info_text += f"  • {self.num_channels} input channels\n"
        info_text += f"  • {self.csae_model.hidden_dim} CSAE features\n"
        info_text += f"  • Top-k: {self.csae_model.top_k}\n"
        info_text += f"  • GradCAM: {num_selected}/{self.num_channels} channels"

        ax_info.text(0.05, 0.5, info_text, fontsize=9, family='monospace',
                    verticalalignment='center', transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round',
                             facecolor='lightgreen' if correct else 'lightcoral',
                             alpha=0.3))

        # Feature importance bar chart
        ax_bar = fig.add_subplot(gs[0, 4:])
        importances = [imp for _, imp, _ in top_features]
        feature_indices = [f"F{idx}" for idx, _, _ in top_features]
        ax_bar.bar(range(len(importances)), importances,
                  color='steelblue', alpha=0.8, edgecolor='navy')
        ax_bar.set_xlabel('Feature Index', fontsize=10)
        ax_bar.set_ylabel('Importance', fontsize=10)
        ax_bar.set_title(f'Top-{n_features} CSAE Feature Importance',
                        fontsize=12, fontweight='bold')
        ax_bar.set_xticks(range(len(importances)))
        ax_bar.set_xticklabels(feature_indices, rotation=45, ha='right', fontsize=8)
        ax_bar.grid(True, alpha=0.3, axis='y')

        # ===== Rows 1+: Feature maps =====
        for i, (feat_idx, importance, activation_map) in enumerate(top_features):
            row = 1 + i // 4
            col = (i % 4) * 2

            ax_feat = fig.add_subplot(gs[row, col:col+2])
            im = ax_feat.imshow(activation_map.numpy(), cmap='hot', interpolation='bilinear')
            ax_feat.set_title(f"Feature {feat_idx}\nImp: {importance:.2f}",
                             fontsize=10, fontweight='bold')
            ax_feat.axis('off')

            cbar = plt.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)

        status_str = "CORRECT" if correct else "WRONG"
        plt.suptitle(f'ImageNet-1k Test Set Visualization ({status_str})\n' +
                    f'Multi-Channel ConvSAE ({self.model_name}): {self.num_channels} channels → {self.csae_model.hidden_dim} features',
                    fontsize=13, fontweight='bold', y=0.998)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_class_consistency(self, images: List[Image.Image], label: int,
                                   top_k: int = 12, save_path: str = None):
        """
        Visualize feature consistency across multiple images from the same class.

        Args:
            images: List of PIL Images from the same class (typically 3)
            label: Class label
            top_k: Number of top features to show per image
            save_path: Path to save visualization
        """
        n_images = len(images)
        print(f"\nAnalyzing {n_images} images from class {label}...")

        # Get class name
        class_name = list(IMAGENET2012_CLASSES.values())[label]

        # Extract features for all images
        all_results = []
        all_feature_indices = set()

        for i, image in enumerate(images):
            results = self.extract_features(image, label, top_k=top_k)
            all_results.append(results)

            # Collect all feature indices
            for feat_idx, _, _ in results['top_features']:
                all_feature_indices.add(feat_idx)

            print(f"  Image {i+1}: {len(results['top_features'])} top features, "
                  f"prediction={'✓' if results['correct'] else '✗'}")

        # Analyze feature overlap
        common_features = self._find_common_features(all_results, top_k)
        print(f"  Common features across all images: {len(common_features)}")

        # Create visualization
        print("Generating class consistency visualization...")
        self._plot_class_consistency(all_results, label, class_name, common_features, save_path)

        print(f"✓ Class consistency visualization complete!")
        if save_path:
            print(f"  Saved to: {save_path}")

    def _find_common_features(self, all_results: List[Dict], top_k: int) -> List[int]:
        """Find features that appear in top-k for all images."""
        if not all_results:
            return []

        # Get feature sets for each image
        feature_sets = []
        for results in all_results:
            feature_set = set([feat_idx for feat_idx, _, _ in results['top_features']])
            feature_sets.append(feature_set)

        # Find intersection
        common = feature_sets[0]
        for fs in feature_sets[1:]:
            common = common.intersection(fs)

        return sorted(list(common))

    def _plot_class_consistency(self, all_results: List[Dict], label: int,
                                class_name: str, common_features: List[int],
                                save_path: str = None):
        """
        Plot feature consistency visualization for multiple images from same class.

        Layout:
        - Row 0: Class info + 3 input images + common features bar chart
        - Rows 1-3: Top features for each image (3 rows, one per image)
        """
        n_images = len(all_results)
        n_features_per_image = min(12, len(all_results[0]['top_features']))

        fig = plt.figure(figsize=(28, 4 + 3.5 * n_images))
        gs = fig.add_gridspec(n_images + 1, 16, hspace=0.5, wspace=0.4)

        # ===== Row 0: Overview =====
        # Class info
        ax_info = fig.add_subplot(gs[0, 0:2])
        ax_info.axis('off')

        info_text = f"Class {label}\n"
        info_text += f"{class_name[:50]}...\n\n" if len(class_name) > 50 else f"{class_name}\n\n"
        info_text += f"Analyzing:\n"
        info_text += f"  • {n_images} test images\n"
        info_text += f"  • Top-{n_features_per_image} features each\n"
        info_text += f"  • {len(common_features)} common features\n\n"
        info_text += f"Backbone: {self.model_name}\n"

        correct_count = sum(1 for r in all_results if r['correct'])
        info_text += f"Predictions: {correct_count}/{n_images} correct"

        ax_info.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # Input images
        for i, results in enumerate(all_results):
            ax_img = fig.add_subplot(gs[0, 2+i*2:2+(i+1)*2])
            ax_img.imshow(results['image'])
            status = "✓" if results['correct'] else "✗"
            ax_img.set_title(f"Image {i+1} {status}", fontsize=11, fontweight='bold')
            ax_img.axis('off')

        # Common features analysis
        ax_common = fig.add_subplot(gs[0, 8:])
        if common_features:
            # Get average importance for common features
            common_importances = []
            for feat_idx in common_features[:15]:  # Show top 15 common
                avg_imp = np.mean([
                    next((imp for idx, imp, _ in r['top_features'] if idx == feat_idx), 0)
                    for r in all_results
                ])
                common_importances.append(avg_imp)

            ax_common.barh(range(len(common_features[:15])), common_importances,
                          color='green', alpha=0.7, edgecolor='darkgreen')
            ax_common.set_yticks(range(len(common_features[:15])))
            ax_common.set_yticklabels([f'F{f}' for f in common_features[:15]], fontsize=8)
            ax_common.set_xlabel('Avg Importance', fontsize=10)
            ax_common.set_title(f'Common Features ({len(common_features)} total)',
                              fontsize=11, fontweight='bold')
            ax_common.grid(True, alpha=0.3, axis='x')
            ax_common.invert_yaxis()
        else:
            ax_common.text(0.5, 0.5, 'No common features\nin top-k across all images',
                          ha='center', va='center', fontsize=10,
                          transform=ax_common.transAxes)
            ax_common.axis('off')

        # ===== Rows 1-3: Top features for each image =====
        for img_idx, results in enumerate(all_results):
            row = img_idx + 1
            top_features = results['top_features'][:n_features_per_image]

            for feat_idx, (f_idx, importance, activation_map) in enumerate(top_features):
                if feat_idx >= 12:  # Max 12 features per row
                    break

                col_start = feat_idx
                col_end = col_start + 1

                ax_feat = fig.add_subplot(gs[row, col_start:col_end])

                # Highlight common features
                is_common = f_idx in common_features
                border_color = 'green' if is_common else 'none'

                im = ax_feat.imshow(activation_map.numpy(), cmap='hot', interpolation='bilinear')
                title_color = 'green' if is_common else 'black'
                ax_feat.set_title(f"F{f_idx}\n{importance:.1f}",
                                fontsize=9, fontweight='bold' if is_common else 'normal',
                                color=title_color)
                ax_feat.axis('off')

                # Add border for common features
                if is_common:
                    for spine in ax_feat.spines.values():
                        spine.set_edgecolor('green')
                        spine.set_linewidth(3)
                        spine.set_visible(True)

        # Title
        plt.suptitle(f'Feature Consistency Analysis ({self.model_name}): Class {label} ({class_name[:40]}...)\n' +
                    f'{n_images} Test Images | Top-{n_features_per_image} Features | ' +
                    f'{len(common_features)} Common Features (highlighted in green)',
                    fontsize=14, fontweight='bold', y=0.998)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize Multi-Channel ConvSAE on ImageNet-1k Test Set (Multi-Model Support)'
    )
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Backbone model (default: resnet50)')
    parser.add_argument('--target_layer', type=str, default=None,
                       help='Target layer name (default: model-specific default)')
    parser.add_argument('--csae_model', type=str, default=None,
                       help='Path to trained ConvSAE model (auto-detected if not specified)')
    parser.add_argument('--raw_data_dir', type=str,
                       default='/data/imagenet_raw/data',
                       help='Path to raw ImageNet parquet files')
    parser.add_argument('--test_data_dir', type=str,
                       default='/data/imagenet1k_sampletest',
                       help='Path to cached test samples')
    parser.add_argument('--test_images_per_class', type=int, default=5,
                       help='Number of test images to sample per class')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of random test images to visualize (random mode)')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes to analyze (consistency mode, default: analyze by class)')
    parser.add_argument('--images_per_class_viz', type=int, default=3,
                       help='Images per class for consistency analysis (default: 3)')
    parser.add_argument('--top_k_features', type=int, default=16,
                       help='Number of top features to visualize per image')
    parser.add_argument('--output_dir', type=str,
                       default='imagenet1k_test_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--force_resample', action='store_true',
                       help='Force resampling of test images')

    args = parser.parse_args()

    # Auto-detect target layer if not specified
    if args.target_layer is None:
        args.target_layer = MODEL_CONFIGS[args.model]['default_target_layer']

    # Auto-detect CSAE model path if not specified
    if args.csae_model is None:
        args.csae_model = f'imagenet1k_csae_{args.model}_model.pkl'

    print("="*80)
    print(f"Multi-Channel ConvSAE Test Set Visualization (ImageNet-1k Full)")
    print(f"Backbone: {args.model} ({MODEL_CONFIGS[args.model]['description']})")
    print(f"Target layer: {args.target_layer}")
    print(f"CSAE model: {args.csae_model}")
    print("="*80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Sample test images
    print("\nStep 1: Sampling test images...")
    sampler = ImageNet1kTestSampler(
        raw_dir=Path(args.raw_data_dir),
        test_dir=Path(args.test_data_dir),
        images_per_class=args.test_images_per_class,
        force_resample=args.force_resample
    )

    # Create visualizer
    print("\nStep 2: Loading ConvSAE model...")
    visualizer = MultiModelSAEVisualizerTestMF(
        model_name=args.model,
        target_layer_name=args.target_layer,
        csae_model_path=args.csae_model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Determine mode
    if args.num_classes is not None:
        # ===== CONSISTENCY MODE: Analyze multiple images per class =====
        print(f"\n{'='*80}")
        print(f"MODE: Class Consistency Analysis")
        print(f"  - Analyzing {args.num_classes} classes")
        print(f"  - {args.images_per_class_viz} images per class")
        print(f"  - Top-{args.top_k_features} features per image")
        print(f"{'='*80}")

        # Get samples by class
        print(f"\nStep 3: Sampling {args.images_per_class_viz} images from {args.num_classes} classes...")
        class_samples = sampler.get_samples_by_class(
            num_classes=args.num_classes,
            images_per_class=args.images_per_class_viz
        )
        print(f"Selected {len(class_samples)} classes")

        # Visualize each class
        print(f"\nStep 4: Generating class consistency visualizations...")
        correct_count = 0
        total_images = 0

        for class_idx, images in class_samples.items():
            class_name = list(IMAGENET2012_CLASSES.values())[class_idx]
            print(f"\n{'='*80}")
            print(f"Class {class_idx}: {class_name[:60]}...")
            print(f"{'='*80}")

            save_path = output_dir / f"class_{class_idx}_consistency_{args.model}.png"

            # Visualize consistency
            visualizer.visualize_class_consistency(
                images, class_idx,
                top_k=args.top_k_features,
                save_path=str(save_path)
            )

            # Count predictions (for statistics)
            for img in images:
                results = visualizer.extract_features(img, class_idx, top_k=1)
                if results['correct']:
                    correct_count += 1
                total_images += 1

        # Summary
        accuracy = (correct_count / total_images) * 100 if total_images > 0 else 0
        print(f"\n{'='*80}")
        print(f"✓ All class consistency visualizations complete!")
        print(f"  Output directory: {output_dir}")
        print(f"  Classes analyzed: {len(class_samples)}")
        print(f"  Total images: {total_images}")
        print(f"  Accuracy: {correct_count}/{total_images} ({accuracy:.1f}%)")
        print(f"{'='*80}")

    else:
        # ===== RANDOM MODE: Visualize random individual images =====
        print(f"\n{'='*80}")
        print(f"MODE: Random Sample Visualization")
        print(f"  - {args.num_samples} random test images")
        print(f"  - Top-{args.top_k_features} features per image")
        print(f"{'='*80}")

        # Get random test samples
        print(f"\nStep 3: Selecting {args.num_samples} random test images...")
        test_samples = sampler.get_random_samples(args.num_samples)
        print(f"Selected {len(test_samples)} test images")

        # Visualize each test image
        print(f"\nStep 4: Generating visualizations...")
        correct_count = 0

        for i, (image, label) in enumerate(test_samples):
            print(f"\n{'='*80}")
            print(f"Test Image {i+1}/{len(test_samples)} (label={label})")
            print(f"{'='*80}\n")

            save_path = output_dir / f"test_sample_{i+1}_label{label}_{args.model}.png"

            # Extract features to check prediction
            results = visualizer.extract_features(image, label, top_k=args.top_k_features)
            if results['correct']:
                correct_count += 1

            # Visualize
            visualizer.visualize_features(
                image, label,
                top_k=args.top_k_features,
                save_path=str(save_path)
            )

        # Summary
        accuracy = (correct_count / len(test_samples)) * 100
        print(f"\n{'='*80}")
        print(f"✓ All visualizations complete!")
        print(f"  Output directory: {output_dir}")
        print(f"  Accuracy on visualized samples: {correct_count}/{len(test_samples)} ({accuracy:.1f}%)")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
