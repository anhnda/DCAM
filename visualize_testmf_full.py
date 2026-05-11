"""
Multi-Channel ConvSAE Visualization on ImageNet-1k Test Set (Multi-Model Support)

UPDATE: Each feature-map row now begins with the per-image Grad-CAM heatmap,
making the decomposition visually explicit:

    [Grad-CAM map]  [Feature 1]  [Feature 2]  ...  [Feature N]

This matches the DCAM framing -- the Grad-CAM map should look like the sum of
the displayed sub-pattern features.

Modes:
1. RANDOM MODE: Visualize random individual test images.
   In this mode the Grad-CAM heatmap is shown in the overview row (once per
   image, since the feature panels for a single image span multiple rows).

2. CONSISTENCY MODE: Analyze feature consistency across multiple images per
   class. Each per-image row begins with that image's Grad-CAM heatmap, then
   the per-image top features (with common features highlighted in green).

Supports multiple backbone architectures:
- ResNet50 (default): layer3, 1024 channels, 14x14 resolution
- ResNet18: layer3, 256 channels, 14x14 resolution
- VGG16: features[16], 256 channels, 28x28 resolution
- EfficientNet-B0: features[4], ~80 channels, 14x14 resolution

Usage:
    python visualize_testmf_full.py --num_classes 10 --top_k_features 12
    python visualize_testmf_full.py --num_samples 10 --top_k_features 16
    python visualize_testmf_full.py --model vgg16 --num_classes 5

Output:
    - Consistency mode: class_{label}_consistency_{model}.png
    - Random mode: test_sample_{n}_label{label}_{model}.png
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
        'description': 'ResNet50 (layer3: 1024ch, 14x14)'
    },
    'resnet18': {
        'model_fn': lambda: models.resnet18(pretrained=True),
        'default_target_layer': 'layer3',
        'description': 'ResNet18 (layer3: 256ch, 14x14)'
    },
    'vgg16': {
        'model_fn': lambda: models.vgg16(pretrained=True),
        'default_target_layer': 'features[16]',
        'description': 'VGG16 (features[16]: 256ch, 28x28)'
    },
    'efficientnet': {
        'model_fn': lambda: models.efficientnet_b0(pretrained=True),
        'default_target_layer': 'features[4]',
        'description': 'EfficientNet-B0 (features[4]: ~80ch, 14x14)'
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
        self.raw_dir = raw_dir
        self.test_dir = test_dir
        self.images_per_class = images_per_class

        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.test_dir / "test_metadata.pkl"

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

        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(IMAGENET2012_CLASSES.keys())}

        class_samples = defaultdict(list)

        val_parquet_files = sorted(self.raw_dir.glob("validation-*.parquet"))

        if len(val_parquet_files) == 0:
            raise FileNotFoundError(f"No validation parquet files found in {self.raw_dir}")

        print(f"Found {len(val_parquet_files)} validation parquet files")

        for parquet_file in tqdm(val_parquet_files, desc="Reading validation parquet files"):
            df = pd.read_parquet(parquet_file)

            for idx, row in df.iterrows():
                label = row['label']

                if len(class_samples[label]) < self.images_per_class:
                    image_bytes = row['image']['bytes']
                    class_samples[label].append((image_bytes, label))

            min_samples = min(len(samples) for samples in class_samples.values()) if class_samples else 0
            if min_samples >= self.images_per_class and len(class_samples) == 1000:
                print(f"\nCollected {self.images_per_class} samples for all 1000 classes!")
                break

        self.samples = []
        for class_idx in range(1000):
            if len(class_samples[class_idx]) >= self.images_per_class:
                sampled = random.sample(class_samples[class_idx], self.images_per_class)
                self.samples.extend(sampled)
            else:
                print(f"WARNING: Class {class_idx} has only {len(class_samples[class_idx])} test samples")
                self.samples.extend(class_samples[class_idx])

        print(f"\nTotal sampled test images: {len(self.samples)}")

        print(f"Saving test samples to {self.test_dir}...")
        joblib.dump({
            'samples': self.samples,
            'images_per_class': self.images_per_class,
            'num_classes': 1000,
            'wnid_to_idx': self.wnid_to_idx
        }, self.metadata_path)
        print(f"Test samples cached!")

    def load_cached_samples(self):
        metadata = joblib.load(self.metadata_path)
        self.samples = metadata['samples']
        self.wnid_to_idx = metadata['wnid_to_idx']

        print(f"Loaded {len(self.samples)} test images from cache")
        print(f"  Images per class: {metadata['images_per_class']}")
        print(f"  Number of classes: {metadata['num_classes']}")

    def get_random_samples(self, n: int) -> List[Tuple[Image.Image, int]]:
        sampled_indices = random.sample(range(len(self.samples)), min(n, len(self.samples)))

        result = []
        for idx in sampled_indices:
            image_bytes, label = self.samples[idx]
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            result.append((image, label))

        return result

    def get_samples_by_class(self, num_classes: int, images_per_class: int = 3) -> Dict[int, List[Image.Image]]:
        class_to_samples = defaultdict(list)
        for image_bytes, label in self.samples:
            class_to_samples[label].append(image_bytes)

        available_classes = [cls for cls, samples in class_to_samples.items()
                           if len(samples) >= images_per_class]

        if len(available_classes) < num_classes:
            print(f"Warning: Only {len(available_classes)} classes have {images_per_class}+ samples")
            num_classes = len(available_classes)

        selected_classes = random.sample(available_classes, num_classes)

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
    Each feature row is prefixed with the per-image Grad-CAM heatmap.
    """

    def __init__(self,
                 model_name: str,
                 target_layer_name: str,
                 csae_model_path: str,
                 device='cuda',
                 cumulative_threshold=0.85):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.target_layer_name = target_layer_name
        self.cumulative_threshold = cumulative_threshold

        # Load ConvSAE
        print(f"Loading Multi-Channel ConvSAE from {csae_model_path}...")
        self.csae_model = joblib.load(csae_model_path).to(self.device)
        self.csae_model.eval()
        print(f"  Model loaded: {self.csae_model.in_channels}->{self.csae_model.hidden_dim}, top_k={self.csae_model.top_k}")

        # Load backbone model
        print(f"Loading {model_name} backbone...")
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

        self.backbone = MODEL_CONFIGS[model_name]['model_fn']().to(self.device)
        self.backbone.eval()

        self.target_layer = self._get_target_layer()
        self._detect_layer_dimensions()

        print(f"  Target layer: {target_layer_name}, {self.num_channels} channels, {self.spatial_size}x{self.spatial_size}")

        self.layer_activations = None
        self.target_layer.register_forward_hook(self._save_layer_activation)

        self.gradcam = GradCAM(self.backbone, self.target_layer)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("Visualizer ready!\n")

    def _get_target_layer(self):
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
        self.layer_activations = output.detach()

    def _normalize_layer_activations(self, acts: torch.Tensor) -> torch.Tensor:
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

    def _select_channels_and_gradcam_map(self, image: torch.Tensor
                                          ) -> Tuple[torch.Tensor, int, torch.Tensor, int, torch.Tensor]:
        """Use Grad-CAM to:
          (1) select important channels (binary mask),
          (2) return the raw alpha weights,
          (3) return the predicted class,
          (4) compute the spatial Grad-CAM map L = ReLU(sum_k alpha_k * A_k).
        """
        weights, _, pred_class = self.gradcam.forward(image, class_idx=None, verbose=False)
        # self.layer_activations: [1, C, H, W], populated by the forward hook
        # during gradcam.forward()

        # --- Channel selection ---
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

        # --- Spatial Grad-CAM map ---
        with torch.no_grad():
            acts = self.layer_activations[0]                       # [C, H, W]
            weighted = (weights.view(-1, 1, 1) * acts).sum(dim=0)  # [H, W]
            gradcam_map = F.relu(weighted)
            # Min-max normalize for display
            gmin, gmax = gradcam_map.min(), gradcam_map.max()
            if (gmax - gmin) > 1e-8:
                gradcam_map = (gradcam_map - gmin) / (gmax - gmin)
            else:
                gradcam_map = torch.zeros_like(gradcam_map)

        return channel_mask, num_selected, weights, pred_class, gradcam_map.cpu()

    def get_top_input_channels_for_feature(self, feature_id: int,
                                           top_n: int = 6,
                                           view: str = 'encoder'
                                           ) -> Tuple[List[int], List[float]]:
        """For a given SAE latent feature index, return the input-channel indices
        that contribute most to it, along with their (signed) weight values.

        Args:
            feature_id: index d into the SAE hidden dimension (0..D-1).
            top_n: how many input channels to return.
            view: 'encoder' (default) uses row d of W_e -- "which input channels
                  cause z_d to fire". 'decoder' uses column d of W_d -- "which
                  input channels z_d reconstructs into". For most interpretability
                  questions, 'encoder' is the right choice.

        Returns:
            (channel_ids, weights) -- two parallel lists of length top_n.
            channel_ids: input channel indices (0..C-1).
            weights: the (signed) connection weights, sorted by magnitude
                     (largest |weight| first).
        """
        with torch.no_grad():
            if view == 'encoder':
                # encoder.weight: [D, C, kH, kW]; for 1x1 kernel, kH=kW=1
                w = self.csae_model.encoder.weight[feature_id]      # [C, kH, kW]
            elif view == 'decoder':
                # decoder.weight: [C, D, kH, kW]; column d
                w = self.csae_model.decoder.weight[:, feature_id]   # [C, kH, kW]
            else:
                raise ValueError(f"view must be 'encoder' or 'decoder', got {view}")

            # Average over the (typically 1x1) kernel so we get one scalar per channel
            w_per_channel = w.view(w.shape[0], -1).mean(dim=1)      # [C]

            # Rank by |weight|, then read off the signed value
            mag = w_per_channel.abs()
            top_n = min(top_n, mag.numel())
            top_vals, top_idx = torch.topk(mag, k=top_n)
            channel_ids = top_idx.cpu().tolist()
            weights = w_per_channel[top_idx].cpu().tolist()

        return channel_ids, weights

    def extract_features(self, image: Image.Image, label: int, top_k: int = 16,
                         feature_id: int = None,
                         top_input_channels: int = 6,
                         input_channel_view: str = 'encoder') -> Dict:
        """
        Extract top-k activated features for a test image, plus the Grad-CAM
        spatial map.
        """
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.backbone(image_tensor)
            layer_acts = self.layer_activations.clone()
            pred_label = logits.argmax(dim=1).item()

        # Grad-CAM channel mask + spatial map
        (channel_mask, num_selected, channel_weights,
         gradcam_pred, gradcam_map) = self._select_channels_and_gradcam_map(image_tensor)

        layer_acts_norm = self._normalize_layer_activations(layer_acts)

        with torch.no_grad():
            _, sparse_features = self.csae_model(layer_acts_norm, use_topk=True)

        feature_importance = sparse_features.sum(dim=(2, 3)).squeeze()
        top_k_values, top_k_indices = torch.topk(feature_importance, k=min(top_k, len(feature_importance)))

        top_features = []
        for idx, importance in zip(top_k_indices, top_k_values):
            activation_map = sparse_features[0, idx, :, :].cpu()
            top_features.append((idx.item(), importance.item(), activation_map))

        # ------------------------------------------------------------------
        # Sum-of-features maps:
        #   z_sum_all_map: sum over ALL D features (what the constraint sees)
        #   z_sum_topk_map: sum over only the displayed top-k features
        # Both are L1-normalized per-map (sum to 1) for direct visual
        # comparison with the (also L1-normalized) Grad-CAM map.
        # We also compute the cosine similarity between each sum and the
        # Grad-CAM map -- that's the scalar "how well did the constraint
        # work" diagnostic.
        # ------------------------------------------------------------------
        with torch.no_grad():
            z_sum_all = sparse_features[0].sum(dim=0).cpu()       # [H, W]
            z_sum_topk = torch.zeros_like(z_sum_all)
            for _, _, m in top_features:
                z_sum_topk = z_sum_topk + m

            # L1-normalize both for display
            def _l1_normalize(m):
                s = m.sum()
                if s.abs() > 1e-8:
                    return m / s
                return m

            z_sum_all_n = _l1_normalize(z_sum_all)
            z_sum_topk_n = _l1_normalize(z_sum_topk)
            # Grad-CAM map was already normalized for display in
            # _select_channels_and_gradcam_map -- min-max to [0,1]. Re-normalize
            # to sum-1 for fair cosine comparison.
            gc_n = _l1_normalize(gradcam_map.clone())

            # Cosine similarity (treat maps as flat vectors)
            def _cos(a, b):
                a = a.flatten(); b = b.flatten()
                na = a.norm().clamp(min=1e-8)
                nb = b.norm().clamp(min=1e-8)
                return float((a @ b) / (na * nb))

            cos_all = _cos(z_sum_all_n, gc_n)
            cos_topk = _cos(z_sum_topk_n, gc_n)

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
            'gradcam_map': gradcam_map,                # H x W grad-cam heatmap (min-max [0,1])
            'top_features': top_features,
            'feature_importance': feature_importance.cpu(),
            # NEW: sum-of-z diagnostics
            'z_sum_all_map': z_sum_all_n,              # sum over ALL D features, L1-normalized
            'z_sum_topk_map': z_sum_topk_n,            # sum over the displayed top-k, L1-normalized
            'z_sum_all_cos': cos_all,                  # cosine sim vs Grad-CAM
            'z_sum_topk_cos': cos_topk,
        }

        # ------------------------------------------------------------------
        # If a specific feature_id is requested, also collect:
        #   - that feature's own activation map for this image
        #   - the top input channels that feed into it (encoder view)
        #   - the actual normalized activation maps of those input channels
        #     on THIS image (so the user sees the per-image evidence)
        # ------------------------------------------------------------------
        if feature_id is not None:
            if feature_id < 0 or feature_id >= self.csae_model.hidden_dim:
                raise ValueError(
                    f"feature_id={feature_id} out of range "
                    f"[0, {self.csae_model.hidden_dim})"
                )

            # The feature's own activation map for this image
            feature_act_map = sparse_features[0, feature_id, :, :].cpu()
            feature_self_importance = feature_act_map.sum().item()

            # Top input channels by encoder-weight magnitude
            top_ch_ids, top_ch_weights = self.get_top_input_channels_for_feature(
                feature_id=feature_id,
                top_n=top_input_channels,
                view=input_channel_view
            )

            # Their actual activation maps on THIS image (normalized version,
            # since that's what the SAE sees)
            input_channel_maps = []
            for ch_id, w_val in zip(top_ch_ids, top_ch_weights):
                ch_map = layer_acts_norm[0, ch_id, :, :].cpu()
                input_channel_maps.append((ch_id, w_val, ch_map))

            results['feature_id'] = feature_id
            results['feature_act_map'] = feature_act_map
            results['feature_self_importance'] = feature_self_importance
            results['input_channel_maps'] = input_channel_maps  # list of (ch_id, weight, map)
            results['input_channel_view'] = input_channel_view

        return results

    def visualize_features(self, image: Image.Image, label: int,
                          top_k: int = 16, save_path: str = None,
                          feature_id: int = None,
                          top_input_channels: int = 6,
                          input_channel_view: str = 'encoder'):
        print(f"Processing test image (label={label})...")
        results = self.extract_features(
            image, label, top_k=top_k,
            feature_id=feature_id,
            top_input_channels=top_input_channels,
            input_channel_view=input_channel_view,
        )

        print("Generating visualization...")
        self._plot_feature_grid(results, save_path)

        print(f"Visualization complete!")
        if save_path:
            print(f"  Saved to: {save_path}")

    # ----------------------------------------------------------------------
    # RANDOM MODE PLOTTING
    # Layout:
    #   Row 0: [Input image] [Grad-CAM heatmap] [Prediction info] [Importance bar]
    #   Rows 1+: feature maps (4 per row)
    # ----------------------------------------------------------------------

    def _plot_feature_grid(self, results: Dict, save_path: str = None):
        """Plot grid of CSAE features with Grad-CAM heatmap in overview row.
        If results contains 'feature_id', an extra bottom row shows the input
        activation channels that contribute most to that feature.
        """
        image = results['image']
        gradcam_map = results['gradcam_map']
        top_features = results['top_features']
        true_class = results['true_class']
        pred_class = results['pred_class']
        correct = results['correct']
        num_selected = results['num_selected_channels']

        n_features = len(top_features)
        n_cols = 10  # was 8; +2 cols to fit a Sum-of-z panel in the overview row
        n_feature_rows = (n_features + 3) // 4
        has_feature_input = 'feature_id' in results
        # 1 overview row + feature rows + (optionally) 1 input-channel sub-row
        n_rows = 1 + n_feature_rows + (1 if has_feature_input else 0)

        fig = plt.figure(figsize=(28, 3.5 * n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.5, wspace=0.3)

        # ===== Row 0: Input image | Grad-CAM | Sum-of-z | Info | Importance =====
        ax_img = fig.add_subplot(gs[0, 0:2])
        ax_img.imshow(image)
        ax_img.set_title("Test Image", fontsize=12, fontweight='bold')
        ax_img.axis('off')

        ax_gc = fig.add_subplot(gs[0, 2:4])
        ax_gc.imshow(gradcam_map.numpy(), cmap='jet', interpolation='bilinear')
        ax_gc.set_title("Grad-CAM Heatmap\n(target for sum of features)",
                        fontsize=11, fontweight='bold', color='royalblue')
        for spine in ax_gc.spines.values():
            spine.set_edgecolor('royalblue')
            spine.set_linewidth(2)
            spine.set_visible(True)
        ax_gc.set_xticks([])
        ax_gc.set_yticks([])

        # NEW: Sum-of-z panel (sum over all D features, L1-normalized)
        ax_sum = fig.add_subplot(gs[0, 4:6])
        z_sum_all = results['z_sum_all_map']
        cos_all = results['z_sum_all_cos']
        ax_sum.imshow(z_sum_all.numpy(), cmap='jet', interpolation='bilinear')
        if cos_all >= 0.8:
            sum_color = 'forestgreen'
        elif cos_all >= 0.5:
            sum_color = 'darkorange'
        else:
            sum_color = 'firebrick'
        for spine in ax_sum.spines.values():
            spine.set_edgecolor(sum_color)
            spine.set_linewidth(2)
            spine.set_visible(True)
        ax_sum.set_title(f"Sum of z (all D)\ncos vs Grad-CAM = {cos_all:.3f}",
                         fontsize=11, fontweight='bold', color=sum_color)
        ax_sum.set_xticks([])
        ax_sum.set_yticks([])

        ax_info = fig.add_subplot(gs[0, 6:8])
        ax_info.axis('off')

        status = "CORRECT" if correct else "WRONG"
        info_text = f"Prediction: {status}\n"
        info_text += f"  True: {true_class[:30]}...\n" if len(true_class) > 30 else f"  True: {true_class}\n"
        info_text += f"  Pred: {pred_class[:30]}...\n\n" if len(pred_class) > 30 else f"  Pred: {pred_class}\n\n"
        info_text += f"Backbone: {self.model_name}\n"
        info_text += f"  - {self.num_channels} input channels\n"
        info_text += f"  - {self.csae_model.hidden_dim} CSAE features\n"
        info_text += f"  - Top-k: {self.csae_model.top_k}\n"
        info_text += f"  - GradCAM: {num_selected}/{self.num_channels} channels\n"
        info_text += f"  - Sum(z) cos vs GC: {results['z_sum_all_cos']:.3f}\n"
        info_text += f"  - Sum(top-k z) cos vs GC: {results['z_sum_topk_cos']:.3f}"
        if has_feature_input:
            info_text += f"\n  - Queried feature: F{results['feature_id']}"
            info_text += f"\n  - Input view: {results['input_channel_view']}"

        ax_info.text(0.05, 0.5, info_text, fontsize=8, family='monospace',
                    verticalalignment='center', transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round',
                             facecolor='lightgreen' if correct else 'lightcoral',
                             alpha=0.3))

        ax_bar = fig.add_subplot(gs[0, 8:])
        importances = [imp for _, imp, _ in top_features]
        feature_indices = [f"F{idx}" for idx, _, _ in top_features]
        bar_colors = ['crimson' if has_feature_input and idx == results['feature_id']
                      else 'steelblue'
                      for idx, _, _ in top_features]
        ax_bar.bar(range(len(importances)), importances,
                  color=bar_colors, alpha=0.85, edgecolor='navy')
        ax_bar.set_xlabel('Feature', fontsize=9)
        ax_bar.set_ylabel('Importance', fontsize=9)
        ax_bar.set_title(f'Top-{n_features} Importance', fontsize=11, fontweight='bold')
        ax_bar.set_xticks(range(len(importances)))
        ax_bar.set_xticklabels(feature_indices, rotation=45, ha='right', fontsize=7)
        ax_bar.grid(True, alpha=0.3, axis='y')

        # ===== Rows 1..n_feature_rows: Feature maps =====
        for i, (feat_idx, importance, activation_map) in enumerate(top_features):
            row = 1 + i // 4
            col = (i % 4) * 2

            ax_feat = fig.add_subplot(gs[row, col:col+2])
            im = ax_feat.imshow(activation_map.numpy(), cmap='hot', interpolation='bilinear')
            highlight = (has_feature_input and feat_idx == results['feature_id'])
            title_prefix = "[Q] " if highlight else ""
            title_color = 'crimson' if highlight else 'black'
            ax_feat.set_title(f"{title_prefix}Feature {feat_idx}\nImp: {importance:.2f}",
                             fontsize=10, fontweight='bold', color=title_color)
            ax_feat.axis('off')
            if highlight:
                for spine in ax_feat.spines.values():
                    spine.set_edgecolor('crimson')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
                ax_feat.set_xticks([])
                ax_feat.set_yticks([])
                ax_feat.set_frame_on(True)

            cbar = plt.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)

        # ===== Final row (if feature_id given): input channels feeding F_d =====
        if has_feature_input:
            self._render_feature_input_subrow_full(
                fig, gs, results,
                row_index=n_rows - 1,
                n_cols=n_cols,
            )

        status_str = "CORRECT" if correct else "WRONG"
        plt.suptitle(f'ImageNet-1k Test Visualization ({status_str})  --  ' +
                    f'Grad-CAM = sum of feature maps  --  ' +
                    f'{self.model_name}: {self.num_channels}ch -> {self.csae_model.hidden_dim} features',
                    fontsize=13, fontweight='bold', y=0.998)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    # ----------------------------------------------------------------------
    # Helper: draw the "input channels contributing to feature F_d" sub-row
    # ----------------------------------------------------------------------

    def _render_feature_input_subrow_full(self, fig, gs, results: Dict,
                                           row_index: int, n_cols: int):
        """Draw a row showing:
            [F_d itself]  [ch_a, w=W]  [ch_b, w=W]  ...
        for the queried feature in `results['feature_id']`.

        Uses the same gridspec `gs` row at `row_index`, spanning n_cols columns.
        """
        feature_id = results['feature_id']
        feature_act_map = results['feature_act_map']
        feature_self_imp = results['feature_self_importance']
        input_channel_maps = results['input_channel_maps']  # [(ch_id, w, map)]
        view_name = results.get('input_channel_view', 'encoder')

        n_inputs = len(input_channel_maps)
        # Layout within the row: feature occupies cols 0:2, inputs share cols 2..n_cols
        # Each input panel takes 1 column.
        ax_feat = fig.add_subplot(gs[row_index, 0:2])
        im = ax_feat.imshow(feature_act_map.numpy(), cmap='hot', interpolation='bilinear')
        ax_feat.set_title(f"[Q] Feature F{feature_id}\n(self-importance: {feature_self_imp:.2f})",
                         fontsize=10, fontweight='bold', color='crimson')
        for spine in ax_feat.spines.values():
            spine.set_edgecolor('crimson')
            spine.set_linewidth(3)
            spine.set_visible(True)
        ax_feat.set_xticks([])
        ax_feat.set_yticks([])
        ax_feat.set_frame_on(True)
        cbar = plt.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

        # Available columns for input channel panels: 2 .. n_cols
        avail_cols = n_cols - 2
        if n_inputs == 0 or avail_cols <= 0:
            return
        # Allocate roughly one column per input; if more inputs than cols, cap.
        n_shown = min(n_inputs, avail_cols)

        for k in range(n_shown):
            ch_id, w_val, ch_map = input_channel_maps[k]
            col_start = 2 + k * (avail_cols // n_shown)
            col_end = 2 + (k + 1) * (avail_cols // n_shown)
            # Last panel absorbs remaining columns
            if k == n_shown - 1:
                col_end = n_cols

            ax_ch = fig.add_subplot(gs[row_index, col_start:col_end])
            im_ch = ax_ch.imshow(ch_map.numpy(), cmap='viridis', interpolation='bilinear')
            sign = "+" if w_val >= 0 else "-"
            # Color the title by sign of contribution
            t_color = 'darkgreen' if w_val >= 0 else 'darkred'
            ax_ch.set_title(f"ch {ch_id}  ({view_name} w={sign}{abs(w_val):.3f})",
                            fontsize=9, fontweight='bold', color=t_color)
            ax_ch.set_xticks([])
            ax_ch.set_yticks([])
            # Border by sign
            for spine in ax_ch.spines.values():
                spine.set_edgecolor(t_color)
                spine.set_linewidth(1.5)
                spine.set_visible(True)
            ax_ch.set_frame_on(True)
            cbar = plt.colorbar(im_ch, ax=ax_ch, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)

    # ----------------------------------------------------------------------
    # CONSISTENCY MODE
    # ----------------------------------------------------------------------

    def visualize_class_consistency(self, images: List[Image.Image], label: int,
                                   top_k: int = 12, save_path: str = None,
                                   feature_id: int = None,
                                   top_input_channels: int = 6,
                                   input_channel_view: str = 'encoder'):
        n_images = len(images)
        print(f"\nAnalyzing {n_images} images from class {label}...")

        class_name = list(IMAGENET2012_CLASSES.values())[label]

        all_results = []
        for i, image in enumerate(images):
            results = self.extract_features(
                image, label, top_k=top_k,
                feature_id=feature_id,
                top_input_channels=top_input_channels,
                input_channel_view=input_channel_view,
            )
            all_results.append(results)
            print(f"  Image {i+1}: {len(results['top_features'])} top features, "
                  f"prediction={'OK' if results['correct'] else 'WRONG'}")

        common_features = self._find_common_features(all_results, top_k)
        print(f"  Common features across all images: {len(common_features)}")

        print("Generating class consistency visualization...")
        self._plot_class_consistency(all_results, label, class_name,
                                      common_features, save_path,
                                      feature_id=feature_id)

        print(f"Class consistency visualization complete!")
        if save_path:
            print(f"  Saved to: {save_path}")

    def _find_common_features(self, all_results: List[Dict], top_k: int) -> List[int]:
        if not all_results:
            return []

        feature_sets = []
        for results in all_results:
            feature_set = set([feat_idx for feat_idx, _, _ in results['top_features']])
            feature_sets.append(feature_set)

        common = feature_sets[0]
        for fs in feature_sets[1:]:
            common = common.intersection(fs)

        return sorted(list(common))

    def _plot_class_consistency(self, all_results: List[Dict], label: int,
                                class_name: str, common_features: List[int],
                                save_path: str = None,
                                feature_id: int = None):
        """
        Layout:
        - Row 0 (overview): class info | input images | common-features bar chart
        - Per image:
            Feature row:     [Grad-CAM]  [F1]  [F2]  ...  [Fk]
            Input sub-row    (only if feature_id given):
                             [Feature F_d (recap)]  [ch_a, w]  [ch_b, w]  ...
        """
        n_images = len(all_results)
        n_features_per_image = min(12, len(all_results[0]['top_features']))
        has_feature_id = feature_id is not None

        # Each image takes 1 row normally, or 2 rows (feature row + input row)
        # when feature_id is set.
        rows_per_image = 2 if has_feature_id else 1

        # Total columns:
        # Without feature_id: 1 (grad-cam) + n_features_per_image, min 16
        # With feature_id: same total columns; the input row reuses the same
        # column structure (feature recap in cols 0..1, input channels spread
        # across cols 2..total_cols).
        # +1 extra column for the [Sum of z] panel beside Grad-CAM.
        total_cols = max(2 + n_features_per_image, 16)

        col_width = 1.7
        fig_width = max(28, total_cols * col_width)
        fig_height = 4 + 3.5 * rows_per_image * n_images

        total_rows = 1 + rows_per_image * n_images
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(total_rows, total_cols, hspace=0.6, wspace=0.4)

        # ===== Row 0: Overview =====
        ax_info = fig.add_subplot(gs[0, 0:2])
        ax_info.axis('off')

        info_text = f"Class {label}\n"
        info_text += f"{class_name[:50]}...\n\n" if len(class_name) > 50 else f"{class_name}\n\n"
        info_text += f"Analyzing:\n"
        info_text += f"  - {n_images} test images\n"
        info_text += f"  - Top-{n_features_per_image} features each\n"
        info_text += f"  - {len(common_features)} common features\n"
        if has_feature_id:
            view_name = all_results[0].get('input_channel_view', 'encoder')
            info_text += f"  - Queried feature: F{feature_id}\n"
            info_text += f"  - Input view: {view_name}\n\n"
        else:
            info_text += "\n"
        info_text += f"Backbone: {self.model_name}\n"

        correct_count = sum(1 for r in all_results if r['correct'])
        info_text += f"Predictions: {correct_count}/{n_images} correct"

        ax_info.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round',
                             facecolor='lavender' if has_feature_id else 'lightblue',
                             alpha=0.3))

        # Input images in overview row
        for i, results in enumerate(all_results):
            col_start = 2 + i * 2
            col_end = col_start + 2
            if col_end > total_cols - 4:
                break
            ax_img = fig.add_subplot(gs[0, col_start:col_end])
            ax_img.imshow(results['image'])
            status = "OK" if results['correct'] else "X"
            ax_img.set_title(f"Image {i+1} [{status}]", fontsize=11, fontweight='bold')
            ax_img.axis('off')

        # Common-features bar chart
        bar_col_start = max(2 + 2 * n_images, total_cols // 2)
        if bar_col_start < total_cols:
            ax_common = fig.add_subplot(gs[0, bar_col_start:])
            if common_features:
                common_importances = []
                for feat_idx in common_features[:15]:
                    avg_imp = np.mean([
                        next((imp for idx, imp, _ in r['top_features'] if idx == feat_idx), 0)
                        for r in all_results
                    ])
                    common_importances.append(avg_imp)

                bar_colors = ['crimson' if has_feature_id and f == feature_id else 'green'
                              for f in common_features[:15]]
                ax_common.barh(range(len(common_features[:15])), common_importances,
                              color=bar_colors, alpha=0.8, edgecolor='darkgreen')
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

        # ===== Per-image blocks =====
        for img_idx, results in enumerate(all_results):
            # Feature row for this image
            feat_row = 1 + img_idx * rows_per_image
            top_features = results['top_features'][:n_features_per_image]
            gradcam_map = results['gradcam_map']
            z_sum_all = results['z_sum_all_map']
            cos_all = results['z_sum_all_cos']

            # Column 0: Grad-CAM heatmap (the target)
            ax_gc = fig.add_subplot(gs[feat_row, 0])
            ax_gc.imshow(gradcam_map.numpy(), cmap='jet', interpolation='bilinear')
            for spine in ax_gc.spines.values():
                spine.set_edgecolor('royalblue')
                spine.set_linewidth(2.5)
                spine.set_visible(True)
            status = "OK" if results['correct'] else "X"
            ax_gc.set_title(f"Grad-CAM\nImg {img_idx+1} [{status}]",
                           fontsize=9, fontweight='bold', color='royalblue')
            ax_gc.set_xticks([])
            ax_gc.set_yticks([])

            # Column 1: Sum of ALL z features (what the constraint pushes toward Grad-CAM)
            ax_sum = fig.add_subplot(gs[feat_row, 1])
            ax_sum.imshow(z_sum_all.numpy(), cmap='jet', interpolation='bilinear')
            # Color the border by how close to Grad-CAM the sum is
            # cos in [-1, 1]; with both maps non-negative it lives in [0, 1]
            if cos_all >= 0.8:
                sum_color = 'forestgreen'
            elif cos_all >= 0.5:
                sum_color = 'darkorange'
            else:
                sum_color = 'firebrick'
            for spine in ax_sum.spines.values():
                spine.set_edgecolor(sum_color)
                spine.set_linewidth(2.5)
                spine.set_visible(True)
            ax_sum.set_title(f"Sum of z\ncos={cos_all:.2f}",
                            fontsize=9, fontweight='bold', color=sum_color)
            ax_sum.set_xticks([])
            ax_sum.set_yticks([])

            # Columns 2..k+1: top feature maps
            for feat_pos, (f_idx, importance, activation_map) in enumerate(top_features):
                col = 2 + feat_pos
                if col >= total_cols:
                    break

                ax_feat = fig.add_subplot(gs[feat_row, col])
                is_common = f_idx in common_features
                is_queried = has_feature_id and f_idx == feature_id

                im = ax_feat.imshow(activation_map.numpy(), cmap='hot', interpolation='bilinear')
                if is_queried:
                    title_color = 'crimson'
                    title_prefix = "[Q] "
                elif is_common:
                    title_color = 'green'
                    title_prefix = ""
                else:
                    title_color = 'black'
                    title_prefix = ""
                ax_feat.set_title(f"{title_prefix}F{f_idx}\n{importance:.1f}",
                                fontsize=9,
                                fontweight='bold' if (is_common or is_queried) else 'normal',
                                color=title_color)
                ax_feat.axis('off')

                if is_queried:
                    for spine in ax_feat.spines.values():
                        spine.set_edgecolor('crimson')
                        spine.set_linewidth(3)
                        spine.set_visible(True)
                    ax_feat.set_xticks([])
                    ax_feat.set_yticks([])
                    ax_feat.set_frame_on(True)
                elif is_common:
                    for spine in ax_feat.spines.values():
                        spine.set_edgecolor('green')
                        spine.set_linewidth(3)
                        spine.set_visible(True)
                    ax_feat.set_xticks([])
                    ax_feat.set_yticks([])
                    ax_feat.set_frame_on(True)

            # Input sub-row (only if feature_id was provided AND results has it)
            if has_feature_id and 'input_channel_maps' in results:
                input_row = feat_row + 1
                self._render_feature_input_subrow_full(
                    fig, gs, results,
                    row_index=input_row,
                    n_cols=total_cols,
                )

        # Title
        avg_cos = float(np.mean([r['z_sum_all_cos'] for r in all_results]))
        title = (f'Feature Consistency ({self.model_name}): '
                 f'Class {label} ({class_name[:40]}...)\n'
                 f'Each row: [Grad-CAM | Sum-of-z | top-{n_features_per_image} features]  --  '
                 f'avg cos(Sum z, Grad-CAM) = {avg_cos:.3f}')
        if has_feature_id:
            title += (f'\nQueried F{feature_id} in crimson, '
                      f'per-image input-channel sub-row beneath')
        else:
            title += f'  --  {len(common_features)} common features in green'
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.998)

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
        description='Visualize Multi-Channel ConvSAE on ImageNet-1k Test Set (with Grad-CAM per row)'
    )
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--target_layer', type=str, default=None)
    parser.add_argument('--csae_model', type=str, default=None)
    parser.add_argument('--raw_data_dir', type=str,
                       default='/data/imagenet_raw/data')
    parser.add_argument('--test_data_dir', type=str,
                       default='/data/imagenet1k_sampletest')
    parser.add_argument('--test_images_per_class', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--images_per_class_viz', type=int, default=3)
    parser.add_argument('--top_k_features', type=int, default=16)
    parser.add_argument('--output_dir', type=str,
                       default='imagenet1k_test_visualizations')
    parser.add_argument('--force_resample', action='store_true')

    # ----- New: feature-input inspection -----
    parser.add_argument('--feature_id', type=int, default=None,
                       help='If set, add a sub-row beneath each image showing the '
                            'input activation channels that contribute most to '
                            'SAE feature z_d (with channel ids and weights).')
    parser.add_argument('--top_input_channels', type=int, default=6,
                       help='How many top input channels to display per feature '
                            '(default: 6).')
    parser.add_argument('--input_channel_view', type=str, default='encoder',
                       choices=['encoder', 'decoder'],
                       help='Which weight matrix to inspect: '
                            "'encoder' = inputs that drive z_d (default), "
                            "'decoder' = inputs that z_d reconstructs into.")

    args = parser.parse_args()

    if args.target_layer is None:
        args.target_layer = MODEL_CONFIGS[args.model]['default_target_layer']

    if args.csae_model is None:
        args.csae_model = f'imagenet1k_csae_{args.model}_model.pkl'

    print("="*80)
    print(f"Multi-Channel ConvSAE Test Set Visualization (ImageNet-1k Full)")
    print(f"Backbone: {args.model} ({MODEL_CONFIGS[args.model]['description']})")
    print(f"Target layer: {args.target_layer}")
    print(f"CSAE model: {args.csae_model}")
    print(f"Each feature row starts with that image's Grad-CAM heatmap.")
    if args.feature_id is not None:
        print(f"Queried feature: F{args.feature_id}  "
              f"(top {args.top_input_channels} input channels via "
              f"{args.input_channel_view} weights)")
    print("="*80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\nStep 1: Sampling test images...")
    sampler = ImageNet1kTestSampler(
        raw_dir=Path(args.raw_data_dir),
        test_dir=Path(args.test_data_dir),
        images_per_class=args.test_images_per_class,
        force_resample=args.force_resample
    )

    print("\nStep 2: Loading ConvSAE model...")
    visualizer = MultiModelSAEVisualizerTestMF(
        model_name=args.model,
        target_layer_name=args.target_layer,
        csae_model_path=args.csae_model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    if args.num_classes is not None:
        print(f"\n{'='*80}")
        print(f"MODE: Class Consistency Analysis")
        print(f"  - Analyzing {args.num_classes} classes")
        print(f"  - {args.images_per_class_viz} images per class")
        print(f"  - Top-{args.top_k_features} features per image")
        print(f"  - Each row begins with the image's Grad-CAM heatmap")
        print(f"{'='*80}")

        print(f"\nStep 3: Sampling {args.images_per_class_viz} images from {args.num_classes} classes...")
        class_samples = sampler.get_samples_by_class(
            num_classes=args.num_classes,
            images_per_class=args.images_per_class_viz
        )
        print(f"Selected {len(class_samples)} classes")

        print(f"\nStep 4: Generating class consistency visualizations...")
        correct_count = 0
        total_images = 0

        for class_idx, images in class_samples.items():
            class_name = list(IMAGENET2012_CLASSES.values())[class_idx]
            print(f"\n{'='*80}")
            print(f"Class {class_idx}: {class_name[:60]}...")
            print(f"{'='*80}")

            save_path = output_dir / f"class_{class_idx}_consistency_{args.model}.png"

            visualizer.visualize_class_consistency(
                images, class_idx,
                top_k=args.top_k_features,
                save_path=str(save_path),
                feature_id=args.feature_id,
                top_input_channels=args.top_input_channels,
                input_channel_view=args.input_channel_view,
            )

            for img in images:
                results = visualizer.extract_features(img, class_idx, top_k=1)
                if results['correct']:
                    correct_count += 1
                total_images += 1

        accuracy = (correct_count / total_images) * 100 if total_images > 0 else 0
        print(f"\n{'='*80}")
        print(f"All class consistency visualizations complete!")
        print(f"  Output directory: {output_dir}")
        print(f"  Classes analyzed: {len(class_samples)}")
        print(f"  Total images: {total_images}")
        print(f"  Accuracy: {correct_count}/{total_images} ({accuracy:.1f}%)")
        print(f"{'='*80}")

    else:
        print(f"\n{'='*80}")
        print(f"MODE: Random Sample Visualization")
        print(f"  - {args.num_samples} random test images")
        print(f"  - Top-{args.top_k_features} features per image")
        print(f"  - Grad-CAM heatmap shown in overview row")
        print(f"{'='*80}")

        print(f"\nStep 3: Selecting {args.num_samples} random test images...")
        test_samples = sampler.get_random_samples(args.num_samples)
        print(f"Selected {len(test_samples)} test images")

        print(f"\nStep 4: Generating visualizations...")
        correct_count = 0

        for i, (image, label) in enumerate(test_samples):
            print(f"\n{'='*80}")
            print(f"Test Image {i+1}/{len(test_samples)} (label={label})")
            print(f"{'='*80}\n")

            save_path = output_dir / f"test_sample_{i+1}_label{label}_{args.model}.png"

            results = visualizer.extract_features(image, label, top_k=args.top_k_features)
            if results['correct']:
                correct_count += 1

            visualizer.visualize_features(
                image, label,
                top_k=args.top_k_features,
                save_path=str(save_path),
                feature_id=args.feature_id,
                top_input_channels=args.top_input_channels,
                input_channel_view=args.input_channel_view,
            )

        accuracy = (correct_count / len(test_samples)) * 100
        print(f"\n{'='*80}")
        print(f"All visualizations complete!")
        print(f"  Output directory: {output_dir}")
        print(f"  Accuracy on visualized samples: {correct_count}/{len(test_samples)} ({accuracy:.1f}%)")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()