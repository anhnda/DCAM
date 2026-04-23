"""
Accuracy Drop Analysis for Multi-Model ImageNet-1k CSAE

This script evaluates the impact of replacing original backbone activations
with CSAE-reconstructed activations on classification accuracy.

Supports multiple backbone architectures:
- ResNet50 (default): layer3, 1024 channels, 14×14 resolution
- ResNet18: layer3, 256 channels, 14×14 resolution
- VGG16: features[16], 256 channels, 28×28 resolution
- EfficientNet-B0: features[4], ~80 channels, 14×14 resolution

Methodology:
1. Load pretrained backbone model
2. Load trained ImageNet-1k Multi-Channel ConvSAE
3. Sample 5K test images (5 images per class from 1000 classes)
4. For each test image:
   a. Extract original activations at target layer
   b. Pass through CSAE to get reconstruction
   c. Replace original with reconstructed activations
   d. Continue forward pass through rest of model
   e. Compare predictions with/without reconstruction

Metrics:
- Original Accuracy: Classification accuracy with original activations
- Reconstruction Accuracy: Classification accuracy with CSAE reconstructions
- Accuracy Drop: Difference between original and reconstruction accuracy
- Only Top-1 accuracy is reported (no Top-5)

Usage:
    # ResNet50 (default)
    python check_acc_drop_full.py

    # ResNet18
    python check_acc_drop_full.py --model resnet18

    # VGG16
    python check_acc_drop_full.py --model vgg16 --csae_model imagenet1k_csae_vgg16_model.pkl

    # EfficientNet
    python check_acc_drop_full.py --model efficientnet --csae_model imagenet1k_csae_efficientnet_model.pkl

    # Custom target layer for ResNet50
    python check_acc_drop_full.py --model resnet50 --target_layer layer2 --csae_model imagenet1k_csae_resnet50_layer2_model.pkl

    # Force resample test set
    python check_acc_drop_full.py --force_resample
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import joblib
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Tuple, List
import sys
from pathlib import Path
from PIL import Image
import io
from collections import defaultdict

sys.path.append('.')
from run_xcsae_full import MultiChannelConvSAE
from full_classes import IMAGENET2012_CLASSES

import pandas as pd
import random


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
# Test Dataset Loader
# ==========================================

class ImageNet1kTestDataset(Dataset):
    """Dataset that loads sampled test images from cached samples."""

    def __init__(self, test_metadata_path: Path, transform=None):
        self.transform = transform

        print(f"Loading cached test samples from {test_metadata_path}...")
        metadata = joblib.load(test_metadata_path)
        self.samples = metadata['samples']

        print(f"  ✓ Loaded {len(self.samples)} test images")
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


def create_test_samples_if_needed(raw_dir: Path, test_dir: Path,
                                  images_per_class: int, force_resample: bool):
    """Create test samples if cache doesn't exist or force_resample is True."""
    metadata_path = test_dir / "test_metadata.pkl"

    if metadata_path.exists() and not force_resample:
        print(f"Test samples already cached at {metadata_path}")
        return metadata_path

    print(f"\n{'='*80}")
    print(f"Creating test sample dataset...")
    print(f"{'='*80}")
    print(f"Sampling {images_per_class} test images per class from validation set...")

    test_dir.mkdir(parents=True, exist_ok=True)

    wnid_to_idx = {wnid: idx for idx, wnid in enumerate(IMAGENET2012_CLASSES.keys())}
    class_samples = defaultdict(list)

    val_parquet_files = sorted(raw_dir.glob("validation-*.parquet"))

    if len(val_parquet_files) == 0:
        raise FileNotFoundError(f"No validation parquet files found in {raw_dir}")

    print(f"Found {len(val_parquet_files)} validation parquet files")

    for parquet_file in tqdm(val_parquet_files, desc="Reading validation parquet files"):
        df = pd.read_parquet(parquet_file)

        for idx, row in df.iterrows():
            label = row['label']

            if len(class_samples[label]) < images_per_class:
                image_bytes = row['image']['bytes']
                class_samples[label].append((image_bytes, label))

        min_samples = min(len(samples) for samples in class_samples.values()) if class_samples else 0
        if min_samples >= images_per_class and len(class_samples) == 1000:
            print(f"\nCollected {images_per_class} samples for all 1000 classes!")
            break

    samples = []
    for class_idx in range(1000):
        if len(class_samples[class_idx]) >= images_per_class:
            sampled = random.sample(class_samples[class_idx], images_per_class)
            samples.extend(sampled)
        else:
            print(f"WARNING: Class {class_idx} has only {len(class_samples[class_idx])} test samples")
            samples.extend(class_samples[class_idx])

    print(f"\nTotal sampled test images: {len(samples)}")

    print(f"Saving test samples to {test_dir}...")
    joblib.dump({
        'samples': samples,
        'images_per_class': images_per_class,
        'num_classes': 1000,
        'wnid_to_idx': wnid_to_idx
    }, metadata_path)
    print(f"✓ Test samples cached!")

    return metadata_path


# ==========================================
# Multi-Model with CSAE Reconstruction
# ==========================================

class MultiModelWithCSAEReconstruction:
    """
    Multi-model support with CSAE reconstruction at specified layer.

    Supports ResNet18, VGG16, EfficientNet-B0.
    """

    def __init__(self, model_name: str, target_layer_name: str,
                 csae_model: MultiChannelConvSAE, device='cuda'):
        """
        Args:
            model_name: Model name ('resnet18', 'vgg16', 'efficientnet')
            target_layer_name: Target layer name
            csae_model: Trained Multi-Channel ConvSAE
            device: Device to run on
        """
        self.model_name = model_name
        self.target_layer_name = target_layer_name
        self.device = device

        # Load model
        config = MODEL_CONFIGS[model_name]
        self.model = config['model_fn']().to(device)
        self.model.eval()

        self.csae = csae_model.to(device)
        self.csae.eval()

        # Get target layer
        self.target_layer = self._get_layer_by_name(target_layer_name)

        # Hook
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

        print(f"✓ Model: {config['description']}")
        print(f"✓ Target layer: {target_layer_name}")
        print(f"✓ CSAE: {csae_model.in_channels}→{csae_model.hidden_dim}, top_k={csae_model.top_k}")

    def _get_layer_by_name(self, layer_name: str):
        """Get layer by name."""
        if '[' in layer_name:
            parts = layer_name.split('[')
            attr_name = parts[0]
            index = int(parts[1].rstrip(']'))
            return getattr(self.model, attr_name)[index]
        else:
            return getattr(self.model, layer_name)

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output

    def _normalize_activations(self, acts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize activations using per-channel 99th percentile."""
        B, C, H, W = acts.shape
        normalized = acts.clone()
        scale_factors = torch.ones(B, C, device=acts.device)

        for b in range(B):
            for c in range(C):
                channel_data = acts[b, c, :, :]

                if channel_data.abs().sum() < 1e-8:
                    continue

                non_zero_vals = channel_data[channel_data > 1e-8]
                if len(non_zero_vals) > 0:
                    scale_factor = torch.quantile(non_zero_vals, 0.99)

                    if scale_factor > 1e-8:
                        channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                        normalized[b, c, :, :] = channel_data / (scale_factor + 1e-8)
                        scale_factors[b, c] = scale_factor

        return normalized, scale_factors

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

    def _forward_from_target_layer(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from target layer to output."""
        if self.model_name in ['resnet50', 'resnet18']:
            # Continue from current layer
            if 'layer1' in self.target_layer_name:
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)
            elif 'layer2' in self.target_layer_name:
                x = self.model.layer3(x)
                x = self.model.layer4(x)
            elif 'layer3' in self.target_layer_name:
                x = self.model.layer4(x)
            # layer4 doesn't need additional processing

            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
            return x

        elif self.model_name == 'vgg16':
            target_idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            # Continue through remaining features
            for i in range(target_idx + 1, len(self.model.features)):
                x = self.model.features[i](x)

            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x

        elif self.model_name == 'efficientnet':
            target_idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            # Continue through remaining features
            for i in range(target_idx + 1, len(self.model.features)):
                x = self.model.features[i](x)

            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x

        return x

    def forward_original(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with original activations."""
        with torch.no_grad():
            return self.model(x)

    def forward_with_reconstruction(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with CSAE-reconstructed activations."""
        with torch.no_grad():
            # Forward to target layer
            x = self._forward_to_target_layer(x)

            # Capture original activations
            original_acts = x.clone()

            # Normalize
            normalized_acts, scale_factors = self._normalize_activations(original_acts)

            # CSAE reconstruction
            reconstruction, sparse_features = self.csae(normalized_acts, use_topk=True)

            # Denormalize
            scale_factors_4d = scale_factors.unsqueeze(2).unsqueeze(3)
            reconstructed_acts = reconstruction * scale_factors_4d

            # Continue with reconstructed activations
            logits = self._forward_from_target_layer(reconstructed_acts)

            # Compute stats
            mse = F.mse_loss(reconstructed_acts, original_acts).item()
            relative_error = (reconstructed_acts - original_acts).abs().mean() / (original_acts.abs().mean() + 1e-8)
            relative_error = relative_error.item()
            sparsity = (sparse_features > 0).float().mean().item()

            stats = {
                'mse': mse,
                'relative_error': relative_error,
                'sparsity': sparsity
            }

            return logits, stats


# ==========================================
# Evaluation
# ==========================================

def evaluate_accuracy_drop(
    model: MultiModelWithCSAEReconstruction,
    data_loader: DataLoader,
    device: str = 'cuda'
) -> Dict:
    """Evaluate classification accuracy with original vs reconstructed activations."""

    total_samples = 0
    correct_original = 0
    correct_reconstructed = 0

    mse_list = []
    relative_error_list = []
    sparsity_list = []

    per_class_correct_original = defaultdict(int)
    per_class_correct_reconstructed = defaultdict(int)
    per_class_total = defaultdict(int)

    print("\nEvaluating accuracy with original vs reconstructed activations...")
    print("="*80)

    for images, labels in tqdm(data_loader, desc="Processing batches"):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Original
        logits_original = model.forward_original(images)
        pred_original = logits_original.argmax(dim=1)

        # Reconstructed
        logits_reconstructed, stats = model.forward_with_reconstruction(images)
        pred_reconstructed = logits_reconstructed.argmax(dim=1)

        # Top-1 accuracy
        correct_original += (pred_original == labels).sum().item()
        correct_reconstructed += (pred_reconstructed == labels).sum().item()

        # Per-class
        for i in range(batch_size):
            label = labels[i].item()
            per_class_total[label] += 1
            if pred_original[i] == labels[i]:
                per_class_correct_original[label] += 1
            if pred_reconstructed[i] == labels[i]:
                per_class_correct_reconstructed[label] += 1

        total_samples += batch_size

        mse_list.append(stats['mse'])
        relative_error_list.append(stats['relative_error'])
        sparsity_list.append(stats['sparsity'])

    # Compute accuracies
    acc_original = correct_original / total_samples * 100
    acc_reconstructed = correct_reconstructed / total_samples * 100
    acc_drop = acc_original - acc_reconstructed

    avg_mse = np.mean(mse_list)
    avg_relative_error = np.mean(relative_error_list)
    avg_sparsity = np.mean(sparsity_list)

    # Per-class drops
    per_class_acc_drop = {}
    for label in per_class_total.keys():
        if per_class_total[label] > 0:
            acc_orig = per_class_correct_original[label] / per_class_total[label] * 100
            acc_recon = per_class_correct_reconstructed[label] / per_class_total[label] * 100
            per_class_acc_drop[label] = acc_orig - acc_recon

    # Worst classes
    sorted_classes = sorted(per_class_acc_drop.items(), key=lambda x: x[1], reverse=True)
    worst_classes = sorted_classes[:10] if len(sorted_classes) >= 10 else sorted_classes

    results = {
        'total_samples': total_samples,
        'acc_original': acc_original,
        'acc_reconstructed': acc_reconstructed,
        'acc_drop': acc_drop,
        'avg_mse': avg_mse,
        'avg_relative_error': avg_relative_error,
        'avg_sparsity': avg_sparsity,
        'per_class_total': dict(per_class_total),
        'worst_classes': worst_classes
    }

    return results


def print_results(results: Dict, model_name: str):
    """Pretty-print accuracy drop analysis results."""
    print("\n" + "="*80)
    print(f"ACCURACY DROP ANALYSIS RESULTS ({model_name.upper()} - ImageNet-1k)")
    print("="*80)

    print(f"\nDataset:")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Number of classes: {len(results['per_class_total'])}")

    print(f"\nTop-1 Accuracy:")
    print(f"  Original (no reconstruction):     {results['acc_original']:.2f}%")
    print(f"  Reconstructed (CSAE):             {results['acc_reconstructed']:.2f}%")
    print(f"  Accuracy Drop:                    {results['acc_drop']:.2f}%")
    if results['acc_original'] > 0:
        print(f"  Relative Drop:                    {results['acc_drop']/results['acc_original']*100:.2f}%")

    print(f"\nReconstruction Quality:")
    print(f"  Average MSE:                      {results['avg_mse']:.6f}")
    print(f"  Average Relative Error:           {results['avg_relative_error']:.6f}")
    print(f"  Average Sparsity (active %):      {results['avg_sparsity']*100:.2f}%")

    if results['worst_classes']:
        print(f"\nTop 10 Classes with Largest Accuracy Drop:")
        for i, (label, drop) in enumerate(results['worst_classes'], 1):
            class_name = list(IMAGENET2012_CLASSES.values())[label]
            class_name_short = class_name[:50] + "..." if len(class_name) > 50 else class_name
            print(f"  {i:2d}. Class {label:3d} ({class_name_short}): {drop:+.2f}%")

    print("\n" + "="*80)

    print("\nInterpretation:")
    if results['acc_drop'] < 1.0:
        print("  ✓ Excellent: CSAE preserves almost all classification information (<1% drop)")
    elif results['acc_drop'] < 3.0:
        print("  ✓ Very Good: CSAE preserves most classification information (<3% drop)")
    elif results['acc_drop'] < 5.0:
        print("  ✓ Good: CSAE preserves significant classification information (<5% drop)")
    elif results['acc_drop'] < 10.0:
        print("  ⚠ Moderate: Notable accuracy drop (5-10%), but reconstruction is still useful")
    else:
        print("  ⚠ High: Significant accuracy drop (>10%), CSAE may be too lossy")

    print("="*80 + "\n")


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate accuracy drop with CSAE-reconstructed activations (Multi-Model Support)'
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
                       help='Number of test images per class (default: 5, total: 5000)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--force_resample', action='store_true',
                       help='Force resampling of test images')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file to save results (auto-named if not specified)')

    args = parser.parse_args()

    # Auto-detect CSAE model path if not specified
    if args.csae_model is None:
        config = MODEL_CONFIGS[args.model]
        target_layer = args.target_layer if args.target_layer else config['default_target_layer']
        layer_suffix = target_layer.replace('[', '_').replace(']', '')

        if target_layer == config['default_target_layer']:
            args.csae_model = f'imagenet1k_csae_{args.model}_model.pkl'
        else:
            args.csae_model = f'imagenet1k_csae_{args.model}_{layer_suffix}_model.pkl'

        print(f"Auto-detected CSAE model: {args.csae_model}")

    # Auto-detect output file if not specified
    if args.output_file is None:
        args.output_file = f'accuracy_drop_{args.model}_results.txt'

    print("="*80)
    print(f"ImageNet-1k CSAE Accuracy Drop Analysis")
    print(f"Model: {args.model.upper()}")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Test samples
    print("\nStep 1: Preparing test dataset...")
    test_metadata_path = create_test_samples_if_needed(
        raw_dir=Path(args.raw_data_dir),
        test_dir=Path(args.test_data_dir),
        images_per_class=args.test_images_per_class,
        force_resample=args.force_resample
    )

    # Load model
    print("\nStep 2: Loading backbone model...")
    config = MODEL_CONFIGS[args.model]
    target_layer = args.target_layer if args.target_layer else config['default_target_layer']

    # Load CSAE
    print(f"\nStep 3: Loading CSAE model from {args.csae_model}...")
    csae_model = joblib.load(args.csae_model)
    csae_model.eval()

    # Create combined model
    print("\nStep 4: Creating model with CSAE reconstruction...")
    model = MultiModelWithCSAEReconstruction(
        model_name=args.model,
        target_layer_name=target_layer,
        csae_model=csae_model,
        device=device
    )

    # Setup data
    print(f"\nStep 5: Loading test dataset...")
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageNet1kTestDataset(test_metadata_path, transform=data_transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total test images: {len(dataset)}")

    # Evaluate
    print("\nStep 6: Evaluating accuracy drop...")
    results = evaluate_accuracy_drop(model, data_loader, device=device)

    # Print results
    print_results(results, args.model)

    # Save results
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"{args.model.upper()} CSAE - Accuracy Drop Analysis\n")
        f.write("="*80 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  Model: {args.model}\n")
        f.write(f"  Target Layer: {target_layer}\n")
        f.write(f"  CSAE Model: {args.csae_model}\n")
        f.write(f"  Test Images: {results['total_samples']}\n")
        f.write(f"  Classes: {len(results['per_class_total'])}\n\n")

        f.write(f"Top-1 Accuracy:\n")
        f.write(f"  Original:        {results['acc_original']:.2f}%\n")
        f.write(f"  Reconstructed:   {results['acc_reconstructed']:.2f}%\n")
        f.write(f"  Drop:            {results['acc_drop']:.2f}%\n")
        if results['acc_original'] > 0:
            f.write(f"  Relative Drop:   {results['acc_drop']/results['acc_original']*100:.2f}%\n\n")
        else:
            f.write(f"  Relative Drop:   N/A\n\n")

        f.write(f"Reconstruction Quality:\n")
        f.write(f"  Average MSE:             {results['avg_mse']:.6f}\n")
        f.write(f"  Average Relative Error:  {results['avg_relative_error']:.6f}\n")
        f.write(f"  Average Sparsity:        {results['avg_sparsity']*100:.2f}%\n\n")

        if results['worst_classes']:
            f.write(f"Top 10 Classes with Largest Accuracy Drop:\n")
            for i, (label, drop) in enumerate(results['worst_classes'], 1):
                class_name = list(IMAGENET2012_CLASSES.values())[label]
                f.write(f"  {i:2d}. Class {label:3d} ({class_name}): {drop:+.2f}%\n")
            f.write("\n")

        f.write("="*80 + "\n")

    print(f"✓ Results saved to {args.output_file}")
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
