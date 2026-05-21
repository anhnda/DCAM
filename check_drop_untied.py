"""
Accuracy Drop Analysis for Multi-Model ImageNet-1k DCAM
========================================================

This script evaluates the impact of replacing original backbone activations
with DCAM-reconstructed activations on classification accuracy.

DCAM (Decomposed Class Activation Maps) reconstructs backbone activations via
a single membership matrix Pi in R^{D x C} (D <= C) and its tied pseudo-inverse:

    z'    = ReLU(Pi A)              # encode  C -> D
    z     = TopK(z', K)             # sparsify over concepts
    A_hat = Pi_dagger z             # decode  D -> C   (tied: Pi_dagger from Pi)

The substitution pipeline (extract A at target layer, normalize per-channel
99th-percentile, run DCAM, denormalize, continue forward) is identical to the
CSAE pipeline because DCAM was trained on the same normalized activations.

Supports the same backbones as run_dcam_full.py:
- ResNet50  (default): layer3, 1024ch, 14x14
- ResNet18           : layer3,  256ch, 14x14
- VGG16              : features[16], 256ch, 28x28
- EfficientNet-B0    : features[4],  ~80ch, 14x14

What's different vs the CSAE script
-----------------------------------
* The DCAM "model" is saved by run_dcam_full.py as a joblib dict with keys
  {'Pi', 'Pi0', 'config', 'logs', 'final_metrics'}, NOT as a pickled nn.Module.
  We reconstruct a `DCAM` instance from Pi + config here.
* Default checkpoint filename pattern:
    imagenet1k_dcam_{model}[_<layer>]_D{D}_seed{data_seed}-{model_seed}_result.pkl

Usage:
    # ResNet50, default seeds (data=42, model=42), default D
    python check_drop_dcam.py

    # Explicit DCAM checkpoint
    python check_drop_dcam.py --model resnet18 \\
        --dcam_model imagenet1k_dcam_resnet18_D128_seed42-42_result.pkl

    # Compare two model seeds (run twice and diff the printouts)
    python check_drop_dcam.py --dcam_model imagenet1k_dcam_resnet50_D512_seed42-0_result.pkl
    python check_drop_dcam.py --dcam_model imagenet1k_dcam_resnet50_D512_seed42-1_result.pkl
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
# Pull the DCAM module class straight from the training script so the
# encode/topk/pinv-decode forward is byte-identical to training.
from run_dcam_full import DCAM, project_rows_to_simplex
# Untied diagnostic variant. Import is best-effort: if the user hasn't created
# the file yet, fall back gracefully (tied checkpoints will still load fine).
try:
    from run_dcam_untied import DCAMUntied
    _HAVE_UNTIED = True
except ImportError:
    DCAMUntied = None
    _HAVE_UNTIED = False
from full_classes import IMAGENET2012_CLASSES

import pandas as pd
import random


# ==========================================
# Model Configurations  (mirrors run_dcam_full.py)
# ==========================================

MODEL_CONFIGS = {
    'resnet50': {
        'model_fn': lambda: models.resnet50(pretrained=True),
        'default_target_layer': 'layer3',
        'description': 'ResNet50 (layer3: 1024ch, 14x14)',
        'default_D': 512,
    },
    'resnet18': {
        'model_fn': lambda: models.resnet18(pretrained=True),
        'default_target_layer': 'layer3',
        'description': 'ResNet18 (layer3: 256ch, 14x14)',
        'default_D': 128,
    },
    'vgg16': {
        'model_fn': lambda: models.vgg16(pretrained=True),
        'default_target_layer': 'features[16]',
        'description': 'VGG16 (features[16]: 256ch, 28x28)',
        'default_D': 128,
    },
    'efficientnet': {
        'model_fn': lambda: models.efficientnet_b0(pretrained=True),
        'default_target_layer': 'features[4]',
        'description': 'EfficientNet-B0 (features[4]: ~80ch, 14x14)',
        'default_D': 40,
    },
}


# ==========================================
# Test Dataset Loader (unchanged from CSAE script)
# ==========================================

class ImageNet1kTestDataset(Dataset):
    """Dataset that loads sampled test images from cached samples."""

    def __init__(self, test_metadata_path: Path, transform=None):
        self.transform = transform
        print(f"Loading cached test samples from {test_metadata_path}...")
        metadata = joblib.load(test_metadata_path)
        self.samples = metadata['samples']
        print(f"  Loaded {len(self.samples)} test images")
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

    print(f"\n{'='*80}\nCreating test sample dataset\n{'='*80}")
    print(f"Sampling {images_per_class} test images per class from validation set...")
    test_dir.mkdir(parents=True, exist_ok=True)

    wnid_to_idx = {wnid: idx for idx, wnid in enumerate(IMAGENET2012_CLASSES.keys())}
    class_samples = defaultdict(list)
    val_parquet_files = sorted(raw_dir.glob("validation-*.parquet"))
    if len(val_parquet_files) == 0:
        raise FileNotFoundError(f"No validation parquet files found in {raw_dir}")
    print(f"Found {len(val_parquet_files)} validation parquet files")

    for parquet_file in tqdm(val_parquet_files, desc="Reading validation parquet"):
        df = pd.read_parquet(parquet_file)
        for _, row in df.iterrows():
            label = row['label']
            if len(class_samples[label]) < images_per_class:
                class_samples[label].append((row['image']['bytes'], label))
        min_s = min(len(v) for v in class_samples.values()) if class_samples else 0
        if min_s >= images_per_class and len(class_samples) == 1000:
            print(f"\nCollected {images_per_class} samples for all 1000 classes.")
            break

    samples = []
    for class_idx in range(1000):
        if len(class_samples[class_idx]) >= images_per_class:
            samples.extend(random.sample(class_samples[class_idx], images_per_class))
        else:
            print(f"WARNING: Class {class_idx} has only "
                  f"{len(class_samples[class_idx])} test samples")
            samples.extend(class_samples[class_idx])

    print(f"\nTotal sampled test images: {len(samples)}")
    joblib.dump({'samples': samples, 'images_per_class': images_per_class,
                 'num_classes': 1000, 'wnid_to_idx': wnid_to_idx},
                metadata_path)
    print("Test samples cached.")
    return metadata_path


# ==========================================
# DCAM checkpoint loader
# ==========================================

def load_dcam_from_result(result_path: str, device: str
                          ) -> Tuple[nn.Module, dict]:
    """Reconstruct a DCAM (tied) or DCAMUntied (diagnostic) module from a
    result .pkl produced by run_dcam_full.py or run_dcam_untied.py.

    Both training scripts save:
        {'Pi': [D, C] tensor, 'config': {...}, ...}
    Untied checkpoints additionally save 'W' (the free decoder) and set
    config['decoder'] = 'untied'. We use the 'decoder' marker to pick the
    right module; absence of the marker (or value 'tied') means standard
    DCAM with the pinv decoder.
    """
    print(f"Loading DCAM checkpoint: {result_path}")
    blob = joblib.load(result_path)

    if not isinstance(blob, dict) or 'Pi' not in blob or 'config' not in blob:
        raise ValueError(
            f"{result_path} is not a DCAM result file. "
            f"Expected keys 'Pi' and 'config'; got "
            f"{list(blob.keys()) if isinstance(blob, dict) else type(blob)}")

    Pi = blob['Pi']
    cfg = blob['config']
    C, D = cfg['C'], cfg['D']
    top_k = cfg.get('top_k', min(32, D))
    decoder_kind = cfg.get('decoder', 'tied')

    if Pi.shape != (D, C):
        raise ValueError(
            f"Pi shape {tuple(Pi.shape)} != config (D, C) = ({D}, {C})")

    if decoder_kind == 'untied':
        # Diagnostic variant: encoder Pi + free decoder W.
        if not _HAVE_UNTIED:
            raise ImportError(
                "Checkpoint has decoder='untied' but run_dcam_untied.py "
                "(providing DCAMUntied) was not importable. Add it to the "
                "same directory as run_dcam_full.py and retry.")
        if 'W' not in blob:
            raise ValueError(
                "Untied checkpoint missing 'W' tensor. The result file is "
                "corrupted or was produced by an old script version.")
        W = blob['W']
        if W.shape != (C, D):
            raise ValueError(
                f"W shape {tuple(W.shape)} != expected (C, D) = ({C}, {D})")
        module = DCAMUntied(in_channels=C, num_concepts=D, top_k=top_k).to(device)
        with torch.no_grad():
            module.Pi.data.copy_(project_rows_to_simplex(Pi.clone().to(device)))
            module.W.data.copy_(W.clone().to(device))
        module.eval()
        print(f"  [UNTIED]  C={C}  D={D}  top_k={top_k}  "
              f"(decoder W is free, no ridge)")
    else:
        # Standard tied DCAM.
        ridge = cfg.get('ridge', 1e-4)
        module = DCAM(in_channels=C, num_concepts=D, top_k=top_k,
                      ridge=ridge).to(device)
        with torch.no_grad():
            # Re-project as a defensive measure; well-trained Pi should already
            # satisfy the simplex constraint up to numerical noise.
            Pi_proj = project_rows_to_simplex(Pi.clone().to(device))
            module.Pi.data.copy_(Pi_proj)
        module.eval()
        print(f"  [TIED]    C={C}  D={D}  top_k={top_k}  ridge={ridge}")

    print(f"  Trained on backbone={cfg['model']}  layer={cfg['target_layer']}")
    print(f"  Anchor non-degeneracy nu={cfg.get('nu', float('nan')):.4e}")
    print(f"  Seeds: data={cfg.get('data_seed','?')}  "
          f"model={cfg.get('model_seed','?')}  "
          f"anchor={cfg.get('anchor_seed','?')}")
    return module, cfg


# ==========================================
# Multi-Model with DCAM Reconstruction
# ==========================================

class MultiModelWithDCAMReconstruction:
    """Run a backbone with DCAM reconstruction substituted at the target layer."""

    def __init__(self, model_name: str, target_layer_name: str,
                 dcam_module: nn.Module, device='cuda'):
        # dcam_module is either DCAM (tied) or DCAMUntied (diagnostic). Both
        # expose .C, .D, .top_k and the same forward signature, so the
        # downstream substitution code doesn't need to branch.
        self.model_name = model_name
        self.target_layer_name = target_layer_name
        self.device = device

        config = MODEL_CONFIGS[model_name]
        self.model = config['model_fn']().to(device).eval()

        self.dcam = dcam_module.to(device).eval()

        # Register the hook just for sanity; we actually use the explicit
        # forward_to/from helpers below to substitute activations.
        self.target_layer = self._get_layer_by_name(target_layer_name)
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

        print(f"Model: {config['description']}")
        print(f"Target layer: {target_layer_name}")
        print(f"DCAM: C={self.dcam.C} -> D={self.dcam.D}, top_k={self.dcam.top_k}")

    def _get_layer_by_name(self, layer_name: str):
        if '[' in layer_name:
            attr_name, idx = layer_name.split('[')
            return getattr(self.model, attr_name)[int(idx.rstrip(']'))]
        return getattr(self.model, layer_name)

    def _save_activation(self, module, input, output):
        self.activations = output

    # -- per-channel 99th-percentile normalization (matches training-time) --
    def _normalize_activations(self, acts: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-channel 99th-percentile normalization, vectorized.

        Matches the per-channel clip-and-rescale used by
        run_dcam_full.py::ActivationExtractor.collect (`_norm_chunk`):
            scale_c = quantile_0.99( {A[c,i,j] : A[c,i,j] > 1e-8} )
            A_norm[c] = clamp(A[c], 0, scale_c) / (scale_c + 1e-8)
        Channels with no positive activations are left unchanged.
        """
        B, C, H, W = acts.shape
        flat = acts.view(B, C, H * W)

        pos_mask = flat > 1e-8
        neg_inf = torch.finfo(flat.dtype).min
        masked = torch.where(pos_mask, flat, torch.full_like(flat, neg_inf))
        sorted_vals, _ = torch.sort(masked, dim=-1)            # asc; -inf at bottom

        n_pos = pos_mask.sum(dim=-1)                            # [B, C]
        start = (H * W) - n_pos
        q_offset = ((n_pos - 1).clamp(min=0).float() * 0.99).long()
        q_idx = (start + q_offset).clamp(max=H * W - 1)
        scale_factors = sorted_vals.gather(-1, q_idx.unsqueeze(-1)).squeeze(-1)

        valid = (n_pos > 0) & (scale_factors > 1e-8)
        scale_factors = torch.where(valid, scale_factors,
                                    torch.ones_like(scale_factors))
        scale_4d = scale_factors.view(B, C, 1, 1)
        normalized = torch.clamp(acts, min=0.0)
        normalized = torch.minimum(normalized, scale_4d)
        normalized = normalized / (scale_4d + 1e-8)

        valid_4d = valid.view(B, C, 1, 1)
        normalized = torch.where(valid_4d, normalized, acts)
        return normalized, scale_factors

    # -- explicit two-half forwards (so we can substitute activations) ----
    def _forward_to_target_layer(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name in ['resnet50', 'resnet18']:
            x = self.model.conv1(x); x = self.model.bn1(x)
            x = self.model.relu(x);  x = self.model.maxpool(x)
            x = self.model.layer1(x)
            if 'layer1' in self.target_layer_name: return x
            x = self.model.layer2(x)
            if 'layer2' in self.target_layer_name: return x
            x = self.model.layer3(x)
            if 'layer3' in self.target_layer_name: return x
            x = self.model.layer4(x)
            return x
        elif self.model_name in ['vgg16', 'efficientnet']:
            target_idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            for i in range(target_idx + 1):
                x = self.model.features[i](x)
            return x
        return x

    def _forward_from_target_layer(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name in ['resnet50', 'resnet18']:
            if 'layer1' in self.target_layer_name:
                x = self.model.layer2(x); x = self.model.layer3(x); x = self.model.layer4(x)
            elif 'layer2' in self.target_layer_name:
                x = self.model.layer3(x); x = self.model.layer4(x)
            elif 'layer3' in self.target_layer_name:
                x = self.model.layer4(x)
            # layer4 → straight to head
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
            return x
        elif self.model_name == 'vgg16':
            target_idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            for i in range(target_idx + 1, len(self.model.features)):
                x = self.model.features[i](x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        elif self.model_name == 'efficientnet':
            target_idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            for i in range(target_idx + 1, len(self.model.features)):
                x = self.model.features[i](x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        return x

    # -- public --
    def forward_original(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)

    def forward_with_reconstruction(self, x: torch.Tensor
                                    ) -> Tuple[torch.Tensor, Dict]:
        """Forward with DCAM reconstruction substituted at the target layer."""
        with torch.no_grad():
            x = self._forward_to_target_layer(x)
            original_acts = x.clone()

            normalized_acts, scale_factors = self._normalize_activations(original_acts)

            # DCAM forward: A_hat (normalized scale), z (sparse concept maps)
            A_hat_norm, z = self.dcam(normalized_acts, use_topk=True)

            # Denormalize back to the original activation scale.
            scale_4d = scale_factors.unsqueeze(2).unsqueeze(3)
            reconstructed_acts = A_hat_norm * scale_4d

            logits = self._forward_from_target_layer(reconstructed_acts)

            mse = F.mse_loss(reconstructed_acts, original_acts).item()
            rel_err = ((reconstructed_acts - original_acts).abs().mean()
                       / (original_acts.abs().mean() + 1e-8)).item()
            # Sparsity here = fraction of active CONCEPT entries (DCAM analogue
            # of CSAE feature sparsity). Useful sanity check: after TopK over
            # D, this should be roughly top_k / D for the spatial-mean cells.
            sparsity = (z > 0).float().mean().item()

            stats = {'mse': mse, 'relative_error': rel_err, 'sparsity': sparsity}
            return logits, stats


# ==========================================
# Evaluation
# ==========================================

def evaluate_accuracy_drop(model: MultiModelWithDCAMReconstruction,
                           data_loader: DataLoader, device: str = 'cuda') -> Dict:
    """Evaluate classification accuracy with original vs reconstructed activations."""
    total_samples = 0
    correct_original = 0
    correct_reconstructed = 0
    mse_list, relerr_list, sparsity_list = [], [], []

    per_class_correct_original = defaultdict(int)
    per_class_correct_reconstructed = defaultdict(int)
    per_class_total = defaultdict(int)

    print("\nEvaluating accuracy: original vs DCAM-reconstructed activations")
    print("=" * 80)

    for images, labels in tqdm(data_loader, desc="Processing batches"):
        images, labels = images.to(device), labels.to(device)
        B = images.size(0)

        logits_o = model.forward_original(images)
        pred_o = logits_o.argmax(dim=1)

        logits_r, stats = model.forward_with_reconstruction(images)
        pred_r = logits_r.argmax(dim=1)

        correct_original += (pred_o == labels).sum().item()
        correct_reconstructed += (pred_r == labels).sum().item()

        for i in range(B):
            lbl = labels[i].item()
            per_class_total[lbl] += 1
            if pred_o[i] == labels[i]:
                per_class_correct_original[lbl] += 1
            if pred_r[i] == labels[i]:
                per_class_correct_reconstructed[lbl] += 1

        total_samples += B
        mse_list.append(stats['mse'])
        relerr_list.append(stats['relative_error'])
        sparsity_list.append(stats['sparsity'])

    acc_o = correct_original / total_samples * 100
    acc_r = correct_reconstructed / total_samples * 100

    per_class_drop = {}
    for lbl, n in per_class_total.items():
        if n > 0:
            ao = per_class_correct_original[lbl] / n * 100
            ar = per_class_correct_reconstructed[lbl] / n * 100
            per_class_drop[lbl] = ao - ar
    worst = sorted(per_class_drop.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        'total_samples': total_samples,
        'acc_original': acc_o,
        'acc_reconstructed': acc_r,
        'acc_drop': acc_o - acc_r,
        'avg_mse': float(np.mean(mse_list)),
        'avg_relative_error': float(np.mean(relerr_list)),
        'avg_sparsity': float(np.mean(sparsity_list)),
        'per_class_total': dict(per_class_total),
        'worst_classes': worst,
    }


def print_results(results: Dict, model_name: str, dcam_cfg: dict):
    print("\n" + "=" * 80)
    print(f"DCAM ACCURACY DROP ANALYSIS  ({model_name.upper()} - ImageNet-1k)")
    print("=" * 80)
    print(f"\nDCAM config:")
    print(f"  C={dcam_cfg['C']}  D={dcam_cfg['D']}  top_k={dcam_cfg.get('top_k','?')}")
    print(f"  lambda_anchor={dcam_cfg.get('lambda_anchor','?')}  "
          f"lambda_l1={dcam_cfg.get('lambda_l1','?')}  "
          f"ridge={dcam_cfg.get('ridge','?')}")
    print(f"  anchor nu={dcam_cfg.get('nu', float('nan')):.4e}  "
          f"seeds data={dcam_cfg.get('data_seed','?')} "
          f"model={dcam_cfg.get('model_seed','?')} "
          f"anchor={dcam_cfg.get('anchor_seed','?')}")
    print(f"\nDataset:")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Number of classes: {len(results['per_class_total'])}")

    print(f"\nTop-1 Accuracy:")
    print(f"  Original (no reconstruction):     {results['acc_original']:.2f}%")
    print(f"  Reconstructed (DCAM):             {results['acc_reconstructed']:.2f}%")
    print(f"  Accuracy Drop:                    {results['acc_drop']:.2f}%")
    if results['acc_original'] > 0:
        print(f"  Relative Drop:                    "
              f"{results['acc_drop']/results['acc_original']*100:.2f}%")

    print(f"\nReconstruction Quality:")
    print(f"  Average MSE:                      {results['avg_mse']:.6f}")
    print(f"  Average Relative Error:           {results['avg_relative_error']:.6f}")
    print(f"  Average Concept Activity (%):     {results['avg_sparsity']*100:.2f}%")

    if results['worst_classes']:
        print(f"\nTop 10 Classes with Largest Accuracy Drop:")
        for i, (label, drop) in enumerate(results['worst_classes'], 1):
            name = list(IMAGENET2012_CLASSES.values())[label]
            name = name[:50] + "..." if len(name) > 50 else name
            print(f"  {i:2d}. Class {label:3d} ({name}): {drop:+.2f}%")

    print("\n" + "=" * 80)
    print("Interpretation (Top-1 absolute drop):")
    d = results['acc_drop']
    if d < 1.0:
        print("  Excellent: DCAM preserves nearly all classification info (<1% drop)")
    elif d < 3.0:
        print("  Very good: <3% drop")
    elif d < 5.0:
        print("  Good: <5% drop")
    elif d < 10.0:
        print("  Moderate: 5-10% drop")
    else:
        print("  High: >10% drop -- DCAM bottleneck is lossy at this (D, top_k).")
    print("=" * 80 + "\n")


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate accuracy drop with DCAM-reconstructed activations')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--target_layer', type=str, default=None,
                        help='Target layer (default: per-model default). Must match '
                             'the layer the DCAM checkpoint was trained on.')
    parser.add_argument('--dcam_model', type=str, default=None,
                        help='Path to trained DCAM result .pkl '
                             '(auto-detected if not specified).')
    parser.add_argument('--data_seed', type=int, default=42,
                        help='Data seed used at training (only used for auto-naming).')
    parser.add_argument('--model_seed', type=int, default=42,
                        help='Model seed used at training (only used for auto-naming).')
    parser.add_argument('--D', type=int, default=None,
                        help='Atom count D used at training (only used for auto-naming). '
                             'Default: per-model default_D.')

    parser.add_argument('--raw_data_dir', type=str,
                        default='/data/imagenet_raw/data')
    parser.add_argument('--test_data_dir', type=str,
                        default='/data/imagenet1k_sampletest')
    parser.add_argument('--test_images_per_class', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--force_resample', action='store_true')
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()

    cfg_proto = MODEL_CONFIGS[args.model]
    target_layer = args.target_layer or cfg_proto['default_target_layer']
    D = args.D if args.D is not None else cfg_proto['default_D']

    # ---- auto-detect DCAM checkpoint path ------------------------------
    if args.dcam_model is None:
        prefix = f"imagenet1k_dcam_{args.model}"
        if args.target_layer is not None:
            prefix += "_" + args.target_layer.replace('[', '_').replace(']', '')
        prefix += f"_D{D}_seed{args.data_seed}-{args.model_seed}"
        args.dcam_model = f"{prefix}_result.pkl"
        print(f"Auto-detected DCAM checkpoint: {args.dcam_model}")

    if args.output_file is None:
        stem = Path(args.dcam_model).stem.replace('_result', '')
        args.output_file = f"accuracy_drop_dcam_{stem}.txt"

    print("=" * 80)
    print(f"ImageNet-1k DCAM Accuracy Drop Analysis -- {args.model.upper()}")
    print("=" * 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. test samples
    print("\nStep 1: Preparing test dataset...")
    test_metadata_path = create_test_samples_if_needed(
        raw_dir=Path(args.raw_data_dir),
        test_dir=Path(args.test_data_dir),
        images_per_class=args.test_images_per_class,
        force_resample=args.force_resample)

    # 2/3. load DCAM (and read its trained config for cross-checks)
    print("\nStep 2: Loading DCAM checkpoint...")
    dcam_module, dcam_cfg = load_dcam_from_result(args.dcam_model, device)

    # Sanity-check: backbone/layer in the checkpoint MUST match what we run.
    # A Pi trained on backbone X cannot be substituted into backbone Y, since
    # the C dimension (and the activation distribution) differ. So if there's
    # a mismatch, we *adopt the checkpoint's backbone/layer* and override the
    # CLI, rather than silently crashing inside DCAM.forward.
    ckpt_model = dcam_cfg.get('model')
    if ckpt_model is not None and ckpt_model != args.model:
        if ckpt_model not in MODEL_CONFIGS:
            raise ValueError(
                f"Checkpoint was trained on backbone '{ckpt_model}', which is "
                f"not in MODEL_CONFIGS. Cannot evaluate.")
        print(f"  NOTE: checkpoint backbone '{ckpt_model}' != --model "
              f"'{args.model}'. Overriding to '{ckpt_model}' "
              f"(the Pi was trained on it; the CLI value is ignored).")
        args.model = ckpt_model
        cfg_proto = MODEL_CONFIGS[args.model]
        # If the user didn't pin --target_layer, fall back to the new backbone's
        # default; otherwise keep their explicit choice and let the next check
        # handle it.
        if args.target_layer is None:
            target_layer = cfg_proto['default_target_layer']

    ckpt_layer = dcam_cfg.get('target_layer')
    if ckpt_layer is not None and ckpt_layer != target_layer:
        print(f"  NOTE: checkpoint target_layer '{ckpt_layer}' != "
              f"'{target_layer}'. Overriding to '{ckpt_layer}'.")
        target_layer = ckpt_layer

    # Final hard check: the Pi we loaded must agree on C with the backbone we
    # are about to instantiate. We rebuild a one-pass dummy to compute C from
    # the actual backbone, since spelling out C per-model duplicates state.
    expected_C = dcam_cfg['C']
    print(f"  Pi expects C={expected_C} channels at layer '{target_layer}'.")

    # 4. combined model
    print("\nStep 3: Building backbone + DCAM substitution pipeline...")
    model = MultiModelWithDCAMReconstruction(
        model_name=args.model,
        target_layer_name=target_layer,
        dcam_module=dcam_module,
        device=device)

    # 5. data
    print("\nStep 4: Loading test dataset...")
    data_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    dataset = ImageNet1kTestDataset(test_metadata_path, transform=data_transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total test images: {len(dataset)}")

    # 6. evaluate
    print("\nStep 5: Evaluating accuracy drop...")
    results = evaluate_accuracy_drop(model, data_loader, device=device)

    # 7. report
    print_results(results, args.model, dcam_cfg)

    # 8. save
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{args.model.upper()} DCAM - Accuracy Drop Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Model:           {args.model}\n")
        f.write(f"  Target Layer:    {target_layer}\n")
        f.write(f"  DCAM Checkpoint: {args.dcam_model}\n")
        f.write(f"  C={dcam_cfg['C']}  D={dcam_cfg['D']}  "
                f"top_k={dcam_cfg.get('top_k','?')}\n")
        f.write(f"  lambda_anchor={dcam_cfg.get('lambda_anchor','?')}  "
                f"lambda_l1={dcam_cfg.get('lambda_l1','?')}  "
                f"ridge={dcam_cfg.get('ridge','?')}\n")
        f.write(f"  Seeds: data={dcam_cfg.get('data_seed','?')} "
                f"model={dcam_cfg.get('model_seed','?')} "
                f"anchor={dcam_cfg.get('anchor_seed','?')}\n")
        f.write(f"  Anchor nu:       {dcam_cfg.get('nu', float('nan')):.4e}\n")
        f.write(f"  Test Images:     {results['total_samples']}\n")
        f.write(f"  Classes:         {len(results['per_class_total'])}\n\n")

        f.write("Top-1 Accuracy:\n")
        f.write(f"  Original:        {results['acc_original']:.2f}%\n")
        f.write(f"  Reconstructed:   {results['acc_reconstructed']:.2f}%\n")
        f.write(f"  Drop:            {results['acc_drop']:.2f}%\n")
        if results['acc_original'] > 0:
            f.write(f"  Relative Drop:   "
                    f"{results['acc_drop']/results['acc_original']*100:.2f}%\n\n")
        else:
            f.write("  Relative Drop:   N/A\n\n")

        f.write("Reconstruction Quality:\n")
        f.write(f"  Average MSE:             {results['avg_mse']:.6f}\n")
        f.write(f"  Average Relative Error:  {results['avg_relative_error']:.6f}\n")
        f.write(f"  Average Concept Active:  {results['avg_sparsity']*100:.2f}%\n\n")

        if results['worst_classes']:
            f.write("Top 10 Classes with Largest Accuracy Drop:\n")
            for i, (label, drop) in enumerate(results['worst_classes'], 1):
                name = list(IMAGENET2012_CLASSES.values())[label]
                f.write(f"  {i:2d}. Class {label:3d} ({name}): {drop:+.2f}%\n")
            f.write("\n")
        f.write("=" * 80 + "\n")

    print(f"Results saved to {args.output_file}")
    print("\n" + "=" * 80)
    print("Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()