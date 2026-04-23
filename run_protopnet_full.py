"""
ProtoPNet Training Script on Full ImageNet-1k
(This Looks Like That: Prototypical Part Network for Interpretable Classification)

Based on: "This Looks Like That: Deep Learning for Interpretable Image Recognition"
Chen et al., NeurIPS 2019

Key Design:
- Backbone: Pretrained CNN (ResNet50/ResNet18/VGG16)
- Prototype Layer: Learns prototypical parts for ALL classes
- Classification: Weighted similarity to class prototypes
- Interpretability: "This part looks like that prototype"

Training Procedure (3 stages):
1. Joint Training: Optimize backbone + prototypes with cluster/separation costs
2. Prototype Projection: Push prototypes to nearest training patches
3. Last Layer Optimization: Sparse convex optimization

Dataset:
- Full ImageNet-1k (1000 classes)
- Samples 50 images per class from training set (50,000 total)
- Caches sampled dataset to /data/imagenet1k_sampled for reuse

Memory Usage Guide (for OOM errors):
    Memory consumption depends on:
    1. Batch size (most direct impact)
    2. Number of prototypes (affects distance computation)
    3. Backbone model (ResNet50 > ResNet18)
    4. Image resolution (224×224 default)

    Recommended configurations by GPU memory:

    8GB GPU (e.g., RTX 2080):
        python run_protopnet_full.py --model resnet18 --batch_size 8 --num_prototypes_per_class 2

    12GB GPU (e.g., RTX 3080):
        python run_protopnet_full.py --model resnet18 --batch_size 16 --num_prototypes_per_class 2
        python run_protopnet_full.py --model resnet50 --batch_size 8 --num_prototypes_per_class 2

    16GB GPU (e.g., RTX 4080, V100):
        python run_protopnet_full.py --model resnet50 --batch_size 16 --num_prototypes_per_class 2

    24GB+ GPU (e.g., RTX 3090, A100):
        python run_protopnet_full.py --model resnet50 --batch_size 32 --num_prototypes_per_class 2

Usage:
    # Minimal memory (8GB GPU)
    python run_protopnet_full.py --model resnet18 --batch_size 8

    # ResNet18 with moderate batch size
    python run_protopnet_full.py --model resnet18 --batch_size 16

    # ResNet50 (requires more memory)
    python run_protopnet_full.py --model resnet50 --batch_size 8

    # Custom number of prototypes per class
    python run_protopnet_full.py --num_prototypes_per_class 10

    # Force resample dataset
    python run_protopnet_full.py --force_resample
"""

import torch
torch.cuda.init()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
import joblib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import sys
import os
from pathlib import Path
from PIL import Image
import io
import pandas as pd
import random
import argparse
from collections import defaultdict

sys.path.append('.')
from src.gradcam import GradCAM
from full_classes import IMAGENET2012_CLASSES

# ==========================================
# Configuration
# ==========================================

# Paths
IMAGENET_RAW_DIR = Path("/data/imagenet_raw/data")
IMAGENET_SAMPLED_DIR = Path("/data/imagenet1k_sampled")
ACTIVATION_CACHE_DIR = Path("cache_protopnet")

# Sampling parameters
IMAGES_PER_CLASS = 50
NUM_CLASSES = 1000

# Model configurations
MODEL_CONFIGS = {
    'resnet50': {
        'model_fn': lambda: models.resnet50(pretrained=True),
        'target_layer': 'layer3',
        'add_on_layers_type': 'regular',
        'prototype_shape': (2000, 1024, 1, 1),  # (num_prototypes, channels, H, W)
        'description': 'ResNet50 (layer3: 1024ch, 14×14)'
    },
    'resnet18': {
        'model_fn': lambda: models.resnet18(pretrained=True),
        'target_layer': 'layer3',
        'add_on_layers_type': 'regular',
        'prototype_shape': (2000, 256, 1, 1),
        'description': 'ResNet18 (layer3: 256ch, 14×14)'
    },
    'vgg16': {
        'model_fn': lambda: models.vgg16(pretrained=True),
        'target_layer': 'features',
        'add_on_layers_type': 'regular',
        'prototype_shape': (2000, 512, 1, 1),
        'description': 'VGG16 (features: 512ch, 14×14)'
    }
}


# ==========================================
# ProtoPNet Architecture
# ==========================================

class ProtoPNet(nn.Module):
    """
    Prototypical Part Network for interpretable classification.

    Architecture:
        Input -> Conv Backbone (f) -> Add-on Layers -> Prototype Layer (gp) -> Last Layer (h) -> Logits

    Each prototype represents a prototypical part of some class.
    Classification is based on similarity to learned prototypes.
    """

    def __init__(self,
                 backbone_name: str = 'resnet50',
                 num_classes: int = 1000,
                 num_prototypes_per_class: int = 2,
                 prototype_shape: Tuple[int, int, int, int] = None,
                 init_weights: bool = True):
        super().__init__()

        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_prototypes = num_classes * num_prototypes_per_class

        # Load pretrained backbone
        config = MODEL_CONFIGS[backbone_name]
        base_model = config['model_fn']()

        # Extract convolutional features (remove classifier)
        if 'resnet' in backbone_name:
            # ResNet: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3
            self.conv_features = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3
            )
        elif 'vgg' in backbone_name:
            # VGG: use features up to certain layer
            self.conv_features = base_model.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Determine feature dimensions
        if prototype_shape is None:
            prototype_shape = config['prototype_shape']

        self.prototype_shape = prototype_shape
        num_prototypes_cfg, prototype_dim, prototype_h, prototype_w = prototype_shape

        # Override num_prototypes from config with actual calculation
        self.num_prototypes = num_classes * num_prototypes_per_class
        self.prototype_shape = (self.num_prototypes, prototype_dim, prototype_h, prototype_w)

        # Add-on layers (1x1 conv to adjust dimensions if needed)
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(prototype_dim, prototype_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(prototype_dim, prototype_dim, kernel_size=1),
            nn.Sigmoid()  # Last layer uses sigmoid as in paper
        )

        # Prototype layer - learnable prototypes
        # Shape: (num_prototypes, prototype_dim, prototype_h, prototype_w)
        self.prototype_vectors = nn.Parameter(
            torch.randn(self.prototype_shape),
            requires_grad=True
        )

        # Last layer - fully connected (no bias)
        # Connects prototype similarity scores to class logits
        self.last_layer = nn.Linear(self.num_prototypes, num_classes, bias=False)

        # Epsilon for numerical stability
        self.epsilon = 1e-4

        # Store class identity of each prototype
        # prototype_class_identity[j] = k means prototype j belongs to class k
        self.prototype_class_identity = torch.zeros(self.num_prototypes, dtype=torch.long)
        for k in range(num_classes):
            start_idx = k * num_prototypes_per_class
            end_idx = (k + 1) * num_prototypes_per_class
            self.prototype_class_identity[start_idx:end_idx] = k

        # Initialize weights (must come after prototype_class_identity is created)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights as described in the paper."""
        # Initialize add-on layers
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize last layer weights
        # Set w_{k,j} = 1 if prototype j belongs to class k, else -0.5
        with torch.no_grad():
            for k in range(self.num_classes):
                for j in range(self.num_prototypes):
                    if self.prototype_class_identity[j] == k:
                        self.last_layer.weight[k, j] = 1.0
                    else:
                        self.last_layer.weight[k, j] = -0.5

    def conv_features_forward(self, x):
        """Forward through convolutional backbone."""
        return self.conv_features(x)

    def prototype_distances(self, x):
        """
        Compute L2 distances between conv features and prototypes.

        Args:
            x: Conv features of shape [B, D, H, W]

        Returns:
            distances: [B, num_prototypes, H, W] - L2 distances to each prototype
            min_distances: [B, num_prototypes] - minimum distance to each prototype
        """
        B, D, H, W = x.shape
        num_prototypes = self.prototype_vectors.shape[0]

        # Unfold conv features into patches
        # x: [B, D, H, W] -> patches: [B, D, H*W]
        patches = x.view(B, D, H * W)

        # Compute L2 distances
        # For each prototype p_j: ||patch - p_j||^2
        distances = torch.zeros(B, num_prototypes, H, W, device=x.device)

        for j in range(num_prototypes):
            # prototype: [D, 1, 1]
            prototype = self.prototype_vectors[j].view(D, 1)

            # Compute squared L2 distance for all patches
            # ||patch - p||^2 = ||patch||^2 + ||p||^2 - 2 * patch^T * p
            patch_norm_sq = (patches ** 2).sum(dim=1, keepdim=True)  # [B, 1, H*W]
            proto_norm_sq = (prototype ** 2).sum()  # scalar
            cross_term = torch.matmul(prototype.t(), patches)  # [1, H*W]

            dist_sq = patch_norm_sq + proto_norm_sq - 2 * cross_term  # [B, 1, H*W]
            dist_sq = dist_sq.view(B, H, W)

            distances[:, j, :, :] = dist_sq

        # Global min pooling - find minimum distance across spatial dimensions
        min_distances = F.adaptive_max_pool2d(-distances, (1, 1)).view(B, num_prototypes)
        min_distances = -min_distances  # Convert back to actual distances

        return distances, min_distances

    def distance_to_similarity(self, distances):
        """
        Convert distances to similarity scores.

        Uses: similarity = log((||z-p||^2 + 1) / (||z-p||^2 + ε))
        """
        return torch.log((distances + 1) / (distances + self.epsilon))

    def forward(self, x, return_distances=False):
        """
        Forward pass through ProtoPNet.

        Args:
            x: Input images [B, 3, H_img, W_img]
            return_distances: If True, return distances and min_distances

        Returns:
            logits: Class logits [B, num_classes]
            (Optional) distances, min_distances
        """
        # Convolutional features
        conv_features = self.conv_features(x)

        # Add-on layers
        x_aug = self.add_on_layers(conv_features)

        # Compute distances to prototypes
        distances, min_distances = self.prototype_distances(x_aug)

        # Convert to similarity scores (apply max pooling already done in prototype_distances)
        # Shape: [B, num_prototypes]
        prototype_activations = self.distance_to_similarity(min_distances)

        # Last layer - compute logits
        logits = self.last_layer(prototype_activations)

        if return_distances:
            return logits, distances, min_distances, conv_features
        else:
            return logits

    def push_forward(self, x):
        """
        Forward pass for prototype projection.
        Returns conv features for finding nearest patches.
        """
        conv_features = self.conv_features(x)
        x_aug = self.add_on_layers(conv_features)
        distances, _ = self.prototype_distances(x_aug)
        return conv_features, x_aug, distances

    def set_last_layer_incorrect_connection(self, incorrect_strength: float = -0.5):
        """
        Set negative connection weights for non-class prototypes.

        Args:
            incorrect_strength: Weight for w_{k,j} when prototype j not in class k
        """
        with torch.no_grad():
            for k in range(self.num_classes):
                for j in range(self.num_prototypes):
                    if self.prototype_class_identity[j] != k:
                        self.last_layer.weight[k, j] = incorrect_strength


# ==========================================
# Loss Functions
# ==========================================

class ClusterLoss(nn.Module):
    """
    Cluster cost: Encourages each image to have a patch close to at least one prototype of its class.

    Clst = (1/n) Σ_i min_{j: p_j ∈ P_{y_i}} min_{z ∈ patches(f(x_i))} ||z - p_j||^2
    """

    def __init__(self, prototype_class_identity):
        super().__init__()
        self.prototype_class_identity = prototype_class_identity

    def forward(self, min_distances, labels):
        """
        Args:
            min_distances: [B, num_prototypes] - min distance to each prototype
            labels: [B] - class labels

        Returns:
            cluster_cost: scalar
        """
        B = min_distances.shape[0]
        cluster_cost = 0.0

        for i in range(B):
            # Find prototypes belonging to the correct class
            class_label = labels[i].item()
            class_prototype_mask = (self.prototype_class_identity == class_label)

            # Get distances to class prototypes only
            class_prototype_distances = min_distances[i, class_prototype_mask]

            # Minimum distance to any class prototype
            min_dist = torch.min(class_prototype_distances)
            cluster_cost += min_dist

        return cluster_cost / B


class SeparationLoss(nn.Module):
    """
    Separation cost: Encourages patches to stay away from prototypes of other classes.

    Sep = -(1/n) Σ_i min_{j: p_j ∉ P_{y_i}} min_{z ∈ patches(f(x_i))} ||z - p_j||^2
    """

    def __init__(self, prototype_class_identity):
        super().__init__()
        self.prototype_class_identity = prototype_class_identity

    def forward(self, min_distances, labels):
        """
        Args:
            min_distances: [B, num_prototypes] - min distance to each prototype
            labels: [B] - class labels

        Returns:
            separation_cost: scalar (negative)
        """
        B = min_distances.shape[0]
        separation_cost = 0.0

        for i in range(B):
            # Find prototypes NOT belonging to the correct class
            class_label = labels[i].item()
            other_class_prototype_mask = (self.prototype_class_identity != class_label)

            # Get distances to other class prototypes
            other_prototype_distances = min_distances[i, other_class_prototype_mask]

            # Minimum distance to any other class prototype
            min_dist = torch.min(other_prototype_distances)
            separation_cost += min_dist

        return -separation_cost / B  # Negative because we want to maximize distance


# ==========================================
# Dataset (reuse from XCSAE)
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
# Training Functions
# ==========================================

def warm_only(model):
    """Set requires_grad=True only for add-on layers and prototypes."""
    for p in model.conv_features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True  # FIX: Last layer must be trainable during warm-up


def joint(model):
    """Set requires_grad=True for all layers including last layer."""
    for p in model.conv_features.parameters():
        p.requires_grad = True
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True  # FIX: Last layer must be trainable during joint training


def last_only(model):
    """Set requires_grad=True only for last layer."""
    for p in model.conv_features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer.parameters():
        p.requires_grad = True


def train_epoch(model, dataloader, optimizer,
                criterion_ce, criterion_clst, criterion_sep,
                lambda_clst=0.8, lambda_sep=-0.08,
                device='cuda'):
    """
    Train for one epoch (Stage 1: Joint Training).

    Loss = CrossEntropy + λ_clst * Clst + λ_sep * Sep
    """
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_clst = 0.0
    total_sep = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, _, min_distances, _ = model(images, return_distances=True)

        # Move prototype class identity to same device
        if model.prototype_class_identity.device != device:
            model.prototype_class_identity = model.prototype_class_identity.to(device)

        # Compute losses
        loss_ce = criterion_ce(logits, labels)
        loss_clst = criterion_clst(min_distances, labels)
        loss_sep = criterion_sep(min_distances, labels)

        # Total loss
        loss = loss_ce + lambda_clst * loss_clst + lambda_sep * loss_sep

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_clst += loss_clst.item()
        total_sep += loss_sep.item()

        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_ce = total_ce / len(dataloader)
    avg_clst = total_clst / len(dataloader)
    avg_sep = total_sep / len(dataloader)
    accuracy = 100.0 * correct / total

    return {
        'loss': avg_loss,
        'ce_loss': avg_ce,
        'clst_loss': avg_clst,
        'sep_loss': avg_sep,
        'accuracy': accuracy
    }


def push_prototypes(model, dataloader, device='cuda'):
    """
    Stage 2: Prototype Projection (Push).

    For each prototype, find the nearest training patch from the same class
    and update the prototype to be that patch.
    """
    print(f"\n{'='*80}")
    print("Prototype Projection (Push)")
    print(f"{'='*80}")

    model.eval()

    # Store global min distances and corresponding patches for each prototype
    global_min_proto_dist = {j: float('inf') for j in range(model.num_prototypes)}
    global_min_patches = {j: None for j in range(model.num_prototypes)}

    # Move prototype class identity to device
    if model.prototype_class_identity.device != device:
        model.prototype_class_identity = model.prototype_class_identity.to(device)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Finding nearest patches"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            conv_features, proto_features, distances = model.push_forward(images)

            # For each prototype
            for j in range(model.num_prototypes):
                # Get class of this prototype
                proto_class = model.prototype_class_identity[j].item()

                # Find images in batch that belong to this class
                class_mask = (labels == proto_class)

                if class_mask.sum() == 0:
                    continue  # No images of this class in batch

                # Get distances for this prototype and these images
                # distances: [B, num_prototypes, H, W]
                proto_distances = distances[class_mask, j, :, :]  # [N_class, H, W]

                # Find minimum distance across all patches and images
                min_dist, min_idx = proto_distances.view(-1).min(0)
                min_dist = min_dist.item()

                # Update global minimum if necessary
                if min_dist < global_min_proto_dist[j]:
                    global_min_proto_dist[j] = min_dist

                    # Find which image and which spatial location
                    n_class_images = proto_distances.shape[0]
                    H, W = proto_distances.shape[1], proto_distances.shape[2]

                    img_idx_in_class = min_idx.item() // (H * W)
                    spatial_idx = min_idx.item() % (H * W)
                    h_idx = spatial_idx // W
                    w_idx = spatial_idx % W

                    # Get the actual image index in batch
                    class_indices = torch.where(class_mask)[0]
                    img_idx = class_indices[img_idx_in_class].item()

                    # Extract the patch from proto_features
                    # proto_features: [B, D, H, W]
                    patch = proto_features[img_idx, :, h_idx:h_idx+1, w_idx:w_idx+1]
                    global_min_patches[j] = patch.clone()

    # Update prototype vectors
    print("\nUpdating prototype vectors...")
    for j in range(model.num_prototypes):
        if global_min_patches[j] is not None:
            model.prototype_vectors[j] = global_min_patches[j].squeeze()
        else:
            print(f"Warning: No patch found for prototype {j} (class {model.prototype_class_identity[j].item()})")

    print(f"✓ Prototype projection complete!")
    print(f"  Average min distance: {np.mean(list(global_min_proto_dist.values())):.4f}")


def last_layer_optimization(model, dataloader, optimizer, criterion_ce,
                            lambda_l1=1e-4, num_epochs=10, device='cuda'):
    """
    Stage 3: Last Layer Optimization.

    Convex optimization with L1 regularization on negative connections.
    Goal: Make w_{k,j} ≈ 0 for prototypes not in class k.
    """
    print(f"\n{'='*80}")
    print("Last Layer Optimization")
    print(f"{'='*80}")

    model.eval()  # Fix all other layers
    last_only(model)  # Only optimize last layer

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_ce = 0.0
        total_l1 = 0.0

        for images, labels in tqdm(dataloader, desc=f"Last layer epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(images)

            # Cross entropy loss
            loss_ce = criterion_ce(logits, labels)

            # L1 regularization on incorrect connections
            # We want to penalize w_{k,j} for j where prototype j is not in class k
            l1_loss = 0.0
            for k in range(model.num_classes):
                incorrect_mask = (model.prototype_class_identity != k).float().to(device)
                l1_loss += torch.sum(torch.abs(model.last_layer.weight[k, :]) * incorrect_mask)

            loss = loss_ce + lambda_l1 * l1_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_l1 += l1_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_ce = total_ce / len(dataloader)
        avg_l1 = total_l1 / len(dataloader)

        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, CE={avg_ce:.4f}, L1={avg_l1:.4f}")

    print(f"✓ Last layer optimization complete!")


# ==========================================
# Visualization
# ==========================================

def plot_training_logs(logs: Dict, model_name: str, save_path: str):
    """Plot training metrics."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'ProtoPNet Training ({model_name.upper()} - ImageNet-1k)',
                 fontsize=14, fontweight='bold')

    # Loss
    axs[0, 0].plot(logs['loss'], color='blue', linewidth=1.5)
    axs[0, 0].set_title("Total Loss")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].grid(True, alpha=0.3)

    # Cross Entropy
    axs[0, 1].plot(logs['ce_loss'], color='red', linewidth=1.5)
    axs[0, 1].set_title("Cross Entropy Loss")
    axs[0, 1].grid(True, alpha=0.3)

    # Cluster Loss
    axs[0, 2].plot(logs['clst_loss'], color='green', linewidth=1.5)
    axs[0, 2].set_title("Cluster Loss")
    axs[0, 2].grid(True, alpha=0.3)

    # Separation Loss
    axs[1, 0].plot(logs['sep_loss'], color='orange', linewidth=1.5)
    axs[1, 0].set_title("Separation Loss")
    axs[1, 0].grid(True, alpha=0.3)

    # Accuracy
    axs[1, 1].plot(logs['accuracy'], color='purple', linewidth=1.5)
    axs[1, 1].set_title("Training Accuracy")
    axs[1, 1].set_ylabel("Accuracy (%)")
    axs[1, 1].grid(True, alpha=0.3)

    # Loss components
    axs[1, 2].plot(logs['ce_loss'], label='CE', alpha=0.7)
    axs[1, 2].plot(logs['clst_loss'], label='Clst', alpha=0.7)
    axs[1, 2].set_title("Loss Components")
    axs[1, 2].legend()
    axs[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


# ==========================================
# Main Training Script
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='ProtoPNet Training on Full ImageNet-1k'
    )
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Backbone model (default: resnet50)')
    parser.add_argument('--num_prototypes_per_class', type=int, default=2,
                       help='Number of prototypes per class (default: 2)')
    parser.add_argument('--force_resample', action='store_true',
                       help='Force resampling of dataset')
    parser.add_argument('--warm_epochs', type=int, default=5,
                       help='Number of warm-up epochs (default: 5)')
    parser.add_argument('--joint_epochs', type=int, default=20,
                       help='Number of joint training epochs (default: 20)')
    parser.add_argument('--last_layer_epochs', type=int, default=10,
                       help='Number of last layer optimization epochs (default: 10)')
    parser.add_argument('--push_every', type=int, default=5,
                       help='Push prototypes every N epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--lambda_clst', type=float, default=0.8,
                       help='Cluster loss weight (default: 0.8)')
    parser.add_argument('--lambda_sep', type=float, default=-0.08,
                       help='Separation loss weight (default: -0.08)')
    parser.add_argument('--lambda_l1', type=float, default=1e-4,
                       help='L1 regularization weight for last layer (default: 1e-4)')

    args = parser.parse_args()

    print("="*80)
    print(f"ProtoPNet Training on Full ImageNet-1k")
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

    train_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)

    # Create model
    print(f"\n{'='*80}")
    print("Creating ProtoPNet model...")
    print(f"{'='*80}")

    model = ProtoPNet(
        backbone_name=args.model,
        num_classes=NUM_CLASSES,
        num_prototypes_per_class=args.num_prototypes_per_class,
        init_weights=True
    ).to(device)

    print(f"  Total prototypes: {model.num_prototypes}")
    print(f"  Prototypes per class: {args.num_prototypes_per_class}")
    print(f"  Prototype shape: {model.prototype_shape}")

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_clst = ClusterLoss(model.prototype_class_identity)
    criterion_sep = SeparationLoss(model.prototype_class_identity)

    # Training logs
    logs = {
        'loss': [],
        'ce_loss': [],
        'clst_loss': [],
        'sep_loss': [],
        'accuracy': []
    }

    # ==========================================
    # Training Loop
    # ==========================================

    print(f"\n{'='*80}")
    print("Starting Training...")
    print(f"{'='*80}")

    # Stage 1a: Warm-up (train only add-on layers and prototypes)
    print(f"\n--- Warm-up Phase ({args.warm_epochs} epochs) ---")
    warm_only(model)
    optimizer_warm = optim.Adam([
        {'params': model.add_on_layers.parameters(), 'lr': 3e-3},  # FIX: Use 3e-3 as in paper
        {'params': [model.prototype_vectors], 'lr': 3e-3},          # FIX: Use 3e-3 as in paper
        {'params': model.last_layer.parameters(), 'lr': args.lr}
    ])

    for epoch in range(args.warm_epochs):
        print(f"\nWarm Epoch {epoch+1}/{args.warm_epochs}")
        metrics = train_epoch(
            model, train_loader, optimizer_warm,
            criterion_ce, criterion_clst, criterion_sep,
            lambda_clst=args.lambda_clst,
            lambda_sep=args.lambda_sep,
            device=device
        )

        print(f"  Loss: {metrics['loss']:.4f} | CE: {metrics['ce_loss']:.4f} | "
              f"Clst: {metrics['clst_loss']:.4f} | Sep: {metrics['sep_loss']:.4f} | "
              f"Acc: {metrics['accuracy']:.2f}%")

        for key in logs:
            logs[key].append(metrics[key])

    # Stage 1b: Joint training (train all layers including last)
    print(f"\n--- Joint Training Phase ({args.joint_epochs} epochs) ---")
    joint(model)
    optimizer_joint = optim.Adam([
        {'params': model.conv_features.parameters(), 'lr': args.lr / 10},  # Keep 1e-5 for backbone
        {'params': model.add_on_layers.parameters(), 'lr': 3e-3},          # FIX: Use 3e-3 as in paper
        {'params': [model.prototype_vectors], 'lr': 3e-3},                 # FIX: Use 3e-3 as in paper
        {'params': model.last_layer.parameters(), 'lr': args.lr}
    ])

    for epoch in range(args.joint_epochs):
        print(f"\nJoint Epoch {epoch+1}/{args.joint_epochs}")
        metrics = train_epoch(
            model, train_loader, optimizer_joint,
            criterion_ce, criterion_clst, criterion_sep,
            lambda_clst=args.lambda_clst,
            lambda_sep=args.lambda_sep,
            device=device
        )

        print(f"  Loss: {metrics['loss']:.4f} | CE: {metrics['ce_loss']:.4f} | "
              f"Clst: {metrics['clst_loss']:.4f} | Sep: {metrics['sep_loss']:.4f} | "
              f"Acc: {metrics['accuracy']:.2f}%")

        for key in logs:
            logs[key].append(metrics[key])

        # Periodic prototype projection
        if (epoch + 1) % args.push_every == 0:
            push_prototypes(model, train_loader, device=device)

    # Stage 2: Final prototype projection
    print(f"\n--- Final Prototype Projection ---")
    push_prototypes(model, train_loader, device=device)

    # Stage 3: Last layer optimization
    print(f"\n--- Last Layer Optimization ({args.last_layer_epochs} epochs) ---")
    optimizer_last = optim.Adam([
        {'params': model.last_layer.parameters(), 'lr': args.lr}
    ])

    last_layer_optimization(
        model, train_loader, optimizer_last, criterion_ce,
        lambda_l1=args.lambda_l1,
        num_epochs=args.last_layer_epochs,
        device=device
    )

    # Save model
    output_prefix = f"protopnet_{args.model}_p{args.num_prototypes_per_class}"

    print(f"\n{'='*80}")
    print("Saving model...")
    print(f"{'='*80}")

    torch.save(model.state_dict(), f'{output_prefix}_model.pth')
    joblib.dump(model.cpu(), f'{output_prefix}_model.pkl')

    training_info = {
        'config': {
            'model': args.model,
            'num_classes': NUM_CLASSES,
            'num_prototypes_per_class': args.num_prototypes_per_class,
            'total_prototypes': model.num_prototypes,
            'prototype_shape': model.prototype_shape,
            'warm_epochs': args.warm_epochs,
            'joint_epochs': args.joint_epochs,
            'last_layer_epochs': args.last_layer_epochs,
            'lr': args.lr,
            'lambda_clst': args.lambda_clst,
            'lambda_sep': args.lambda_sep,
            'lambda_l1': args.lambda_l1
        },
        'logs': logs
    }
    joblib.dump(training_info, f'{output_prefix}_training_info.pkl')

    # Plot training logs
    plot_training_logs(logs, args.model, f'{output_prefix}_logs.png')

    print(f"\n{'='*80}")
    print("✓ Training Complete!")
    print(f"  Model: {output_prefix}_model.pkl")
    print(f"  Logs: {output_prefix}_logs.png")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
