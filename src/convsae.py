import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Model Definitions (Same as before)
# ==========================================

class ConvSAE(nn.Module):
    def __init__(self, in_channels=256, hidden_dim=4096, kernel_size=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.encoder = nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=padding)
        self.decoder = nn.Conv2d(hidden_dim, in_channels, kernel_size, padding=padding)
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.constant_(self.encoder_bias, 2.0)  # Positive bias to prevent encoder collapse (reduced from 5.0 with encoder normalization)
        # Initialize encoder with POSITIVE weights to prevent ReLU death
        # Kaiming can produce negative weights, causing all activations to die
        nn.init.uniform_(self.encoder.weight, a=0.0, b=0.1)

        # Initialize decoder with positive weights (compatible with non-negativity constraint)
        # Use uniform distribution in [0, 0.02] to ensure all weights start positive
        nn.init.uniform_(self.decoder.weight, a=0.0, b=0.02)

    def forward(self, x):
        pre_act = self.encoder(x) + self.encoder_bias.view(1, -1, 1, 1)
        feature_acts = F.relu(pre_act)
        reconstruction = self.decoder(feature_acts)
        return reconstruction, feature_acts

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """
        Normalize decoder weights with non-negativity constraint.
        Since feature_acts = ReLU(encoder(x)) are always non-negative,
        decoder weights should also be non-negative for meaningful reconstruction.
        """
        weight = self.decoder.weight
        # Apply ReLU to enforce non-negativity
        weight = torch.relu(weight)
        # L2 normalization
        norms = weight.norm(p=2, dim=(0, 2, 3), keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        self.decoder.weight.data = weight / norms

    @torch.no_grad()
    def normalize_encoder_weights(self):
        """
        Normalize encoder weights to prevent explosion.
        Applies L2 normalization per output channel.
        """
        weight = self.encoder.weight
        # L2 normalization per output channel
        norms = weight.norm(p=2, dim=(1, 2, 3), keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        self.encoder.weight.data = weight / norms

class LateralInhibitionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        self.kernel = k.view(1, 1, 3, 3)
        self.padding = 1

    def forward(self, feature_map):
        b, c, h, w = feature_map.shape
        device = feature_map.device
        filters = self.kernel.repeat(c, 1, 1, 1).to(device)
        neighbor_sum = F.conv2d(feature_map, filters, padding=self.padding, groups=c)
        loss = (feature_map * neighbor_sum).mean()
        return loss


class ClassDiversityLoss(nn.Module):
    """
    Encourages different classes to activate different features (class-discriminative learning).

    This loss promotes orthogonality between class-specific feature activations while
    allowing some shared features for common patterns.

    The loss works by:
    1. Computing average feature activation for each class
    2. Measuring similarity between different classes' feature usage
    3. Penalizing high similarity (encourages class-specific features)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, feature_acts, labels):
        """
        Args:
            feature_acts: [B, C, H, W] - sparse feature activations
            labels: [B] - class labels (0 to num_classes-1)

        Returns:
            diversity_loss: scalar - penalty for feature overlap between classes
                          (0 = classes use different features, 1 = classes use same features)
        """
        B, C, H, W = feature_acts.shape
        device = feature_acts.device

        # Compute per-class feature activation strength
        # class_features[c, i] = average activation of feature i for class c
        class_features = torch.zeros(self.num_classes, C, device=device)

        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                # Average activation across spatial dims and samples
                class_acts = feature_acts[mask].mean(dim=(0, 2, 3))  # [C]
                class_features[c] = class_acts

        # Normalize feature vectors (L2 normalization)
        class_features_norm = F.normalize(class_features, dim=1, p=2, eps=1e-8)  # [num_classes, C]

        # Compute pairwise cosine similarity between classes
        similarity_matrix = torch.mm(class_features_norm, class_features_norm.t())  # [num_classes, num_classes]

        # Penalize off-diagonal elements (high similarity between different classes)
        # We want different classes to have LOW similarity (orthogonal features)
        mask = torch.ones_like(similarity_matrix) - torch.eye(self.num_classes, device=device)
        diversity_loss = (similarity_matrix.abs() * mask).sum() / (self.num_classes * (self.num_classes - 1))

        return diversity_loss


class DualConvSAE(nn.Module):
    """
    Dual-pathway Convolutional Sparse Autoencoder that learns:
    1. Shared features: Global patterns common across all classes (unsupervised)
    2. Class-specific features: Discriminative patterns for classification (supervised)

    Architecture:
    - Shared pathway: encoder → decoder (reconstruction loss + L1 sparsity)
    - Class pathway: encoder → decoder + classifier (reconstruction + classification + L1 sparsity)
    - Combined reconstruction: shared_recon + class_recon
    """
    def __init__(self, in_channels=1, shared_dim=256, class_dim=256, num_classes=10, kernel_size=1):
        super().__init__()
        self.shared_dim = shared_dim
        self.class_dim = class_dim
        self.num_classes = num_classes

        padding = (kernel_size - 1) // 2

        # ===== Shared Pathway (Global Features) =====
        self.shared_encoder = nn.Conv2d(in_channels, shared_dim, kernel_size, padding=padding)
        self.shared_decoder = nn.Conv2d(shared_dim, in_channels, kernel_size, padding=padding)
        self.shared_encoder_bias = nn.Parameter(torch.zeros(shared_dim))

        # ===== Class-Specific Pathway (Discriminative Features) =====
        self.class_encoder = nn.Conv2d(in_channels, class_dim, kernel_size, padding=padding)
        self.class_decoder = nn.Conv2d(class_dim, in_channels, kernel_size, padding=padding)
        self.class_encoder_bias = nn.Parameter(torch.zeros(class_dim))

        # ===== Classifier Head (for class-specific features) =====
        # Uses global average pooling + linear layer
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(class_dim, num_classes)
        )

        # ===== Initialization =====
        # Initialize biases to positive values to prevent ReLU death
        nn.init.constant_(self.shared_encoder_bias, 2.0)
        nn.init.constant_(self.class_encoder_bias, 2.0)

        # Initialize encoder weights (positive to prevent ReLU death)
        nn.init.uniform_(self.shared_encoder.weight, a=0.0, b=0.1)
        nn.init.uniform_(self.class_encoder.weight, a=0.0, b=0.1)

        # Initialize decoder weights (positive for non-negativity constraint)
        nn.init.uniform_(self.shared_decoder.weight, a=0.0, b=0.02)
        nn.init.uniform_(self.class_decoder.weight, a=0.0, b=0.02)

        # Initialize classifier
        nn.init.kaiming_normal_(self.classifier[2].weight)
        nn.init.zeros_(self.classifier[2].bias)

    def forward(self, x, return_logits=False):
        """
        Forward pass through both pathways.

        Args:
            x: [B, C, H, W] - input activation maps
            return_logits: If True, returns classification logits (for training)

        Returns:
            reconstruction: [B, C, H, W] - combined reconstruction
            shared_features: [B, shared_dim, H, W] - shared feature activations
            class_features: [B, class_dim, H, W] - class-specific feature activations
            class_logits: [B, num_classes] - classification logits (only if return_logits=True)
        """
        # ===== Shared Pathway =====
        shared_pre_act = self.shared_encoder(x) + self.shared_encoder_bias.view(1, -1, 1, 1)
        shared_features = F.relu(shared_pre_act)
        shared_recon = self.shared_decoder(shared_features)

        # ===== Class-Specific Pathway =====
        class_pre_act = self.class_encoder(x) + self.class_encoder_bias.view(1, -1, 1, 1)
        class_features = F.relu(class_pre_act)
        class_recon = self.class_decoder(class_features)

        # ===== Combined Reconstruction =====
        reconstruction = shared_recon + class_recon

        # ===== Classification (optional) =====
        class_logits = None
        if return_logits:
            class_logits = self.classifier(class_features)

        return reconstruction, shared_features, class_features, class_logits

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """Normalize decoder weights with non-negativity constraint."""
        for decoder in [self.shared_decoder, self.class_decoder]:
            weight = decoder.weight
            # Apply ReLU to enforce non-negativity
            weight = torch.relu(weight)
            # L2 normalization
            norms = weight.norm(p=2, dim=(0, 2, 3), keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            decoder.weight.data = weight / norms

    @torch.no_grad()
    def normalize_encoder_weights(self):
        """Normalize encoder weights to prevent explosion."""
        for encoder in [self.shared_encoder, self.class_encoder]:
            weight = encoder.weight
            # L2 normalization per output channel
            norms = weight.norm(p=2, dim=(1, 2, 3), keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            encoder.weight.data = weight / norms
