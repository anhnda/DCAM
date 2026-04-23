"""
GradCAM implementation for visualizing CNN activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GradCAM:
    """
    GradCAM implementation for visualizing CNN activations.
    
    Args:
        model: Pre-trained PyTorch model
        target_layer: Layer to compute GradCAM on
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save activation maps"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients"""
        self.gradients = grad_output[0].detach()
    
    def _compute_weights(self) -> torch.Tensor:
        """
        Compute channel weights using Global Average Pooling of gradients.
        Returns: weights tensor [n_channels]
        """
        # Global Average Pooling: [batch, channels, h, w] -> [batch, channels]
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=False)
        return weights.squeeze()
    
    def forward(self, x: torch.Tensor, class_idx: Optional[int] = None, 
                verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compute GradCAM for input image.
        
        Args:
            x: Input tensor [1, 3, H, W]
            class_idx: Target class index (None = use predicted class)
            verbose: Print debug information
            
        Returns:
            weights: Channel weights after ReLU
            cam: GradCAM heatmap
            pred_class: Predicted/target class index
        """
        self.model.eval()
        
        # Forward pass
        logits = self.model(x)
        
        # Use predicted class if not specified
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Compute weights with ReLU
        weights = F.relu(self._compute_weights())
        
        # Compute CAM
        activations = self.activations.squeeze()  # [channels, h, w]
        cam = torch.einsum('c,chw->hw', weights, activations)
        
        if verbose:
            print(f"Activations - min: {activations.min():.4f}, max: {activations.max():.4f}")
        
        # Normalize CAM
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        
        return weights, cam, class_idx