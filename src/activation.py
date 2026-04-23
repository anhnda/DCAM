import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

# Import GradCAM from the same package
from .gradcam import GradCAM

class ActivationMapCollector:
    """
    Collect activation maps from top contributing channels using GradCAM.
    ... (docstring kept short for brevity)
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module, 
                 device: torch.device, top_k_percent: float = 0.9):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.top_k_percent = top_k_percent
        self.gradcam = GradCAM(model, target_layer)
    
    def collect_maps(self, data_loader: DataLoader,
                     use_true_labels: bool = True, output_tensor=True, device=None,
                     return_labels: bool = False) -> List[torch.Tensor]:
        """
        Collect activation maps from dataset.

        Args:
            return_labels: If True, returns (maps, labels) tuple where labels indicate
                          which class each activation map came from
        """
        maps_list = []
        labels_list = []

        for image_tensor, label in tqdm(data_loader, desc="Collecting activation maps"):
            image_tensor = image_tensor.to(self.device)
            class_idx = label.item() if use_true_labels else None

            # Compute GradCAM weights
            weights, _, _ = self.gradcam.forward(image_tensor, class_idx=class_idx)

            # Get activations (stored by hook)
            activations = self.gradcam.activations.squeeze().cpu()  # [C, H, W]

            # Find top-k% channels by cumulative weight
            top_channel_indices = self._get_top_k_channels(weights.cpu().numpy())

            # Collect maps from top channels
            for idx in top_channel_indices:
                maps_list.append(activations[idx])
                if return_labels:
                    labels_list.append(class_idx)

        if output_tensor:
            maps_list = torch.stack(maps_list).to(device)
            maps_list = maps_list.unsqueeze(1)
            if return_labels:
                labels_list = torch.tensor(labels_list, dtype=torch.long, device=device)

        if return_labels:
            return maps_list, labels_list
        return maps_list
    
    def _get_top_k_channels(self, weights: np.ndarray) -> np.ndarray:
        """
        Find channel indices that contribute to top k% of cumulative weight.
        ... (docstring kept short for brevity)
        """
        sorted_indices = np.argsort(weights)[::-1]
        sorted_weights = weights[sorted_indices]
        
        cumsum = np.cumsum(sorted_weights)
        total_weight = cumsum[-1] if len(cumsum) > 0 else 0
        
        if total_weight <= 0:
            return np.array([], dtype=int)
        
        threshold = total_weight * self.top_k_percent
        cutoff_idx = np.where(cumsum >= threshold)[0]
        
        if len(cutoff_idx) > 0:
            return sorted_indices[:cutoff_idx[0] + 1]
        else:
            return sorted_indices