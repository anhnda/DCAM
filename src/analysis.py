import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional
import os

# Import GradCAM from the same package
from .gradcam import GradCAM

# ============================================================================
# L1 Coding (copied from cell '6e275b82' and '27dddb67')
# ============================================================================

def l1_code(A, D_atoms, lam=1e-2, iters=100, nonneg=False):
    # ... (function body from cell '6e275b82')
    D = np.stack([Di.reshape(-1) for Di in D_atoms], axis=1)  # (d^2, m)
    a = A.reshape(-1)                                         # (d^2,)
    G = D.T @ D                                               # (m, m)
    b = D.T @ a                                               # (m,)
    m = G.shape[0]
    w = np.zeros(m)
    for _ in range(iters):
        for j in range(m):
            zj = b[j] - (G[j] @ w - G[j, j] * w[j])           # partial residual
            if nonneg:
                w[j] = max(0.0, (zj - lam) / (G[j, j] + 1e-12))
            else:
                # soft-threshold
                v = np.sign(zj) * max(abs(zj) - lam, 0.0)
                w[j] = v / (G[j, j] + 1e-12)
    return w

def l1_code_nonneg(A, D_atoms, lam=1e-2, iters=100, tol=1e-6):
    # ... (function body from cell '27dddb67')
    # Build dictionary matrix D: (d^2, m)
    D = np.stack([Di.reshape(-1) for Di in D_atoms], axis=1)
    a = A.reshape(-1)  # (d^2,)
    
    # Precompute for efficiency
    DTD = D.T @ D  # (m, m) - Gram matrix
    DTa = D.T @ a  # (m,) - correlation
    
    # Lipschitz constant for gradient (largest eigenval of DTD)
    L = np.linalg.norm(DTD, ord=2) + 1e-8
    step_size = 1.0 / L
    
    # Initialize
    m = DTD.shape[0]
    w = np.zeros(m)
    w_old = w.copy()
    t = 1.0  # FISTA momentum parameter
    
    # FISTA with non-negative constraint
    for iter_num in range(iters):
        # Momentum extrapolation
        y = w + ((t - 1) / (t + 2)) * (w - w_old)
        
        # Gradient step: grad = DTD @ y - DTa
        grad = DTD @ y - DTa
        z = y - step_size * grad
        
        # Proximal operator: soft-thresholding + non-negative projection
        # prox(z) = max(0, sign(z) * max(|z| - step_size * lam, 0))
        w_new = np.maximum(0.0, z - step_size * lam)
        
        # Check convergence
        change = np.linalg.norm(w_new - w)
        if change < tol:
            w = w_new
            break
        
        # Update
        w_old = w
        w = w_new
        t += 1
    
    return w

# ============================================================================
# TopKActivationAnalyzer (copied from cell '3d6ad0c7')
# ============================================================================

class TopKActivationAnalyzer:
    """
    Analyze top-K most influential activation maps for a given input image.
    
    Args:
        model: Pre-trained model
        target_layer: Layer to extract activations from
        dictionary: Learned dictionary atoms [m, d, d]
        device: Computing device
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module, 
                 dictionary: torch.Tensor, device: torch.device):
        self.model = model
        self.target_layer = target_layer
        self.dictionary = dictionary.cpu().numpy()  # Convert to numpy for l1_code
        self.device = device
        self.gradcam = GradCAM(model, target_layer)
    
    def get_top_k_maps(self, image: torch.Tensor, k: int = 5, 
                       class_idx: Optional[int] = None) -> Tuple[List[torch.Tensor], List[int], torch.Tensor]:
        """
        Get top-K most influential activation maps for input image.
        
        Args:
            image: Input image tensor [1, 3, H, W]
            k: Number of top maps to return
            class_idx: Target class (None = predicted class)
            
        Returns:
            top_maps: List of k activation maps [H, W]
            top_indices: Channel indices of top maps
            weights: All channel weights from GradCAM
        """
        image = image.to(self.device)
        
        # Compute GradCAM weights
        weights, _, pred_class = self.gradcam.forward(image, class_idx=class_idx)
        
        # Get activations [C, H, W]
        activations = self.gradcam.activations.squeeze().cpu()
        
        # Find top-k channels by weight
        top_k_indices = torch.argsort(weights, descending=True)[:k].cpu().numpy()
        
        # Extract top-k activation maps
        top_maps = [activations[idx] for idx in top_k_indices]
        
        return top_maps, top_k_indices.tolist(), weights
    
    def analyze_sparse_codes(self, activation_maps: List[torch.Tensor], 
                            lam: float = 1e-2, iters: int = 100, 
                            nonneg: bool = True) -> List[np.ndarray]:
        """
        Compute sparse codes for activation maps using learned dictionary.
        
        Args:
            activation_maps: List of activation maps [H, W]
            lam: L1 regularization
            iters: Optimization iterations
            nonneg: Non-negativity constraint
            
        Returns:
            List of sparse code vectors
        """
        sparse_codes = []
        
        for A in activation_maps:
            A_np = A.numpy() if torch.is_tensor(A) else A
            w = l1_code(A_np, self.dictionary, lam=lam, iters=iters, nonneg=nonneg)
            sparse_codes.append(w)
        
        return sparse_codes
    
    def visualize_analysis(self, image: torch.Tensor, k: int = 5, 
                          class_idx: Optional[int] = None, 
                          lam: float = 1e-2) -> dict:
        """
        Complete analysis: get top-k maps and their sparse codes.
        
        Args:
            image: Input image [1, 3, H, W]
            k: Number of top maps
            class_idx: Target class
            lam: L1 regularization
            
        Returns:
            Dictionary with analysis results
        """
        # Get top-k activation maps
        top_maps, top_indices, all_weights = self.get_top_k_maps(image, k, class_idx)
        
        # Compute sparse codes
        sparse_codes = self.analyze_sparse_codes(top_maps, lam=lam)
        
        # Compile results
        results = {
            'top_maps': top_maps,
            'top_indices': top_indices,
            'all_weights': all_weights.cpu().numpy(),
            'sparse_codes': sparse_codes,
            'top_weights': all_weights[top_indices].cpu().numpy(),
            'input_image': image  # Store input image for visualization
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Top-{k} Most Influential Activation Maps Analysis")
        print(f"{'='*70}")
        
        for i, (idx, w, code) in enumerate(zip(top_indices, results['top_weights'], sparse_codes)):
            print(f"\nRank {i+1}: Channel {idx}")
            print(f"  GradCAM Weight: {w:.6f}")
            print(f"  Sparse Code Stats:")
            print(f"    - Non-zero atoms: {np.sum(np.abs(code) > 1e-6)}/{len(code)}")
            print(f"    - L1 norm: {np.sum(np.abs(code)):.6f}")
            print(f"    - Max coefficient: {np.max(np.abs(code)):.6f}")
            print(f"    - Top 5 atom indices: {np.argsort(np.abs(code))[-5:][::-1].tolist()}")
            print(f"    - Top 5 coefficients: {code[np.argsort(np.abs(code))[-5:][::-1]]}")
        
        print(f"\n{'='*70}\n")
        
        return results
    
    def visualize_channel_decomposition(self, results: dict, channel_rank: int = 0, 
                                       figsize: Tuple[int, int] = (15, 10),
                                       save_path: Optional[str] = None):
        """
        Visualize a single channel decomposition in 3x2 grid:
        - Top-left: Original image downscaled
        - Top-right: Activation map (grayscale)
        - Bottom 4: Top-4 dictionary atoms
        
        Args:
            results: Results from visualize_analysis()
            channel_rank: Which top channel to visualize (0-indexed)
            figsize: Figure size
            save_path: Path to save figure (None = display only)
        """
        if channel_rank >= len(results['top_maps']):
            raise ValueError(f"channel_rank {channel_rank} out of range (max: {len(results['top_maps'])-1})")
        
        # Extract data
        input_image = results['input_image']
        activation_map = results['top_maps'][channel_rank]
        sparse_code = results['sparse_codes'][channel_rank]
        channel_idx = results['top_indices'][channel_rank]
        gradcam_weight = results['top_weights'][channel_rank]
        
        # Get top-4 atoms
        top_atom_indices = np.argsort(np.abs(sparse_code))[-4:][::-1]
        top_atom_weights = sparse_code[top_atom_indices]
        
        # Convert activation map to numpy
        act_map_np = activation_map.numpy() if torch.is_tensor(activation_map) else activation_map
        target_size = act_map_np.shape  # (H, W)
        
        # Denormalize and prepare input image
        input_np = input_image.squeeze(0).cpu().numpy()  # [3, H, W]
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        input_denorm = input_np * std + mean
        input_denorm = np.clip(input_denorm, 0, 1)
        input_denorm = np.transpose(input_denorm, (1, 2, 0))  # [H, W, 3]
        
        # Downscale input image to activation map size
        input_pil = Image.fromarray((input_denorm * 255).astype(np.uint8))
        input_downscaled = input_pil.resize(target_size[::-1], Image.BILINEAR)  # (W, H)
        input_downscaled = np.array(input_downscaled) / 255.0
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(f'Channel {channel_idx} Decomposition (Rank {channel_rank + 1})\n'
                    f'GradCAM Weight: {gradcam_weight:.4f}', 
                    fontsize=16, fontweight='bold')
        
        # Top-left: Downscaled input image
        axes[0, 0].imshow(input_downscaled)
        axes[0, 0].set_title('Original Image\n(Downscaled)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Top-right: Activation map (grayscale)
        im1 = axes[0, 1].imshow(act_map_np, cmap='gray')
        axes[0, 1].set_title(f'Activation Map\n(Channel {channel_idx})', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        # plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Bottom 4: Top-4 dictionary atoms
        dict_np = self.dictionary
        for i, (atom_idx, weight) in enumerate(zip(top_atom_indices, top_atom_weights)):
            row = (i // 2) + 1
            col = i % 2
            
            atom = dict_np[atom_idx]
            im = axes[row, col].imshow(atom, cmap='gray')
            axes[row, col].set_title(f'Atom {atom_idx}\nWeight: {weight:.4f}', 
                                     fontsize=11, fontweight='bold')
            axes[row, col].axis('off')
            # plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.show()
    
    def visualize_all_top_channels(self, results: dict, 
                                   save_dir: Optional[str] = None):
        """
        Visualize all top-K channels.
        
        Args:
            results: Results from visualize_analysis()
            save_dir: Directory to save figures (None = display only)
        """
        import os
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for i in range(len(results['top_maps'])):
            save_path = None
            if save_dir:
                channel_idx = results['top_indices'][i]
                save_path = os.path.join(save_dir, f'channel_{channel_idx}_rank_{i+1}.png')
            self.visualize_channel_decomposition(results, channel_rank=i, save_path=save_path)

   