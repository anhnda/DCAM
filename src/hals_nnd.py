import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional
from tqdm import tqdm

# Params to be sparse but strange pattern?
# lamD: float = 1e-2,
# lamW: float = 5e-2,
# gamma: float = 5e-2,
def hals_nnd_correct(
    X_list: List[torch.Tensor],
    m: int,
    nsteps: int = 200,
    lamD: float = 5e-4,
    lamW: float = 5e-2,
    gamma: float = 5e-4,
    eta: float = 0.01,
    seed: int = 0,
    batch_size: int = 1000,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    HALS-NND: Hierarchical Alternating Least Squares with Non-Negative Dictionary learning.
    
    Optimizes: min_{D,W} sum_i ||A_i - sum_k w_ik D_k||^2 + lamW*||W||^2 + lamD*||D||_1 + gamma*Var(D)
    
    Algorithm:
        (A) Update W: Block coordinate descent with non-negativity
            w_ik = max(0, (<A_i, D_k> - sum_{l!=k} w_il <D_l, D_k>) / (||D_k||^2 + lamW))
        
        (B) Update D: Proximal gradient descent with spatial regularization
            1. Gradient step: Y_k = D_k - eta * grad_D_k
            2. Spatially-varying soft-threshold based on distance from center of mass
            3. Non-negativity projection
            4. Frobenius norm projection to unit ball
    
    Args:
        X_list: List of n tensors [d, d] - input activation maps
        m: Number of dictionary atoms
        nsteps: Number of training epochs
        lamD: L1 regularization on dictionary
        lamW: L2 regularization on weights
        gamma: Spatial variance penalty coefficient
        eta: Learning rate for proximal gradient
        seed: Random seed for reproducibility
        batch_size: Mini-batch size for SGD
        device: Computing device (GPU/CPU)
        
    Returns:
        D: Dictionary atoms [m, d, d]
        W: Sparse weights [n, m]
        loss_dict: Dictionary containing:
            - 'total': Total loss per epoch
            - 'reconstruction': Reconstruction loss per epoch
            - 'regularization': Regularization loss per epoch (W + D + spatial)
    """
    # ==================== Setup ====================
    if device is None:
        device = X_list[0].device if torch.is_tensor(X_list[0]) else \
                 torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(seed)
    print("HALS v2.2")
    n = len(X_list)
    d = X_list[0].shape[0]
    
    # Stack all input tensors
    X_stack = torch.stack([
        x.to(device) if torch.is_tensor(x) else torch.tensor(x, device=device, dtype=torch.float32)
        for x in X_list
    ])  # [n, d, d]
    
    # ==================== Initialize D ====================
    D = torch.zeros(m, d, d, device=device, dtype=torch.float32)
    
    # Use random samples from data for initialization
    indices = torch.randperm(n, device=device)[:min(m, n)]
    for i, idx in enumerate(indices):
        D[i] = X_stack[idx].clone()
    
    # If m > n, fill remaining with small random values
    if m > n:
        D[n:] = torch.randn(m - n, d, d, device=device) * 0.1
    
    # Normalize each atom to unit Frobenius norm
    D_flat = D.view(m, -1)
    norms = torch.linalg.norm(D_flat, dim=1, keepdim=True).clamp(min=1e-8)
    D = (D_flat / norms).view(m, d, d)
    
    # ==================== Initialize W ====================
    W = torch.rand(n, m, device=device, dtype=torch.float32) * 0.1
    
    # ==================== Spatial coordinates ====================
    ii, jj = torch.meshgrid(
        torch.arange(d, device=device, dtype=torch.float32),
        torch.arange(d, device=device, dtype=torch.float32),
        indexing='ij'
    )
    R_coords = torch.stack([ii, jj], dim=-1)  # [d, d, 2]
    def enforce_W_sparsity(W, density):
        flat = W.view(-1)
        k = int(density * flat.numel())
        if k < 1:
            return torch.zeros_like(W)
        thresh = torch.topk(flat, k).values[-1]
        return torch.where(W >= thresh, W, torch.zeros_like(W))
    # ==================== Loss computation ====================
    def compute_loss_components(X_batch: torch.Tensor, W_batch: torch.Tensor) -> Tuple[float, float, float]:
        """Compute loss components for a batch
        
        Returns:
            recon_loss: Reconstruction error
            reg_loss: Total regularization (W + D + spatial)
            total_loss: Sum of both
        """
        batch_n = X_batch.shape[0]
        
        # Reconstruction error
        recon = torch.einsum('ik,kde->ide', W_batch, D)
        recon_error = torch.sum((X_batch - recon) ** 2).item()
        
        # Weight regularization
        w_penalty = lamW * torch.sum(W_batch ** 2).item()
        
        # Dictionary L1 penalty
        d_l1_penalty = lamD * torch.sum(torch.abs(D)).item()
        
        # Spatial variance penalty (vectorized)
        atom_abs = torch.abs(D)  # [m, d, d]
        weight_sum = atom_abs.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)  # [m, 1, 1]
        
        # Center of mass for each atom
        mu = (atom_abs.unsqueeze(-1) * R_coords).sum(dim=(1, 2)) / weight_sum.squeeze(-1)  # [m, 2]
        
        # Distance squared from center
        dist2 = ((R_coords.unsqueeze(0) - mu.view(m, 1, 1, 2)) ** 2).sum(dim=-1)  # [m, d, d]
        
        spatial_penalty = gamma * (atom_abs * dist2).sum().item()
        
        # Regularization = W penalty + D penalty + spatial penalty
        reg_loss = w_penalty + d_l1_penalty + spatial_penalty
        total_loss = recon_error + reg_loss
        
        # Average per sample
        return recon_error / batch_n, reg_loss / batch_n, total_loss / batch_n
    
    # ==================== Training loop ====================
    loss_history = {
        'total': [],
        'reconstruction': [],
        'regularization': []
    }
    best_loss = float('inf')
    best_D = D.clone()
    best_W = W.clone()
    
    print(f"{'='*70}")
    print(f"Starting HALS-NND optimization: v0")
    print(f"  Data: {n} samples of size {d}x{d}")
    print(f"  Dictionary: {m} atoms")
    print(f"  Epochs: {nsteps}, Batch size: {batch_size}")
    print(f"  λ_D={lamD}, λ_W={lamW}, γ={gamma}, η={eta}")
    print(f"{'='*70}")
    
    dataset = TensorDataset(X_stack, torch.arange(n, device=device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    pbar = tqdm(range(nsteps), desc="Training")
    for epoch in pbar:
        epoch_recon_losses = []
        epoch_reg_losses = []
        epoch_total_losses = []
        
        # ==============================
        # (A) UPDATE W for ALL batches
        # ==============================
                    # Precompute G and B for this batch
        D_flat = D.view(m, -1)
        G = D_flat @ D_flat.t()            # [m, m]
        diagG = torch.diagonal(G)
        denom = diagG + lamW    
        for X_batch, indices_batch in loader:
            curr_batch_size = X_batch.shape[0]

           # [m]

            X_flat = X_batch.view(curr_batch_size, -1)
            B = X_flat @ D_flat.t()            # [batch, m]
            
            W_batch = W[indices_batch]

            # TRUE HALS Gauss–Seidel update
            for k in range(m):
                # Compute reconstruction of atom k using other atoms
                residual_k = B[:, k] - (W_batch @ G[:, k] - W_batch[:, k] * G[k, k])

                W_batch[:, k] = torch.clamp(residual_k / denom[k], min=0.0)

            #w_thresh = 1e-3
            #W_batch[W_batch < w_thresh] = 0.0
            W_batch = enforce_W_sparsity(W_batch, density=0.08)
            W[indices_batch] = W_batch
            #W[indices_batch] = W_batch
            
            # (Optional) track batch loss for W updates only
            recon_loss, reg_loss, total_loss = compute_loss_components(X_batch, W_batch)
            epoch_recon_losses.append(recon_loss)
            epoch_reg_losses.append(reg_loss)
            epoch_total_losses.append(total_loss)

        # ==============================
        # (B) UPDATE D once per epoch
        # ==============================
        # FULL reconstruction residuals, using *all* updated W
        recon_full = torch.einsum('ik,kde->ide', W, D)
        residuals = X_stack - recon_full

        # FULL gradient over dataset
        grad_D = -torch.einsum('ik,ide->kde', W, residuals) / n

        # Gradient descent step
        Y = D - eta * grad_D

        # Spatial soft threshold
        #atom_abs = torch.abs(D)
        atom_abs = torch.abs(Y)    # better approximation for prox step

        weight_sum = atom_abs.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        mu = (atom_abs.unsqueeze(-1) * R_coords).sum(dim=(1, 2)) / weight_sum.squeeze(-1)
        dist2 = ((R_coords.unsqueeze(0) - mu.view(m, 1, 1, 2)) ** 2).sum(dim=-1)

        threshold = eta * (lamD + gamma * dist2)
        Z = torch.sign(Y) * torch.clamp(torch.abs(Y) - threshold, min=0.0)

        # Non-negativity projection
        Z = torch.clamp(Z, min=0.0)

        # Frobenius norm projection
        Z_flat = Z.view(m, -1)
        dead = (Z_flat.abs().sum(dim=1) < 1e-6)
        if dead.any():
            Z_flat[dead] = 0.1 * torch.randn_like(Z_flat[dead])
            W[:, dead] = 0.1 * torch.rand(W.shape[0], dead.sum(), device=device)

        norms = torch.linalg.norm(Z_flat, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)   # avoid div0 but don’t force nonzero norm
        D = (Z_flat / norms).view(m, d, d)

        # ==============================
        # (C) Compute epoch loss
        # ==============================
        avg_recon_loss = sum(epoch_recon_losses) / len(epoch_recon_losses)
        avg_reg_loss = sum(epoch_reg_losses) / len(epoch_reg_losses)
        avg_total_loss = sum(epoch_total_losses) / len(epoch_total_losses)
        
        loss_history['reconstruction'].append(avg_recon_loss)
        loss_history['regularization'].append(avg_reg_loss)
        loss_history['total'].append(avg_total_loss)

        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            best_D = D.clone()
            best_W = W.clone()
            status = "✓ new best"
        else:
            status = "→"

        delta = avg_total_loss - loss_history['total'][-2] if len(loss_history['total']) > 1 else 0

        pbar.set_postfix({
            'loss': f'{avg_total_loss:.6f}',
            'recon': f'{avg_recon_loss:.6f}',
            'reg': f'{avg_reg_loss:.6f}',
            'Δ': f'{delta:.2e}',
            'status': status
        })
        if epoch % 1 == 0:
            with torch.no_grad():
                print(f"[epoch {epoch}] ||W||_F={W.norm().item():.3f}, "
                    f"||grad_D||_F={grad_D.norm().item():.3f}")
        
    # ==================== Final report ====================
    print(f"\n{'='*70}")
    print(f"Optimization completed:")
    print(f"  Final total loss:  {avg_total_loss:.6f}")
    print(f"  Final recon loss:  {avg_recon_loss:.6f}")
    print(f"  Final reg loss:    {avg_reg_loss:.6f}")
    print(f"  Best total loss:   {best_loss:.6f}")
    print(f"  W sparsity:        {(best_W > 1e-4).float().mean().item()*100:.1f}% non-zero")
    print(f"  D sparsity:        {(best_D > 1e-3).float().mean().item()*100:.1f}% non-zero")
    print(f"{'='*70}")
    
    return best_D, best_W, loss_history

