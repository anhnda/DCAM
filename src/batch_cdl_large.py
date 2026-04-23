import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import joblib
from torchvision import datasets, transforms
from tqdm import tqdm
import importlib

# =========================================================
# PART 1: ROTO-CDL LIBRARY
# =========================================================

def rotated_atoms(phi, n_rotations):
    J, _, _, _ = phi.shape
    phis = []
    for k_rot in range(n_rotations):
        phis.append(torch.rot90(phi, k_rot, dims=(2, 3)))
    return torch.stack(phis, dim=1)

def reconstruct(phi, Z):
    N, J, K, H, W = Z.shape
    d = phi.shape[-1]
    C = J * K
    phi_rot = rotated_atoms(phi, K).reshape(C, 1, d, d)
    Z_flat = Z.view(N, C, H, W)
    out = F.conv2d(Z_flat, phi_rot, padding=d // 2, groups=C)
    return out.view(N, J, K, H, W).sum(dim=(1, 2)).unsqueeze(1)

def recenter_atoms(phi):
    J, _, d, d = phi.shape
    device = phi.device
    yy, xx = torch.meshgrid(torch.arange(d, device=device), torch.arange(d, device=device), indexing='ij')
    center = (d - 1) / 2.0
    phi_new = phi.clone()

    for j in range(J):
        atom = phi[j, 0]
        mass = atom.sum()
        if mass < 1e-6: continue

        com_y = (atom * yy).sum() / mass
        com_x = (atom * xx).sum() / mass

        shift_y = int(round(center - com_y.item()))
        shift_x = int(round(center - com_x.item()))

        if shift_x != 0 or shift_y != 0:
            shifted = torch.roll(atom, shifts=(shift_y, shift_x), dims=(0, 1))
            if shift_y > 0:
                shifted[:shift_y, :] = 0
            elif shift_y < 0:
                shifted[shift_y:, :] = 0
            if shift_x > 0:
                shifted[:, :shift_x] = 0
            elif shift_x < 0:
                shifted[:, shift_x:] = 0
            phi_new[j, 0] = shifted

    return phi_new

def solve_z_prox_adam(X_batch, phi, n_rotations, lr_z, sparsity, n_steps=30):
    N, _, n, _ = X_batch.shape
    J = phi.shape[0]
    device = X_batch.device

    # Random Init (Positive)
    Z = torch.randn(N, J, n_rotations, n, n, device=device) * 0.01
    Z = Z.abs()
    Z.requires_grad = True
    
    # Use SGD with Momentum for better convergence in short burst
    optimizer = torch.optim.SGD([Z], lr=lr_z, momentum=0.5) 

    for k in range(n_steps):
        optimizer.zero_grad()
        recon = reconstruct(phi, Z)
        
        loss = 0.5 * (recon - X_batch).pow(2).sum() / N
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            Z.data = F.relu(Z.data - sparsity)

    return Z.detach()

def roto_cdl_large_scale(
        X_full, n_atoms=64, n_rotations=4, d=9,
        batch_size=1500, n_epochs=20,
        lr_phi=0.05, lr_z=1.0, sparsity=0.002,
        device='cuda'
):
    print(f"--- Initialization [Device: {device}] ---")
    device = torch.device(device)

    try:
        X_full = X_full.to(device)
        print("Data loaded to VRAM.")
    except:
        print("Data kept on CPU.")

    dataset = TensorDataset(X_full)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # --- Init Dictionary (Gaussian Noise) ---
    # Crucial for Feature Maps: Start with noise, not data patches
    print("Initializing Dictionary with Gaussian Noise...")
    phi_init = torch.randn(n_atoms, 1, d, d, device=device)
    
    # Normalize immediately
    norm = phi_init.view(n_atoms, -1).norm(dim=1, keepdim=True).view(n_atoms, 1, 1, 1)
    phi_init = phi_init / (norm + 1e-8)
    phi_raw = torch.nn.Parameter(phi_init)

    # --- Optimizer Setup ---
    def get_optimizer(params, lr):
        return torch.optim.Adam(params, lr=lr)

    opt_phi = get_optimizer([phi_raw], lr_phi)
    scheduler = torch.optim.lr_scheduler.StepLR(opt_phi, step_size=10, gamma=0.5)

    def get_unit_phi(p):
        p_pos = F.relu(p)
        norm = p_pos.view(n_atoms, -1).norm(dim=1, keepdim=True).view(n_atoms, 1, 1, 1)
        return p_pos / (norm + 1e-8)

    print(f"--- Training: {len(X_full)} imgs, {n_atoms} atoms ---")

    target_sparsity = sparsity

    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_sparsity = 0
        steps = len(dataloader)

        # Sparsity Warmup
        if epoch < 1:
            current_sparsity = 0.0
        elif epoch < 5:
            current_sparsity = target_sparsity * (epoch / 5.0)
        else:
            current_sparsity = target_sparsity
        
        for batch_idx, (X_batch,) in enumerate(dataloader):
            X_batch = X_batch.to(device)

            # A. Sparse Coding
            curr_phi = get_unit_phi(phi_raw).detach()
            # Increased n_steps to 60 for better convergence
            Z_batch = solve_z_prox_adam(X_batch, curr_phi, n_rotations, lr_z, current_sparsity, n_steps=30)
            
            # B. Dictionary Update
            opt_phi.zero_grad()
            phi_unit = get_unit_phi(phi_raw)
            recon = reconstruct(phi_unit, Z_batch)
            loss = 0.5 * (recon - X_batch).pow(2).sum() / X_batch.shape[0]
            loss.backward()
            opt_phi.step()

            # C. Projection
            with torch.no_grad():
                phi_raw.data.clamp_(min=0.0)

            # Metrics
            non_zeros = (Z_batch > 1e-4).float().mean().item()
            epoch_loss += loss.item()
            epoch_sparsity += non_zeros

            if batch_idx % 20 == 0:
                print(f"\rEp {epoch} [{batch_idx}/{steps}] MSE: {loss.item():.5f} | Spar: {non_zeros:.4f} (Target: {current_sparsity:.4f})", end="")

        # D. Re-Centering
        if epoch % 5 == 0 and epoch > 0:
            print("\n[Re-centering Atoms...]")
            with torch.no_grad():
                phi_raw.data = recenter_atoms(phi_raw.data)
            opt_phi = get_optimizer([phi_raw], lr_phi) # Reset Momentum

        scheduler.step()
        avg_loss = epoch_loss / steps
        avg_spar = epoch_sparsity / steps
        print(f"\nEpoch {epoch} Done. Avg MSE: {avg_loss:.5f} | Avg Sparsity: {avg_spar:.4f}")

    return get_unit_phi(phi_raw).detach()

def vis_large_dict(phi):
    n = phi.shape[0]
    cols = 16
    rows = int(np.ceil(n / cols))
    phi_cpu = phi.detach().cpu()

    plt.figure(figsize=(cols, rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(phi_cpu[i, 0], cmap='viridis')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('large_dictionary.png', dpi=300)


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import joblib

# 1. Load your learned dictionary
# phi_learned = joblib.load('phi_learned_imagenette.pkl') 
# phi_learned = phi_learned.to('cuda')

def solve_sparse_code(X, phi, sparsity=0.005, lr=3.0, n_steps=200):
    """
    Finds Z such that X ≈ Phi * Z
    Minimizes: 0.5 * ||X - Rec(Z)||² + sparsity * ||Z||_1
    """
    device = X.device
    N, C, H, W = X.shape
    n_atoms, _, d, _ = phi.shape
    n_rotations = 4 # Match your training setting
    
    # 1. Initialize Z (Latent Code)
    # Shape: [Batch, Atoms, Rotations, Height, Width]
    Z = torch.zeros(N, n_atoms, n_rotations, H, W, device=device, requires_grad=True)
    
    # 2. Optimizer (Use SGD or Adam)
    # We use a high LR because Z needs to grow from 0 to signal magnitude quickly
    optimizer = torch.optim.SGD([Z], lr=lr, momentum=0.9)
    
    # 3. Optimization Loop (FISTA / Proximal Gradient)
    phi_unit = F.relu(phi) 
    phi_unit = phi_unit / (phi_unit.view(n_atoms, -1).norm(dim=1, keepdim=True, p=2).unsqueeze(-1) + 1e-8)
    
    loss_history = []
    
    for i in range(n_steps):
        optimizer.zero_grad()
        
        # Forward: Reconstruct X from current Z
        # (Reusing your reconstruct function)
        recon = reconstruct(phi_unit, Z)
        
        # Loss: MSE + L1 Penalty
        mse = 0.5 * (recon - X).pow(2).sum()
        l1 = sparsity * Z.abs().sum() # For monitoring only
        total_loss = mse # Gradient for MSE only
        
        total_loss.backward()
        optimizer.step()
        
        # PROXIMAL STEP: Soft Thresholding
        # This creates the zeros!
        with torch.no_grad():
            Z.data = F.relu(Z.data - sparsity * lr) # Shrinkage
            
        loss_history.append(mse.item())
        
    return Z.detach(), loss_history

# --- Helper to visualize Z ---
def visualize_Z(Z):
    # Z shape: [1, 64, 4, 13, 13]
    # We collapse rotations and atoms to see "Where are the activations?"
    
    # 1. Sum over rotations (we don't care about angle for density)
    Z_no_rot = Z.sum(dim=2) # [1, 64, 13, 13]
    
    # 2. Max projection over atoms (Heatmap of "Something is here")
    Z_heatmap = Z_no_rot.max(dim=1)[0].squeeze() # [13, 13]
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.title("Activation Heatmap (All Atoms)")
    plt.imshow(Z_heatmap.cpu(), cmap='hot')
    plt.colorbar()
    
    # 3. Show top 5 active atoms
    # Which atoms are used?
    atom_usage = Z_no_rot.view(64, -1).sum(dim=1)
    top_indices = torch.topk(atom_usage, 5).indices
    print(f"Top 5 Active Atoms: {top_indices.tolist()}")
    
    plt.show()

# --- USAGE ---
# Ensure X is scaled similarly to training (X * 10 if you used that)
# Z_star, history = solve_sparse_code(X_input * 10, phi_learned)
# visualize_Z(Z_star)