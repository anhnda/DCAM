import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader



# ---------------------------------------------------------
# 1. Core Helper Functions (Keep these same as before)
# ---------------------------------------------------------

def rotated_atoms(phi, n_rotations):
    J, _, _, _ = phi.shape
    phis = []
    for j in range(J):
        row = []
        for k_rot in range(n_rotations):
            row.append(torch.rot90(phi[j], k_rot, dims=(1, 2)))
        phis.append(torch.stack(row, dim=0))
    return torch.stack(phis, dim=0)


def reconstruct(phi, Z):
    N, J, K, H, W = Z.shape
    phi_rot = rotated_atoms(phi, K)
    C = J * K
    d = phi.shape[-1]
    phi_flat = phi_rot.view(C, 1, d, d)
    Z_flat = Z.view(N, C, H, W)
    out = F.conv2d(Z_flat, phi_flat, padding=d // 2, groups=C)
    return out.sum(dim=1, keepdim=True)


def recenter_atoms(phi):
    J, _, d, _ = phi.shape
    device = phi.device
    mid = d // 2
    grid_y, grid_x = torch.meshgrid(torch.arange(d, device=device),
                                    torch.arange(d, device=device), indexing='ij')
    phi_new = phi.clone()
    for j in range(J):
        atom = phi[j, 0]
        mass = atom.sum()
        if mass < 1e-5: continue
        cy = (atom * grid_y).sum() / mass
        cx = (atom * grid_x).sum() / mass
        shift_y = int(mid - torch.round(cy).item())
        shift_x = int(mid - torch.round(cx).item())
        if shift_y != 0 or shift_x != 0:
            atom_shifted = torch.roll(atom, shifts=(shift_y, shift_x), dims=(0, 1))
            mask = torch.ones_like(atom)
            if shift_y > 0: mask[:shift_y, :] = 0
            if shift_y < 0: mask[shift_y:, :] = 0
            if shift_x > 0: mask[:, :shift_x] = 0
            if shift_x < 0: mask[:, shift_x:] = 0
            phi_new[j, 0] = atom_shifted * mask
    return phi_new


def compactness_penalty(phi):
    J, _, d, _ = phi.shape
    device = phi.device
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, d, device=device),
        torch.linspace(-1, 1, d, device=device),
        indexing="ij"
    )
    r2 = yy ** 2 + xx ** 2
    return (r2.unsqueeze(0).unsqueeze(0) * (phi ** 2)).mean()


# ---------------------------------------------------------
# 2. The Transient Z Solver (Inference Step)
# ---------------------------------------------------------
def solve_z_for_batch(X_batch, phi, n_rotations, lr_z=0.1, sparsity=0.1, n_steps=20):
    """
    Solves for Z given a fixed Phi and X_batch.
    Returns the optimal Z for this batch.
    """
    N, _, n, _ = X_batch.shape
    J = phi.shape[0]
    device = X_batch.device

    # Initialize Z with zeros (standard for batch processing)
    Z = torch.zeros(N, J, n_rotations, n, n, device=device, requires_grad=True)

    # Normalize atoms for the forward pass (crucial for stability)
    with torch.no_grad():
        norm = phi.view(J, -1).norm(dim=1, keepdim=True).view(J, 1, 1, 1)
        phi_unit = phi / (norm + 1e-8)

    # ISTA Loop
    for _ in range(n_steps):
        recon = reconstruct(phi_unit, Z)
        loss = 0.5 * (recon - X_batch).pow(2).sum()
        grad_Z = torch.autograd.grad(loss, Z)[0]

        # Gradient Descent
        Z_temp = Z - lr_z * grad_Z

        # Soft Thresholding (ReLU for Non-Negative SC)
        thresh = lr_z * sparsity
        Z = F.relu(Z_temp - thresh)

    return Z.detach()  # Detach! We don't backprop through the ISTA unrolling here.


# ---------------------------------------------------------
# 3. Batch Training Loop
# ---------------------------------------------------------
def roto_cdl_batch_train(
        X_full,
        n_atoms=5,
        n_rotations=4,
        d=7,
        batch_size=64,
        n_epochs=20,  # Fewer epochs needed because we have many updates per epoch
        lr_phi=0.02,
        lr_z=0.1,
        sparsity=0.2,
        alpha=0.5,  # Compactness
        device=None
):
    if device is None: device = X_full.device
    device = torch.device(device)
    X_full = X_full.to(device).float()

    # Normalize Data
    X_full = (X_full - X_full.min()) / (X_full.max() - X_full.min())

    # Create DataLoader
    dataset = TensorDataset(X_full)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Initialize Phi (Smart Init) ---
    # Take a random subset to find bright patches
    subset_size = min(1000, X_full.shape[0])
    patches = F.unfold(X_full[:subset_size], kernel_size=d).permute(0, 2, 1).reshape(-1, d, d)
    energies = patches.sum(dim=(1, 2))
    _, top_idx = torch.topk(energies, k=n_atoms * 20)
    phi_init = patches[top_idx[:n_atoms]].unsqueeze(1).to(device)
    phi_raw = torch.nn.Parameter(phi_init)

    opt_phi = torch.optim.Adam([phi_raw], lr=lr_phi)

    # Helper for unit norm
    def get_unit_phi(p):
        p_pos = F.relu(p)
        norm = p_pos.view(n_atoms, -1).norm(dim=1, keepdim=True).view(n_atoms, 1, 1, 1)
        return p_pos / (norm + 1e-8)

    print(f"Starting Batch Training on {len(X_full)} images...")

    global_step = 0

    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_sparsity = 0

        for batch_idx, (X_batch,) in enumerate(dataloader):
            X_batch = X_batch.to(device)

            # --- Step 1: Solve Z (Inference) ---
            # We treat Phi as constant here
            Z_batch = solve_z_for_batch(X_batch, get_unit_phi(phi_raw), n_rotations, lr_z, sparsity)

            # --- Step 2: Update Phi (Learning) ---
            opt_phi.zero_grad()
            phi_unit = get_unit_phi(phi_raw)

            # We must re-compute reconstruction to get the gradient flow into Phi
            # (Z_batch is treated as a constant constant input now)
            recon = reconstruct(phi_unit, Z_batch)

            mse = 0.5 * (recon - X_batch).pow(2).sum() / X_batch.shape[0]
            reg = alpha * compactness_penalty(phi_unit)

            loss = mse + reg
            loss.backward()
            opt_phi.step()

            # --- Step 3: Atomic Constraints (Projected Gradient) ---
            with torch.no_grad():
                phi_raw.data.clamp_(min=0.0)

            # Stats
            epoch_loss += mse.item()
            epoch_sparsity += (Z_batch > 1e-3).float().mean().item()
            global_step += 1

        # --- End of Epoch Cleanups ---
        with torch.no_grad():
            # Recenter atoms (Only do this once per epoch to avoid jitter)
            if epoch % 2 == 0:
                phi_raw.data = recenter_atoms(phi_raw.data)

            # Weak pixel pruning
            if epoch > 5:
                p_norm = get_unit_phi(phi_raw)
                phi_raw.data *= (p_norm > 0.05).float()

        # Print Average Epoch Stats
        avg_loss = epoch_loss / len(dataloader)
        avg_spar = epoch_sparsity / len(dataloader)
        print(f"Epoch {epoch}: Avg MSE={avg_loss:.4f} | Avg Sparsity={avg_spar:.4f}")

    return get_unit_phi(phi_raw).detach()


# ---------------------------------------------------------
# 4. Usage Example
# ---------------------------------------------------------
if __name__ == "__main__":
    from gen_toy_data import generate_toy_dataset

    # Simulate Large Dataset
    N_large = 20000  # Try 2000 or 50000
    n, d = 13, 7

    print("Generating Data...")
    X, phi_gt = generate_toy_dataset(N_large, n, d)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # 1. Train Dictionary
    phi_learned = roto_cdl_batch_train(X, n_atoms=5, batch_size=64, n_epochs=15, device=device)


    # 2. Visualize Dictionary
    def visualize_dictionary(phi):
        n = phi.shape[0]
        plt.figure(figsize=(n * 2, 2))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(phi[i, 0].cpu(), cmap='viridis')
            plt.axis('off')
            plt.title(f"Atom {i}")
        plt.show()


    print("Learned Atoms:")
    visualize_dictionary(phi_learned)
    print("Ground Truth Atoms:")
    visualize_dictionary(phi_gt)
    # 3. How to get Z for specific images later?
    print("Running Inference on first 5 images...")
    X_sample = X[:5].to(device)
    Z_sample = solve_z_for_batch(X_sample, phi_learned, n_rotations=4, sparsity=0.2)

    print("Z Sample Shape:", Z_sample.shape)  # (5, 5, 4, 32, 32)