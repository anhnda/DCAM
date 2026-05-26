"""
DCAM-Untied: Diagnostic variant of DCAM with a FREE, unconstrained decoder.
===========================================================================

Purpose
-------
Isolate the question: "Is the tied-pinv decoder the binding bottleneck in DCAM,
or is the simplex-constrained Pi encoder the binding bottleneck?"

The original DCAM ties decoder to encoder via the Moore-Penrose pseudo-inverse:

    Pi_dagger = Pi^T (Pi Pi^T + ridge I)^{-1}        # tied to Pi
    A_hat     = Pi_dagger z

This script REPLACES that with an UNTIED, UNCONSTRAINED decoder:

    A_hat     = W z                                  # W in R^{C x D}, free

Everything else is byte-identical to run_dcam_full.py:
- Pi encoder: ReLU(Pi A) with the same simplex-projected Pi in R^{D x C}.
- TopK over concepts: same.
- Activation cache: same key format, so the cached (A, M_tau, L_GC) tuples
  produced by run_dcam_full.py are reused. No re-extraction.
- Reconstruction loss: same masked MSE on Grad-CAM-selected channels.
- Anchor + L1: kept around but DEFAULT TO ZERO. The diagnostic question is
  "can ANY decoder do well given the encoder?", so regularizers are off.

What this rules in / out
------------------------
* If untied DCAM gets recon and accuracy close to CSAE's:
    -> The tied pinv is the binding constraint in the original DCAM.
    -> Paper needs to drop the "only Pi is learned" claim, or find a richer
       tied parametrization (e.g. A_hat = (Pi^T diag(s)) z).
* If untied DCAM also fails to recover accuracy:
    -> The simplex-constrained encoder is the binding constraint.
    -> The soft-membership story itself doesn't fit ResNet50 layer3.

Diagnostics added (that run_dcam_full.py was missing)
-----------------------------------------------------
* Per-batch RELATIVE reconstruction error  ||A_hat - A||_1 / ||A||_1 .
  This is the metric that mattered in the eval: training-time MSE looked
  small but rel_err was > 1, which means the model lost direction info.
* Concept utilization: fraction of D concepts that are EVER activated across
  an epoch. This catches dead concepts that K/D-floored cell-level sparsity
  cannot.
* sigma_min of BOTH Pi (encoder conditioning) and W (decoder conditioning).

Compatibility with check_drop_dcam.py
-------------------------------------
The result .pkl carries an extra 'W' tensor and 'decoder' = 'untied' in
config. check_drop_dcam.py needs a small patch (handled separately) to look
for W and route through the untied forward.

Usage
-----
    # Drop-in: trains 5 epochs at D=512, K=32, no regularizers.
    python run_dcam_untied.py

    # Match a previous DCAM (tied) run for direct comparison:
    python run_dcam_untied.py --model resnet50 --D 512 --top_k 32 \\
        --lambda_anchor 0.0 --lambda_l1 0.0 --epochs 5
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
from collections import defaultdict
import random
import argparse

sys.path.append('.')
# Reuse the heavy pieces from the tied DCAM script so we share the activation
# cache and stay byte-identical on the data side.
from misc.run_dcam_full import (
    MODEL_CONFIGS,                  # same backbone configs
    set_seed,
    project_rows_to_simplex,
    compute_coactivation_matrix,
    build_anchor,                   # only for diagnostic Pi0 distance, optional
    ImageNet1kSampledDataset,
    ActivationExtractor,
    ChunkedActivationDataset,
    masked_reconstruction_loss,
    anchor_loss,
    l1_sparsity,
    atom_distance,
    IMAGENET_RAW_DIR,
    IMAGENET_SAMPLED_DIR,
    ACTIVATION_CACHE_DIR,
    IMAGES_PER_CLASS,
    NUM_CLASSES,
    ACTIVATION_CHUNK_SIZE,
    BATCH_SIZE_COLLECTION,
    DEFAULT_DATA_SEED,
    DEFAULT_MODEL_SEED,
)


# ==========================================
# DCAM-Untied: simplex Pi encoder + free W decoder
# ==========================================

class DCAMUntied(nn.Module):
    """DCAM with an untied, unconstrained decoder W.

    Encoder (identical to DCAM):
        z'  = ReLU(Pi A)        with Pi in R^{D x C}, rows on the L1-simplex
        z   = TopK(z', K)       sparsify over the D concepts

    Decoder (DIFFERENT from DCAM):
        A_hat = W z             with W in R^{C x D}, fully free (any sign,
                                any magnitude). Initialized via Kaiming.

    The two parameter groups are intentionally trained together with the same
    optimizer for simplicity. Pi is still projected to the simplex after each
    step; W is unconstrained throughout. This is what makes this version a
    clean "is the decoder the problem?" test.
    """

    def __init__(self, in_channels: int, num_concepts: int, top_k: int,
                 decoder_init_scale: float = 1.0):
        super().__init__()
        # Unlike DCAM, we DO NOT require D <= C; an untied decoder can absorb
        # any D. Default D in MODEL_CONFIGS is still C/2 for parity with the
        # tied version, but you can pass --D 2048 etc.
        self.C = in_channels
        self.D = num_concepts
        self.top_k = top_k

        # Pi: simplex-constrained encoder, identical to DCAM.
        self.Pi = nn.Parameter(torch.empty(self.D, self.C))
        with torch.no_grad():
            self.Pi.uniform_(0.0, 1.0)
            self.Pi.data = project_rows_to_simplex(self.Pi.data)

        # W: free decoder. Kaiming-normal init scaled to match the variance
        # of the pinv-decoder's typical column norm so the early training
        # dynamics are roughly comparable to tied DCAM.
        # Empirically the pinv columns have norm ~ 1/sqrt(C); we match that.
        self.W = nn.Parameter(torch.empty(self.C, self.D))
        nn.init.kaiming_normal_(self.W, mode='fan_in', nonlinearity='relu')
        with torch.no_grad():
            self.W.data *= decoder_init_scale

    # ---- initialization -------------------------------------------------
    def init_pi_from_anchor(self, Pi0: torch.Tensor):
        """Initialize Pi at a seed-free anchor Pi0 (optional, diagnostic only).

        Note: We do NOT touch W here. The anchor concept only applies to the
        encoder; the decoder is free.
        """
        assert Pi0.shape == (self.D, self.C), \
            f"anchor shape {tuple(Pi0.shape)} != Pi shape {(self.D, self.C)}"
        with torch.no_grad():
            self.Pi.data = project_rows_to_simplex(Pi0.clone().to(self.Pi.device))

    # ---- top-k over concepts (identical to DCAM) -----------------------
    def topk_over_concepts(self, zp: torch.Tensor) -> torch.Tensor:
        B, D, H, W = zp.shape
        score = zp.sum(dim=(2, 3))
        k = min(self.top_k, D)
        _, idx = torch.topk(score, k=k, dim=1)
        mask = torch.zeros(B, D, device=zp.device, dtype=zp.dtype)
        mask.scatter_(1, idx, 1.0)
        return zp * mask.unsqueeze(-1).unsqueeze(-1)

    # ---- forward --------------------------------------------------------
    def forward(self, A: torch.Tensor, use_topk: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = A.shape
        assert C == self.C, f"input has {C} channels, Pi expects {self.C}"

        zp = torch.einsum('dc,bchw->bdhw', self.Pi, A)
        zp = F.relu(zp)
        z = self.topk_over_concepts(zp) if use_topk else zp

        # The ONLY architectural change: free W instead of pinv(Pi).
        A_hat = torch.einsum('cd,bdhw->bchw', self.W, z)
        return A_hat, z

    # ---- projection (Pi only; W is unconstrained) ----------------------
    @torch.no_grad()
    def project(self):
        self.Pi.data = project_rows_to_simplex(self.Pi.data)


# ==========================================
# Diagnostic metrics
# ==========================================

@torch.no_grad()
def relative_reconstruction_error(A_hat: torch.Tensor,
                                  A: torch.Tensor) -> float:
    """||A_hat - A||_1 / ||A||_1 .  This is the metric that exposed the
    tied-DCAM failure mode: MSE looked fine while rel_err > 1, meaning the
    reconstruction lost the signal's direction even as it tracked its mean
    magnitude. We use L1 (not L2) because it's the metric that downstream
    classification cares about (linear in activations).
    """
    num = (A_hat - A).abs().mean()
    den = A.abs().mean().clamp(min=1e-8)
    return float((num / den).item())


@torch.no_grad()
def concept_utilization(z: torch.Tensor) -> Tuple[float, float]:
    """Two granularities of concept-life:
        live_per_batch: fraction of D concepts that are activated by AT LEAST
                        ONE image in the batch. Catches "globally dead" atoms
                        that the K/D cell-level readout cannot.
        live_per_image: mean over images of the fraction of D concepts each
                        image activates. By construction, after TopK each
                        image activates min(K, D) atoms, so this is just
                        K/D unless the encoder is producing fewer non-zero
                        pre-TopK activations.
    """
    # z: [B, D, H, W]
    per_concept_mass = z.sum(dim=(0, 2, 3))          # [D]
    live_per_batch = (per_concept_mass > 0).float().mean().item()
    per_image_live = (z.sum(dim=(2, 3)) > 0).float().mean().item()
    return live_per_batch, per_image_live


# ==========================================
# Visualization
# ==========================================

def plot_training_logs(logs: Dict[str, List], model_name: str,
                       save_path: str):
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'DCAM-Untied Training ({model_name.upper()} - ImageNet-1k)',
                 fontsize=14, fontweight='bold')

    axs[0, 0].plot(logs["recon_loss"], color='blue', lw=1.5)
    axs[0, 0].set_title("Masked Reconstruction Loss")
    axs[0, 0].set_ylabel("MSE (masked)"); axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(logs["rel_err"], color='crimson', lw=1.5)
    axs[0, 1].axhline(1.0, color='black', ls='--', lw=0.8, alpha=0.5)
    axs[0, 1].set_title("Relative Recon Error  ||A_hat - A||_1 / ||A||_1")
    axs[0, 1].grid(True, alpha=0.3)

    axs[0, 2].plot(logs["l1_loss"], color='green', lw=1.5)
    axs[0, 2].set_title("L1 Concept-Sparsity Loss"); axs[0, 2].grid(True, alpha=0.3)

    axs[1, 0].plot(logs["sigma_min_Pi"], color='red', lw=1.5)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title("sigma_min(Pi)  (encoder conditioning)")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(logs["sigma_min_W"], color='purple', lw=1.5)
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title("sigma_min(W)  (decoder conditioning)")
    axs[1, 1].grid(True, alpha=0.3)

    axs[1, 2].plot(logs["live_per_batch"], color='teal', lw=1.5,
                   label='per-batch live')
    axs[1, 2].plot(logs["live_per_image"], color='lightseagreen', lw=1.5,
                   label='per-image live', alpha=0.7)
    axs[1, 2].set_ylim(0, 1.05)
    axs[1, 2].set_title("Concept utilization")
    axs[1, 2].legend(fontsize=8); axs[1, 2].grid(True, alpha=0.3)

    axs[2, 0].plot(logs["total_loss"], color='black', lw=2)
    axs[2, 0].set_title("Total Loss"); axs[2, 0].grid(True, alpha=0.3)

    axs[2, 1].plot(logs["recon_loss"], label='recon', alpha=0.8)
    axs[2, 1].plot(logs["l1_loss"], label='L1', alpha=0.8)
    if any(v > 0 for v in logs["anchor_loss"]):
        axs[2, 1].plot(logs["anchor_loss"], label='anchor', alpha=0.8)
    axs[2, 1].set_yscale('log')
    axs[2, 1].set_title("Loss components (log)")
    axs[2, 1].legend(fontsize=8); axs[2, 1].grid(True, alpha=0.3)

    axs[2, 2].plot(logs["W_norm"], color='orange', lw=1.5)
    axs[2, 2].set_title("||W||_F  (free decoder norm)")
    axs[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='DCAM-Untied training on ImageNet-1k '
                    '(diagnostic: free decoder W)')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--target_layer', type=str, default=None)
    parser.add_argument('--force_resample', action='store_true')
    parser.add_argument('--force_reextract', action='store_true')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Defaults to 5: diagnostic runs converge fast.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=None)

    # DCAM hyperparameters (Pi side, same defaults as DCAM for parity)
    parser.add_argument('--D', type=int, default=None,
                        help='Atom-vocabulary size D. With the free decoder, '
                             'D > C is allowed; defaults to per-backbone C/2 '
                             'for direct comparison with tied DCAM.')
    parser.add_argument('--top_k', type=int, default=32)
    parser.add_argument('--lambda_anchor', type=float, default=0.0,
                        help='DEFAULT 0.0 -- the diagnostic uses an unanchored '
                             'encoder. Set > 0 only for ablation runs.')
    parser.add_argument('--lambda_l1', type=float, default=0.0,
                        help='DEFAULT 0.0 -- diagnostic. The recon loss alone '
                             'is what we want to see drop.')
    parser.add_argument('--cumulative_threshold', type=float, default=0.85)

    # Anchor construction (optional, only used if lambda_anchor > 0)
    parser.add_argument('--use_anchor_init', action='store_true',
                        help='Initialize Pi from the seed-free anchor Pi0 '
                             '(symmetric NMF of co-activation S). Off by '
                             'default; for the diagnostic we want a clean '
                             'random init to avoid biasing toward Pi0.')
    parser.add_argument('--nmf_iters', type=int, default=500)
    parser.add_argument('--merge_tol', type=float, default=1e-2)
    parser.add_argument('--anchor_seed', type=int, default=0)

    parser.add_argument('--decoder_init_scale', type=float, default=1.0,
                        help='Multiplier on Kaiming-init of W. Lower if early '
                             'training is unstable; raise if recon is sluggish.')

    parser.add_argument('--model_suffix', type=str, default='')
    parser.add_argument('--data_seed', type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument('--model_seed', type=int, default=DEFAULT_MODEL_SEED)
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = 16 if args.model in ('resnet50', 'vgg16') else 32

    print("=" * 80)
    print(f"DCAM-UNTIED Training -- backbone {args.model.upper()}")
    print("  (diagnostic: free decoder W, simplex Pi encoder)")
    print(f"  Seeds: data={args.data_seed}  model={args.model_seed}")
    print("=" * 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- data ----------------------------------------------------------
    print(f"\nApplying data seed:")
    set_seed(args.data_seed)
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    dataset = ImageNet1kSampledDataset(
        IMAGENET_RAW_DIR, IMAGENET_SAMPLED_DIR, IMAGES_PER_CLASS,
        transform=tfm, force_resample=args.force_resample)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_COLLECTION,
                        shuffle=False, num_workers=4)

    # ---- extraction (reuses the same cache as run_dcam_full.py) --------
    extractor = ActivationExtractor(
        args.model, args.target_layer, device,
        cumulative_threshold=args.cumulative_threshold)
    act_chunks, mask_chunks, label_chunks, gcmap_chunks = extractor.collect(
        loader, normalize=True, chunk_size=ACTIVATION_CHUNK_SIZE,
        use_cache=not args.force_reextract)

    C = extractor.num_channels
    D = args.D if args.D is not None else MODEL_CONFIGS[args.model]['default_D']
    print(f"\nC = {C},  D = {D}  (untied decoder allows D > C; "
          f"D/C ratio = {D/C:.2f})")

    # ---- optional anchor (off by default) ------------------------------
    Pi0 = None
    nu = float('nan')
    if args.use_anchor_init or args.lambda_anchor > 0:
        print(f"\n{'='*80}\nBuilding seed-free anchor Pi0 "
              f"(for {'init' if args.use_anchor_init else 'loss'})\n"
              f"{'='*80}")
        S = compute_coactivation_matrix(act_chunks)
        Pi0_tensor, nu, D_eff = build_anchor(
            S, D, nmf_iters=args.nmf_iters, merge_tol=args.merge_tol,
            seed=args.anchor_seed)
        if D_eff != D:
            print(f"  NOTE: D reduced {D} -> {D_eff} after non-degeneracy merge.")
            D = D_eff
        Pi0 = Pi0_tensor.to(device)

    # ---- model ---------------------------------------------------------
    print(f"\nApplying model seed:")
    set_seed(args.model_seed)
    top_k = min(args.top_k, D)
    model = DCAMUntied(in_channels=C, num_concepts=D, top_k=top_k,
                       decoder_init_scale=args.decoder_init_scale).to(device)
    if args.use_anchor_init and Pi0 is not None:
        model.init_pi_from_anchor(Pi0)
        print("  Pi initialized at anchor Pi0.")
    else:
        print("  Pi initialized at random (no anchor).")

    print(f"\nTraining Configuration:")
    print(f"  C={C}  D={D}  Top-K={top_k}")
    print(f"  decoder: UNTIED (W in R^{{{C} x {D}}}, free)")
    print(f"  lambda_anchor={args.lambda_anchor}  lambda_l1={args.lambda_l1}")
    print(f"  epochs={args.epochs}  lr={args.lr}  batch_size={args.batch_size}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    train_dataset = ChunkedActivationDataset(
        act_chunks, mask_chunks, label_chunks, gcmap_chunks)
    gen = torch.Generator().manual_seed(args.model_seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, generator=gen)

    logs = {"total_loss": [], "recon_loss": [], "anchor_loss": [],
            "l1_loss": [], "rel_err": [],
            "sigma_min_Pi": [], "sigma_min_W": [], "W_norm": [],
            "live_per_batch": [], "live_per_image": []}

    print(f"\n{'='*80}\nStarting Training\n{'='*80}")
    for epoch in range(args.epochs):
        ep = {k: 0.0 for k in logs}
        nb = 0
        # Track epoch-wide concept utilization (union over all batches).
        epoch_concept_mass = torch.zeros(D, device=device)

        for bi, (A, M, _lbl, _gc) in enumerate(train_loader):
            A = A.to(device); M = M.to(device)

            A_hat, z = model(A, use_topk=True)

            l_recon = masked_reconstruction_loss(A_hat, A, M)
            l_l1 = l1_sparsity(z)
            # Anchor is computed only if requested; otherwise dummy zero.
            if args.lambda_anchor > 0 and Pi0 is not None:
                l_anchor = anchor_loss(model.Pi, Pi0)
            else:
                l_anchor = torch.zeros((), device=device)

            loss = (l_recon
                    + args.lambda_anchor * l_anchor
                    + args.lambda_l1 * l_l1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.project()   # Pi back to simplex; W is left alone

            # diagnostics
            with torch.no_grad():
                rel_err = relative_reconstruction_error(A_hat, A)
                sv_pi = torch.linalg.svdvals(model.Pi)
                sigma_min_Pi = float(sv_pi.min().item())
                sv_w = torch.linalg.svdvals(model.W)
                sigma_min_W = float(sv_w.min().item())
                W_norm = float(model.W.norm().item())
                live_b, live_i = concept_utilization(z)
                epoch_concept_mass += z.sum(dim=(0, 2, 3))

            logs["total_loss"].append(loss.item())
            logs["recon_loss"].append(l_recon.item())
            logs["anchor_loss"].append(float(l_anchor.item()))
            logs["l1_loss"].append(l_l1.item())
            logs["rel_err"].append(rel_err)
            logs["sigma_min_Pi"].append(sigma_min_Pi)
            logs["sigma_min_W"].append(sigma_min_W)
            logs["W_norm"].append(W_norm)
            logs["live_per_batch"].append(live_b)
            logs["live_per_image"].append(live_i)
            for k in ep:
                ep[k] += logs[k][-1]
            nb += 1

            if bi % 20 == 0:
                print(f"\rEpoch {epoch+1}/{args.epochs} "
                      f"[{bi}/{len(train_loader)}] "
                      f"L={loss.item():.4f} | recon={l_recon.item():.4f} | "
                      f"rel_err={rel_err:.3f} | "
                      f"sigW={sigma_min_W:.2e} | "
                      f"live={live_b:.2f}", end="")

        avg = {k: v / max(nb, 1) for k, v in ep.items()}
        live_epoch = (epoch_concept_mass > 0).float().mean().item()
        print(f"\n[Epoch {epoch+1}/{args.epochs}] "
              f"total={avg['total_loss']:.4f}  "
              f"recon={avg['recon_loss']:.4f}  "
              f"rel_err={avg['rel_err']:.3f}  "
              f"sigW={avg['sigma_min_W']:.2e}  "
              f"live(epoch)={live_epoch:.3f}")
        print("-" * 80)

    # ---- save (format readable by check_drop_dcam.py with small patch) -
    prefix = f"imagenet1k_dcam_untied_{args.model}"
    if args.target_layer:
        prefix += "_" + args.target_layer.replace('[', '_').replace(']', '')
    prefix += f"_D{D}_seed{args.data_seed}-{args.model_seed}{args.model_suffix}"

    torch.save(model.state_dict(), f"{prefix}_model.pth")
    save_dict = {
        'Pi': model.Pi.detach().cpu(),
        'W':  model.W.detach().cpu(),
        'Pi0': Pi0.detach().cpu() if Pi0 is not None else None,
        'config': {
            'decoder': 'untied',                       # <-- key marker
            'model': args.model,
            'target_layer': extractor.target_layer_name,
            'C': C, 'D': D, 'top_k': top_k,
            'lambda_anchor': args.lambda_anchor,
            'lambda_l1': args.lambda_l1,
            'decoder_init_scale': args.decoder_init_scale,
            'use_anchor_init': args.use_anchor_init,
            'cumulative_threshold': args.cumulative_threshold,
            'nu': nu,
            'data_seed': args.data_seed,
            'model_seed': args.model_seed,
            'anchor_seed': args.anchor_seed,
        },
        'logs': logs, 'final_metrics': avg,
    }
    joblib.dump(save_dict, f"{prefix}_result.pkl")
    plot_training_logs(logs, args.model, f"{prefix}_logs.png")

    print(f"\n{'='*80}\nTraining Complete")
    print(f"  Result:    {prefix}_result.pkl")
    print(f"  Logs:      {prefix}_logs.png")
    print(f"  Final recon error: {avg['recon_loss']:.4f}")
    print(f"  Final relative error: {avg['rel_err']:.3f}  "
          f"({'OK' if avg['rel_err'] < 0.5 else 'BAD: rel_err>=0.5'})")
    print(f"  Concept utilization (final epoch): {live_epoch:.3f}")
    print()
    print("Next step: evaluate this checkpoint with the (patched) "
          "check_drop_dcam.py. If accuracy approaches the original backbone, "
          "the tied pinv decoder is the binding constraint in standard DCAM.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()