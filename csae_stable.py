"""
csae_stable.py
==============
CSAE training with an ICA-anchor STABILITY LOSS on the encoder.

Motivation
----------
Plain CSAE (run_xcsae_full.py) reconstructs ResNet50 layer3 well (~68% top-1
after substitution) but its atoms are seed-UNSTABLE: encoder filters split and
absorb across model seeds, because a free Conv2d encoder trained by Adam has
no identifiability constraint.

DCAM tried to fix this with a non-negative simplex anchor + tied pinv decoder.
That destroyed reconstruction (rel_err > 3, ~0.4% accuracy): the non-negative
constraint cannot represent the signed, ~200-effective-rank per-cell structure
of layer3.

This script takes the OTHER route: keep CSAE's free signed encoder and free
decoder (so reconstruction is preserved), and add ONLY a soft stability prior
-- an anchor loss pulling the encoder toward a fixed, seed-free SIGNED ICA
basis W0 (built by csae_ica_anchor.py). Signed, so no DCAM-style constraint
mismatch. Soft, so it shapes the solution without forbidding reconstruction.

    L_total = L_recon + lambda_l1*L1 + lambda_lat*L_lat + lambda_compact*L_tv
              + lambda_gradcam*L_gc + LAMBDA_ANCHOR * L_anchor

    L_anchor = || W_enc_slice - W0 ||_F^2 / D     (mean per anchored atom)

KEY DESIGN CHOICE  -- --anchor_mode
-----------------------------------
CSAE's encoder has hidden_dim = C*8 (8192 for ResNet50 layer3) but the ICA
anchor has D ~ 200 atoms (the per-cell effective rank). They are not the same
size, so "anchor the encoder to W0" is ambiguous. Two honest options:

  --anchor_mode subspace  (DEFAULT)
      Keep hidden_dim = C*8. Anchor ONLY the first D encoder units to W0;
      leave the other (hidden_dim - D) units free. Preserves CSAE's
      overcomplete capacity (and its ~68%); the anchored units act as a
      stable "spine". OPEN QUESTION: whether pinning 200/8192 units is
      enough to stabilise the rest.

  --anchor_mode full
      Shrink hidden_dim to D, anchor ALL encoder units to W0. Every atom is
      anchored -> strongest stability claim, but abandons overcompleteness,
      so it is a DIFFERENT model from CSAE and may lose accuracy.

Run BOTH. They answer different questions; the paper needs both points.

This is a HYPOTHESIS test, not a known fix. Possible outcomes:
  * accuracy stays ~68% AND cross-seed atom distance drops sharply -> the
    constructive result: signed ICA anchor stabilises CSAE for free.
  * accuracy drops -> LAMBDA_ANCHOR too high, or the anchor over-constrains;
    sweep LAMBDA_ANCHOR down.
  * stability does not improve (subspace mode) -> 200 anchored units cannot
    discipline 8192; use --anchor_mode full or raise D.
Measure, do not assume.

Prerequisite
------------
Run csae_ica_anchor.py first to build + verify the anchor:
    python csae_ica_anchor.py --cache_dir cache_activations \\
        --cache_key <gcmap1 cache dir> --D 200 --verify_seeds 0 1 2
That writes csae_ica_anchor_W0.npy. Pass it here as --ica_anchor.

Usage
-----
    python csae_stable.py --ica_anchor csae_ica_anchor_W0.npy \\
        --anchor_mode subspace --lambda_anchor 1.0 --model_seed 0
    # repeat with --model_seed 1, 2 to measure seed-stability

Then compare encoders across seeds with atom_distance (see compare_seeds()).
"""

import torch
torch.cuda.init()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import joblib
import argparse
import sys
from pathlib import Path

sys.path.append('.')
# Reuse every data/model/loss piece from the CSAE script unchanged, so the
# ONLY difference from plain CSAE is the anchor loss. This keeps the
# comparison clean and avoids transcription drift.
from run_xcsae_full import (
    MODEL_CONFIGS, set_seed,
    MultiChannelConvSAE,
    LateralInhibitionLoss, SpatialCompactnessLoss,
    FeatureChannelSparsityLoss, GradCAMDecompositionLoss,
    masked_reconstruction_loss,
    ImageNet1kSampledDataset, MultiModelActivationExtractor,
    ChunkedActivationDataset, ClassBalancedBatchSampler,
    IMAGENET_RAW_DIR, IMAGENET_SAMPLED_DIR,
    IMAGES_PER_CLASS, ACTIVATION_CHUNK_SIZE, BATCH_SIZE_COLLECTION,
    DEFAULT_DATA_SEED, DEFAULT_MODEL_SEED,
)
from torchvision import transforms


# ==========================================
# Anchor loss
# ==========================================

def encoder_weight_slice(model: MultiChannelConvSAE) -> torch.Tensor:
    """The [hidden_dim, C] view of the encoder's 1x1 conv weight.

    encoder.weight is [hidden_dim, C, kH, kW]. For the 1x1 conv used by CSAE
    (kernel_size=1) this squeezes to [hidden_dim, C] -- one signed C-vector
    per hidden unit, directly comparable to an anchor row.
    """
    w = model.encoder.weight                       # [H, C, kH, kW]
    if w.shape[2] == 1 and w.shape[3] == 1:
        return w[:, :, 0, 0]                        # [H, C]
    # non-1x1 kernel: average over spatial kernel positions as the
    # channel-mixing summary (CSAE default is 1x1, so this is a fallback).
    return w.mean(dim=(2, 3))                       # [H, C]


def anchor_loss(model: MultiChannelConvSAE, W0: torch.Tensor,
                mode: str) -> torch.Tensor:
    """Soft stability prior: pull encoder weights toward the fixed signed
    ICA anchor W0 [D, C].

    mode 'subspace': anchor the FIRST D hidden units; others unconstrained.
    mode 'full'    : anchor ALL hidden units (requires hidden_dim == D).

    Returned value is the mean squared row-difference over anchored atoms,
    so its scale does not grow with D and LAMBDA_ANCHOR is comparable across
    D settings.
    """
    W_enc = encoder_weight_slice(model)             # [H, C]
    D = W0.shape[0]
    if mode == "full":
        if W_enc.shape[0] != D:
            raise ValueError(
                f"--anchor_mode full needs hidden_dim == D ({D}); "
                f"encoder has {W_enc.shape[0]} hidden units. "
                f"Set --hidden_dim {D} or use --anchor_mode subspace.")
        diff = W_enc - W0
    else:  # subspace
        if W_enc.shape[0] < D:
            raise ValueError(
                f"hidden_dim ({W_enc.shape[0]}) < D ({D}); cannot anchor a "
                f"D-row subspace. Raise hidden_dim or lower D.")
        diff = W_enc[:D] - W0                       # first D units only
    return (diff ** 2).sum(dim=1).mean()            # mean over anchored atoms


# ==========================================
# Seed-comparison helper (for after training)
# ==========================================

@torch.no_grad()
def atom_distance(W_a: torch.Tensor, W_b: torch.Tensor) -> float:
    """Sign/permutation-invariant matched distance between two encoder
    weight slices [H, C]. Hungarian matching on (1 - |cosine|); returns the
    mean matched value. This is the seed-stability metric: run two model
    seeds, compare their encoders, lower = more stable."""
    from scipy.optimize import linear_sum_assignment
    A = W_a.detach().cpu().numpy().astype(np.float64)
    B = W_b.detach().cpu().numpy().astype(np.float64)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    cos = An @ Bn.T
    cost = 1.0 - np.abs(cos)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def compare_seeds(model_paths):
    """Load >=2 saved csae_stable encoders and print pairwise atom_distance.
    Use after training several --model_seed runs."""
    slices = []
    for p in model_paths:
        blob = joblib.load(p)
        # saved as a dict with 'encoder_weight_slice'
        slices.append(torch.as_tensor(blob['encoder_weight_slice']))
    print(f"\nSeed-stability: pairwise matched atom distance "
          f"({len(slices)} runs)")
    for i in range(len(slices)):
        for j in range(i + 1, len(slices)):
            d = atom_distance(slices[i], slices[j])
            print(f"  run {i} <-> run {j}: {d:.4f}")


# ==========================================
# Main
# ==========================================

def main():
    ap = argparse.ArgumentParser(
        description="CSAE training with ICA-anchor stability loss.")
    ap.add_argument('--model', type=str, default='resnet50',
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--target_layer', type=str, default=None)
    ap.add_argument('--force_resample', action='store_true')
    ap.add_argument('--force_reextract', action='store_true')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--accumulation_steps', type=int, default=1)

    # CSAE losses (same defaults as run_xcsae_full.py)
    ap.add_argument('--lambda_gradcam', type=float, default=1.0)
    ap.add_argument('--gradcam_no_normalize', action='store_true')
    ap.add_argument('--cumulative_threshold', type=float, default=0.95)
    ap.add_argument('--top_k', type=int, default=32)

    # NEW: the anchor
    ap.add_argument('--ica_anchor', type=str, required=True,
                    help='Path to the .npy ICA anchor W0 [D, C] produced by '
                         'csae_ica_anchor.py.')
    ap.add_argument('--anchor_mode', type=str, default='subspace',
                    choices=['subspace', 'full'],
                    help="'subspace': anchor first D of hidden_dim units, "
                         "rest free, keeps overcomplete capacity. "
                         "'full': anchor all units, requires hidden_dim==D.")
    ap.add_argument('--lambda_anchor', type=float, default=1.0,
                    help='Weight of the encoder anchor loss. Sweep this: too '
                         'high trades reconstruction for stability.')
    ap.add_argument('--hidden_dim', type=int, default=None,
                    help='Encoder hidden dim. Default: C*8 (CSAE default) for '
                         'subspace mode; MUST equal D for full mode.')

    ap.add_argument('--data_seed', type=int, default=DEFAULT_DATA_SEED)
    ap.add_argument('--model_seed', type=int, default=DEFAULT_MODEL_SEED)
    ap.add_argument('--model_suffix', type=str, default='')
    args = ap.parse_args()

    if args.batch_size is None:
        args.batch_size = 16 if args.model in ('resnet50', 'vgg16') else 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(f"CSAE-STABLE -- backbone {args.model.upper()}")
    print(f"  anchor_mode={args.anchor_mode}  lambda_anchor={args.lambda_anchor}")
    print(f"  seeds: data={args.data_seed} model={args.model_seed}")
    print("=" * 80)
    print(f"Device: {device}")

    # ---- load + check the ICA anchor ----------------------------------
    W0_np = np.load(args.ica_anchor).astype(np.float32)
    D_anchor, C_anchor = W0_np.shape
    print(f"\nLoaded ICA anchor: W0 shape [{D_anchor}, {C_anchor}]")
    W0 = torch.as_tensor(W0_np, device=device)

    # ---- data ----------------------------------------------------------
    set_seed(args.data_seed)
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    dataset = ImageNet1kSampledDataset(
        IMAGENET_RAW_DIR, IMAGENET_SAMPLED_DIR, IMAGES_PER_CLASS,
        transform=tfm, force_resample=args.force_resample)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE_COLLECTION,
                             shuffle=False, num_workers=4)

    extractor = MultiModelActivationExtractor(
        model_name=args.model, target_layer=args.target_layer,
        device=device, cumulative_threshold=args.cumulative_threshold)
    (act_chunks, mask_chunks, label_chunks,
     gcmap_chunks) = extractor.collect_activation_maps_chunked(
        data_loader, normalize=True, chunk_size=ACTIVATION_CHUNK_SIZE,
        use_cache=not args.force_reextract)

    C = extractor.num_channels
    if C != C_anchor:
        raise ValueError(
            f"anchor C={C_anchor} != backbone C={C}. The anchor was built "
            f"for a different layer/backbone. Rebuild with csae_ica_anchor.py.")

    # ---- hidden_dim resolution ----------------------------------------
    if args.anchor_mode == 'full':
        hidden_dim = D_anchor
        if args.hidden_dim is not None and args.hidden_dim != D_anchor:
            print(f"  NOTE: --anchor_mode full forces hidden_dim = D = "
                  f"{D_anchor} (ignoring --hidden_dim {args.hidden_dim}).")
    else:
        hidden_dim = args.hidden_dim if args.hidden_dim is not None else C * 8
        if hidden_dim < D_anchor:
            raise ValueError(f"hidden_dim {hidden_dim} < D {D_anchor}.")
    print(f"\nEncoder hidden_dim = {hidden_dim}  "
          f"(anchor covers {D_anchor} units, "
          f"{'all' if args.anchor_mode=='full' else 'subspace'})")

    # ---- model ---------------------------------------------------------
    set_seed(args.model_seed)
    top_k = min(args.top_k, hidden_dim)
    model = MultiChannelConvSAE(
        in_channels=C, hidden_dim=hidden_dim, kernel_size=1,
        top_k=top_k).to(device)

    # initialise the anchored units AT the anchor (init + loss, as discussed:
    # init alone does not survive SGD, the loss alone starts far away --
    # doing both gives the anchor the best chance).
    with torch.no_grad():
        w = model.encoder.weight                    # [H, C, 1, 1]
        n_init = D_anchor if args.anchor_mode == 'subspace' else hidden_dim
        w[:n_init, :, 0, 0] = W0[:n_init]
        print(f"  encoder: first {n_init} units initialised at the anchor.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    lat_loss = LateralInhibitionLoss().to(device)
    tv_loss = SpatialCompactnessLoss().to(device)
    gc_loss = GradCAMDecompositionLoss(
        normalize=not args.gradcam_no_normalize).to(device)

    LAMBDA_L1, LAMBDA_LAT, LAMBDA_COMPACT = 0.3, 0.01, 0.01

    train_dataset = ChunkedActivationDataset(
        act_chunks, mask_chunks, label_chunks, gcmap_chunks)
    num_classes = len(train_dataset.class_to_indices)
    gen = torch.Generator().manual_seed(args.model_seed)
    if num_classes <= 100:
        sampler = ClassBalancedBatchSampler(
            train_dataset.class_to_indices, args.batch_size, drop_last=True)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True, generator=gen)

    logs = {"total": [], "recon": [], "l1": [], "anchor": [],
            "gradcam": [], "active_pct": []}

    print(f"\n{'='*80}\nTraining (CSAE + ICA anchor)\n{'='*80}")
    for epoch in range(args.epochs):
        ep = {k: 0.0 for k in logs}
        nb = 0
        optimizer.zero_grad()
        for bi, (A, M, _lbl, GC) in enumerate(train_loader):
            A = A.to(device); M = M.to(device); GC = GC.to(device)

            recon, z = model(A, use_topk=True)

            l_recon = masked_reconstruction_loss(recon, A, M)
            l_l1 = z.abs().mean()
            l_lat = lat_loss(z)
            l_tv = tv_loss(z)
            l_gc = gc_loss(z, GC)
            l_anchor = anchor_loss(model, W0, args.anchor_mode)

            loss = (l_recon
                    + LAMBDA_L1 * l_l1
                    + LAMBDA_LAT * l_lat
                    + LAMBDA_COMPACT * l_tv
                    + args.lambda_gradcam * l_gc
                    + args.lambda_anchor * l_anchor)
            loss = loss / args.accumulation_steps
            loss.backward()

            if ((bi + 1) % args.accumulation_steps == 0
                    or (bi + 1) == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                model.normalize_decoder_weights()

            with torch.no_grad():
                active = (z > 0).float().mean().item() * 100.0
            logs["total"].append(loss.item() * args.accumulation_steps)
            logs["recon"].append(l_recon.item())
            logs["l1"].append(l_l1.item())
            logs["anchor"].append(l_anchor.item())
            logs["gradcam"].append(l_gc.item())
            logs["active_pct"].append(active)
            for k in ep:
                ep[k] += logs[k][-1]
            nb += 1

            if bi % 20 == 0:
                print(f"\rEpoch {epoch+1}/{args.epochs} [{bi}/"
                      f"{len(train_loader)}] recon={l_recon.item():.4f} "
                      f"anchor={l_anchor.item():.4f} "
                      f"active={active:.1f}%", end="")

        avg = {k: v / max(nb, 1) for k, v in ep.items()}
        print(f"\n[Epoch {epoch+1}] recon={avg['recon']:.4f}  "
              f"anchor={avg['anchor']:.4f}  gradcam={avg['gradcam']:.6f}  "
              f"active={avg['active_pct']:.2f}%")
        print("-" * 80)

    # ---- save ----------------------------------------------------------
    prefix = (f"imagenet1k_csae_stable_{args.model}_"
              f"{args.anchor_mode}_la{args.lambda_anchor}_"
              f"seed{args.data_seed}-{args.model_seed}{args.model_suffix}")
    torch.save(model.state_dict(), f"{prefix}_model.pth")
    joblib.dump({
        'encoder_weight_slice': encoder_weight_slice(model).detach().cpu(),
        'config': {
            'model': args.model, 'C': C, 'hidden_dim': hidden_dim,
            'D_anchor': D_anchor, 'anchor_mode': args.anchor_mode,
            'lambda_anchor': args.lambda_anchor, 'top_k': top_k,
            'data_seed': args.data_seed, 'model_seed': args.model_seed,
        },
        'logs': logs, 'final_metrics': avg,
    }, f"{prefix}_result.pkl")
    print(f"\n{'='*80}\nTraining complete.")
    print(f"  Result: {prefix}_result.pkl")
    print(f"  Final recon={avg['recon']:.4f}  anchor={avg['anchor']:.4f}")
    print(f"\nNext steps:")
    print(f"  1. Train >=2 more --model_seed runs.")
    print(f"  2. compare_seeds([...]) -> cross-seed atom distance "
          f"(the stability number).")
    print(f"  3. Evaluate accuracy with the FIXED eval "
          f"(check_drop_dcam_fixed.py-style, normalized-space rel_err).")
    print(f"  Constructive result requires BOTH: accuracy ~ plain CSAE's 68% "
          f"AND atom distance well below plain CSAE's.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()