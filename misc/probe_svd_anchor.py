"""
probe_svd_anchor.py
===================
Train a lightweight MLP that predicts the ImageNet class from the SVD-anchor
projection of a backbone's layer-3 activations, save it, and report two
accuracy-drop numbers.

WHY THIS SCRIPT EXISTS
----------------------
csae_svd_anchor.py builds a seed-free SIGNED basis W0 in R^{D x C} -- the
signed top-D PCA eigenvectors of the per-cell channel covariance of (e.g.)
ResNet50 layer3. csae_stable.py uses W0 only as a SOFT prior on a much larger
trained CSAE encoder. Neither script answers a more basic question:

    How much of the layer-3 signal that the classifier actually USES
    survives a projection onto the D-dimensional anchor subspace?

A linear projection onto W0 throws away (C - D) channel directions. If the
discarded directions carry class information, no amount of CSAE training on
top can recover it -- the anchor would be a bad scaffold. This script
measures that directly with a cheap, honest probe instead of assuming.

WHAT IT DOES
------------
1. Reuses run_xcsae_full's extractor to collect normalized layer-3
   activations + labels (the SAME activations CSAE / csae_stable train on).
2. Projects each activation onto the anchor: codes = A_cell @ W0.T, giving a
   [D]-dim representation per spatial cell. Three --probe_input options:
     'codes'   (default) -- spatially average-pooled anchor codes -> [D]
     'spatial'            -- keep [D, H, W], small global-pool MLP head
     'recon'              -- reconstruct A_hat = (A @ W0.T) @ W0  -> [C]
                             (what the anchor subspace can represent at all)
3. Trains a lightweight MLP (D|C -> hidden -> 1000) on those features.
4. Saves the probe (state dict + config) so it can be reloaded.
5. Reports TWO accuracy-drop numbers:
     (a) PROBE vs BACKBONE -- probe top-1 from anchor codes vs the backbone's
         own top-1 on the same images. Drop = how much class signal the
         anchor subspace + a tiny head fails to capture.
     (b) RECONSTRUCTION DROP -- substitute A_hat = (A @ W0.T) @ W0 back into
         the backbone and run it to the logits, exactly like
         check_drop_csae_fixed.py does for CSAE. Drop = how much the backbone
         classifier itself loses when its layer-3 input is squashed onto the
         D-dim anchor subspace. This is the apples-to-apples comparison
         against CSAE's ~68%.

   (a) and (b) answer different questions; both are printed. (b) needs no
   training and is the cleaner anchor-quality number; (a) tells you whether
   the anchor codes are a usable feature space on their own.

NORMALIZATION
-------------
Test-time activations are normalized with normalize_like_training, copied
verbatim from check_drop_csae_fixed.py, so the probe sees the same input
distribution it was trained on and the reconstruction-drop number is
comparable to the CSAE eval. rel_err is reported in BOTH normalized and
denormalized space, same as the CSAE eval.

PREREQUISITE
------------
Build the anchor first:
    python csae_svd_anchor.py --cache_dir cache_activations \\
        --cache_key resnet50_layer3_thresh0p85_samples50000_chunk100_gcmap1 \\
        --D 200 --verify_seeds 0 1 2
That writes csae_svd_anchor_W0.npy. Pass it here as --svd_anchor.

USAGE
-----
  # train the probe + measure both drops
  python probe_svd_anchor.py --svd_anchor csae_svd_anchor_W0.npy \\
      --model resnet50 --probe_input codes --epochs 20

  # reuse a saved probe, skip training, just re-run the drop eval
  python probe_svd_anchor.py --svd_anchor csae_svd_anchor_W0.npy \\
      --load_probe probe_svd_anchor_resnet50_codes_D200.pth --no_train

OUTCOMES TO EXPECT (measure, do not assume)
-------------------------------------------
  * reconstruction drop small (acc stays near backbone top-1): the D-dim
    anchor subspace already preserves the class-relevant signal -> the anchor
    is a sound scaffold for CSAE, and CSAE's job is just sparsity/structure.
  * reconstruction drop large: the anchor discards directions the classifier
    needs -> raising D, or the fact that CSAE's free encoder is overcomplete,
    is doing the real work. Either way, an important caveat for the paper.
  * probe drop >> reconstruction drop: anchor codes carry the signal but a
    tiny MLP cannot read it linearly enough -> expected, codes are a rotated
    PCA space, not a classification space.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import io
import random

import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd

sys.path.append('.')
# Data + extractor come from the CSAE training script so the probe trains on
# exactly the activations CSAE / csae_stable see (same normalization, same
# layer, same sampling).
from run_xcsae_full import (
    set_seed,
    ImageNet1kSampledDataset, MultiModelActivationExtractor,
    ChunkedActivationDataset,
    IMAGENET_RAW_DIR, IMAGENET_SAMPLED_DIR,
    IMAGES_PER_CLASS, ACTIVATION_CHUNK_SIZE, BATCH_SIZE_COLLECTION,
    DEFAULT_DATA_SEED, DEFAULT_MODEL_SEED,
)
from full_classes import IMAGENET2012_CLASSES


# ==========================================
# Backbone registry (same as check_drop_csae_fixed.py)
# ==========================================

MODEL_CONFIGS = {
    'resnet50': {'model_fn': lambda: models.resnet50(pretrained=True),
                 'default_target_layer': 'layer3',
                 'description': 'ResNet50 (layer3: 1024ch, 14x14)'},
    'resnet18': {'model_fn': lambda: models.resnet18(pretrained=True),
                 'default_target_layer': 'layer3',
                 'description': 'ResNet18 (layer3: 256ch, 14x14)'},
    'vgg16': {'model_fn': lambda: models.vgg16(pretrained=True),
              'default_target_layer': 'features[16]',
              'description': 'VGG16 (features[16]: 256ch, 28x28)'},
    'efficientnet': {'model_fn': lambda: models.efficientnet_b0(pretrained=True),
                     'default_target_layer': 'features[4]',
                     'description': 'EfficientNet-B0 (features[4]: ~80ch)'},
}


# ==========================================
# The lightweight probe
# ==========================================

class AnchorProbe(nn.Module):
    """Small MLP: anchor-code vector -> class logits.

    Two BN+ReLU hidden layers with dropout. Deliberately lightweight: the
    point of the probe is to read whatever class signal is LINEARLY-ish
    available in the anchor codes, not to be a strong classifier. If a probe
    this small recovers the backbone's accuracy, the anchor subspace clearly
    preserves the class signal.
    """

    def __init__(self, in_dim: int, num_classes: int = 1000,
                 hidden: int = 1024, depth: int = 2, dropout: float = 0.3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, num_classes)]
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden = hidden
        self.depth = depth

    def forward(self, x):
        return self.net(x)


# ==========================================
# Anchor projection helpers
# ==========================================

def project_to_codes(A: torch.Tensor, W0: torch.Tensor,
                      probe_input: str) -> torch.Tensor:
    """Turn a batch of activations [B, C, H, W] into probe features.

    W0 is [D, C], signed unit-norm rows. For each spatial cell the anchor
    code is  c = W0 @ a   (a in R^C  ->  c in R^D).

    probe_input:
      'codes'   -> mean over space of the [D] codes      -> [B, D]
      'spatial' -> codes kept as [B, D, H, W]             -> [B, D, H, W]
      'recon'   -> A_hat = W0.T @ (W0 @ a), mean over sp. -> [B, C]
    """
    B, C, H, W = A.shape
    cells = A.permute(0, 2, 3, 1).reshape(B, H * W, C)      # [B, HW, C]
    codes = cells @ W0.T                                    # [B, HW, D]

    if probe_input == 'codes':
        return codes.mean(dim=1)                            # [B, D]
    if probe_input == 'spatial':
        D = W0.shape[0]
        return codes.permute(0, 2, 1).reshape(B, D, H, W)   # [B, D, H, W]
    if probe_input == 'recon':
        recon_cells = codes @ W0                            # [B, HW, C]
        return recon_cells.mean(dim=1)                      # [B, C]
    raise ValueError(f"unknown probe_input {probe_input}")


def reconstruct_activation(A: torch.Tensor, W0: torch.Tensor) -> torch.Tensor:
    """Project layer-3 activations onto the anchor subspace and back:
        A_hat = (A @ W0.T) @ W0      (per spatial cell)
    Returns [B, C, H, W], same shape as A. This is the anchor's best
    reconstruction of the activation -- exactly what gets substituted back
    into the backbone for the reconstruction-drop metric.
    """
    B, C, H, W = A.shape
    cells = A.permute(0, 2, 3, 1).reshape(-1, C)            # [B*HW, C]
    recon = (cells @ W0.T) @ W0                             # [B*HW, C]
    return recon.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


# ==========================================
# Normalization -- VERBATIM from check_drop_csae_fixed.py
# (matches run_xcsae_full.ActivationExtractor._norm_chunk)
# ==========================================

def _vectorized_q99_of_positives(a_flat: torch.Tensor) -> torch.Tensor:
    """0.99-quantile of the positive (>1e-8) entries along the last dim,
    fully vectorized -- no Python loop over channels."""
    pos = a_flat > 1e-8
    n_pos = pos.sum(dim=-1)
    neg_inf = torch.finfo(a_flat.dtype).min
    masked = torch.where(pos, a_flat, torch.full_like(a_flat, neg_inf))
    sorted_vals, _ = torch.sort(masked, dim=-1)

    N = a_flat.shape[-1]
    pos_float = (n_pos - 1).clamp(min=0).to(a_flat.dtype) * 0.99
    lo_off = pos_float.floor().long()
    hi_off = torch.minimum(lo_off + 1, (n_pos - 1).clamp(min=0))
    frac = pos_float - lo_off.to(a_flat.dtype)

    start = N - n_pos
    lo_idx = (start + lo_off).clamp(max=N - 1)
    hi_idx = (start + hi_off).clamp(max=N - 1)

    lo_vals = sorted_vals.gather(-1, lo_idx.unsqueeze(-1)).squeeze(-1)
    hi_vals = sorted_vals.gather(-1, hi_idx.unsqueeze(-1)).squeeze(-1)
    q = lo_vals + frac * (hi_vals - lo_vals)

    q = torch.where(n_pos > 0, q, torch.zeros_like(q))
    return q


def normalize_like_training(acts: torch.Tensor, mode: str = "per_image"):
    """Per-channel 0.99-quantile clip+rescale, matching
    run_xcsae_full.ActivationExtractor._norm_chunk. Returns
    (A_norm, scale_factors)."""
    B, C, H, W = acts.shape
    a = torch.clamp(acts, min=0.0)

    if mode == "per_batch":
        a_flat = a.permute(1, 0, 2, 3).reshape(C, -1)
        q = _vectorized_q99_of_positives(a_flat)
        valid = q > 1e-8
        sf = torch.where(valid, q, torch.ones_like(q))
        sf_b = sf.view(1, C, 1, 1)
        normalized = torch.minimum(a, sf_b) / (sf_b + 1e-8)
        normalized = torch.where(valid.view(1, C, 1, 1), normalized, acts)
        return normalized, sf_b

    a_flat = a.view(B, C, H * W)
    q = _vectorized_q99_of_positives(a_flat)
    valid = q > 1e-8
    sf = torch.where(valid, q, torch.ones_like(q))
    sf_b = sf.view(B, C, 1, 1)
    normalized = torch.minimum(a, sf_b) / (sf_b + 1e-8)
    normalized = torch.where(valid.view(B, C, 1, 1), normalized, acts)
    return normalized, sf_b


# ==========================================
# Test dataset -- VERBATIM from check_drop_csae_fixed.py
# ==========================================

class ImageNet1kTestDataset(Dataset):
    def __init__(self, test_metadata_path: Path, transform=None):
        self.transform = transform
        metadata = joblib.load(test_metadata_path)
        self.samples = metadata['samples']
        print(f"  Loaded {len(self.samples)} test images "
              f"({metadata['images_per_class']}/class, "
              f"{metadata['num_classes']} classes)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_bytes, label = self.samples[idx]
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def create_test_samples_if_needed(raw_dir, test_dir, images_per_class,
                                  force_resample):
    metadata_path = test_dir / "test_metadata.pkl"
    if metadata_path.exists() and not force_resample:
        print(f"Test samples already cached at {metadata_path}")
        return metadata_path
    print(f"\nCreating test sample dataset ({images_per_class}/class)...")
    test_dir.mkdir(parents=True, exist_ok=True)
    wnid_to_idx = {w: i for i, w in enumerate(IMAGENET2012_CLASSES.keys())}
    class_samples = defaultdict(list)
    val_files = sorted(raw_dir.glob("validation-*.parquet"))
    if not val_files:
        raise FileNotFoundError(f"No validation parquet files in {raw_dir}")
    for pf in tqdm(val_files, desc="Reading validation parquet"):
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            lbl = row['label']
            if len(class_samples[lbl]) < images_per_class:
                class_samples[lbl].append((row['image']['bytes'], lbl))
        min_s = (min(len(v) for v in class_samples.values())
                 if class_samples else 0)
        if min_s >= images_per_class and len(class_samples) == 1000:
            break
    samples = []
    for c in range(1000):
        pool = class_samples[c]
        samples.extend(random.sample(pool, images_per_class)
                       if len(pool) >= images_per_class else pool)
    joblib.dump({'samples': samples, 'images_per_class': images_per_class,
                 'num_classes': 1000, 'wnid_to_idx': wnid_to_idx},
                metadata_path)
    print(f"Test samples cached ({len(samples)} images).")
    return metadata_path


# ==========================================
# Backbone with layer-3 split (for the reconstruction-drop metric)
# ==========================================

class BackboneSplit:
    """Runs a backbone to the target layer (_forward_to) and from it
    (_forward_from). Same logic as check_drop_csae_fixed's pipeline, lifted
    out so both the probe-feature extraction and the reconstruction-drop
    eval can share it."""

    def __init__(self, model_name, target_layer_name, device):
        self.model_name = model_name
        self.target_layer_name = target_layer_name
        self.device = device
        self.model = MODEL_CONFIGS[model_name]['model_fn']().to(device).eval()

    def _forward_to(self, x):
        m = self.model
        if self.model_name in ('resnet50', 'resnet18'):
            x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
            x = m.layer1(x)
            if 'layer1' in self.target_layer_name:
                return x
            x = m.layer2(x)
            if 'layer2' in self.target_layer_name:
                return x
            x = m.layer3(x)
            if 'layer3' in self.target_layer_name:
                return x
            x = m.layer4(x)
            return x
        idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
        for i in range(idx + 1):
            x = m.features[i](x)
        return x

    def _forward_from(self, x):
        m = self.model
        if self.model_name in ('resnet50', 'resnet18'):
            tl = self.target_layer_name
            if 'layer1' in tl:
                x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
            elif 'layer2' in tl:
                x = m.layer3(x); x = m.layer4(x)
            elif 'layer3' in tl:
                x = m.layer4(x)
            x = m.avgpool(x); x = torch.flatten(x, 1); x = m.fc(x)
            return x
        idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
        for i in range(idx + 1, len(m.features)):
            x = m.features[i](x)
        x = m.avgpool(x); x = torch.flatten(x, 1)
        x = m.classifier(x)
        return x

    @torch.no_grad()
    def forward_original(self, x):
        return self.model(x)


# ==========================================
# Build probe-training features from cached activations
# ==========================================

def build_probe_features(act_chunks, label_chunks, gcmap_chunks, mask_chunks,
                          W0, probe_input, device, feat_batch=64):
    """Stream the cached (already-normalized) training activations, project
    each through the anchor, and stack into one feature tensor + label
    tensor for the probe's DataLoader.

    Activations from collect_activation_maps_chunked are already normalized
    (normalize=True at collection), so we do NOT re-normalize here -- that
    matches how CSAE consumes them. Test-time activations in the drop eval
    DO get normalize_like_training, because they are extracted raw there.
    """
    ds = ChunkedActivationDataset(act_chunks, mask_chunks,
                                  label_chunks, gcmap_chunks)
    loader = DataLoader(ds, batch_size=feat_batch, shuffle=False,
                        num_workers=2)
    feats, labels = [], []
    print(f"Projecting {len(ds)} training activations onto the anchor "
          f"(probe_input={probe_input})...")
    for A, _M, lbl, _GC in tqdm(loader, desc="Anchor projection"):
        A = A.to(device)
        with torch.no_grad():
            f = project_to_codes(A, W0, probe_input)
        feats.append(f.cpu())
        labels.append(lbl.clone())
    X = torch.cat(feats, dim=0)
    y = torch.cat(labels, dim=0).long()
    print(f"  probe feature tensor: {tuple(X.shape)}  labels: {tuple(y.shape)}")
    return X, y


# ==========================================
# Probe training
# ==========================================

def train_probe(probe, X, y, device, epochs, lr, batch_size,
                 val_frac=0.1, seed=0):
    """Train the MLP probe on (X, y). Holds out a small validation split to
    report honest probe accuracy. 'spatial' probe_input is pooled here
    before the MLP (the probe is an MLP, not a conv net -- pooling keeps it
    lightweight as intended)."""
    if X.dim() == 4:                       # [N, D, H, W] -> global avg pool
        X = X.mean(dim=(2, 3))
    set_seed(seed)
    N = X.shape[0]
    perm = torch.randperm(N)
    X, y = X[perm], y[perm]
    n_val = max(int(N * val_frac), 1)
    Xtr, ytr = X[n_val:], y[n_val:]
    Xva, yva = X[:n_val], y[:n_val]

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                           shuffle=True, drop_last=True)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size,
                           shuffle=False)

    probe = probe.to(device)
    opt = optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    print(f"\n{'='*80}\nTraining probe "
          f"({sum(p.numel() for p in probe.parameters())/1e6:.2f}M params)"
          f"\n{'='*80}")
    for ep in range(epochs):
        probe.train()
        tot, run_loss = 0, 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = probe(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * xb.size(0)
            tot += xb.size(0)
        sched.step()

        probe.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = probe(xb).argmax(1)
                vc += (pred == yb).sum().item()
                vt += yb.size(0)
        val_acc = vc / max(vt, 1) * 100
        print(f"[Epoch {ep+1:>2}/{epochs}] train_loss={run_loss/tot:.4f}  "
              f"val_acc={val_acc:.2f}%  lr={sched.get_last_lr()[0]:.2e}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)
    print(f"\nBest probe val accuracy: {best_val:.2f}%")
    return probe, best_val


# ==========================================
# Accuracy-drop evaluation
# ==========================================

@torch.no_grad()
def evaluate_drops(probe, W0, backbone, data_loader, device,
                   probe_input, norm_mode, debug_batches=3):
    """Compute BOTH drop metrics on the test set in a single pass:

      (a) probe vs backbone : probe top-1 (from anchor codes) vs backbone
          top-1 on the same images.
      (b) reconstruction drop : backbone top-1 when layer-3 activations are
          replaced by A_hat = (A @ W0.T) @ W0.

    rel_err is reported in normalized and denormalized space, mirroring
    check_drop_csae_fixed.py.
    """
    probe = probe.to(device).eval()
    total = 0
    correct_backbone = 0
    correct_probe = 0
    correct_recon = 0
    agree_probe = 0       # probe argmax == backbone argmax
    agree_recon = 0       # recon argmax == backbone argmax
    reln_l, reld_l = [], []

    print("\nEvaluating: backbone vs probe vs anchor-reconstruction")
    print("=" * 80)
    for bi, (images, labels) in enumerate(tqdm(data_loader, desc="Batches")):
        images, labels = images.to(device), labels.to(device)
        B = images.size(0)

        # --- backbone reference ---
        logits_o = backbone.forward_original(images)
        pred_o = logits_o.argmax(1)

        # --- raw layer-3 activations, then training-style normalization ---
        A_raw = backbone._forward_to(images)                  # [B,C,H,W]
        A_norm, sf = normalize_like_training(A_raw, mode=norm_mode)

        # --- (a) probe on anchor codes of the NORMALIZED activation ---
        feats = project_to_codes(A_norm, W0, probe_input)
        if feats.dim() == 4:
            feats = feats.mean(dim=(2, 3))
        logits_p = probe(feats)
        pred_p = logits_p.argmax(1)

        # --- (b) reconstruction drop: A_hat back through the backbone ---
        A_hat_norm = reconstruct_activation(A_norm, W0)        # normalized sp
        A_hat = A_hat_norm * sf                                # denormalize
        logits_r = backbone._forward_from(A_hat)
        pred_r = logits_r.argmax(1)

        relerr_norm = ((A_hat_norm - A_norm).abs().mean()
                       / (A_norm.abs().mean() + 1e-8)).item()
        relerr_denorm = ((A_hat - A_raw).abs().mean()
                         / (A_raw.abs().mean() + 1e-8)).item()
        reln_l.append(relerr_norm)
        reld_l.append(relerr_denorm)

        correct_backbone += (pred_o == labels).sum().item()
        correct_probe += (pred_p == labels).sum().item()
        correct_recon += (pred_r == labels).sum().item()
        agree_probe += (pred_p == pred_o).sum().item()
        agree_recon += (pred_r == pred_o).sum().item()
        total += B

        if bi < debug_batches:
            print(f"\n   [debug] ---- batch {bi} ----")
            print(f"   raw A      : min={A_raw.min():.3f} "
                  f"max={A_raw.max():.3f} absmean={A_raw.abs().mean():.3f}")
            print(f"   A_norm     : min={A_norm.min():.3f} "
                  f"max={A_norm.max():.3f} mean={A_norm.mean():.3f}")
            print(f"   A_hat_norm : min={A_hat_norm.min():.3f} "
                  f"max={A_hat_norm.max():.3f}")
            print(f"   rel_err (normalized space)   = {relerr_norm:.4f}"
                  f"   <-- anchor subspace quality")
            print(f"   rel_err (denormalized space) = {relerr_denorm:.4f}"
                  f"   <-- what the classifier sees")
            print(f"   backbone top-1 (batch) = "
                  f"{(pred_o==labels).float().mean()*100:.1f}%  "
                  f"probe = {(pred_p==labels).float().mean()*100:.1f}%  "
                  f"recon = {(pred_r==labels).float().mean()*100:.1f}%")

    acc_b = correct_backbone / total * 100
    acc_p = correct_probe / total * 100
    acc_r = correct_recon / total * 100
    return {
        'total_samples': total,
        'acc_backbone': acc_b,
        'acc_probe': acc_p,
        'acc_recon': acc_r,
        'probe_drop': acc_b - acc_p,
        'recon_drop': acc_b - acc_r,
        'probe_agreement': agree_probe / total * 100,
        'recon_agreement': agree_recon / total * 100,
        'avg_relerr_normalized': float(np.mean(reln_l)),
        'avg_relerr_denorm': float(np.mean(reld_l)),
    }


def print_drops(res, model_name, probe_input, D, norm_mode):
    print("\n" + "=" * 80)
    print(f"SVD-ANCHOR PROBE -- ACCURACY DROP  ({model_name.upper()} "
          f"- ImageNet-1k)")
    print("=" * 80)
    print(f"\nanchor D={D}   probe_input={probe_input}   "
          f"norm_mode={norm_mode}")
    print(f"\nTop-1 accuracy:")
    print(f"  Backbone (reference):          {res['acc_backbone']:.2f}%")
    print(f"  Probe on anchor codes:         {res['acc_probe']:.2f}%")
    print(f"  Anchor reconstruction:         {res['acc_recon']:.2f}%")
    print(f"\nDrops vs backbone:")
    print(f"  (a) probe drop:                {res['probe_drop']:.2f}%   "
          f"<-- class signal a tiny MLP reads from the D-dim codes")
    print(f"  (b) reconstruction drop:       {res['recon_drop']:.2f}%   "
          f"<-- backbone loss from squashing layer-3 onto the anchor")
    if res['acc_backbone'] > 0:
        print(f"  relative probe drop:           "
              f"{res['probe_drop']/res['acc_backbone']*100:.2f}%")
        print(f"  relative reconstruction drop:  "
              f"{res['recon_drop']/res['acc_backbone']*100:.2f}%")
    print(f"\nArgmax agreement with backbone:")
    print(f"  probe == backbone:             {res['probe_agreement']:.2f}%")
    print(f"  reconstruction == backbone:    {res['recon_agreement']:.2f}%")
    print(f"\nAnchor reconstruction quality:")
    print(f"  rel_err (normalized space):    "
          f"{res['avg_relerr_normalized']:.4f}")
    print(f"  rel_err (denormalized space):  {res['avg_relerr_denorm']:.4f}")

    print(f"\n  INTERPRETATION:")
    rn = res['avg_relerr_normalized']
    rd = res['recon_drop']
    pd_ = res['probe_drop']
    if rd < 5:
        print("   Reconstruction drop is small: the D-dim anchor subspace "
              "preserves the class-relevant layer-3 signal -> the SVD anchor "
              "is a sound scaffold for CSAE.")
    elif rd > 30:
        print("   Reconstruction drop is large: the anchor subspace discards "
              "directions the classifier needs -> D is too small, or CSAE's "
              "overcomplete free encoder is doing the real work. Raise D and "
              "re-measure.")
    else:
        print("   Reconstruction drop is intermediate: the anchor keeps most "
              "but not all class signal -- report the number, consider a D "
              "sweep.")
    if pd_ - rd > 20:
        print("   Probe drop >> reconstruction drop: the codes CARRY the "
              "signal (reconstruction proves it) but the lightweight MLP "
              "cannot linearly read it from the rotated PCA space -- "
              "expected; the probe is intentionally small.")
    if rn > 0.7:
        print("   NOTE: normalized-space rel_err is high -- a D-dim linear "
              "projection simply cannot span layer-3. Compare against CSAE's "
              "overcomplete encoder.")
    print("=" * 80 + "\n")


# ==========================================
# Probe persistence
# ==========================================

def save_probe(probe, path, cfg):
    """Save state dict + the config needed to rebuild the probe."""
    torch.save({'state_dict': probe.state_dict(), 'config': cfg}, path)
    print(f"Probe saved -> {path}")


def load_probe(path, device):
    """Rebuild an AnchorProbe from a saved checkpoint."""
    blob = torch.load(path, map_location='cpu')
    cfg = blob['config']
    probe = AnchorProbe(in_dim=cfg['in_dim'], num_classes=cfg['num_classes'],
                        hidden=cfg['hidden'], depth=cfg['depth'],
                        dropout=cfg.get('dropout', 0.3))
    probe.load_state_dict(blob['state_dict'])
    probe = probe.to(device).eval()
    print(f"Loaded probe from {path}  "
          f"(in_dim={cfg['in_dim']}, hidden={cfg['hidden']}, "
          f"depth={cfg['depth']})")
    return probe, cfg


# ==========================================
# Main
# ==========================================

def main():
    ap = argparse.ArgumentParser(
        description="Lightweight MLP probe on the SVD anchor, with "
                    "two-way accuracy-drop evaluation.")
    ap.add_argument('--svd_anchor', type=str, required=True,
                    help='Path to the .npy SVD anchor W0 [D, C] from '
                         'csae_svd_anchor.py.')
    ap.add_argument('--model', type=str, default='resnet50',
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--target_layer', type=str, default=None)
    ap.add_argument('--probe_input', type=str, default='codes',
                    choices=['codes', 'spatial', 'recon'],
                    help="'codes': spatially-pooled D-dim anchor codes "
                         "(default). 'spatial': D x H x W codes, pooled "
                         "before the MLP. 'recon': anchor-reconstructed "
                         "C-dim activation.")

    # probe architecture
    ap.add_argument('--hidden', type=int, default=1024,
                    help='Probe MLP hidden width.')
    ap.add_argument('--depth', type=int, default=2,
                    help='Number of hidden layers in the probe MLP.')
    ap.add_argument('--dropout', type=float, default=0.3)

    # probe training
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--no_train', action='store_true',
                    help='Skip training; requires --load_probe.')
    ap.add_argument('--load_probe', type=str, default=None,
                    help='Path to a saved probe .pth to reuse instead of '
                         'training a fresh one.')

    # data / extraction
    ap.add_argument('--force_resample', action='store_true')
    ap.add_argument('--force_reextract', action='store_true')
    ap.add_argument('--cumulative_threshold', type=float, default=0.95)
    ap.add_argument('--data_seed', type=int, default=DEFAULT_DATA_SEED)
    ap.add_argument('--model_seed', type=int, default=DEFAULT_MODEL_SEED)

    # drop eval
    ap.add_argument('--norm_mode', type=str, default='per_image',
                    choices=['per_image', 'per_batch'],
                    help="Test-time normalization granularity. Run both to "
                         "check it is not the dominant effect.")
    ap.add_argument('--debug_batches', type=int, default=3)
    ap.add_argument('--raw_data_dir', type=str,
                    default='/data/imagenet_raw/data')
    ap.add_argument('--test_data_dir', type=str,
                    default='/data/imagenet1k_sampletest')
    ap.add_argument('--test_images_per_class', type=int, default=5)
    ap.add_argument('--eval_batch_size', type=int, default=32)

    ap.add_argument('--model_suffix', type=str, default='')
    ap.add_argument('--output_file', type=str, default=None)
    args = ap.parse_args()

    if args.no_train and args.load_probe is None:
        ap.error("--no_train requires --load_probe.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_layer = (args.target_layer
                    or MODEL_CONFIGS[args.model]['default_target_layer'])

    print("=" * 80)
    print(f"SVD-ANCHOR PROBE -- backbone {args.model.upper()}")
    print(f"  probe_input={args.probe_input}  norm_mode={args.norm_mode}")
    print(f"  seeds: data={args.data_seed} model={args.model_seed}")
    print("=" * 80)
    print(f"Device: {device}")

    # ---- load the SVD anchor ------------------------------------------
    W0_np = np.load(args.svd_anchor).astype(np.float32)
    D_anchor, C_anchor = W0_np.shape
    print(f"\nLoaded SVD anchor: W0 shape [{D_anchor}, {C_anchor}]")
    W0 = torch.as_tensor(W0_np, device=device)
    # rows should be unit-norm and signed already; report so a wrong file
    # is caught early.
    rn = W0.norm(dim=1)
    print(f"  anchor row-norm range: [{rn.min():.4f}, {rn.max():.4f}]")

    # ---- collect training activations (same path as CSAE) -------------
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
            f"for a different layer/backbone -- rebuild with "
            f"csae_svd_anchor.py.")

    in_dim = D_anchor if args.probe_input in ('codes', 'spatial') else C
    probe_cfg = {
        'in_dim': in_dim, 'num_classes': 1000,
        'hidden': args.hidden, 'depth': args.depth, 'dropout': args.dropout,
        'probe_input': args.probe_input, 'model': args.model,
        'D_anchor': D_anchor, 'C': C, 'target_layer': target_layer,
    }

    # ---- probe: load or train -----------------------------------------
    if args.load_probe is not None:
        probe, loaded_cfg = load_probe(args.load_probe, device)
        if loaded_cfg['in_dim'] != in_dim:
            raise ValueError(
                f"loaded probe in_dim={loaded_cfg['in_dim']} != expected "
                f"{in_dim} for probe_input={args.probe_input}. The probe "
                f"was trained for a different feature type.")
        best_val = None
    else:
        probe = AnchorProbe(in_dim=in_dim, num_classes=1000,
                            hidden=args.hidden, depth=args.depth,
                            dropout=args.dropout)

    if not args.no_train:
        X, y = build_probe_features(
            act_chunks, label_chunks, gcmap_chunks, mask_chunks,
            W0, args.probe_input, device)
        probe, best_val = train_probe(
            probe, X, y, device, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, seed=args.model_seed)

        prefix = (f"probe_svd_anchor_{args.model}_{args.probe_input}_"
                  f"D{D_anchor}{args.model_suffix}")
        save_probe(probe, f"{prefix}.pth",
                   {**probe_cfg, 'best_val_acc': best_val})

    # ---- accuracy-drop evaluation -------------------------------------
    test_metadata_path = create_test_samples_if_needed(
        Path(args.raw_data_dir), Path(args.test_data_dir),
        args.test_images_per_class, args.force_resample)
    test_ds = ImageNet1kTestDataset(test_metadata_path, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size,
                             shuffle=False, num_workers=4)

    backbone = BackboneSplit(args.model, target_layer, device)
    res = evaluate_drops(
        probe, W0, backbone, test_loader, device,
        probe_input=args.probe_input, norm_mode=args.norm_mode,
        debug_batches=args.debug_batches)
    print_drops(res, args.model, args.probe_input, D_anchor, args.norm_mode)

    # ---- save the report ----------------------------------------------
    if args.output_file is None:
        args.output_file = (f"probedrop_{args.model}_{args.probe_input}_"
                            f"D{D_anchor}_{args.norm_mode}.txt")
    with open(args.output_file, 'w') as f:
        f.write(f"SVD-anchor probe accuracy drop -- {args.model}\n")
        f.write(f"anchor: {args.svd_anchor}  D={D_anchor} C={C}\n")
        f.write(f"probe_input={args.probe_input}  "
                f"norm_mode={args.norm_mode}\n")
        f.write(f"probe: hidden={args.hidden} depth={args.depth}\n\n")
        f.write(f"acc_backbone           {res['acc_backbone']:.2f}%\n")
        f.write(f"acc_probe              {res['acc_probe']:.2f}%\n")
        f.write(f"acc_recon              {res['acc_recon']:.2f}%\n")
        f.write(f"probe_drop             {res['probe_drop']:.2f}%\n")
        f.write(f"recon_drop             {res['recon_drop']:.2f}%\n")
        f.write(f"probe_agreement        {res['probe_agreement']:.2f}%\n")
        f.write(f"recon_agreement        {res['recon_agreement']:.2f}%\n")
        f.write(f"relerr_normalized      "
                f"{res['avg_relerr_normalized']:.4f}\n")
        f.write(f"relerr_denorm          {res['avg_relerr_denorm']:.4f}\n")
    print(f"Report saved to {args.output_file}")

    print(f"\nNext steps:")
    print(f"  1. Sweep --probe_input (codes / recon) to separate "
          f"'codes carry signal' from 'MLP can read it'.")
    print(f"  2. Sweep anchor D (rebuild with csae_svd_anchor.py) and "
          f"re-run -- reconstruction drop vs D is the key curve.")
    print(f"  3. Compare reconstruction drop here against CSAE's ~68% "
          f"(check_drop_csae_fixed.py): the gap is what CSAE's "
          f"overcomplete free encoder buys over a pure linear anchor.")


if __name__ == "__main__":
    main()