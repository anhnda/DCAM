"""
check_drop_csae_fixed.py
========================
Corrected accuracy-drop evaluation for CSAE / csae-stable, with the same
diagnostics as check_drop_dcam_fixed.py.

WHY THIS EXISTS
---------------
check_acc_drop_full.py (the previous CSAE eval) used a hand-rolled per-image
quantile via sort-and-gather. CSAE was trained on activations normalized by
ActivationExtractor._norm_chunk in run_xcsae_full.py, which uses
torch.quantile(positive, 0.99) per channel over an ENTIRE chunk (100 images).
The mismatch makes the denormalized rel_err number meaningless -- it can sit
near 1.0 even when accuracy is perfectly fine -- because the numerator and
denominator end up in incompatible scales.

This script:
  * normalizes test activations with the SAME torch.quantile(positive, 0.99)
    used at training time;
  * reports rel_err in BOTH normalized space (pure model quality, scale-
    independent) and denormalized space (what the classifier sees);
  * supports per_image (default, honest eval analogue when test images are
    processed independently) and per_batch (closer to training's 100-image
    chunk) normalization;
  * prints stage-by-stage debug for the first --debug_batches batches so
    the diagnosis is explicit, not inferred.

This is a CSAE port of check_drop_dcam_fixed.py. The normalization function
and the debug scaffolding are copied verbatim; only the model-loading and
the forward pass differ (CSAE returns reconstruction directly, no Pi-row
projection step, no Gram condition diagnostic).

Usage
-----
    python check_drop_csae_fixed.py \\
        --model resnet50 \\
        --csae_model imagenet1k_csae_stable_resnet50_subspace_la0.01_seed42-0_model.pkl \\
        --norm_mode per_image --debug_batches 3

Run both --norm_mode per_image and --norm_mode per_batch on the same
checkpoint to confirm that the normalization granularity is not the dominant
source of the accuracy drop (the diagnostic from check_drop_dcam_fixed.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import joblib
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Tuple, List
import sys
from pathlib import Path
from PIL import Image
import io
from collections import defaultdict

sys.path.append('.')
from run_xcsae_full import MultiChannelConvSAE
from full_classes import IMAGENET2012_CLASSES

import pandas as pd
import random
from csae_pca_baseline import PCAReconstructor   # noqa: F401  (needed for unpickling)

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
# Test dataset (verbatim from the dcam fixed script)
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
        min_s = min(len(v) for v in class_samples.values()) if class_samples else 0
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
# CSAE checkpoint loader
# ==========================================

def load_csae_module(csae_path, device):
    """Load a CSAE module from either:
      * a joblib-pickled MultiChannelConvSAE (the *_model.pkl produced by
        the .pth -> .pkl fixup), OR
      * a state-dict .pth file -- in which case we auto-find the matching
        *_result.pkl alongside it (replacing 'model.pth' with 'result.pkl'),
        read C/hidden_dim/top_k from its config dict, build the module, and
        load the state dict. No preprocessing step required.

    Print a few sanity checks so a wrong-shape checkpoint is caught early.
    """
    print(f"Loading CSAE checkpoint: {csae_path}")
    p = Path(csae_path)

    if p.suffix == ".pth" or p.name.endswith("_model.pth"):
        # state-dict path: find the result.pkl with the matching stem
        result_path = Path(str(p).replace("_model.pth", "_result.pkl"))
        if not result_path.exists():
            raise FileNotFoundError(
                f"Expected metadata {result_path} alongside {p}. The "
                f"result.pkl holds C/hidden_dim/top_k needed to rebuild "
                f"the module. If you have a non-standard layout, pass the "
                f"joblib-pickled module path instead.")
        print(f"  found metadata: {result_path}")
        meta = joblib.load(result_path)
        cfg = meta['config']
        module = MultiChannelConvSAE(
            in_channels=cfg['C'], hidden_dim=cfg['hidden_dim'],
            kernel_size=1, top_k=cfg['top_k'])
        module.load_state_dict(torch.load(str(p), map_location='cpu'))
    else:
        # joblib-pickled module
        obj = joblib.load(csae_path)
        if not isinstance(obj, MultiChannelConvSAE):
            raise ValueError(
                f"{csae_path} is not a MultiChannelConvSAE instance "
                f"(got {type(obj).__name__}). Pass the *_model.pth state "
                f"dict instead -- the loader will auto-find the matching "
                f"*_result.pkl and rebuild the module.")
        module = obj

    module = module.to(device).eval()
    with torch.no_grad():
        W = module.encoder.weight                         # [H, C, 1, 1]
        row_norms = W[:, :, 0, 0].norm(dim=1)
        print(f"  C={module.in_channels}  hidden_dim={module.hidden_dim}  "
              f"top_k={module.top_k}")
        print(f"  encoder row-norm range: "
              f"[{row_norms.min():.4f}, {row_norms.max():.4f}]  "
              f"mean={row_norms.mean():.4f}")
        if row_norms.min() < 1e-6:
            print("  [debug] WARNING: some encoder rows have ~zero norm -- "
                  "those units are dead.")
    return module


# ==========================================
# Normalization -- matches run_xcsae_full.ActivationExtractor._norm_chunk
# ==========================================

def _vectorized_q99_of_positives(a_flat: torch.Tensor) -> torch.Tensor:
    """0.99-quantile of the positive (>1e-8) entries along the last dim,
    fully vectorized -- no Python loop over channels.

    Mimics torch.quantile(positives, 0.99) with linear interpolation, which
    is what run_xcsae_full._norm_chunk uses (it calls torch.quantile, not a
    nearest-rank approximation, so we match that here even though it's a
    few extra ops over a hard floor).

    Input:  a_flat shape [..., N]  (non-negative; non-positives are masked out
            by the sort-with-sentinel trick before computing the percentile).
    Output: q shape [...] -- one quantile per leading-dim cell. Cells with
            no positive entries return 0 (caller should treat that as
            'leave channel untouched, set sf=1', matching _norm_chunk).
    """
    # Mask non-positives with -inf so they sort to the bottom and never enter
    # the percentile index range.
    pos = a_flat > 1e-8
    n_pos = pos.sum(dim=-1)                              # [...]
    neg_inf = torch.finfo(a_flat.dtype).min
    masked = torch.where(pos, a_flat, torch.full_like(a_flat, neg_inf))
    sorted_vals, _ = torch.sort(masked, dim=-1)          # ascending; -inf at bottom

    N = a_flat.shape[-1]
    # positives occupy indices [N - n_pos, N-1]; in those n_pos slots the
    # 0.99-quantile with linear interp lives at offset 0.99*(n_pos - 1).
    pos_float = (n_pos - 1).clamp(min=0).to(a_flat.dtype) * 0.99
    lo_off = pos_float.floor().long()
    # cap hi_off so we never index past the last positive
    hi_off = torch.minimum(lo_off + 1, (n_pos - 1).clamp(min=0))
    frac = pos_float - lo_off.to(a_flat.dtype)

    start = N - n_pos                                    # [...]
    lo_idx = (start + lo_off).clamp(max=N - 1)
    hi_idx = (start + hi_off).clamp(max=N - 1)

    lo_vals = sorted_vals.gather(-1, lo_idx.unsqueeze(-1)).squeeze(-1)
    hi_vals = sorted_vals.gather(-1, hi_idx.unsqueeze(-1)).squeeze(-1)
    q = lo_vals + frac * (hi_vals - lo_vals)

    # cells with no positives: return 0 (sentinel for "no scale")
    q = torch.where(n_pos > 0, q, torch.zeros_like(q))
    return q


def normalize_like_training(acts: torch.Tensor, mode: str = "per_image"
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel 0.99-quantile clip+rescale, matching
    run_xcsae_full.ActivationExtractor._norm_chunk.

    Training code, per channel c (over a 100-image chunk):
        nz = {A[c,i,j] : A[c,i,j] > 1e-8}
        sf = torch.quantile(nz, 0.99)
        A_norm[c] = clamp(A[c], 0, sf) / (sf + 1e-8)        if sf > 1e-8
        A_norm[c] = A[c]                                    otherwise

    mode:
      'per_image' -- sf computed from each image's own positive values
                     (honest eval-time analogue when test images are
                     processed independently).
      'per_batch' -- sf pooled over the whole batch (closer to training's
                     100-image chunk; run both to test granularity effects).

    Returns (A_norm, scale_factors) broadcastable to A:
      per_image -> [B, C, 1, 1];  per_batch -> [1, C, 1, 1].

    Fully vectorized -- one sort+gather over all (B*)C rows, no Python loop.
    """
    B, C, H, W = acts.shape
    a = torch.clamp(acts, min=0.0)

    if mode == "per_batch":
        # pool over batch+spatial: shape [C, B*H*W]
        a_flat = a.permute(1, 0, 2, 3).reshape(C, -1)
        q = _vectorized_q99_of_positives(a_flat)         # [C]
        # match _norm_chunk: if q<=1e-8, leave channel raw (sf=1, no clip)
        valid = q > 1e-8
        sf = torch.where(valid, q, torch.ones_like(q))   # [C]
        sf_b = sf.view(1, C, 1, 1)
        normalized = torch.minimum(a, sf_b) / (sf_b + 1e-8)
        # restore raw acts on invalid channels (no quantile to clip to)
        normalized = torch.where(valid.view(1, C, 1, 1),
                                 normalized, acts)
        return normalized, sf_b

    # per_image: shape [B, C, H*W]
    a_flat = a.view(B, C, H * W)
    q = _vectorized_q99_of_positives(a_flat)             # [B, C]
    valid = q > 1e-8
    sf = torch.where(valid, q, torch.ones_like(q))       # [B, C]
    sf_b = sf.view(B, C, 1, 1)
    normalized = torch.minimum(a, sf_b) / (sf_b + 1e-8)
    normalized = torch.where(valid.view(B, C, 1, 1),
                             normalized, acts)
    return normalized, sf_b


# ==========================================
# Backbone + CSAE substitution
# ==========================================

class CSAEReconstructionPipeline:
    def __init__(self, model_name, target_layer_name, csae_module,
                 device='cuda', norm_mode='per_image'):
        self.model_name = model_name
        self.target_layer_name = target_layer_name
        self.device = device
        self.norm_mode = norm_mode
        self.model = MODEL_CONFIGS[model_name]['model_fn']().to(device).eval()
        self.csae = csae_module.to(device).eval()
        print(f"Model: {MODEL_CONFIGS[model_name]['description']}")
        print(f"Target layer: {target_layer_name}  | norm_mode: {norm_mode}")
        print(f"CSAE: C={self.csae.in_channels} -> "
              f"hidden_dim={self.csae.hidden_dim}, "
              f"top_k={self.csae.top_k}")

    def _forward_to(self, x):
        m = self.model
        if self.model_name in ['resnet50', 'resnet18']:
            x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
            x = m.layer1(x)
            if 'layer1' in self.target_layer_name: return x
            x = m.layer2(x)
            if 'layer2' in self.target_layer_name: return x
            x = m.layer3(x)
            if 'layer3' in self.target_layer_name: return x
            x = m.layer4(x)
            return x
        else:
            idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            for i in range(idx + 1):
                x = m.features[i](x)
            return x

    def _forward_from(self, x):
        m = self.model
        if self.model_name in ['resnet50', 'resnet18']:
            tl = self.target_layer_name
            if 'layer1' in tl:
                x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
            elif 'layer2' in tl:
                x = m.layer3(x); x = m.layer4(x)
            elif 'layer3' in tl:
                x = m.layer4(x)
            x = m.avgpool(x); x = torch.flatten(x, 1); x = m.fc(x)
            return x
        else:
            idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            for i in range(idx + 1, len(m.features)):
                x = m.features[i](x)
            x = m.avgpool(x); x = torch.flatten(x, 1)
            x = m.classifier(x)
            return x

    def forward_original(self, x):
        with torch.no_grad():
            return self.model(x)

    def forward_with_reconstruction(self, x, debug=False):
        with torch.no_grad():
            A = self._forward_to(x)                       # [B,C,H,W] raw
            A_norm, sf = normalize_like_training(A, mode=self.norm_mode)
            A_hat_norm, z = self.csae(A_norm, use_topk=True)
            A_hat = A_hat_norm * sf                       # denormalize
            logits = self._forward_from(A_hat)

            relerr_norm = ((A_hat_norm - A_norm).abs().mean()
                           / (A_norm.abs().mean() + 1e-8)).item()
            relerr_denorm = ((A_hat - A).abs().mean()
                             / (A.abs().mean() + 1e-8)).item()
            mse_denorm = F.mse_loss(A_hat, A).item()
            sparsity = (z > 0).float().mean().item()

            stats = {'mse': mse_denorm,
                     'relative_error': relerr_denorm,
                     'relative_error_normalized': relerr_norm,
                     'sparsity': sparsity}

            if debug:
                print("\n   [debug] ---- one batch, stage by stage ----")
                print(f"   raw A      : min={A.min():.4f} max={A.max():.4f} "
                      f"mean={A.mean():.4f} absmean={A.abs().mean():.4f}")
                print(f"   A_norm     : min={A_norm.min():.4f} "
                      f"max={A_norm.max():.4f} mean={A_norm.mean():.4f}")
                print(f"   scale sf   : min={sf.min():.4f} max={sf.max():.4f} "
                      f"mean={sf.mean():.4f}")
                print(f"   A_hat_norm : min={A_hat_norm.min():.4f} "
                      f"max={A_hat_norm.max():.4f} "
                      f"mean={A_hat_norm.mean():.4f}")
                print(f"   A_hat      : min={A_hat.min():.4f} "
                      f"max={A_hat.max():.4f} mean={A_hat.mean():.4f}")
                print(f"   rel_err (normalized space)   = "
                      f"{relerr_norm:.4f}   <-- model quality, "
                      f"normalization-independent")
                print(f"   rel_err (denormalized space) = "
                      f"{relerr_denorm:.4f}   <-- what the classifier sees")
                print(f"   active units = {sparsity*100:.2f}%  "
                      f"(expect ~ top_k/hidden_dim = "
                      f"{self.csae.top_k}/{self.csae.hidden_dim} = "
                      f"{100*self.csae.top_k/self.csae.hidden_dim:.2f}%)")
                if relerr_norm < 0.3 and relerr_denorm > 0.7:
                    print("   [debug] DIAGNOSIS: model reconstructs well in "
                          "normalized space but denormalized rel_err is "
                          "large -> the SCALE/denormalization is the bug, "
                          "not the model.")
                elif relerr_norm > 0.7:
                    print("   [debug] DIAGNOSIS: rel_err is already large in "
                          "NORMALIZED space -> the CSAE model itself cannot "
                          "reconstruct. Not a normalization artifact.")
                else:
                    print("   [debug] DIAGNOSIS: intermediate -- inspect "
                          "further.")

            return logits, stats


# ==========================================
# Evaluation
# ==========================================

def evaluate(pipeline, data_loader, device, debug_batches=3):
    total = 0
    correct_o = correct_r = 0
    agree = 0
    mse_l, rel_l, reln_l, sp_l = [], [], [], []
    per_class_total = defaultdict(int)
    per_class_correct_o = defaultdict(int)
    per_class_correct_r = defaultdict(int)

    print("\nEvaluating: original vs CSAE-reconstructed")
    print("=" * 80)
    for bi, (images, labels) in enumerate(tqdm(data_loader, desc="Batches")):
        images, labels = images.to(device), labels.to(device)
        B = images.size(0)
        logits_o = pipeline.forward_original(images)
        pred_o = logits_o.argmax(1)

        do_debug = bi < debug_batches
        logits_r, stats = pipeline.forward_with_reconstruction(
            images, debug=do_debug)
        pred_r = logits_r.argmax(1)

        correct_o += (pred_o == labels).sum().item()
        correct_r += (pred_r == labels).sum().item()
        agree += (pred_o == pred_r).sum().item()

        for i in range(B):
            lbl = labels[i].item()
            per_class_total[lbl] += 1
            if pred_o[i] == labels[i]:
                per_class_correct_o[lbl] += 1
            if pred_r[i] == labels[i]:
                per_class_correct_r[lbl] += 1

        total += B
        mse_l.append(stats['mse'])
        rel_l.append(stats['relative_error'])
        reln_l.append(stats['relative_error_normalized'])
        sp_l.append(stats['sparsity'])

    acc_o = correct_o / total * 100
    acc_r = correct_r / total * 100
    per_class_drop = {}
    for lbl, n in per_class_total.items():
        if n > 0:
            per_class_drop[lbl] = (per_class_correct_o[lbl] / n * 100
                                   - per_class_correct_r[lbl] / n * 100)
    worst = sorted(per_class_drop.items(), key=lambda x: x[1],
                   reverse=True)[:10]

    return {
        'total_samples': total,
        'acc_original': acc_o, 'acc_reconstructed': acc_r,
        'acc_drop': acc_o - acc_r,
        'agreement': agree / total * 100,
        'avg_mse': float(np.mean(mse_l)),
        'avg_relative_error': float(np.mean(rel_l)),
        'avg_relative_error_normalized': float(np.mean(reln_l)),
        'avg_sparsity': float(np.mean(sp_l)),
        'per_class_total': dict(per_class_total),
        'worst_classes': worst,
    }


def print_results(results, model_name, csae_module, norm_mode):
    print("\n" + "=" * 80)
    print(f"CSAE ACCURACY DROP  ({model_name.upper()} - ImageNet-1k)")
    print("=" * 80)
    print(f"\nCSAE config:  C={csae_module.in_channels}  "
          f"hidden_dim={csae_module.hidden_dim}  "
          f"top_k={csae_module.top_k}")
    print(f"norm_mode: {norm_mode}")
    print(f"\nTop-1 Accuracy:")
    print(f"  Original:               {results['acc_original']:.2f}%")
    print(f"  Reconstructed (CSAE):   {results['acc_reconstructed']:.2f}%")
    print(f"  Accuracy Drop:          {results['acc_drop']:.2f}%")
    if results['acc_original'] > 0:
        print(f"  Relative Drop:          "
              f"{results['acc_drop']/results['acc_original']*100:.2f}%")
    print(f"  Pred agreement (recon==orig argmax): "
          f"{results['agreement']:.2f}%")

    print(f"\nReconstruction Quality:")
    print(f"  Avg MSE (denormalized):           {results['avg_mse']:.6f}")
    print(f"  Avg rel_err (denormalized):       "
          f"{results['avg_relative_error']:.4f}   <-- classifier sees this")
    print(f"  Avg rel_err (NORMALIZED space):   "
          f"{results['avg_relative_error_normalized']:.4f}   <-- pure "
          f"model quality")
    print(f"  Avg active units:                 "
          f"{results['avg_sparsity']*100:.2f}%")

    print(f"\n  INTERPRETATION:")
    rn = results['avg_relative_error_normalized']
    rd = results['avg_relative_error']
    if rn < 0.3 and rd > 0.7:
        print("   Normalized-space rel_err is small but denormalized is "
              "large: the DENORMALIZATION/scale handling is the dominant "
              "source of the gap. Try --norm_mode per_batch.")
    elif rn > 0.7:
        print("   rel_err is large even in NORMALIZED space: the CSAE model "
              "genuinely cannot reconstruct the activations well. Not a "
              "pipeline artifact.")
    elif rn < 0.5 and results['acc_drop'] > 50:
        print("   Model reconstructs moderately well yet accuracy collapses: "
              "the classifier is highly sensitive to the residual error -- "
              "inspect per-class drops and logit agreement.")
    else:
        print("   Intermediate regime -- read the per-batch debug above.")

    if results['worst_classes']:
        print(f"\n  Top classes by accuracy drop:")
        for i, (lbl, drop) in enumerate(results['worst_classes'][:5], 1):
            name = list(IMAGENET2012_CLASSES.values())[lbl]
            print(f"   {i}. class {lbl} ({name[:40]}): {drop:+.1f}%")
    print("=" * 80 + "\n")


# ==========================================
# Main
# ==========================================

def main():
    ap = argparse.ArgumentParser(
        description='CSAE accuracy-drop eval (fixed normalization + debug)')
    ap.add_argument('--model', type=str, default='resnet50',
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--target_layer', type=str, default=None)
    ap.add_argument('--csae_model', type=str, required=True,
                    help='Path to a joblib-pickled MultiChannelConvSAE '
                         '(the *_model.pkl produced by the .pth -> .pkl '
                         'fixup, NOT the *_result.pkl metadata blob).')
    ap.add_argument('--norm_mode', type=str, default='per_image',
                    choices=['per_image', 'per_batch'],
                    help="'per_image': each test image normalized on its "
                         "own 0.99-quantile (honest eval analogue). "
                         "'per_batch': quantile pooled over the batch "
                         "(closer to training's 100-image chunk). Run BOTH "
                         "to test whether normalization granularity moved "
                         "the result.")
    ap.add_argument('--debug_batches', type=int, default=3,
                    help='Print stage-by-stage debug for the first N '
                         'batches.')
    ap.add_argument('--raw_data_dir', type=str,
                    default='/data/imagenet_raw/data')
    ap.add_argument('--test_data_dir', type=str,
                    default='/data/imagenet1k_sampletest')
    ap.add_argument('--test_images_per_class', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--force_resample', action='store_true')
    ap.add_argument('--output_file', type=str, default=None)
    args = ap.parse_args()

    cfg_proto = MODEL_CONFIGS[args.model]
    target_layer = args.target_layer or cfg_proto['default_target_layer']

    if args.output_file is None:
        stem = Path(args.csae_model).stem.replace('_model', '')
        args.output_file = f"accdrop_{stem}_{args.norm_mode}.txt"

    print("=" * 80)
    print(f"CSAE Accuracy Drop -- {args.model.upper()}  (FIXED eval)")
    print("=" * 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_metadata_path = create_test_samples_if_needed(
        Path(args.raw_data_dir), Path(args.test_data_dir),
        args.test_images_per_class, args.force_resample)

    csae_module = load_csae_module(args.csae_model, device)
    if csae_module.in_channels not in (None,):
        # cross-check vs the chosen backbone's layer C
        # (we can't easily look up the backbone's C without running it once;
        # if the dims mismatch, the forward pass will raise at the conv,
        # which is clearer than a guess here. Leave as-is.)
        pass

    pipeline = CSAEReconstructionPipeline(
        args.model, target_layer, csae_module, device=device,
        norm_mode=args.norm_mode)

    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    dataset = ImageNet1kTestDataset(test_metadata_path, transform=tfm)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    results = evaluate(pipeline, data_loader, device,
                       debug_batches=args.debug_batches)
    print_results(results, args.model, csae_module, args.norm_mode)

    with open(args.output_file, 'w') as f:
        f.write(f"CSAE accuracy drop -- {args.model} -- "
                f"norm_mode={args.norm_mode}\n")
        f.write(f"checkpoint: {args.csae_model}\n")
        f.write(f"C={csae_module.in_channels} "
                f"hidden_dim={csae_module.hidden_dim} "
                f"top_k={csae_module.top_k}\n\n")
        f.write(f"acc_original              {results['acc_original']:.2f}%\n")
        f.write(f"acc_reconstructed         "
                f"{results['acc_reconstructed']:.2f}%\n")
        f.write(f"acc_drop                  {results['acc_drop']:.2f}%\n")
        f.write(f"pred_agreement            {results['agreement']:.2f}%\n")
        f.write(f"avg_mse_denorm            {results['avg_mse']:.6f}\n")
        f.write(f"avg_relerr_denorm         "
                f"{results['avg_relative_error']:.4f}\n")
        f.write(f"avg_relerr_normalized     "
                f"{results['avg_relative_error_normalized']:.4f}\n")
        f.write(f"avg_active_units          "
                f"{results['avg_sparsity']*100:.2f}%\n")
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()