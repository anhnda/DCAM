"""
check_drop_dcam_fixed.py
========================
Corrected accuracy-drop evaluation for DCAM, with extensive diagnostics.

WHAT WAS WRONG IN check_drop_dcam.py
------------------------------------
The eval-time normalization did not match training-time normalization, so
DCAM was fed out-of-distribution input and accuracy collapsed to near-chance
for reasons unrelated to the model's actual quality.

Two concrete mismatches:

1. GRANULARITY. Training (run_dcam_full.ActivationExtractor._norm_chunk)
   computes ONE scale per channel from the 0.99-quantile of that channel's
   positive values pooled over an ENTIRE 100-image chunk. The old eval
   computed a SEPARATE scale per image (the [B,C] scale_factors). So a test
   image was normalized by its own statistics, not the population's.

2. QUANTILE MATH. Training uses torch.quantile (linear interpolation between
   order statistics). The old eval used a hand-rolled sorted-gather with a
   floored index ((n_pos-1)*0.99). Not identical even per-image.

WHAT THIS SCRIPT DOES
---------------------
* Normalization is now done with the SAME torch.quantile(positive, 0.99)
  used by training's _norm_chunk. By default it is applied per-image
  (`--norm_mode per_image`), which is the honest eval-time analogue of a
  per-chunk population statistic when each test image is processed alone.
  `--norm_mode per_batch` pools the quantile over the whole eval batch,
  which is closer to the 100-image training chunk; use it to check whether
  granularity is what moved the number.
* DEBUG OUTPUT. For the first --debug_batches batches it prints, per stage:
  activation ranges, normalized ranges, scale-factor stats, the rel_err of
  A_hat vs A in BOTH normalized and denormalized space, concept activity,
  and how often the reconstructed argmax matches the original argmax.
  This is what tells you whether a low accuracy is a normalization artifact
  (normalized-space rel_err small, denormalized large -> scale bug) or a
  genuine model failure (normalized-space rel_err already large).

This script does not change run_dcam_full.py or the DCAM model.

Usage
-----
    python check_drop_dcam_fixed.py \\
        --dcam_model imagenet1k_dcam_resnet50_D512_seed42-42_result.pkl \\
        --norm_mode per_image --debug_batches 3
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
from misc.run_dcam_full import DCAM, project_rows_to_simplex
from full_classes import IMAGENET2012_CLASSES

import pandas as pd
import random


MODEL_CONFIGS = {
    'resnet50': {'model_fn': lambda: models.resnet50(pretrained=True),
                 'default_target_layer': 'layer3', 'default_D': 512,
                 'description': 'ResNet50 (layer3: 1024ch, 14x14)'},
    'resnet18': {'model_fn': lambda: models.resnet18(pretrained=True),
                 'default_target_layer': 'layer3', 'default_D': 128,
                 'description': 'ResNet18 (layer3: 256ch, 14x14)'},
    'vgg16': {'model_fn': lambda: models.vgg16(pretrained=True),
              'default_target_layer': 'features[16]', 'default_D': 128,
              'description': 'VGG16 (features[16]: 256ch, 28x28)'},
    'efficientnet': {'model_fn': lambda: models.efficientnet_b0(pretrained=True),
                     'default_target_layer': 'features[4]', 'default_D': 40,
                     'description': 'EfficientNet-B0 (features[4]: ~80ch)'},
}


# ==========================================
# Test dataset
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
# DCAM checkpoint loader
# ==========================================

def load_dcam_from_result(result_path, device):
    print(f"Loading DCAM checkpoint: {result_path}")
    blob = joblib.load(result_path)
    if not isinstance(blob, dict) or 'Pi' not in blob or 'config' not in blob:
        raise ValueError(f"{result_path} is not a DCAM result file.")
    Pi = blob['Pi']
    cfg = blob['config']
    C, D = cfg['C'], cfg['D']
    top_k = cfg.get('top_k', min(32, D))
    ridge = cfg.get('ridge', 1e-4)
    if Pi.shape != (D, C):
        raise ValueError(f"Pi shape {tuple(Pi.shape)} != (D,C)=({D},{C})")
    module = DCAM(in_channels=C, num_concepts=D, top_k=top_k, ridge=ridge).to(device)
    with torch.no_grad():
        module.Pi.data.copy_(project_rows_to_simplex(Pi.clone().to(device)))
    module.eval()

    # ---- DEBUG: report the trained Pi's health -------------------------
    with torch.no_grad():
        Pi_t = module.Pi.data
        row_sums = Pi_t.sum(dim=1)
        sv = torch.linalg.svdvals(Pi_t)
        gram = Pi_t @ Pi_t.t()
        gram = gram + ridge * torch.eye(D, device=device, dtype=Pi_t.dtype)
        cond = float((torch.linalg.svdvals(gram)[0]
                      / torch.linalg.svdvals(gram)[-1]).item())
    print(f"  C={C}  D={D}  top_k={top_k}  ridge={ridge}")
    print(f"  Trained on backbone={cfg['model']} layer={cfg['target_layer']}")
    print(f"  [debug] Pi row-sum range: [{row_sums.min():.4f}, "
          f"{row_sums.max():.4f}]  (simplex => should be ~1.0)")
    print(f"  [debug] Pi sigma_min={sv.min():.4e}  sigma_max={sv.max():.4e}")
    print(f"  [debug] tied-decoder Gram condition number = {cond:.3e}")
    if cond > 1e6:
        print(f"  [debug] WARNING: Gram is very ill-conditioned -- the tied "
              f"pinv decoder will amplify noise heavily.")
    print(f"  anchor nu={cfg.get('nu', float('nan')):.4e}  "
          f"seeds data={cfg.get('data_seed','?')} "
          f"model={cfg.get('model_seed','?')}")
    return module, cfg


# ==========================================
# Normalization -- matches run_dcam_full._norm_chunk
# ==========================================

def normalize_like_training(acts: torch.Tensor, mode: str = "per_image"
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel 0.99-quantile clip+rescale, using torch.quantile exactly
    as run_dcam_full.ActivationExtractor._norm_chunk does.

    _norm_chunk, per channel c:
        nz = {A[c,i,j] over the chunk : A[c,i,j] > 1e-8}
        sf = torch.quantile(nz, 0.99)
        A_norm[c] = clamp(A[c], 0, sf) / (sf + 1e-8)        if sf > 1e-8
        A_norm[c] = A[c]                                    otherwise

    mode:
      'per_image' -- sf computed from each image's own positive values.
                     The honest eval-time analogue when test images are
                     processed independently.
      'per_batch' -- sf computed from positive values pooled over the whole
                     batch. Closer to training's 100-image chunk; use it to
                     test whether normalization granularity moved the result.

    Returns (A_norm, scale_factors) where scale_factors is broadcastable to
    A for the later denormalize step:
      per_image -> [B, C, 1, 1];  per_batch -> [1, C, 1, 1].
    """
    B, C, H, W = acts.shape
    a = torch.clamp(acts, min=0.0)

    if mode == "per_batch":
        sf = torch.ones(C, device=acts.device, dtype=acts.dtype)
        for c in range(C):
            col = a[:, c].flatten()
            nz = col[col > 1e-8]
            if nz.numel() > 0:
                q = torch.quantile(nz, 0.99)
                if q > 1e-8:
                    sf[c] = q
        sf_b = sf.view(1, C, 1, 1)
        normalized = torch.minimum(a, sf_b) / (sf_b + 1e-8)
        # channels with no positive values: leave raw (sf stayed 1.0, but
        # _norm_chunk leaves them untouched -> replicate that)
        return normalized, sf_b

    # per_image
    sf = torch.ones(B, C, device=acts.device, dtype=acts.dtype)
    for b in range(B):
        for c in range(C):
            col = a[b, c].flatten()
            nz = col[col > 1e-8]
            if nz.numel() > 0:
                q = torch.quantile(nz, 0.99)
                if q > 1e-8:
                    sf[b, c] = q
    sf_b = sf.view(B, C, 1, 1)
    normalized = torch.minimum(a, sf_b) / (sf_b + 1e-8)
    return normalized, sf_b


# ==========================================
# Backbone + DCAM substitution
# ==========================================

class DCAMReconstructionPipeline:
    def __init__(self, model_name, target_layer_name, dcam_module,
                 device='cuda', norm_mode='per_image'):
        self.model_name = model_name
        self.target_layer_name = target_layer_name
        self.device = device
        self.norm_mode = norm_mode
        self.model = MODEL_CONFIGS[model_name]['model_fn']().to(device).eval()
        self.dcam = dcam_module.to(device).eval()
        print(f"Model: {MODEL_CONFIGS[model_name]['description']}")
        print(f"Target layer: {target_layer_name}  | norm_mode: {norm_mode}")
        print(f"DCAM: C={self.dcam.C} -> D={self.dcam.D}, top_k={self.dcam.top_k}")

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

            A_hat_norm, z = self.dcam(A_norm, use_topk=True)
            A_hat = A_hat_norm * sf                       # denormalize

            logits = self._forward_from(A_hat)

            # --- metrics, in BOTH spaces ---
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
                print(f"   concept activity = {sparsity*100:.2f}%  "
                      f"(expect ~ top_k/D = "
                      f"{self.dcam.top_k}/{self.dcam.D} = "
                      f"{100*self.dcam.top_k/self.dcam.D:.2f}%)")
                if relerr_norm < 0.3 and relerr_denorm > 0.7:
                    print("   [debug] DIAGNOSIS: model reconstructs well in "
                          "normalized space but denormalized rel_err is "
                          "large -> the SCALE/denormalization is the bug, "
                          "not the model.")
                elif relerr_norm > 0.7:
                    print("   [debug] DIAGNOSIS: rel_err is already large in "
                          "NORMALIZED space -> the DCAM model itself cannot "
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
    agree = 0                       # recon argmax == original argmax
    mse_l, rel_l, reln_l, sp_l = [], [], [], []
    per_class_total = defaultdict(int)
    per_class_correct_o = defaultdict(int)
    per_class_correct_r = defaultdict(int)

    print("\nEvaluating: original vs DCAM-reconstructed")
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


def print_results(results, model_name, dcam_cfg, norm_mode):
    print("\n" + "=" * 80)
    print(f"DCAM ACCURACY DROP  ({model_name.upper()} - ImageNet-1k)")
    print("=" * 80)
    print(f"\nDCAM config:  C={dcam_cfg['C']}  D={dcam_cfg['D']}  "
          f"top_k={dcam_cfg.get('top_k','?')}  "
          f"ridge={dcam_cfg.get('ridge','?')}")
    print(f"norm_mode: {norm_mode}")
    print(f"\nTop-1 Accuracy:")
    print(f"  Original:               {results['acc_original']:.2f}%")
    print(f"  Reconstructed (DCAM):   {results['acc_reconstructed']:.2f}%")
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
    print(f"  Avg concept activity:             "
          f"{results['avg_sparsity']*100:.2f}%")

    print(f"\n  INTERPRETATION:")
    rn = results['avg_relative_error_normalized']
    rd = results['avg_relative_error']
    if rn < 0.3 and rd > 0.7:
        print("   Normalized-space rel_err is small but denormalized is "
              "large: the DENORMALIZATION/scale handling is the dominant "
              "problem, not the DCAM model. Try --norm_mode per_batch.")
    elif rn > 0.7:
        print("   rel_err is large even in NORMALIZED space: the DCAM model "
              "genuinely cannot reconstruct layer3. This is an architecture "
              "result, not a pipeline artifact.")
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
        description='DCAM accuracy-drop eval (fixed normalization + debug)')
    ap.add_argument('--model', type=str, default='resnet50',
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--target_layer', type=str, default=None)
    ap.add_argument('--dcam_model', type=str, default=None)
    ap.add_argument('--data_seed', type=int, default=42)
    ap.add_argument('--model_seed', type=int, default=42)
    ap.add_argument('--D', type=int, default=None)
    ap.add_argument('--norm_mode', type=str, default='per_image',
                    choices=['per_image', 'per_batch'],
                    help="'per_image': each test image normalized on its own "
                         "0.99-quantile (honest eval analogue). 'per_batch': "
                         "quantile pooled over the batch (closer to training's "
                         "100-image chunk). Run BOTH to see if normalization "
                         "granularity moved the result.")
    ap.add_argument('--debug_batches', type=int, default=3,
                    help='Print stage-by-stage debug for the first N batches.')
    ap.add_argument('--raw_data_dir', type=str, default='/data/imagenet_raw/data')
    ap.add_argument('--test_data_dir', type=str,
                    default='/data/imagenet1k_sampletest')
    ap.add_argument('--test_images_per_class', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--force_resample', action='store_true')
    ap.add_argument('--output_file', type=str, default=None)
    args = ap.parse_args()

    cfg_proto = MODEL_CONFIGS[args.model]
    target_layer = args.target_layer or cfg_proto['default_target_layer']
    D = args.D if args.D is not None else cfg_proto['default_D']

    if args.dcam_model is None:
        prefix = f"imagenet1k_dcam_{args.model}"
        if args.target_layer is not None:
            prefix += "_" + args.target_layer.replace('[', '_').replace(']', '')
        prefix += f"_D{D}_seed{args.data_seed}-{args.model_seed}"
        args.dcam_model = f"{prefix}_result.pkl"
        print(f"Auto-detected DCAM checkpoint: {args.dcam_model}")
    if args.output_file is None:
        stem = Path(args.dcam_model).stem.replace('_result', '')
        args.output_file = f"accdrop_{stem}_{args.norm_mode}.txt"

    print("=" * 80)
    print(f"DCAM Accuracy Drop -- {args.model.upper()}  (FIXED eval)")
    print("=" * 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_metadata_path = create_test_samples_if_needed(
        Path(args.raw_data_dir), Path(args.test_data_dir),
        args.test_images_per_class, args.force_resample)

    dcam_module, dcam_cfg = load_dcam_from_result(args.dcam_model, device)
    if dcam_cfg.get('target_layer') not in (None, target_layer):
        print(f"  WARNING: checkpoint layer '{dcam_cfg.get('target_layer')}' "
              f"!= '{target_layer}'.")

    pipeline = DCAMReconstructionPipeline(
        args.model, target_layer, dcam_module, device=device,
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
    print_results(results, args.model, dcam_cfg, args.norm_mode)

    with open(args.output_file, 'w') as f:
        f.write(f"DCAM accuracy drop -- {args.model} -- norm_mode={args.norm_mode}\n")
        f.write(f"checkpoint: {args.dcam_model}\n")
        f.write(f"C={dcam_cfg['C']} D={dcam_cfg['D']} "
                f"top_k={dcam_cfg.get('top_k','?')} "
                f"ridge={dcam_cfg.get('ridge','?')}\n\n")
        f.write(f"acc_original              {results['acc_original']:.2f}%\n")
        f.write(f"acc_reconstructed         {results['acc_reconstructed']:.2f}%\n")
        f.write(f"acc_drop                  {results['acc_drop']:.2f}%\n")
        f.write(f"pred_agreement            {results['agreement']:.2f}%\n")
        f.write(f"avg_mse_denorm            {results['avg_mse']:.6f}\n")
        f.write(f"avg_relerr_denorm         {results['avg_relative_error']:.4f}\n")
        f.write(f"avg_relerr_normalized     "
                f"{results['avg_relative_error_normalized']:.4f}\n")
        f.write(f"avg_concept_activity      "
                f"{results['avg_sparsity']*100:.2f}%\n")
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()