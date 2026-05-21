"""
DCAM Training Script on Full ImageNet-1k
========================================
Decomposed Class Activation Maps via a single soft channel-membership matrix.

This REPLACES the old ConvSAE script. The method, per the corrected paper:

  * The only learned object is a membership matrix  Pi in R^{D x C}, D <= C,
    non-negative, with each row on the L1-simplex (sum_k Pi[d,k] = 1).
    Pi[d,k] = soft, overlapping participation of backbone channel k in concept d.

  * Encoder (channels -> concepts), per spatial cell:
        z'_d = ReLU( sum_k Pi[d,k] A_k )          z' in R^{D x H x W}
        z    = TopK(z', K)                        sparse over concepts

  * Decoder (concepts -> channels) is TIED: the Moore-Penrose pseudo-inverse
    of the SAME Pi (no free decoder weights):
        Pi_dagger = Pi^T (Pi Pi^T)^{-1}  in R^{C x D}   (valid because D <= C)
        A_hat     = Pi_dagger z

  * Loss has THREE terms only:
        L_total = L_recon  +  lambda_A * L_anchor  +  lambda_1 * L1
    - L_recon  : masked MSE on Grad-CAM-selected channels  (faithfulness)
    - L_anchor : ||Pi - Pi0||_F^2   (seed-stability; Pi0 is the seed-free anchor)
    - L1       : ||z||_1 / (D H W)  (concept sparsity)
    There is NO Grad-CAM-sum loss, NO lateral inhibition, NO compactness,
    NO channel-sparsity term. Grad-CAM recovery is a consequence, not a loss.

  * Anchor Pi0: built ONCE, seed-free, by symmetric NMF of the channel
    co-activation matrix S (Eq. coact), followed by the non-degeneracy check
    (Eq. nondegeneracy) which lower-bounds nu = min_{d!=d'} ||Pi0_d - Pi0_d'||
    and merges near-duplicate rows (reducing D).

  * After every optimizer step, the rows of Pi are projected back onto the
    non-negative L1-simplex.

Extraction stage:
  Unchanged in spirit from the old script -- it still produces, per image,
  the tuple (A, M_tau, L_GC). L_GC (the Grad-CAM spatial map) is kept for
  EVALUATION ONLY (faithfulness / Grad-CAM-recovery); it is no longer a
  training target. New cache key suffix "dcam_pi_v1" so old caches are ignored.

Reproducibility:
  --data_seed  : dataset sampling.
  --model_seed : Pi initialization + training stochasticity (shuffling).
  The anchor Pi0 is independent of model_seed by construction; this is what
  makes the learned atoms reproducible across runs (see paper, Thm. seed-stability).

Usage:
    python run_dcam_full.py
    python run_dcam_full.py --model resnet18
    python run_dcam_full.py --D 512 --top_k 32 --lambda_anchor 1.0
    python run_dcam_full.py --model_seed 1     # second seed, to test stability
    python run_dcam_full.py --force_reextract
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
from src.gradcam import GradCAM
from full_classes import IMAGENET2012_CLASSES

# ==========================================
# Configuration
# ==========================================

IMAGENET_RAW_DIR = Path("/data/imagenet_raw/data")
IMAGENET_SAMPLED_DIR = Path("/data/imagenet1k_sampled")
ACTIVATION_CACHE_DIR = Path("cache_activations")

IMAGES_PER_CLASS = 50
NUM_CLASSES = 1000

ACTIVATION_CHUNK_SIZE = 100
BATCH_SIZE_COLLECTION = 32

DEFAULT_DATA_SEED = 42
DEFAULT_MODEL_SEED = 42

# Cache version tag. DCAM caches (A, M_tau, L_GC). Distinct from the old
# "gcmap1" tag so old caches are never silently reused.
CACHE_TAG = "dcam_pi_v1"

MODEL_CONFIGS = {
    'resnet50': {
        'model_fn': lambda: models.resnet50(pretrained=True),
        'default_target_layer': 'layer3',
        'valid_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'description': 'ResNet50 (layer3: 1024ch, 14x14)',
        'default_D': 512,          # D <= C ; C = 1024 -> D = C/2
    },
    'resnet18': {
        'model_fn': lambda: models.resnet18(pretrained=True),
        'default_target_layer': 'layer3',
        'valid_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'description': 'ResNet18 (layer3: 256ch, 14x14)',
        'default_D': 128,          # C = 256 -> D = C/2
    },
    'vgg16': {
        'model_fn': lambda: models.vgg16(pretrained=True),
        'default_target_layer': 'features[16]',
        'valid_layers': ['features[10]', 'features[16]', 'features[23]', 'features[30]'],
        'description': 'VGG16 (features[16]: 256ch, 28x28)',
        'default_D': 128,          # C = 256 -> D = C/2
    },
    'efficientnet': {
        'model_fn': lambda: models.efficientnet_b0(pretrained=True),
        'default_target_layer': 'features[4]',
        'valid_layers': ['features[2]', 'features[3]', 'features[4]', 'features[5]'],
        'description': 'EfficientNet-B0 (features[4]: ~80ch, 14x14)',
        'default_D': 40,           # C = 80 -> D = C/2
    },
}


# ==========================================
# Reproducibility
# ==========================================

def set_seed(seed: int, deterministic: bool = True):
    """Seed Python / NumPy / PyTorch RNGs (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"  [seed] RNGs seeded with {seed} "
          f"(cudnn deterministic={'on' if deterministic else 'off'})")


# ==========================================
# Simplex projection (Euclidean projection onto the L1-simplex)
# ==========================================

def project_rows_to_simplex(M: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    """Project each ROW of M onto the simplex {x >= 0, sum x = z}.

    Implements the standard sorted-threshold algorithm
    (Wang & Carreira-Perpinan, 2013), applied row-wise. Used to keep the
    membership matrix Pi non-negative with unit-L1 rows after each step.

    Args:
        M: [D, C] tensor (rows to be projected).
        z: target row sum (1.0 for the probability simplex).
    Returns:
        [D, C] tensor with each row on the simplex.
    """
    D, C = M.shape
    # Sort each row descending.
    u, _ = torch.sort(M, dim=1, descending=True)
    css = torch.cumsum(u, dim=1)
    # rho_d = largest index j (1-based) with  u[d,j] + (z - css[d,j]) / j > 0
    idx = torch.arange(1, C + 1, device=M.device, dtype=M.dtype).unsqueeze(0)  # [1, C]
    cond = u + (z - css) / idx > 0
    rho = cond.float().sum(dim=1).clamp(min=1.0)                               # [D]
    # theta_d = (css[d, rho_d] - z) / rho_d
    rho_idx = (rho.long() - 1).clamp(min=0)
    css_at_rho = css.gather(1, rho_idx.unsqueeze(1)).squeeze(1)                 # [D]
    theta = (css_at_rho - z) / rho                                             # [D]
    out = torch.clamp(M - theta.unsqueeze(1), min=0.0)
    return out


# ==========================================
# DCAM model: a single membership matrix Pi with a tied pseudo-inverse decoder
# ==========================================

class DCAM(nn.Module):
    """Decomposed Class Activation Maps.

    The single learnable parameter is the membership matrix
        Pi  in  R^{D x C},   D <= C,   non-negative,   rows on the L1-simplex.

    Forward path (per spatial cell, vectorized over H, W and batch):
        z'  = ReLU(Pi @ A)            encode    C -> D
        z   = TopK(z', K)             sparsify  over the D concepts
        Ahat = Pi_dagger @ z          decode    D -> C   (TIED decoder)

    where  Pi_dagger = Pi^T (Pi Pi^T)^{-1}  is recomputed from Pi every call.
    There is no independent decoder weight.
    """

    def __init__(self, in_channels: int, num_concepts: int, top_k: int,
                 ridge: float = 1e-4):
        super().__init__()
        assert num_concepts <= in_channels, \
            f"DCAM requires D <= C (compressive regime); got D={num_concepts}, C={in_channels}"
        self.C = in_channels        # backbone channel count
        self.D = num_concepts       # atom-vocabulary size
        self.top_k = top_k
        self.ridge = ridge          # damping for (Pi Pi^T)^{-1} conditioning

        # The one and only parameter. Initialized later from the anchor Pi0
        # via init_from_anchor(); a fallback random init is set here so the
        # module is valid even before anchoring.
        self.Pi = nn.Parameter(torch.empty(self.D, self.C))
        with torch.no_grad():
            self.Pi.uniform_(0.0, 1.0)
            self.Pi.data = project_rows_to_simplex(self.Pi.data)

    # ---- initialization -------------------------------------------------
    def init_from_anchor(self, Pi0: torch.Tensor):
        """Initialize Pi at the seed-free anchor Pi0 (paper: deterministic init)."""
        assert Pi0.shape == (self.D, self.C), \
            f"anchor shape {tuple(Pi0.shape)} != Pi shape {(self.D, self.C)}"
        with torch.no_grad():
            self.Pi.data = project_rows_to_simplex(Pi0.clone().to(self.Pi.device))

    # ---- pseudo-inverse decoder ----------------------------------------
    def pinv(self) -> torch.Tensor:
        """Tied decoder  Pi_dagger = Pi^T (Pi Pi^T + ridge I)^{-1}  in R^{C x D}.

        The ridge term is a small damping that keeps the D x D inverse
        well-conditioned even if two rows drift close together during a step
        (the non-degeneracy check keeps it genuinely small / mostly inactive).
        """
        Pi = self.Pi                                  # [D, C]
        gram = Pi @ Pi.t()                            # [D, D]
        gram = gram + self.ridge * torch.eye(
            self.D, device=Pi.device, dtype=Pi.dtype)
        # solve  gram X = Pi  for X = (Pi Pi^T)^{-1} Pi  -> [D, C]
        X = torch.linalg.solve(gram, Pi)              # [D, C]
        return X.t()                                  # Pi_dagger : [C, D]

    # ---- top-k over concepts -------------------------------------------
    def topk_over_concepts(self, zp: torch.Tensor) -> torch.Tensor:
        """Keep, per image, the K concepts with the highest spatial mass.

        Args:
            zp: [B, D, H, W]  pre-sparsified concept maps (post-ReLU).
        Returns:
            [B, D, H, W]  with all but the top-K concept channels zeroed.
        """
        B, D, H, W = zp.shape
        score = zp.sum(dim=(2, 3))                    # [B, D] concept importance
        k = min(self.top_k, D)
        _, idx = torch.topk(score, k=k, dim=1)        # [B, k]
        mask = torch.zeros(B, D, device=zp.device, dtype=zp.dtype)
        mask.scatter_(1, idx, 1.0)
        return zp * mask.unsqueeze(-1).unsqueeze(-1)

    # ---- forward --------------------------------------------------------
    def forward(self, A: torch.Tensor, use_topk: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            A: [B, C, H, W] backbone activations.
        Returns:
            A_hat: [B, C, H, W] reconstruction  Pi_dagger z
            z:     [B, D, H, W] sparse concept maps (post-ReLU, post-TopK)
        """
        B, C, H, W = A.shape
        assert C == self.C, f"input has {C} channels, Pi expects {self.C}"

        # encode: z' = ReLU(Pi A), applied as a 1x1 conv with weight Pi
        # einsum over channels keeps it explicit: [D,C] x [B,C,H,W] -> [B,D,H,W]
        zp = torch.einsum('dc,bchw->bdhw', self.Pi, A)
        zp = F.relu(zp)

        z = self.topk_over_concepts(zp) if use_topk else zp

        # decode: A_hat = Pi_dagger z   ([C,D] x [B,D,H,W] -> [B,C,H,W])
        Pi_dag = self.pinv()
        A_hat = torch.einsum('cd,bdhw->bchw', Pi_dag, z)
        return A_hat, z

    # ---- projection (call after each optimizer step) -------------------
    @torch.no_grad()
    def project(self):
        """Project Pi's rows back onto the non-negative L1-simplex."""
        self.Pi.data = project_rows_to_simplex(self.Pi.data)


# ==========================================
# Anchor construction: Pi0 from the channel co-activation matrix
# ==========================================

def compute_coactivation_matrix(activation_chunks: List[torch.Tensor]
                                ) -> torch.Tensor:
    """Channel co-activation matrix S (Eq. coact).

        S[k,k'] = E_x [ Abar_k(x) Abar_k'(x) ],   Abar_k = mean_{i,j} A[k,i,j]

    Computed from the cached activations -- no backbone needed.

    Args:
        activation_chunks: list of [n, C, H, W] tensors.
    Returns:
        S: [C, C] symmetric PSD co-activation matrix.
    """
    C = activation_chunks[0].shape[1]
    S = torch.zeros(C, C, dtype=torch.float64)
    n_total = 0
    for chunk in tqdm(activation_chunks, desc="Co-activation matrix"):
        # spatial mean -> [n, C]
        abar = chunk.mean(dim=(2, 3)).double()
        S += abar.t() @ abar
        n_total += abar.shape[0]
    S /= max(n_total, 1)
    return S.float()


def build_anchor(S: torch.Tensor, D: int, nmf_iters: int = 500,
                 merge_tol: float = 1e-2, seed: int = 0
                 ) -> Tuple[torch.Tensor, float, int]:
    """Build the seed-free anchor Pi0 by symmetric NMF of S, then run the
    non-degeneracy check (Eq. nondegeneracy) and merge near-duplicate rows.

    Symmetric NMF:  S ~ W W^T,  W >= 0,  W in R^{C x D}.
    The anchor atom d is column d of W, transposed to a row over channels and
    L1-normalized:  Pi0[d, :] = normalize(W[:, d]).

    NOTE on determinism: the NMF is seeded with a FIXED seed (default 0),
    independent of --model_seed, so Pi0 is identical for every training run.
    This is the property the stability theorem relies on.

    Args:
        S: [C, C] co-activation matrix.
        D: requested number of anchor atoms (<= C).
        nmf_iters: multiplicative-update iterations.
        merge_tol: rows closer than this (L2) are merged.
        seed: FIXED rng seed for the NMF init (NOT the model seed).
    Returns:
        Pi0:   [D_eff, C] anchor, rows on the L1-simplex, D_eff <= D after merge.
        nu:    non-degeneracy constant min_{d!=d'} ||Pi0_d - Pi0_d'||_2.
        D_eff: number of atoms after merging.
    """
    C = S.shape[0]
    assert D <= C, f"anchor needs D <= C; got D={D}, C={C}"
    g = torch.Generator().manual_seed(seed)

    # --- symmetric NMF  S ~ W W^T  via multiplicative updates ------------
    W = torch.rand(C, D, generator=g).clamp(min=1e-4)
    S = S.clamp(min=0.0)  # S is PSD; clamp tiny negatives from numerics
    for it in range(nmf_iters):
        SW = S @ W                                   # [C, D]
        WWtW = W @ (W.t() @ W)                       # [C, D]
        W = W * (SW / (WWtW + 1e-9))
        W = W.clamp(min=1e-9)
    if (it + 1) % 100 == 0 or it == nmf_iters - 1:
        recon_err = torch.norm(S - W @ W.t()) / (torch.norm(S) + 1e-9)
        print(f"  [anchor NMF] iter {it+1}/{nmf_iters}  "
              f"rel. recon err = {recon_err:.4f}")

    # --- atoms = columns of W, L1-normalized as rows over channels -------
    Pi0 = W.t().clone()                              # [D, C]
    Pi0 = project_rows_to_simplex(Pi0)

    # --- non-degeneracy check + merge (Eq. nondegeneracy) ----------------
    keep = list(range(D))
    merged = True
    while merged:
        merged = False
        for a_i in range(len(keep)):
            for b_i in range(a_i + 1, len(keep)):
                da, db = keep[a_i], keep[b_i]
                dist = torch.norm(Pi0[da] - Pi0[db]).item()
                if dist < merge_tol:
                    # merge db into da (average, re-project), drop db
                    Pi0[da] = project_rows_to_simplex(
                        ((Pi0[da] + Pi0[db]) * 0.5).unsqueeze(0)).squeeze(0)
                    keep.pop(b_i)
                    merged = True
                    break
            if merged:
                break
    Pi0 = Pi0[keep]                                  # [D_eff, C]
    D_eff = Pi0.shape[0]

    # final non-degeneracy constant nu
    if D_eff >= 2:
        dmat = torch.cdist(Pi0, Pi0)
        dmat = dmat + torch.eye(D_eff) * 1e9         # ignore the diagonal
        nu = dmat.min().item()
    else:
        nu = float('inf')

    print(f"  [anchor] requested D={D}, after merge D_eff={D_eff}, "
          f"non-degeneracy nu={nu:.4e}")
    if D_eff < D:
        print(f"  [anchor] {D - D_eff} near-duplicate atom(s) merged "
              f"(merge_tol={merge_tol}).")
    return Pi0, nu, D_eff


# ==========================================
# Losses
# ==========================================

def masked_reconstruction_loss(A_hat: torch.Tensor, A: torch.Tensor,
                               masks: torch.Tensor) -> torch.Tensor:
    """Masked MSE on the Grad-CAM-selected channels (paper Eq. recon-loss).

        L_recon = (1 / (|M| H W)) sum_{c in M} sum_{i,j} (A - A_hat)^2

    Args:
        A_hat: [B, C, H, W] reconstruction.
        A:     [B, C, H, W] target activations.
        masks: [B, C] bool/float, the Grad-CAM channel mask M_tau per image.
    """
    m = masks.unsqueeze(-1).unsqueeze(-1).float()    # [B, C, 1, 1]
    sq = (A_hat - A) ** 2 * m                        # [B, C, H, W]
    denom = masks.sum(dim=1).clamp(min=1.0).float()  # [B] = |M|
    H, W = A.shape[2], A.shape[3]
    per_sample = sq.sum(dim=(1, 2, 3)) / (denom * H * W)
    return per_sample.mean()


def anchor_loss(Pi: torch.Tensor, Pi0: torch.Tensor) -> torch.Tensor:
    """L_anchor = ||Pi - Pi0||_F^2  (paper Eq. anchor-loss)."""
    return ((Pi - Pi0) ** 2).sum()


def l1_sparsity(z: torch.Tensor) -> torch.Tensor:
    """L1 = ||z||_1 / (D H W)  -- mean absolute concept activation."""
    return z.abs().mean()


# ==========================================
# ImageNet-1k sampled dataset  (unchanged from old script)
# ==========================================

class ImageNet1kSampledDataset(Dataset):
    """Loads sampled ImageNet-1k images from parquet files."""

    def __init__(self, raw_dir: Path, sampled_dir: Path,
                 images_per_class: int = 50, transform=None,
                 force_resample: bool = False):
        self.raw_dir = raw_dir
        self.sampled_dir = sampled_dir
        self.images_per_class = images_per_class
        self.transform = transform
        self.sampled_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.sampled_dir / "metadata.pkl"

        if self.metadata_path.exists() and not force_resample:
            print(f"\nLoading cached sampled dataset from {self.sampled_dir}")
            self.load_cached_dataset()
        else:
            print(f"\nCreating new sampled dataset...")
            self.create_sampled_dataset()

    def create_sampled_dataset(self):
        self.wnid_to_idx = {w: i for i, w in enumerate(IMAGENET2012_CLASSES.keys())}
        self.idx_to_wnid = {i: w for w, i in self.wnid_to_idx.items()}
        class_samples = defaultdict(list)
        train_parquet_files = sorted(self.raw_dir.glob("train-*.parquet"))
        if len(train_parquet_files) == 0:
            raise FileNotFoundError(f"No train parquet files in {self.raw_dir}")
        print(f"Found {len(train_parquet_files)} train parquet files")

        for parquet_file in tqdm(train_parquet_files, desc="Reading parquet"):
            df = pd.read_parquet(parquet_file)
            for _, row in df.iterrows():
                label = row['label']
                if len(class_samples[label]) < self.images_per_class:
                    class_samples[label].append((row['image']['bytes'], label))
            min_s = min(len(v) for v in class_samples.values()) \
                if len(class_samples) == NUM_CLASSES else 0
            if min_s >= self.images_per_class and len(class_samples) == NUM_CLASSES:
                print(f"\nCollected {self.images_per_class} per class.")
                break

        self.samples = []
        for c in range(NUM_CLASSES):
            pool = class_samples[c]
            if len(pool) >= self.images_per_class:
                self.samples.extend(random.sample(pool, self.images_per_class))
            else:
                print(f"WARNING: class {c} has only {len(pool)} samples")
                self.samples.extend(pool)
        print(f"Total sampled images: {len(self.samples)}")

        joblib.dump({
            'samples': self.samples, 'images_per_class': self.images_per_class,
            'num_classes': NUM_CLASSES, 'wnid_to_idx': self.wnid_to_idx,
            'idx_to_wnid': self.idx_to_wnid,
        }, self.metadata_path)
        print("Sampled dataset cached.")

    def load_cached_dataset(self):
        md = joblib.load(self.metadata_path)
        self.samples = md['samples']
        self.wnid_to_idx = md['wnid_to_idx']
        self.idx_to_wnid = md['idx_to_wnid']
        print(f"Loaded {len(self.samples)} images "
              f"({md['images_per_class']}/class, {md['num_classes']} classes)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_bytes, label = self.samples[idx]
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ==========================================
# Activation extractor: caches (A, M_tau, L_GC) per image
# ==========================================

class ActivationExtractor:
    """Extracts, per image: the activation tensor A, the Grad-CAM channel mask
    M_tau, and the Grad-CAM spatial map L_GC.

    M_tau is used by L_recon during training.
    L_GC  is cached for EVALUATION ONLY (faithfulness / Grad-CAM-recovery).
    """

    def __init__(self, model_name: str, target_layer: Optional[str],
                 device: str, cumulative_threshold: float = 0.85,
                 cache_dir: Path = None):
        self.device = device
        self.cumulative_threshold = cumulative_threshold
        self.model_name = model_name
        self.cache_dir = cache_dir or ACTIVATION_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model {model_name}")
        cfg = MODEL_CONFIGS[model_name]
        self.target_layer_name = target_layer or cfg['default_target_layer']

        print(f"\n{'='*80}\nActivation Extractor: {model_name.upper()}\n{'='*80}")
        print(f"Model: {cfg['description']}   target layer: {self.target_layer_name}")

        self.model = cfg['model_fn']().to(device).eval()
        self.target_layer = self._get_layer(self.target_layer_name)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            out = self._forward_to_target(dummy)
            self.num_channels = out.shape[1]
            self.spatial_size = out.shape[2]
        print(f"  C = {self.num_channels} channels, "
              f"{self.spatial_size}x{self.spatial_size} resolution")

        self.gradcam = GradCAM(self.model, self.target_layer)
        self.activations = None
        self.target_layer.register_forward_hook(self._hook)

    def _get_layer(self, name: str):
        if '[' in name:
            attr, idx = name.split('[')
            return getattr(self.model, attr)[int(idx.rstrip(']'))]
        return getattr(self.model, name)

    def _forward_to_target(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name in ['resnet50', 'resnet18']:
            x = self.model.conv1(x); x = self.model.bn1(x)
            x = self.model.relu(x);  x = self.model.maxpool(x)
            x = self.model.layer1(x)
            if 'layer1' in self.target_layer_name: return x
            x = self.model.layer2(x)
            if 'layer2' in self.target_layer_name: return x
            x = self.model.layer3(x)
            if 'layer3' in self.target_layer_name: return x
            x = self.model.layer4(x)
            return x
        elif self.model_name in ['vgg16', 'efficientnet']:
            idx = int(self.target_layer_name.split('[')[1].rstrip(']'))
            for i in range(idx + 1):
                x = self.model.features[i](x)
            return x
        return x

    def _hook(self, module, inp, out):
        self.activations = out.detach()

    def _mask_and_gcmap(self, image: torch.Tensor
                        ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Grad-CAM channel mask M_tau and spatial map L_GC for one image."""
        weights, _, _ = self.gradcam.forward(image, class_idx=None, verbose=False)
        # weights: [C] Grad-CAM alphas ; self.activations: [1, C, H, W]

        # --- channel mask M_tau (cumulative-threshold over positive alphas) --
        order = torch.argsort(weights, descending=True)
        w_sorted = weights[order]
        total = w_sorted.clamp(min=0).sum()
        if total > 0:
            cum = torch.cumsum(w_sorted.clamp(min=0) / total, dim=0)
            n_sel = int((cum < self.cumulative_threshold).sum().item()) + 1
            n_sel = min(n_sel, len(order))
        else:
            n_sel = max(1, int(0.1 * len(order)))
        mask = torch.zeros(self.num_channels, dtype=torch.bool, device=self.device)
        mask[order[:n_sel]] = True

        # --- spatial Grad-CAM map L_GC (eval-only target) --------------------
        with torch.no_grad():
            acts = self.activations[0]                       # [C, H, W]
            weighted = (weights.view(-1, 1, 1) * acts).sum(0) # [H, W]
            gcmap = F.relu(weighted)
            s = gcmap.sum()
            gcmap = gcmap / s if s > 1e-8 else \
                torch.full_like(gcmap, 1.0 / gcmap.numel())
        return mask, n_sel, gcmap

    def _cache_key(self, num_samples: int, chunk_size: int) -> str:
        s = (f"{self.model_name}_{self.target_layer_name}_"
             f"thresh{self.cumulative_threshold}_samples{num_samples}_"
             f"chunk{chunk_size}_{CACHE_TAG}")
        return s.replace('[', '_').replace(']', '').replace('.', 'p')

    def _save_part(self, cache_key, part_idx, act, mask, label, gcmap):
        d = self.cache_dir / cache_key
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump({'activation': act, 'mask': mask, 'label': label,
                     'gradcam_map': gcmap},
                    d / f"part_{part_idx:04d}.pkl", compress=3)

    def _cache_exists(self, cache_key) -> bool:
        return (self.cache_dir / cache_key / "metadata.pkl").exists()

    def _load_cache(self, cache_key):
        d = self.cache_dir / cache_key
        md = joblib.load(d / "metadata.pkl")
        acts, masks, labels, gcmaps = [], [], [], []
        print(f"  Loading {md['num_chunks']} cache parts...")
        for p in tqdm(range(md['num_chunks']), desc="Loading cache"):
            part = joblib.load(d / f"part_{p:04d}.pkl")
            acts.append(part['activation'])
            masks.append(part['mask'])
            labels.append(part['label'])
            if 'gradcam_map' not in part:
                raise KeyError(f"Cache part {p} lacks gradcam_map; "
                               "run --force_reextract.")
            gcmaps.append(part['gradcam_map'])
        return acts, masks, labels, gcmaps, md

    def collect(self, data_loader: DataLoader, normalize: bool = True,
                chunk_size: int = 100, use_cache: bool = True):
        """Collect (A, M_tau, L_GC) in chunks. Returns four lists of chunks."""
        n = len(data_loader.dataset)
        key = self._cache_key(n, chunk_size)

        if use_cache and self._cache_exists(key):
            acts, masks, labels, gcmaps, md = self._load_cache(key)
            if (md.get('normalized') == normalize and
                    md.get('cumulative_threshold') == self.cumulative_threshold):
                print("  Cache validated.")
                return acts, masks, labels, gcmaps
            print("  Cache metadata mismatch -- re-extracting.")

        all_a, all_m, all_l, all_g = [], [], [], []
        ca, cm, cl, cg = [], [], [], []
        sel_stats = []
        chunk_idx = 0

        def _norm_chunk(act):
            if not normalize:
                return act
            for c in range(act.shape[1]):
                cd = act[:, c]
                nz = cd.flatten()[cd.flatten() > 1e-8]
                if len(nz) > 0:
                    sf = torch.quantile(nz, 0.99)
                    if sf > 1e-8:
                        act[:, c] = torch.clamp(cd, 0.0, sf) / (sf + 1e-8)
            return act

        print(f"\nExtracting activations + Grad-CAM (chunk size {chunk_size})...")
        for images, labels in tqdm(data_loader, desc="Extracting"):
            for i in range(images.size(0)):
                img = images[i:i+1].to(self.device)
                with torch.no_grad():
                    _ = self.model(img)
                    act = self.activations.clone()
                mask, n_sel, gcmap = self._mask_and_gcmap(img)
                sel_stats.append(n_sel)

                ca.append(act.cpu())
                cm.append(mask.cpu())
                cl.append(labels[i:i+1])
                cg.append(gcmap.cpu().unsqueeze(0))

                if len(ca) >= chunk_size:
                    a = _norm_chunk(torch.cat(ca, 0))
                    m = torch.stack(cm, 0)
                    l = torch.cat(cl, 0)
                    g = torch.cat(cg, 0)
                    if use_cache:
                        self._save_part(key, chunk_idx, a, m, l, g)
                    all_a.append(a); all_m.append(m); all_l.append(l); all_g.append(g)
                    chunk_idx += 1
                    ca, cm, cl, cg = [], [], [], []
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if len(ca) > 0:
            a = _norm_chunk(torch.cat(ca, 0))
            m = torch.stack(cm, 0)
            l = torch.cat(cl, 0)
            g = torch.cat(cg, 0)
            if use_cache:
                self._save_part(key, chunk_idx, a, m, l, g)
            all_a.append(a); all_m.append(m); all_l.append(l); all_g.append(g)

        total = sum(c.shape[0] for c in all_a)
        print(f"\nExtraction complete: {total} samples, {len(all_a)} chunks.")
        print(f"  Avg channels selected: {np.mean(sel_stats):.1f} "
              f"+/- {np.std(sel_stats):.1f} (of {self.num_channels})")

        if use_cache:
            md = {'model_name': self.model_name,
                  'target_layer': self.target_layer_name,
                  'cumulative_threshold': self.cumulative_threshold,
                  'normalized': normalize, 'num_channels': self.num_channels,
                  'spatial_size': self.spatial_size, 'total_samples': total,
                  'num_chunks': len(all_a), 'has_gradcam_map': True}
            d = self.cache_dir / key
            joblib.dump(md, d / "metadata.pkl", compress=3)
            print(f"  Cache metadata saved ({len(all_a)} parts).")

        return all_a, all_m, all_l, all_g


# ==========================================
# Chunked activation dataset
# ==========================================

class ChunkedActivationDataset(Dataset):
    """Dataset over chunked (A, M_tau, label, L_GC) tuples."""

    def __init__(self, act_chunks, mask_chunks, label_chunks, gcmap_chunks):
        self.act_chunks = act_chunks
        self.mask_chunks = mask_chunks
        self.label_chunks = label_chunks
        self.gcmap_chunks = gcmap_chunks
        self.index_map = []
        self.class_to_indices = defaultdict(list)
        gi = 0
        for ci, lc in enumerate(label_chunks):
            for si in range(len(lc)):
                self.index_map.append((ci, si))
                self.class_to_indices[lc[si].item()].append(gi)
                gi += 1
        self.total = len(self.index_map)
        print(f"\nChunkedActivationDataset: {self.total} samples, "
              f"{len(self.class_to_indices)} classes.")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        ci, si = self.index_map[idx]
        return (self.act_chunks[ci][si], self.mask_chunks[ci][si],
                self.label_chunks[ci][si], self.gcmap_chunks[ci][si])


# ==========================================
# Stability diagnostics
# ==========================================

@torch.no_grad()
def atom_distance(Pi_a: torch.Tensor, Pi_b: torch.Tensor) -> float:
    """Permutation-invariant atom distance d_atom (paper Eq. atom-distance).

    Finds the best one-to-one matching of rows (Hungarian) and returns the
    Frobenius norm of the matched difference. Use this between two seeds to
    measure seed-stability of the learned atoms.
    """
    from scipy.optimize import linear_sum_assignment
    D = Pi_a.shape[0]
    cost = torch.cdist(Pi_a, Pi_b).cpu().numpy()      # [D, D] row-to-row L2
    r, c = linear_sum_assignment(cost)
    matched = Pi_a[r] - Pi_b[c]
    return float(torch.norm(matched).item())


# ==========================================
# Visualization
# ==========================================

def plot_training_logs(logs: Dict[str, List], model_name: str, save_path: str):
    fig, axs = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f'DCAM Training ({model_name.upper()} - ImageNet-1k)',
                 fontsize=14, fontweight='bold')

    axs[0, 0].plot(logs["recon_loss"], color='blue', lw=1.5)
    axs[0, 0].set_title("Masked Reconstruction Loss")
    axs[0, 0].set_ylabel("MSE (masked)"); axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(logs["anchor_loss"], color='magenta', lw=1.5)
    axs[0, 1].set_title("Anchor Loss  ||Pi - Pi0||_F^2")
    axs[0, 1].grid(True, alpha=0.3)

    axs[0, 2].plot(logs["l1_loss"], color='green', lw=1.5)
    axs[0, 2].set_title("L1 Sparsity Loss")
    axs[0, 2].grid(True, alpha=0.3)

    axs[1, 0].plot(logs["total_loss"], color='black', lw=2)
    axs[1, 0].set_title("Total Loss"); axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(logs["active_pct"], color='teal', lw=1.5)
    axs[1, 1].set_title("Active Concepts %"); axs[1, 1].grid(True, alpha=0.3)

    axs[1, 2].plot(logs["sigma_min"], color='red', lw=1.5)
    axs[1, 2].set_title("sigma_min(Pi)  (decoder conditioning)")
    axs[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='DCAM training on ImageNet-1k (membership matrix Pi)')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--target_layer', type=str, default=None)
    parser.add_argument('--force_resample', action='store_true')
    parser.add_argument('--force_reextract', action='store_true')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=None)

    # DCAM hyperparameters
    parser.add_argument('--D', type=int, default=None,
                        help='Atom-vocabulary size D (<= C). '
                             'Default: per-backbone C/2.')
    parser.add_argument('--top_k', type=int, default=32,
                        help='Top-K concepts kept active per image.')
    parser.add_argument('--lambda_anchor', type=float, default=1.0,
                        help='Weight of the anchor loss ||Pi - Pi0||_F^2.')
    parser.add_argument('--lambda_l1', type=float, default=0.3,
                        help='Weight of the L1 concept-sparsity loss.')
    parser.add_argument('--ridge', type=float, default=1e-4,
                        help='Damping for (Pi Pi^T)^{-1} in the tied decoder.')
    parser.add_argument('--cumulative_threshold', type=float, default=0.85,
                        help='Grad-CAM channel-mask cumulative threshold tau.')
    parser.add_argument('--nmf_iters', type=int, default=500,
                        help='Iterations for the anchor symmetric-NMF.')
    parser.add_argument('--merge_tol', type=float, default=1e-2,
                        help='Anchor row-merge tolerance (non-degeneracy check).')
    parser.add_argument('--anchor_seed', type=int, default=0,
                        help='FIXED seed for anchor NMF (NOT the model seed). '
                             'Keep constant across runs for reproducibility.')

    parser.add_argument('--model_suffix', type=str, default='')
    parser.add_argument('--data_seed', type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument('--model_seed', type=int, default=DEFAULT_MODEL_SEED)
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = 16 if args.model in ('resnet50', 'vgg16') else 32

    print("=" * 80)
    print(f"DCAM Training -- backbone {args.model.upper()}")
    print(f"Seeds: data={args.data_seed}, model={args.model_seed}, "
          f"anchor={args.anchor_seed} (fixed)")
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

    # ---- extraction ----------------------------------------------------
    extractor = ActivationExtractor(
        args.model, args.target_layer, device,
        cumulative_threshold=args.cumulative_threshold)
    act_chunks, mask_chunks, label_chunks, gcmap_chunks = extractor.collect(
        loader, normalize=True, chunk_size=ACTIVATION_CHUNK_SIZE,
        use_cache=not args.force_reextract)

    C = extractor.num_channels
    D = args.D if args.D is not None else MODEL_CONFIGS[args.model]['default_D']
    if D > C:
        raise ValueError(f"DCAM needs D <= C; got D={D}, C={C}.")
    print(f"\nC = {C}, requested D = {D}  (compressive regime D <= C)")

    # ---- anchor Pi0  (seed-free) ---------------------------------------
    print(f"\n{'='*80}\nBuilding seed-free anchor Pi0\n{'='*80}")
    S = compute_coactivation_matrix(act_chunks)
    Pi0, nu, D_eff = build_anchor(
        S, D, nmf_iters=args.nmf_iters, merge_tol=args.merge_tol,
        seed=args.anchor_seed)
    Pi0 = Pi0.to(device)
    if D_eff != D:
        print(f"  NOTE: D reduced {D} -> {D_eff} after non-degeneracy merge.")
    D = D_eff

    # ---- model  (Pi initialized AT the anchor) -------------------------
    print(f"\nApplying model seed:")
    set_seed(args.model_seed)   # affects training shuffling; Pi starts at Pi0
    top_k = min(args.top_k, D)
    model = DCAM(in_channels=C, num_concepts=D, top_k=top_k,
                 ridge=args.ridge).to(device)
    model.init_from_anchor(Pi0)

    print(f"\nTraining Configuration:")
    print(f"  C={C}  D={D}  Top-K={top_k}  ridge={args.ridge}")
    print(f"  lambda_anchor={args.lambda_anchor}  lambda_l1={args.lambda_l1}")
    print(f"  epochs={args.epochs}  lr={args.lr}  batch_size={args.batch_size}")
    print(f"  anchor non-degeneracy nu={nu:.4e}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    train_dataset = ChunkedActivationDataset(
        act_chunks, mask_chunks, label_chunks, gcmap_chunks)
    gen = torch.Generator().manual_seed(args.model_seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, generator=gen)

    logs = {"total_loss": [], "recon_loss": [], "anchor_loss": [],
            "l1_loss": [], "active_pct": [], "sigma_min": []}

    print(f"\n{'='*80}\nStarting Training\n{'='*80}")
    for epoch in range(args.epochs):
        ep = {k: 0.0 for k in logs}
        nb = 0
        for bi, (A, M, _lbl, _gc) in enumerate(train_loader):
            A = A.to(device)
            M = M.to(device)

            A_hat, z = model(A, use_topk=True)

            l_recon = masked_reconstruction_loss(A_hat, A, M)
            l_anchor = anchor_loss(model.Pi, Pi0)
            l_l1 = l1_sparsity(z)
            loss = l_recon + args.lambda_anchor * l_anchor + args.lambda_l1 * l_l1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.project()   # Pi rows back onto the non-negative L1-simplex

            with torch.no_grad():
                active_pct = (z > 0).float().mean().item() * 100.0
                sv = torch.linalg.svdvals(model.Pi)
                sigma_min = float(sv.min().item())

            logs["total_loss"].append(loss.item())
            logs["recon_loss"].append(l_recon.item())
            logs["anchor_loss"].append(l_anchor.item())
            logs["l1_loss"].append(l_l1.item())
            logs["active_pct"].append(active_pct)
            logs["sigma_min"].append(sigma_min)
            for k in ep:
                ep[k] += logs[k][-1]
            nb += 1

            if bi % 20 == 0:
                print(f"\rEpoch {epoch+1}/{args.epochs} "
                      f"[{bi}/{len(train_loader)}] "
                      f"L={loss.item():.4f} | recon={l_recon.item():.4f} | "
                      f"anchor={l_anchor.item():.4f} | "
                      f"sigma_min={sigma_min:.3e} | "
                      f"active={active_pct:.1f}%", end="")

        avg = {k: v / max(nb, 1) for k, v in ep.items()}
        print(f"\n[Epoch {epoch+1}/{args.epochs}] "
              f"total={avg['total_loss']:.4f}  recon={avg['recon_loss']:.4f}  "
              f"anchor={avg['anchor_loss']:.4f}  l1={avg['l1_loss']:.4f}  "
              f"sigma_min={avg['sigma_min']:.3e}  active={avg['active_pct']:.2f}%")
        print("-" * 80)

    # ---- save ----------------------------------------------------------
    prefix = f"imagenet1k_dcam_{args.model}"
    if args.target_layer:
        prefix += "_" + args.target_layer.replace('[', '_').replace(']', '')
    prefix += f"_D{D}_seed{args.data_seed}-{args.model_seed}{args.model_suffix}"

    torch.save(model.state_dict(), f"{prefix}_model.pth")
    # store the learned Pi and the anchor explicitly -- these ARE the result
    joblib.dump({
        'Pi': model.Pi.detach().cpu(),
        'Pi0': Pi0.detach().cpu(),
        'config': {
            'model': args.model, 'target_layer': extractor.target_layer_name,
            'C': C, 'D': D, 'top_k': top_k,
            'lambda_anchor': args.lambda_anchor, 'lambda_l1': args.lambda_l1,
            'ridge': args.ridge, 'cumulative_threshold': args.cumulative_threshold,
            'nu': nu, 'data_seed': args.data_seed,
            'model_seed': args.model_seed, 'anchor_seed': args.anchor_seed,
        },
        'logs': logs, 'final_metrics': avg,
    }, f"{prefix}_result.pkl")

    plot_training_logs(logs, args.model, f"{prefix}_logs.png")

    # report deviation of the learned Pi from the anchor (a stability proxy)
    dev = atom_distance(model.Pi.detach().cpu(), Pi0.detach().cpu())
    print(f"\n{'='*80}\nTraining Complete")
    print(f"  Result: {prefix}_result.pkl")
    print(f"  Logs:   {prefix}_logs.png")
    print(f"  d_atom(Pi, Pi0) = {dev:.4e}   (anchor non-degeneracy nu={nu:.4e})")
    print(f"  To test seed-stability: train a second --model_seed and compare")
    print(f"  the two saved Pi's with atom_distance().")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()