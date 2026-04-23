# DCAM: Decomposed Class Activation Maps

> **What Does Each Channel See?**  
> Decomposing Grad-CAM into Concept-Level Activation Patterns via Convolutional Sparse Autoencoders

---

## Overview

GradCAM collapses hundreds of activation channels into a single heatmap, losing all information about *which* visual concepts contributed. DCAM fills this gap: it trains a **Convolutional Sparse Autoencoder (ConvSAE)** on CNN intermediate activations, guided by a Grad-CAM-masked reconstruction loss, to recover interpretable concept-level sub-patterns (e.g., "body outline", "head/face", "background texture") that compose into the full Grad-CAM explanation.

```
Input image → CNN backbone → A(x) [C×H×W]
                               ↓
                          Grad-CAM → channel mask M(x)  [top 85%]
                               ↓
                ConvSAE (encoder → ReLU + Top-K → decoder)
                               ↓
                  Sparse latent features z  [D×H×W, D = 8C]
                  Reconstruction Â(x)  [C×H×W]
                               ↓
              Loss: MSE on M(x) channels + L1 + lateral + TV
```

### Supported Backbones

| Backbone | Target Layer | Channels | Resolution |
|---|---|---|---|
| ResNet-50 | `layer3` | 1024 | 14×14 |
| ResNet-18 | `layer3` | 256 | 14×14 |
| VGG-16 | `features[16]` | 256 | 28×28 |
| EfficientNet-B0 | `features[4]` | 80 | 14×14 |

---

## Project Structure

```
.
├── run_xcsae_full.py          # ConvSAE training on ImageNet-1k
├── check_acc_drop_full.py     # Accuracy drop evaluation
├── visualize_testmf_full.py   # Feature visualization (random & consistency modes)
├── src/
│   └── gradcam.py             # GradCAM implementation
├── full_classes.py            # ImageNet-1k class mappings
├── cache_activations/         # Cached activation chunks (auto-created)
├── imagenet1k_csae_*_model.pkl  # Trained ConvSAE models (output)
└── README.md
```

### Data Paths (configure in `run_xcsae_full.py`)

```python
IMAGENET_RAW_DIR    = Path("/data/imagenet_raw/data")       # parquet files
IMAGENET_SAMPLED_DIR = Path("/data/imagenet1k_sampled")     # 50K train cache
# Test samples cached to /data/imagenet1k_sampletest
```

---

## Installation

```bash
pip install torch torchvision joblib matplotlib numpy tqdm pandas pyarrow pillow
```

---

## Quickstart

### 1. Train ConvSAE

```bash
# ResNet-50 (default) — 50K images, 15 epochs
python run_xcsae_full.py

# Other backbones
python run_xcsae_full.py --model resnet18
python run_xcsae_full.py --model vgg16
python run_xcsae_full.py --model efficientnet

# Custom layer
python run_xcsae_full.py --model resnet50 --target_layer layer2

# Gradient accumulation (effective batch 32 with 8 GPU batch)
python run_xcsae_full.py --model resnet50 --batch_size 8 --accumulation_steps 4

# Force re-extract activations (ignore cache)
python run_xcsae_full.py --force_reextract
```

### 2. Evaluate Accuracy Preservation

```bash
# ResNet-50 (auto-detects imagenet1k_csae_resnet50_model.pkl)
python check_acc_drop_full.py

# All backbones
python check_acc_drop_full.py --model resnet18
python check_acc_drop_full.py --model vgg16
python check_acc_drop_full.py --model efficientnet
```

Example output:
```
Top-1 Accuracy:
  Original (no reconstruction):     76.14%
  Reconstructed (CSAE):             72.28%
  Accuracy Drop:                    3.86%
```

### 3. Visualize Features

```bash
# Consistency mode: common features across 3 images of the same class
python visualize_testmf_full.py --num_classes 10 --top_k_features 12

# Random mode: individual test images
python visualize_testmf_full.py --num_samples 10 --top_k_features 16

# Custom backbone
python visualize_testmf_full.py --model vgg16 --num_classes 5
```

### Full Multi-Model Comparison Workflow

```bash
# Train all
python run_xcsae_full.py --model resnet50
python run_xcsae_full.py --model resnet18
python run_xcsae_full.py --model vgg16
python run_xcsae_full.py --model efficientnet

# Evaluate all
python check_acc_drop_full.py --model resnet50
python check_acc_drop_full.py --model resnet18
python check_acc_drop_full.py --model vgg16
python check_acc_drop_full.py --model efficientnet

# Visualize same 5 classes across all backbones
for model in resnet50 resnet18 vgg16 efficientnet; do
    python visualize_testmf_full.py --model $model --num_classes 5
done
```

---

## Architecture Details

**ConvSAE** (`MultiChannelConvSAE`):
- Encoder: 1×1 conv, C → D=8C channels, ReLU
- Top-K selection: retain K=0.4D channels by spatial sum importance
- Decoder: 1×1 conv (no bias), D → C; columns normalized to unit norm after each step

**Training loss**:
```
L = L_recon  +  0.3 * L_L1  +  0.01 * L_lateral  +  0.01 * L_compact
```
- `L_recon`: MSE computed **only on GradCAM-selected channels** (top 85% cumulative weight)
- `L_L1`: activation sparsity in latent space
- `L_lateral`: penalizes co-activation of spatially adjacent features (4-neighbor)
- `L_compact`: total variation over spatial dimensions

**Training**: Adam, lr=1e-3, weight decay=1e-5, grad clip=1.0, 15 epochs.

---

## Current Results

| Backbone | Original Acc. | Reconstructed Acc. | Drop |
|---|---|---|---|
| ResNet-50 | 76.14% | 72.28% | 3.86% |
| ResNet-18 | 69.76% | 67.78% | 1.98% |
| VGG-16 | 71.58% | 71.16% | 0.42% |
| EfficientNet-B0 | — | — | TODO |

Masked reconstruction MSE (ResNet-50: 0.0039, ResNet-18: 0.0016, VGG-16: 0.6469).

---

## TODO

### Experiments & Results
- [ ] Run EfficientNet-B0 training and fill accuracy drop table
- [ ] Fill masked reconstruction MSE for EfficientNet-B0 (Table 2 in paper)
- [ ] Fill active feature % for all backbones (Table 2)
- [ ] Run pattern consistency (Jaccard similarity) analysis across classes for Section 6.2
- [ ] Run stability diagnostics: K-gap γK, decoder coherence µS, dead feature %, CKA between two seeded runs (Section 6.4)
- [ ] Generate qualitative visualization figures for paper (Section 6.5): input image, Grad-CAM heatmap, top-5 DCAM features, reconstruction

### Code
- [ ] Implement Jaccard similarity metric for intra-class vs. inter-class feature consistency
- [ ] Add CKA (Centered Kernel Alignment) metric between two independently trained ConvSAE runs
- [ ] Track K-gap γK and decoder coherence µS during training as diagnostics (per Remark 4.7)
- [ ] Add `--seed` argument to training script for reproducibility experiments
- [ ] VLM-based concept naming pipeline (post-hoc naming of active features via a captioning model)
- [ ] Add ablation script for τ (mask threshold), D/C ratio, K/D ratio

### Paper
- [ ] Complete Section 6.1 (fill all TODO cells in Tables 2–3)
- [ ] Write Section 6.2 (pattern consistency results + table)
- [ ] Write Section 6.4 (stability diagnostic plots)
- [ ] Write Section 6.5 (qualitative figure with real data)
- [ ] Write Section 7.1 (interpreting decompositions: anatomy, structure, texture)
- [ ] Add acknowledgements
- [ ] Add Appendix X: ablation studies on D/C and K/D

### Future Work
- [ ] Extend DCAM to Vision Transformers (ViT) with attention-based attribution
- [ ] Global concept coherence objective across dataset
- [ ] Dataset bias detection application demo
- [ ] Speed up activation extraction pipeline (currently one forward+backward per image)

---

## Citation

```bibtex
@article{dcam2024,
  title   = {What Does Each Channel See? Decomposing Grad-CAM into Concept-Level Activation Patterns},
  author  = {Anonymous Authors},
  year    = {2024},
  note    = {Under review}
}
```