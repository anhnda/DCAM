# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Rotation-invariant Convolutional Dictionary Learning (Roto-CDL)** implementation for analyzing deep neural network activation maps. The project uses a fine-tuned AlexNet-based model on the Imagenette dataset and learns interpretable dictionary atoms from activation maps using rotation-equivariant sparse coding.

## Key Commands

### Setup and Preprocessing
```bash
# Download and preprocess Imagenette dataset
python preprocess.py

# Run the main CDL pipeline
python run_cdl.py

# Run the ConvSAE training pipeline (single pathway)
python run_csae.py

# Run the Dual ConvSAE training pipeline (shared + class-discriminative features)
python run_dual_csae.py

# Run Dual ConvSAE with ResNet18 backbone (14×14 activation maps, no fine-tuning)
python run_dual_csae_resnet18.py

# Generate explanations for model predictions (using CDL dictionary)
python explain.py

# Generate explanations using ConvSAE
python explain_csae.py --image_path data/imagenette/tench/n01440764_1.JPEG
python explain_csae.py --class_name tench --num_images 3

# Analyze same-class atom activation patterns (CDL)
python check_same_class.py --class_name tench --num_images 10

# Compare atom activations across multiple classes (CDL)
python check_same_class.py --compare_classes tench church parachute --num_images 10

# Analyze same-class ConvSAE feature patterns
python check_same_class_csae.py --class_name tench --num_images 10

# Compare ConvSAE features across multiple classes
python check_same_class_csae.py --compare_classes tench church parachute --num_images 10

# Analyze same-class Dual ConvSAE feature patterns (shared + class pathways)
python check_same_class_dual_csae.py --class_name tench --num_images 10

# Compare Dual ConvSAE features across multiple classes
python check_same_class_dual_csae.py --compare_classes tench church parachute --num_images 10

# Use with ResNet18 backbone
python check_same_class_dual_csae.py --compare_classes tench church parachute --num_images 10 --use_resnet18 --dual_csae_path dual_csae_resnet18_model.pkl

# View top activation maps with deconvolution
python view_top_activation.py --image_path data/imagenette/tench/n01440764_1.JPEG
python view_top_activation.py --class_name tench --num_images 3

# Run ProtoPNet training on full ImageNet-1k (interpretable classification)
python run_protopnet_full.py  # ResNet50, 2 prototypes/class
python run_protopnet_full.py --model resnet18 --num_prototypes_per_class 5
python run_protopnet_full.py --warm_epochs 10 --joint_epochs 30
```

### Working with Jupyter Notebooks
The primary development workflow uses `main.ipynb`. There is no test suite in this repository.

## Architecture Overview

### Data Pipeline
1. **Data Preparation** (`preprocess.py`):
   - Downloads Imagenette-160px from Kaggle
   - Merges train/val splits
   - Renames class folders from IDs to readable names (e.g., `n01440764` → `tench`)
   - Final structure: `data/imagenette/{class_name}/`

2. **Feature Extraction** (`run_cdl.py` + `src/activation.py`):
   - Loads fine-tuned AlexNet model from `weights/finetune_weights.pth`
   - Target layer: `model.feature_extractor[5]` (MaxPool after 2nd conv block)
   - Uses GradCAM to identify influential channels
   - Collects activation maps from top-k% channels (default 90%)
   - Applies robust normalization (99th percentile clipping + scaling by 10)
   - Optional Gaussian blur to reduce sparsity

3. **Dictionary Learning** (`src/batch_cdl_large.py`):
   - Learns rotation-invariant atoms via Roto-CDL
   - Alternates between sparse coding (Z-step) and dictionary update (Phi-step)
   - Uses proximal gradient descent with non-negativity constraints

### Core Components

#### Model Architecture (`src/model.py`)
- **FineTunedModel**: AlexNet-style architecture with 10-class classifier
- Feature extractor: 5 conv blocks (64→192→384→256→256 channels)
- Classifier: AdaptiveAvgPool → 3 FC layers (9216→4096→4096→10)

#### Activation Analysis (`src/activation.py`, `src/gradcam.py`)
- **GradCAM**: Computes class-discriminative localization maps
- **ActivationMapCollector**: Extracts top-k% influential channels using GradCAM weights
- Channel selection uses cumulative weight sorting (top 90% by default)

#### Dictionary Learning (`src/batch_cdl_large.py`)
The core algorithm has two versions:

1. **Large-Scale Version** (`roto_cdl_large_scale`):
   - Handles datasets via mini-batch training
   - Gaussian noise initialization (not data patches)
   - Sparsity warmup: 0 for epoch 0, linear ramp to target over 5 epochs
   - Periodic atom re-centering (every 5 epochs)
   - Returns normalized dictionary Phi [n_atoms, 1, d, d]

2. **Standard Version** (`roto_cdl_batch_train` in `src/batch_cdl.py`):
   - Smart initialization from brightest data patches
   - Includes compactness penalty regularization
   - Pixel pruning for dead atoms

#### Convolutional Sparse Autoencoder (`src/convsae.py`, `run_csae.py`)
Alternative approach to dictionary learning using a neural network-based sparse autoencoder:

- **ConvSAE Architecture**:
  - Encoder: Conv2d layer that maps activation maps to sparse feature space
  - Decoder: Conv2d layer that reconstructs activation maps from sparse features
  - Default: 1x1 convolutions for spatial sparsity
  - Hidden dimension: 4096 features (configurable)
  - Encoder bias parameter for fine-grained control

- **Training Objectives**:
  1. **Reconstruction Loss**: MSE between input and reconstructed activation maps
  2. **L1 Sparsity**: Encourages sparse activations in feature space
  3. **Lateral Inhibition**: Penalizes neighboring neurons firing together (prevents blob-like activations)

- **Key Features**:
  - Decoder weight normalization after each optimization step with **non-negativity constraint**
    - Since `feature_acts = ReLU(encoder(x))` are always non-negative, decoder weights are constrained to [0, 1]
    - Prevents negative contributions in reconstruction
    - Applied via `ReLU(weight)` before L2 normalization
  - Active neuron monitoring (tracks % of neurons firing per batch)
  - Comprehensive training diagnostics and visualization

- **Training Pipeline** (`run_csae.py`):
  1. Collects activation maps using same pipeline as CDL (via GradCAM + ActivationMapCollector)
  2. Applies robust normalization (99th percentile scaling)
  3. Trains ConvSAE with Adam optimizer
  4. Saves model state dict, full model (joblib), and training info
  5. Generates training log visualizations and learned feature visualizations

- **Output Files**:
  - `csae_model.pth`: PyTorch state dict for model weights
  - `csae_model.pkl`: Full model saved with joblib
  - `csae_training_info.pkl`: Training configuration and logs
  - `csae_training_logs.png`: 4-panel diagnostic plot (reconstruction, L1, lateral inhibition, active neurons)
  - `csae_features.png`: Visualization of learned decoder features

#### Dual Convolutional Sparse Autoencoder (`src/convsae.py:DualConvSAE`, `run_dual_csae.py`)
**Problem**: Standard ConvSAE learns features that are shared across all classes, making it difficult to identify class-discriminative patterns.

**Solution**: Dual-pathway architecture that explicitly separates shared and class-specific features.

- **DualConvSAE Architecture**:
  - **Shared Pathway**: Learns global features common across all classes (unsupervised)
    - Shared encoder: Conv2d(in_channels → shared_dim)
    - Shared decoder: Conv2d(shared_dim → in_channels)
    - Loss: Reconstruction + L1 sparsity + Lateral inhibition

  - **Class-Specific Pathway**: Learns discriminative features for classification (supervised)
    - Class encoder: Conv2d(in_channels → class_dim)
    - Class decoder: Conv2d(class_dim → in_channels)
    - Classifier head: AdaptiveAvgPool2d → Linear(class_dim → num_classes)
    - Loss: Reconstruction + L1 sparsity + Lateral inhibition + Classification + Diversity

  - **Combined Reconstruction**: `recon = shared_decoder(shared_feats) + class_decoder(class_feats)`

- **Training Objectives**:
  1. **Shared Reconstruction**: MSE between shared pathway reconstruction and input
  2. **Class Reconstruction**: MSE between class pathway reconstruction and input
  3. **Total Reconstruction**: MSE between combined reconstruction and input
  4. **Shared Sparsity**: L1 penalty on shared features
  5. **Class Sparsity**: L1 penalty on class-specific features
  6. **Classification**: CrossEntropy loss on class predictions
  7. **Diversity**: Encourages different classes to activate different features (orthogonality)
  8. **Lateral Inhibition**: Applied to both pathways to prevent blob-like activations

- **Key Advantages**:
  - Explicitly separates global patterns (edges, textures) from class-specific patterns (object parts)
  - Classification head provides supervision for class pathway
  - Diversity loss ensures classes use different feature subsets
  - Shared pathway captures common low-level features, reducing redundancy
  - Class pathway focuses on discriminative high-level patterns

- **Hyperparameters** (`run_dual_csae.py`):
  - `shared_dim=256`: Number of shared features
  - `class_dim=256`: Number of class-specific features
  - `kernel_size=3`: 3×3 convolution for spatial context
  - `lambda_shared_l1=0.01`: Sparsity penalty for shared features
  - `lambda_class_l1=0.01`: Sparsity penalty for class features
  - `lambda_classification=1.0`: Classification loss weight
  - `lambda_diversity=0.1`: Diversity loss weight (encourages class separation)
  - `lr=3e-4`, `epochs=15`, `batch_size=256`

- **Training Pipeline** (`run_dual_csae.py`):
  1. Collects activation maps with class labels
  2. Trains dual pathways simultaneously
  3. Monitors classification accuracy and diversity metrics
  4. Saves both pathways and classifier
  5. Generates comprehensive visualizations

- **Output Files**:
  - `dual_csae_model.pth`: PyTorch state dict
  - `dual_csae_model.pkl`: Full model (joblib)
  - `dual_csae_training_info.pkl`: Training config and logs
  - `dual_csae_training_logs.png`: 3×4 diagnostic plot showing:
    - Row 1: Reconstruction losses, classification loss, accuracy
    - Row 2: Sparsity metrics, active neuron percentages, diversity loss
    - Row 3: Lateral inhibition, total loss, loss components, feature usage ratio
  - `dual_csae_features.png`: Weight distributions for both pathways

- **Monitoring During Training**:
  - **Classification Accuracy**: Should increase to 70-90% (validates class pathway is learning)
  - **Diversity Loss**: Should decrease (classes use different features)
  - **Active Neurons**: Both pathways should maintain 5-15% sparsity
  - **Feature Usage Ratio**: `shared_l1 / class_l1` ≈ 1.0 indicates balanced usage

- **When to Use Dual vs Standard ConvSAE**:
  - Use **Dual ConvSAE** when:
    - You need to identify class-discriminative features
    - Cross-class feature analysis shows too much overlap
    - You want interpretable separation between global and class-specific patterns
  - Use **Standard ConvSAE** when:
    - You want a simpler, faster model
    - Class labels are unavailable
    - Task is unsupervised feature discovery

#### Dual ConvSAE with ResNet18 Backbone (`run_dual_csae_resnet18.py`)
**Alternative to fine-tuned AlexNet**: Uses pretrained ResNet18 (ImageNet-1k) without fine-tuning.

**Key Advantages**:
- **No fine-tuning required**: Uses pretrained ImageNet weights directly
- **Better features**: ResNet18 provides stronger pretrained representations than AlexNet
- **Larger spatial resolution**: 14×14 activation maps (vs 13×13 for AlexNet layer 5)
- **More channels**: 256 channels from layer3 (vs variable for AlexNet)

**Architecture Details**:
- **Backbone**: ResNet18 pretrained on ImageNet-1k (1000 classes)
- **Target Layer**: `layer3` - outputs 256 channels at 14×14 spatial resolution
- **Class Mapping**: Maps ImageNet-1k predictions to Imagenette-10 classes
  - Tench → ImageNet class 0
  - English Springer Spaniel → ImageNet class 217
  - Cassette Player → ImageNet class 482
  - Chain Saw → ImageNet class 491
  - Church → ImageNet class 497
  - French Horn → ImageNet class 566
  - Garbage Truck → ImageNet class 569
  - Gas Pump → ImageNet class 571
  - Golf Ball → ImageNet class 574
  - Parachute → ImageNet class 701

**Activation Extraction Pipeline**:
1. For each image, get predicted Imagenette class from ImageNet-1k predictions
2. Use GradCAM on layer3 to compute channel importance for that class
3. Select top 90% of channels by cumulative GradCAM score
4. Average selected channels to create single-channel class-discriminative activation map
5. Apply robust normalization (99th percentile scaling)

**Training Configuration** (same as AlexNet version):
- `shared_dim=256`, `class_dim=256`
- `kernel_size=3` (3×3 convolution)
- Optimized loss weights for discrimination
- 15 epochs with warmup

**Output Files**:
- `dual_csae_resnet18_model.pth`: Model state dict
- `dual_csae_resnet18_model.pkl`: Full model (joblib)
- `dual_csae_resnet18_training_info.pkl`: Training config and logs
- `dual_csae_resnet18_logs.png`: Training diagnostics
- `dual_csae_resnet18_features.png`: Feature visualizations

**When to Use ResNet18 vs AlexNet**:
- Use **ResNet18** when:
  - You don't want to fine-tune a model
  - You want stronger pretrained features
  - You want faster iteration (no fine-tuning step)
  - You have limited training data
- Use **AlexNet** when:
  - You already have a fine-tuned model
  - You want features specifically adapted to your dataset
  - You need exact compatibility with existing pipelines

#### Sparse Coding Inference (`src/batch_cdl_large.py:solve_sparse_code`)
- Given learned dictionary Phi, solves for sparse code Z
- Uses proximal gradient with soft thresholding
- Supports reconstruction and visualization

#### Analysis Tools (`src/analysis.py`)
- **TopKActivationAnalyzer**: Analyzes influential channels for specific inputs
- L1 sparse coding with FISTA (Fast Iterative Shrinkage-Thresholding)
- Visualization of channel decomposition into dictionary atoms

#### Explanation Pipeline (`explain.py`)
- **ExplanationPipeline**: Complete explainability system for model predictions
- **5-Step Pipeline**:
  1. Uses GradCAM to compute channel importance weights at layer 5
  2. Selects top-k activation maps with cumulative score ≥ 80% of total logit score
  3. For each selected activation map, applies `solve_z_prox_adam` to decompose into learned atoms
  4. Identifies which dictionary atoms activate at each spatial position
  5. Computes receptive field masks to trace patterns back to input image regions
- **Visualization**: Two-level visualization showing overview (input, GradCAM, overlay) and per-channel details (activation map, top atoms, receptive fields)
- **Deconvolution**: Geometric receptive field approximation (effective stride ≈ 16 for layer 5, RF ≈ 51×51)

#### Same-Class Analysis (`check_same_class.py`)
- **SameClassAtomAnalyzer**: Analyzes whether images from the same class activate similar dictionary atoms
- **Single-Class Analysis**:
  - Processes multiple images from one class
  - Identifies most frequently activated atoms
  - Computes atom frequency and importance scores
  - Visualizes: sample images, top atoms, activation heatmap, frequency bar chart
- **Cross-Class Comparison**:
  - Compares atom activation patterns across multiple classes
  - Identifies class-specific atoms (high in one class, low in others)
  - Builds atom-class activation matrix
  - Helps validate semantic meaning of learned atoms
- **Output**: Comprehensive visualizations saved to `same_class_analysis/` directory

#### Top Activation Viewer (`view_top_activation.py`)
- **TopActivationVisualizer**: Visualizes which input regions activate specific channels
- **DeconvNet**: Implements guided backpropagation for input attribution
- **Analysis Pipeline**:
  1. Uses GradCAM to select top activation maps (80% cumulative score)
  2. For each selected channel, computes deconvolution saliency map
  3. Shows which input image regions contribute most to each activation
- **Visualizations**:
  - Detailed view: Shows activation map, deconv saliency, overlay, and masked regions for each channel
  - Grid view: Compact overview of all selected channels
- **Use Case**: Understanding what visual patterns in the input trigger specific feature maps at layer 5

### Key Algorithms

#### Rotation-Equivariant Reconstruction
```python
def reconstruct(phi, Z):
    # phi: [J, 1, d, d] - base atoms
    # Z: [N, J, K, H, W] - sparse codes with K rotations
    # Returns: [N, 1, H, W] - reconstructed maps
```
Uses grouped convolutions with rotated atoms (0°, 90°, 180°, 270°).

#### Sparse Coding (Z-step)
- **Optimizer**: SGD with momentum (lr=3.0 for large-scale)
- **Constraint**: ReLU(Z - λ) after each gradient step
- **Iterations**: 30 steps per batch
- **Initialization**: Small random positive values

#### Dictionary Update (Phi-step)
- **Optimizer**: Adam (lr=0.05)
- **Constraint**: ReLU + L2 normalization per atom
- **Scheduler**: StepLR (gamma=0.5 every 10 epochs)

### Critical Implementation Details

1. **Data Normalization**:
   - Input images: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   - Activation maps: Robust 99th percentile scaling then × 10

2. **Hyperparameters** (from `run_cdl.py`):
   - `n_atoms=64`: Dictionary size
   - `n_rotations=4`: 90° rotation steps
   - `d=9`: Atom kernel size
   - `batch_size=1500`: For dictionary learning
   - `lr_phi=0.05`: Dictionary learning rate
   - `lr_z=3.0`: High LR to escape zero initialization
   - `sparsity=0.005`: L1 penalty coefficient

3. **ConvSAE Hyperparameters** (from `run_csae.py`):
   - `hidden_dim=4096`: Number of sparse features in latent space
   - `kernel_size=1`: Convolution kernel size (1x1 for spatial sparsity)
   - `lambda_l1=0.05`: L1 sparsity penalty coefficient (tuned for 5-15% activation)
   - `lambda_lat=0.02`: Lateral inhibition penalty coefficient
   - `lr=3e-4`: Learning rate for Adam optimizer
   - `epochs=20`: Number of training epochs
   - `batch_size=32`: Mini-batch size for ConvSAE training
   - **Target sparsity**: 5-15% active neurons per batch (monitored during training)

4. **Memory Management**:
   - Activation collection uses `batch_size=1` to save VRAM
   - Dictionary learning loads full data to VRAM if possible
   - Falls back to CPU if OOM

5. **Model Checkpoint**:
   - Required file: `weights/finetune_weights.pth`
   - Must be downloaded separately (see README)

## Development Notes

- The codebase uses PyTorch with CUDA/MPS support
- Vietnamese comments are present in some files (legacy from original authors)
- No explicit test suite; validation done via visualization
- Main development happens in `main.ipynb`, then code is refactored to modules
- **CDL outputs**:
  - Dictionary saved as `phi_learned_imagenette.pkl` (joblib format)
  - Visualizations saved as `large_dictionary.png`
- **ConvSAE outputs**:
  - Model state dict: `csae_model.pth`
  - Full model: `csae_model.pkl`
  - Training info: `csae_training_info.pkl`
  - Visualizations: `csae_training_logs.png`, `csae_features.png`

## Using the Explanation Pipeline

The `explain.py` module provides a complete explainability system. Basic usage:

```python
from explain import ExplanationPipeline, load_learned_dictionary
from src.model import FineTunedModel

# Load model and dictionary
model = FineTunedModel(num_classes=10).to(device)
model.load_state_dict(torch.load('weights/finetune_weights.pth'))
phi_learned = load_learned_dictionary('phi_learned_imagenette.pkl', device)

# Create explainer
explainer = ExplanationPipeline(model, phi_learned, device)

# Generate explanation
results = explainer.explain_prediction(
    image=test_image,
    cumulative_threshold=0.8,  # Select maps with 80% cumulative score
    lr_z=3.0,
    sparsity=0.005,
    n_steps_z=50
)

# Visualize
explainer.visualize_explanation(results, save_path='output.png')
```

The pipeline outputs:
- `explanation_output.png`: Overview showing input, GradCAM, and per-channel decomposition
- `atom_activations_detail.png`: Detailed spatial activation patterns for top atoms

## Using the Trained ConvSAE

After training with `run_csae.py`, you can use the ConvSAE for analyzing activation maps:

```python
import torch
import joblib
from src.convsae import ConvSAE

# Method 1: Load full model (easiest)
csae_model = joblib.load('csae_model.pkl').to(device)

# Method 2: Load state dict (if you need to recreate the model)
csae_model = ConvSAE(in_channels=1, hidden_dim=4096, kernel_size=1).to(device)
csae_model.load_state_dict(torch.load('csae_model.pth'))

# Load training info
training_info = joblib.load('csae_training_info.pkl')
print(f"Training config: {training_info['config']}")
print(f"Final metrics: {training_info['final_metrics']}")

# Use the model for inference
csae_model.eval()
with torch.no_grad():
    # activation_map: [1, 1, H, W] - single activation map
    reconstruction, sparse_features = csae_model(activation_map)

    # sparse_features: [1, 4096, H, W] - sparse representation
    # reconstruction: [1, 1, H, W] - reconstructed activation map

    # Analyze sparsity
    active_features = (sparse_features > 0).float().mean().item()
    print(f"Active features: {active_features * 100:.2f}%")

    # Get top-k activated features
    feature_importance = sparse_features.sum(dim=(2, 3)).squeeze()  # [4096]
    top_k = 10
    top_features = torch.topk(feature_importance, top_k)
    print(f"Top {top_k} features: {top_features.indices.tolist()}")
```

The ConvSAE provides:
- Sparse representation of activation maps in a learned feature space
- Reconstruction for validating the learned representation
- Feature-level analysis (which features activate for which patterns)

## Using the ConvSAE Explanation Pipeline

The `explain_csae.py` module provides a complete explainability system using the trained ConvSAE. Basic usage:

```bash
# Explain a single image
python explain_csae.py --image_path data/imagenette/tench/n01440764_1.JPEG

# Analyze multiple images from a class
python explain_csae.py --class_name tench --num_images 3

# Customize parameters
python explain_csae.py \
  --image_path data/imagenette/church/n03028079_15.JPEG \
  --cumulative_threshold 0.8 \
  --top_k_features 10 \
  --max_channels 5
```

Programmatic usage:

```python
from explain_csae import CSAEExplainer
from src.model import FineTunedModel
import joblib

# Load model and CSAE
model = FineTunedModel(num_classes=10).to(device)
model.load_state_dict(torch.load('weights/finetune_weights.pth'))
csae_model = joblib.load('csae_model.pkl').to(device)

# Create explainer
explainer = CSAEExplainer(model, csae_model, device)

# Generate explanation
results = explainer.explain_prediction(
    image_path='data/imagenette/tench/n01440764_1.JPEG',
    cumulative_threshold=0.8,  # Select maps with 80% cumulative score
    top_k_features=10          # Analyze top 10 CSAE features per channel
)

# Visualize
explainer.visualize_explanation(results, save_path='csae_explanation.png')
```

The CSAE explanation pipeline:
1. Uses GradCAM to compute channel importance weights at layer 5
2. Selects top-k activation maps with cumulative score ≥ threshold (default 80%)
3. For each selected activation map, applies robust normalization (same as training)
4. Passes normalized activation through trained CSAE encoder
5. Identifies top-k activated CSAE features and their spatial patterns
6. Visualizes: input image, GradCAM, per-channel activation maps, CSAE reconstructions, top features, and spatial activation patterns

The pipeline outputs:
- `csae_explanation_{image_name}.png`: Comprehensive visualization showing:
  - Input image and GradCAM overlay
  - Original and normalized activation maps for selected channels
  - CSAE reconstruction quality (MSE)
  - Top 5 activated features per channel (bar chart with decoder weights)
  - Spatial activation patterns of the top feature

## Analyzing Same-Class ConvSAE Features

The `check_same_class_csae.py` script analyzes whether images from the same class activate similar ConvSAE features, validating semantic consistency.

### Single Class Analysis

```bash
python check_same_class_csae.py --class_name tench --num_images 10
```

This will:
1. Sample 10 images from the "tench" class
2. For each image, extract layer 5 activation maps using GradCAM
3. Apply CSAE to get sparse feature activations
4. Identify top-k activated features per image
5. Compute feature frequency (how many images activate each feature)
6. Generate visualization showing:
   - Sample images from the class
   - Top 20 most frequently activated features (bar chart with percentages)
   - Feature activation heatmap (images × features matrix)

Output: `same_class_csae_analysis/{class_name}_csae_analysis.png`

### Cross-Class Comparison

```bash
python check_same_class_csae.py --compare_classes tench church parachute --num_images 10
```

This will:
1. Analyze each class independently (as above)
2. Build feature-class activation matrix
3. Identify class-specific features (high activation in one class, low in others)
4. Generate comparison visualization showing:
   - Feature-class heatmap (which features activate for which classes)
   - Top 20 class-specific features with their dominant class
   - Class specificity scores

Output:
- Individual class analyses: `same_class_csae_analysis/{class_name}_csae_analysis.png`
- Comparison: `same_class_csae_analysis/class_comparison_csae.png`

### Interpretation

- **High-frequency features** (appearing in >70% of same-class images): Represent consistent class-specific patterns
- **Class-specific features** (high specificity score): Strongly associated with one class, useful for discrimination
- **Shared features** (low specificity): General patterns used across multiple classes
- **Heatmap patterns**: Vertical bands indicate features consistently used by specific classes

### Parameters

- `--num_images`: Number of images to sample per class (default: 10)
- `--top_k_features`: Number of top features to track per image (default: 50)
- `--cumulative_threshold`: GradCAM channel selection threshold (default: 0.8)

## Analyzing Same-Class Dual ConvSAE Features

The `check_same_class_dual_csae.py` script analyzes feature activation patterns in BOTH the shared pathway (global features) and class-specific pathway (discriminative features) of the Dual ConvSAE model. This validates that the dual-pathway architecture successfully separates common patterns from class-discriminative patterns.

**Key Differences from Single ConvSAE Analysis:**
- Analyzes **two separate pathways**: shared (global) and class-specific (discriminative)
- Compares how shared vs class features differ in their activation patterns
- Validates architectural design: shared pathway should capture common patterns, class pathway should capture discriminative patterns
- Works with both AlexNet (`dual_csae_model.pkl`) and ResNet18 (`dual_csae_resnet18_model.pkl`) backbones

### Single Class Analysis

```bash
# Analyze with AlexNet backbone
python check_same_class_dual_csae.py --class_name tench --num_images 10

# Analyze with ResNet18 backbone
python check_same_class_dual_csae.py --class_name tench --num_images 10 --use_resnet18 --dual_csae_path dual_csae_resnet18_model.pkl
```

This will:
1. Sample 10 images from the "tench" class
2. For each image, extract activation maps using GradCAM
3. Apply Dual CSAE to get sparse features from BOTH pathways
4. Identify top-k activated features per pathway per image
5. Compute feature frequency for both pathways
6. Generate comprehensive visualization showing:
   - Sample images from the class
   - **Shared pathway**: Top 15 most frequently activated features (bar chart, green theme)
   - **Class pathway**: Top 15 most frequently activated features (bar chart, orange theme)
   - **Shared pathway heatmap**: Feature activation patterns across images (images × features matrix)
   - **Class pathway heatmap**: Feature activation patterns across images (images × features matrix)
   - Decoder weight distributions for both pathways

Output: `same_class_dual_csae_analysis/{class_name}_dual_csae_analysis.png`

**Visualization Structure:**
- **Row 1**: Sample images + Shared features bar chart + Class features bar chart
- **Rows 2-3**: Shared features heatmap (left) + Class features heatmap (right)
- **Row 4**: Shared decoder weights histogram + Class decoder weights histogram

### Cross-Class Comparison

```bash
# Compare with AlexNet backbone
python check_same_class_dual_csae.py --compare_classes tench church parachute --num_images 10

# Compare with ResNet18 backbone
python check_same_class_dual_csae.py --compare_classes tench church parachute --num_images 10 --use_resnet18 --dual_csae_path dual_csae_resnet18_model.pkl
```

This will:
1. Analyze each class independently (as above)
2. Build feature-class activation matrices for BOTH pathways
3. Identify class-specific features in the class pathway
4. Generate comprehensive comparison visualization showing:
   - **Top half**: Shared features across classes (heatmap + per-class summary)
   - **Bottom half**: Class-specific features across classes (heatmap + top discriminative features)

Output:
- Individual class analyses: `same_class_dual_csae_analysis/{class_name}_dual_csae_analysis.png`
- Comparison: `same_class_dual_csae_analysis/class_comparison_dual_csae.png`

**Visualization Structure:**
- **Top row**: Shared features heatmap (left, 60 features × classes) + per-class average activation (right)
- **Bottom row**: Class features heatmap (left, 60 features × classes) + top 12 class-specific features (right)

### Interpretation

**Shared Pathway Analysis:**
- **High cross-class activation**: Features that activate strongly across ALL classes represent global patterns (edges, textures, low-level features)
- **Similar activation across classes**: Validates that shared pathway captures commonalities, not class-discriminative patterns
- **Balanced activation**: All classes should activate shared features roughly equally

**Class Pathway Analysis:**
- **High-frequency features** (appearing in >70% of same-class images): Represent consistent class-specific patterns
- **Class-specific features** (high specificity score): Strongly discriminative, activate for one class but not others
- **Vertical bands in heatmap**: Indicate features consistently used by specific classes (good class separation)
- **Low cross-class activation**: Validates that class pathway learns discriminative patterns

**Validation Metrics:**
- **Good separation**: Shared features have low specificity scores (used by all classes), class features have high specificity scores (used by specific classes)
- **Classification accuracy**: Should be 60-90% if class pathway is learning discriminative features
- **Diversity loss**: Should decrease during training, indicating classes use different features

### Parameters

- `--num_images`: Number of images to sample per class (default: 10)
- `--top_k_features`: Number of top features to track per image (default: 50)
- `--cumulative_threshold`: GradCAM channel selection threshold (default: 0.8)
- `--use_resnet18`: Use ResNet18 backbone instead of AlexNet (flag)
- `--dual_csae_path`: Path to dual CSAE model (default: `dual_csae_model.pkl`)
- `--model_path`: Path to fine-tuned AlexNet model (default: `weights/finetune_weights.pth`, ignored if using ResNet18)
- `--data_dir`: Path to Imagenette dataset (default: `data/imagenette`)

### Use Cases

**When to use Dual ConvSAE analysis:**
1. **Validate architecture**: Confirm that shared and class pathways learn different types of features
2. **Feature interpretability**: Understand which features are global (shared) vs discriminative (class)
3. **Cross-class analysis**: Identify which class-specific features distinguish between classes
4. **Model debugging**: Check if class pathway is learning meaningful discriminative patterns (high specificity scores)
5. **Backbone comparison**: Compare AlexNet vs ResNet18 feature learning

**Expected patterns:**
- **Shared features**: Should show similar activation levels across all classes (low variance)
- **Class features**: Should show high activation for one class, low for others (high variance)
- **Feature overlap**: Minimal overlap between classes in the class pathway (good separation)
- **Decoder weights**: Class pathway weights should be more diverse than shared pathway weights

## Common Modifications

When modifying the sparse coding solver (`solve_z_prox_adam`):
- Line 64 in `src/batch_cdl_large.py`
- Critical parameters: learning rate, momentum, number of iterations
- Must maintain non-negativity constraint and sparsity penalty

When changing the target layer for activation extraction:
- Modify `target_layer = model.feature_extractor[X]` in `run_cdl.py`
- Earlier layers = lower-level features (edges, textures)
- Later layers = higher-level features (object parts)
- Default: Layer 5 (after 2nd conv block) balances spatial resolution and semantic content

When adjusting the explanation pipeline (`explain.py`):
- `cumulative_threshold`: Controls how many activation maps to analyze (default 0.8 = 80%)
- `lr_z` and `sparsity`: Control sparse coding quality (match training values: 3.0 and 0.005)
- `n_steps_z`: More steps = better decomposition but slower (default 50)
- `top_atoms_per_map`: How many atoms to show per activation map (default 5)

When adjusting the CSAE explanation pipeline (`explain_csae.py`):
- `cumulative_threshold`: Controls how many activation maps to analyze (default 0.8 = 80%)
- `top_k_features`: Number of top CSAE features to analyze per channel (default 10)
- `max_channels`: Maximum number of channels to visualize (default 5)
- The robust normalization (99th percentile scaling × 10) is automatically applied, matching training preprocessing
- Each visualization shows: original activation, normalized activation, CSAE reconstruction, top features with decoder weights, and spatial patterns

When adjusting ConvSAE training (`run_csae.py`):
- `hidden_dim`: Increase for more expressive features, decrease for more compact representation (default: 4096)
- `kernel_size`: Use 1 for spatial sparsity (features activate at specific locations), use 3+ for more spatial context
- `lambda_l1`: **Critical for sparsity control** (default: 0.05)
  - If "Active Neurons %" > 30%: **Increase** `lambda_l1` (try 0.1, 0.2, or higher)
  - If "Active Neurons %" < 2%: **Decrease** `lambda_l1` (try 0.01, 0.005)
  - **Target range**: 5-15% active neurons
- `lambda_lat`: Higher values = stronger lateral inhibition, prevents blob-like activations (default: 0.02)
- Training will automatically warn you if sparsity is outside the target range
- If reconstruction loss plateaus early while sparsity is good, the model has converged successfully

## Prototypical Part Network (ProtoPNet) Implementation

The `run_protopnet_full.py` script implements the **"This Looks Like That"** interpretable classification approach from Chen et al. (NeurIPS 2019). Unlike the original paper which focused on fine-grained classification (birds, cars), this implementation is designed for **full ImageNet-1k** with prototypes for ALL classes, enabling direct comparison with ConvSAE and Dual ConvSAE.

### Key Differences from Original Paper

**Original ProtoPNet (Chen et al., 2019)**:
- Designed for fine-grained classification (CUB-200-2011 birds: 200 classes, Stanford Cars: 196 classes)
- 10 prototypes per class
- Total prototypes: 2000 (200 classes × 10) or 1960 (196 classes × 10)
- Cropped images with bounding boxes
- Focused on interpretability for specific domains

**Our Implementation (ProtoPNet Full)**:
- **Designed for ImageNet-1k (1000 classes)**
- **Prototypes for ALL classes** (not class-specific like original)
- Default: 2 prototypes per class = 2000 total prototypes (configurable)
- Full images (no bounding boxes required)
- Enables comparison with ConvSAE/Dual ConvSAE feature learning
- Same architecture but scaled for large-scale classification

### Architecture Overview

```
Input Image
    ↓
Convolutional Backbone (f)
  (ResNet50/ResNet18/VGG16 pretrained on ImageNet)
    ↓
Add-on Layers (1×1 conv)
  (Sigmoid activation on last layer)
    ↓
Prototype Layer (gp)
  - Learns m prototypes P = {p_j}
  - Each prototype: 1×1×D (D = feature channels)
  - Computes L2 distance to all patches
  - Converts distance to similarity score
  - Global max pooling → similarity score per prototype
    ↓
Fully Connected Layer (h) [no bias]
  - Weights connect prototypes to classes
  - w_{k,j} = 1 if prototype j ∈ class k
  - w_{k,j} ≈ 0 for prototypes not in class k (learned via L1)
    ↓
Output Logits (1000 classes)
```

### Prototype Layer Details

For each prototype p_j and input features z = f(x):

1. **Distance Computation**: Compute squared L2 distance to all patches
   ```
   d²(z, p_j) = ||z - p_j||² = ||z||² + ||p_j||² - 2·z^T·p_j
   ```

2. **Similarity Conversion**:
   ```
   similarity = log((d² + 1) / (d² + ε))
   ```
   - Monotonically decreasing with distance
   - High similarity = low distance

3. **Global Max Pooling**: Take maximum similarity across all spatial locations
   ```
   activation_j = max_{all patches} similarity(patch, p_j)
   ```

### Training Procedure (3 Stages)

#### Stage 1: Joint Training (Warm + Joint Optimization)

**Warm-up Phase** (default: 5 epochs):
- Fix: Convolutional backbone
- Train: Add-on layers + Prototypes
- Goal: Initialize prototypes with reasonable values

**Joint Phase** (default: 20 epochs):
- Train: Convolutional backbone (small LR) + Add-on layers + Prototypes
- Fix: Last layer weights
  - w_{k,j} = 1.0 if prototype j belongs to class k
  - w_{k,j} = -0.5 otherwise
- Objective:
  ```
  Loss = CrossEntropy + λ_clst · Clst + λ_sep · Sep
  ```

**Cluster Loss (Clst)**:
```
Clst = (1/n) Σ_i min_{j: p_j ∈ P_{y_i}} min_{z ∈ patches} ||z - p_j||²
```
- Encourages each image to have patches close to its class prototypes
- Pushes same-class images to cluster around class prototypes

**Separation Loss (Sep)**:
```
Sep = -(1/n) Σ_i min_{j: p_j ∉ P_{y_i}} min_{z ∈ patches} ||z - p_j||²
```
- Encourages images to stay far from other-class prototypes
- Negative sign: we minimize negative distance = maximize distance
- Helps prototypes become class-discriminative

#### Stage 2: Prototype Projection (Push)

After joint training, project each prototype to the nearest training patch from its class:

```python
For each prototype p_j of class k:
    1. Find all training images from class k
    2. Compute distances from p_j to all patches in these images
    3. Find the patch with minimum distance
    4. Update p_j ← that patch
```

**Why this matters**:
- Makes prototypes visualizable as actual image patches
- Each prototype = real training patch, not abstract latent vector
- Enables "this looks like that" explanations with concrete examples
- Theorem 2.1 in paper: projection doesn't hurt accuracy if Clst is well-optimized

#### Stage 3: Last Layer Optimization (Convex)

Optimize last layer weights with L1 regularization:

```
Loss = CrossEntropy + λ_L1 · Σ_{k,j: p_j ∉ P_k} |w_{k,j}|
```

**Goal**: Make incorrect connections sparse
- w_{k,j} should be ≈ 0 if prototype j doesn't belong to class k
- Reduces negative reasoning: "this is class k because it's NOT like other classes"
- Encourages positive reasoning: "this is class k because it looks like class k prototypes"

### Usage Examples

#### Basic Training (ResNet50, 2 prototypes/class)
```bash
python run_protopnet_full.py
```

Output:
- Model: `protopnet_resnet50_p2_model.pkl` (full model)
- Weights: `protopnet_resnet50_p2_model.pth` (state dict)
- Logs: `protopnet_resnet50_p2_logs.png` (training curves)
- Training info: `protopnet_resnet50_p2_training_info.pkl` (config + metrics)

#### Custom Configuration
```bash
# ResNet18 with 5 prototypes per class
python run_protopnet_full.py --model resnet18 --num_prototypes_per_class 5

# Custom training schedule
python run_protopnet_full.py \
  --warm_epochs 10 \
  --joint_epochs 30 \
  --last_layer_epochs 20 \
  --push_every 5

# Adjust loss weights
python run_protopnet_full.py \
  --lambda_clst 0.8 \    # Cluster loss weight
  --lambda_sep -0.08 \   # Separation loss weight (negative)
  --lambda_l1 1e-4       # L1 regularization for last layer
```

### Hyperparameters

**Model Architecture**:
- `--model`: Backbone (resnet50, resnet18, vgg16)
- `--num_prototypes_per_class`: Number of prototypes per class (default: 2)
  - Total prototypes = 1000 × num_prototypes_per_class
  - More prototypes = more expressive but slower

**Training Schedule**:
- `--warm_epochs`: Warm-up epochs (default: 5)
  - Train only add-on layers + prototypes
- `--joint_epochs`: Joint training epochs (default: 20)
  - Train backbone + add-on + prototypes
- `--last_layer_epochs`: Last layer optimization epochs (default: 10)
- `--push_every`: Push prototypes every N epochs (default: 5)
  - During joint training, periodically project prototypes

**Optimization**:
- `--lr`: Base learning rate (default: 1e-4)
  - Backbone: lr/10
  - Add-on layers: lr
  - Prototypes: 3·lr (learn faster)
- `--batch_size`: Training batch size (default: 32)

**Loss Weights**:
- `--lambda_clst`: Cluster loss weight (default: 0.8)
  - Higher = stronger clustering
  - Typical range: 0.5 - 1.0
- `--lambda_sep`: Separation loss weight (default: -0.08)
  - More negative = stronger separation
  - Typical range: -0.1 to -0.05
- `--lambda_l1`: L1 regularization for last layer (default: 1e-4)
  - Higher = sparser incorrect connections
  - Typical range: 1e-5 to 1e-3

### Expected Training Behavior

**Warm-up Phase** (epochs 1-5):
- **Cluster Loss**: Should decrease rapidly (from ~100 to ~10)
- **Separation Loss**: Should decrease (become more negative)
- **Accuracy**: Should reach 20-40%
- **Sign of success**: Prototypes start to capture class-specific patterns

**Joint Training Phase** (epochs 6-25):
- **Cross Entropy**: Decreases steadily
- **Cluster Loss**: Continues decreasing (to ~1-5)
- **Separation Loss**: Continues decreasing (more negative)
- **Accuracy**: Should reach 60-75% on ImageNet-1k
- **Sign of success**: After each push, accuracy should not drop significantly

**Last Layer Optimization** (epochs 26-35):
- **Cross Entropy**: Small decrease
- **L1 Penalty**: Decreases as incorrect connections become sparse
- **Accuracy**: May improve slightly (1-2%)
- **Sign of success**: Most w_{k,j} for j ∉ P_k should be near 0

### Interpretability: "This Looks Like That"

**How to interpret predictions**:

1. **Forward pass**: Get logits = h(gp(f(x)))
2. **Prototype activations**: For each prototype, find activation score
3. **Upsampling**: Upsample activation map to image size → find activated region
4. **Visualization**: Show:
   - Input image with bounding box around activated region
   - Prototype image (training patch where prototype was pushed)
   - Activation heatmap
   - Similarity score
5. **Reasoning**:
   ```
   "This image is class k because:
   - This part (bounding box) looks like prototype p_j (similarity: 6.5)
   - This part looks like prototype p_m (similarity: 4.2)
   - ...
   Total evidence for class k: Σ w_{k,j} · activation_j"
   ```

### Comparison with ConvSAE and Dual ConvSAE

| Aspect | ProtoPNet | ConvSAE | Dual ConvSAE |
|--------|-----------|---------|--------------|
| **Learning** | Supervised (class labels) | Unsupervised | Semi-supervised |
| **Prototypes** | Explicit (learned vectors) | Implicit (decoder weights) | Explicit (shared + class) |
| **Interpretability** | Case-based reasoning | Feature reconstruction | Pathway separation |
| **Number of Features** | 2000 (2/class × 1000) | 2048-8192 | 512 (shared + class) |
| **Similarity Metric** | L2 distance in latent space | Reconstruction error | L2 distance + classification |
| **Training Complexity** | 3-stage (complex) | Single-stage | Single-stage |
| **Visualization** | Real image patches | Decoder weights | Decoder weights + class info |
| **Use Case** | When interpretability is critical | When features are primary goal | When class separation is needed |

**When to use ProtoPNet**:
- Need human-interpretable explanations with concrete examples
- Want case-based reasoning ("this looks like that")
- Classification is the primary task
- Have labeled training data

**When to use ConvSAE/Dual ConvSAE**:
- Want to learn features from activations
- Need unsupervised or semi-supervised learning
- Want to analyze feature usage across classes
- Focus on representation learning rather than classification

### Output Files

After training, the following files are saved:

1. **`protopnet_<model>_p<N>_model.pth`**: PyTorch state dict
   - Use for loading weights: `model.load_state_dict(torch.load(...))`

2. **`protopnet_<model>_p<N>_model.pkl`**: Full model (joblib)
   - Use for inference: `model = joblib.load(...)`

3. **`protopnet_<model>_p<N>_training_info.pkl`**: Training config and logs
   ```python
   info = joblib.load('protopnet_resnet50_p2_training_info.pkl')
   print(info['config'])  # Model configuration
   print(info['logs'])    # Training metrics per epoch
   ```

4. **`protopnet_<model>_p<N>_logs.png`**: Training curves (6 subplots)
   - Total loss, Cross entropy, Cluster loss
   - Separation loss, Training accuracy, Loss components

### Common Issues and Solutions

#### Out of Memory (OOM) Errors

**Symptom**: `RuntimeError: CUDA out of memory` or `torch.cuda.OutOfMemoryError`

**Memory Consumption Factors** (in order of impact):

1. **Batch Size** (Primary Factor)
   - Memory scales linearly with batch size
   - **Default**: 32 (requires ~12-16GB GPU for ResNet50)
   - **Recommendation**: Start with batch_size=8, increase if no OOM

2. **Backbone Model**
   - ResNet50: ~23M params, requires ~4-6GB base memory
   - ResNet18: ~11M params, requires ~2-3GB base memory
   - VGG16: ~138M params, requires ~8-10GB base memory
   - **Recommendation**: Use ResNet18 for limited GPU memory

3. **Number of Prototypes**
   - Memory for prototype distance computation: O(B × num_prototypes × H × W)
   - Default: 2000 prototypes (2 per class × 1000)
   - **Recommendation**: Keep at 2 per class unless GPU has >24GB memory

4. **Image Resolution**
   - Default: 224×224 (standard ImageNet)
   - Activation map size: 14×14 for layer3
   - **Not recommended to change** (pretrained weights expect 224×224)

**Quick Fixes by GPU Memory**:

| GPU Memory | Model      | Batch Size | Prototypes/Class | Command |
|-----------|-----------|------------|------------------|---------|
| 8GB       | ResNet18  | 8          | 2                | `--model resnet18 --batch_size 8` |
| 12GB      | ResNet18  | 16         | 2                | `--model resnet18 --batch_size 16` |
| 12GB      | ResNet50  | 8          | 2                | `--model resnet50 --batch_size 8` |
| 16GB      | ResNet50  | 16         | 2                | `--model resnet50 --batch_size 16` |
| 24GB      | ResNet50  | 32         | 2                | `--model resnet50 --batch_size 32` |
| 24GB      | ResNet50  | 16         | 5                | `--model resnet50 --batch_size 16 --num_prototypes_per_class 5` |

**Step-by-Step OOM Debugging**:

1. **First, try reducing batch size**:
   ```bash
   # If batch_size=32 fails, try:
   python run_protopnet_full.py --batch_size 16

   # Still OOM? Try:
   python run_protopnet_full.py --batch_size 8

   # Extreme case (very limited memory):
   python run_protopnet_full.py --batch_size 4
   ```

2. **Second, switch to smaller backbone**:
   ```bash
   # If ResNet50 fails, use ResNet18:
   python run_protopnet_full.py --model resnet18 --batch_size 16
   ```

3. **Third, reduce prototypes per class** (only if still OOM):
   ```bash
   # Default is 2, can reduce to 1 (not recommended for quality):
   python run_protopnet_full.py --model resnet18 --batch_size 8 --num_prototypes_per_class 1
   ```

4. **Monitor GPU memory during training**:
   ```bash
   # In another terminal, watch GPU usage:
   watch -n 0.5 nvidia-smi
   ```

**OOM During Prototype Projection (Push)**:

If OOM occurs specifically during the push operation:

```python
# In run_protopnet_full.py, modify push_prototypes() function:
# Process images in smaller batches during push

# Current code processes full batch, change to:
for images, labels in tqdm(dataloader, desc="Finding nearest patches"):
    # Split batch into smaller chunks if needed
    chunk_size = 8  # Process 8 images at a time
    for i in range(0, images.size(0), chunk_size):
        chunk_images = images[i:i+chunk_size].to(device)
        chunk_labels = labels[i:i+chunk_size].to(device)
        # ... rest of code
```

**Memory Optimization Tips**:

1. **Clear cache periodically**:
   - Already implemented in the code
   - `torch.cuda.empty_cache()` called during training

2. **Use mixed precision training** (for compatible GPUs):
   ```python
   # Add to training loop:
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()

   with autocast():
       logits, _, min_distances, _ = model(images, return_distances=True)
       loss = ...

   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **Gradient accumulation** (simulate larger batch with less memory):
   ```python
   # Effective batch size = batch_size × accumulation_steps
   # E.g., batch_size=8, accumulation_steps=4 → effective batch_size=32

   # Add to code:
   accumulation_steps = 4
   for batch_idx, (images, labels) in enumerate(dataloader):
       loss = ... / accumulation_steps
       loss.backward()

       if (batch_idx + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

**Expected Memory Usage** (approximate):

| Configuration | GPU Memory Required |
|--------------|---------------------|
| ResNet18, batch=8, proto=2/class | ~6 GB |
| ResNet18, batch=16, proto=2/class | ~8 GB |
| ResNet50, batch=8, proto=2/class | ~10 GB |
| ResNet50, batch=16, proto=2/class | ~14 GB |
| ResNet50, batch=32, proto=2/class | ~20 GB |

#### Training Issues

**Issue**: Cluster loss not decreasing
- **Cause**: Learning rate too low for prototypes
- **Solution**: Increase prototype learning rate (try 5·lr instead of 3·lr)

**Issue**: Accuracy drops after prototype projection (push)
- **Cause**: Cluster loss not well-optimized before push
- **Solution**: Train longer before first push, or push less frequently

**Issue**: Last layer has many non-zero incorrect connections
- **Cause**: L1 penalty too weak
- **Solution**: Increase `--lambda_l1` (try 1e-3 or 1e-2)

**Issue**: Training very slow
- **Cause**: Too many prototypes or large batch size
- **Solution**: Reduce `--num_prototypes_per_class` or `--batch_size`

**Issue**: Prototypes not class-discriminative
- **Cause**: Separation loss too weak
- **Solution**: Increase magnitude of `--lambda_sep` (make more negative, e.g., -0.1)

### Advanced: Multi-Scale Prototypes

The current implementation uses 1×1×D prototypes (point prototypes). To use larger spatial prototypes:

```python
# Modify in run_protopnet_full.py
prototype_shape = (num_prototypes, channels, 3, 3)  # 3×3 prototypes
```

**Trade-offs**:
- Larger prototypes: Capture more spatial context, slower computation
- Smaller prototypes (1×1): Faster, more flexible, less spatial context
- Original paper uses 1×1 for fine-grained tasks
