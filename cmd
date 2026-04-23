PROTOPNET: 
python run_protopnet_full.py --model resnet50 --batch_size 8   

################################################################################
# ImageNet-1k Multi-Model ConvSAE Commands
# Updated: 2024 - ResNet50 is now the default backbone
################################################################################

################################################################################
# 1. TRAINING: run_xcsae_full.py
#    Train Multi-Channel ConvSAE on full ImageNet-1k (1000 classes, 50 imgs/class)
################################################################################

# ResNet50 (default) - 1024 channels at layer3
python run_xcsae_full.py

# ResNet50 with custom target layer (layer2: higher resolution, fewer channels)
python run_xcsae_full.py --model resnet50 --target_layer layer2

# ResNet18 - 256 channels at layer3
python run_xcsae_full.py --model resnet18

# ResNet18 early layer (layer2: higher resolution, fewer channels)
python run_xcsae_full.py --model resnet18 --target_layer layer2

# VGG16 - 256 channels at features[16]
python run_xcsae_full.py --model vgg16

# VGG16 deeper layer (more semantic features)
python run_xcsae_full.py --model vgg16 --target_layer features[23]

# EfficientNet-B0 - ~80 channels at features[4]
python run_xcsae_full.py --model efficientnet

# Force resample training dataset (re-extract 50 images/class)
python run_xcsae_full.py --model resnet50 --force_resample


################################################################################
# 2. EVALUATION: check_acc_drop_full.py
#    Evaluate accuracy drop with CSAE reconstruction (5K test samples)
################################################################################

# ResNet50 (default) - auto-detects imagenet1k_csae_resnet50_model.pkl
python check_acc_drop_full.py

# ResNet18 - auto-detects imagenet1k_csae_resnet18_model.pkl
python check_acc_drop_full.py --model resnet18

# VGG16 - auto-detects imagenet1k_csae_vgg16_model.pkl
python check_acc_drop_full.py --model vgg16

# EfficientNet - auto-detects imagenet1k_csae_efficientnet_model.pkl
python check_acc_drop_full.py --model efficientnet

# Force resample test set (re-extract 5 images/class from validation set)
python check_acc_drop_full.py --model resnet50 --force_resample

# Custom CSAE model path
python check_acc_drop_full.py \
  --model resnet50 \
  --csae_model my_custom_csae_model.pkl

# Custom batch size for evaluation
python check_acc_drop_full.py --model vgg16 --batch_size 64

# Custom output file for results
python check_acc_drop_full.py \
  --model efficientnet \
  --output_file my_results.txt


################################################################################
# 3. VISUALIZATION: visualize_testmf_full.py
#    Visualize CSAE features on test images (2 modes: consistency & random)
################################################################################

# ===== CONSISTENCY MODE: Analyze feature consistency across same-class images =====

# ResNet50 (default) - Analyze 10 classes, 3 images per class
python visualize_testmf_full.py --num_classes 10 --top_k_features 12

# ResNet50 - Custom images per class (e.g., 5 images per class)
python visualize_testmf_full.py \
  --num_classes 5 \
  --images_per_class_viz 5 \
  --top_k_features 16

# ResNet18 - Consistency analysis
python visualize_testmf_full.py \
  --model resnet18 \
  --num_classes 5 \
  --images_per_class_viz 3

# VGG16 - Consistency analysis
python visualize_testmf_full.py \
  --model vgg16 \
  --num_classes 5 \
  --top_k_features 12

# EfficientNet - Consistency analysis
python visualize_testmf_full.py \
  --model efficientnet \
  --num_classes 5 \
  --top_k_features 16

# ===== RANDOM MODE: Visualize random individual test images =====

# ResNet50 (default) - Random mode: 10 individual images
python visualize_testmf_full.py --num_samples 10 --top_k_features 16

# ResNet18 - Random mode
python visualize_testmf_full.py --model resnet18 --num_samples 10 --top_k_features 12

# VGG16 - Random mode
python visualize_testmf_full.py \
  --model vgg16 \
  --num_samples 15 \
  --top_k_features 16

# EfficientNet - Random mode
python visualize_testmf_full.py \
  --model efficientnet \
  --num_samples 10 \
  --top_k_features 12

# ===== ADVANCED OPTIONS =====

# Force resample test images (re-extract from validation set)
python visualize_testmf_full.py \
  --force_resample \
  --test_images_per_class 10 \
  --num_classes 5

# Custom output directory
python visualize_testmf_full.py \
  --num_classes 5 \
  --output_dir my_visualizations/

# Custom CSAE model path
python visualize_testmf_full.py \
  --model resnet50 \
  --csae_model my_custom_csae_model.pkl \
  --num_classes 5


################################################################################
# 4. TYPICAL WORKFLOW
################################################################################

# Step 1: Train ConvSAE on ResNet50
python run_xcsae_full.py --model resnet50

# Step 2: Evaluate accuracy drop
python check_acc_drop_full.py --model resnet50

# Step 3: Visualize feature consistency (10 classes)
python visualize_testmf_full.py --model resnet50 --num_classes 10

# Step 4: Visualize random samples
python visualize_testmf_full.py --model resnet50 --num_samples 20


################################################################################
# 5. MULTI-MODEL COMPARISON WORKFLOW
################################################################################

# Train all models
python run_xcsae_full.py --model resnet50
python run_xcsae_full.py --model resnet18
python run_xcsae_full.py --model vgg16
python run_xcsae_full.py --model efficientnet

# Evaluate all models
python check_acc_drop_full.py --model resnet50
python check_acc_drop_full.py --model resnet18
python check_acc_drop_full.py --model vgg16
python check_acc_drop_full.py --model efficientnet

# Compare visualizations (same 5 classes for all models)
python visualize_testmf_full.py --model resnet50 --num_classes 5
python visualize_testmf_full.py --model resnet18 --num_classes 5
python visualize_testmf_full.py --model vgg16 --num_classes 5
python visualize_testmf_full.py --model efficientnet --num_classes 5


################################################################################
# 6. NOTES
################################################################################

# - Default model is now ResNet50 (1024 channels at layer3)
# - CSAE model paths are auto-detected: imagenet1k_csae_{model}_model.pkl
# - Training samples: 50 images per class (50K total) cached to /data/imagenet1k_sampled
# - Test samples: 5 images per class (5K total) cached to /data/imagenet1k_sampletest
# - Use --force_resample to re-extract samples from parquet files
# - Consistency mode shows features common across images of the same class (green highlights)
# - Random mode visualizes individual test images independently
