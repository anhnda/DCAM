python csae_svd_anchor.py \
    --cache_dir cache_activations \
    --cache_key resnet50_layer3_thresh0p95_samples50000_chunk100_gcmap1 \
    --D 500 --verify_seeds 0 1 2 --device cuda

python csae_stable.py --ica_anchor csae_svd_anchor_W0.npy     --anchor_mode full --lambda_anchor 0.01 --model_seed 0 --epochs 5
python check_drop_csae_fixed.py     --model resnet50     --csae_model imagenet1k_csae_stable_resnet50_subspace_la0.01_seed42-0_model.pkl     --norm_mode per_image --debug_batches 3

python compare_csae_stable_seeds.py \
    imagenet1k_csae_stable_resnet50_subspace_la0.01_seed42-0_model.pth \
    imagenet1k_csae_stable_resnet50_subspace_la0.01_seed42-1_model.pth \
    imagenet1k_csae_stable_resnet50_subspace_la0.01_seed42-2_model.pth


import torch, joblib
from run_xcsae_full import MultiChannelConvSAE
m = MultiChannelConvSAE(in_channels=1024, hidden_dim=8192, kernel_size=1, top_k=128)
m.load_state_dict(torch.load("imagenet1k_csae_stable_resnet50_subspace_la0.01_seed42-0_model.pth"))
m.eval()
joblib.dump(m, "imagenet1k_csae_stable_resnet50_subspace_la0.01_seed42-0_model.pkl")