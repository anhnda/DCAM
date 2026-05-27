python df_decomposition.py --model resnet50 --target_layer layer3 --basis bn --D 200
# good quality, ~2-3 min
python df_decomposition.py --model resnet50 --target_layer layer4 \
    --basis distill --D 200 \
    --distill_batches 8 --distill_bs 16 --distill_iters 100 \
    --distill_harvest_passes 4 --distill_harvest_bs 64

# publication-grade, ~10 min
python df_decomposition.py --model resnet50 --target_layer layer3 \
    --basis distill --D 200 \
    --distill_batches 16 --distill_bs 16 --distill_iters 1500 \
    --distill_harvest_passes 8 --distill_harvest_bs 64
python df_hier_visualization.py  --df_basis df_basis_resnet50_layer3_kernel_D200.pkl --class_id 281 --offset 3
python df_hier_visualization.py  --df_basis df_basis_resnet50_layer3_distill.pkl --class_id 281 --offset 3
python df_hier_visualization.py  --df_basis df_basis_resnet50_layer3_bn_D200.pkl  --offset 1 --class_id 101
python df_hier_visualization.py  --pca  --offset 1 --class_id 101 --target_layer 3
python df_decomposition.py --model resnet50 --target_layer layer3 --basis distill --D 200

python df_hier_visualization.py  --df_basis df_basis_resnet50_layer4_distill_D200.pkl  --offset 1 --class_id 101
python df_hier_visualization.py  --pca  --offset 1 --class_id 101 --target_layer layer4


    df_basis_resnet50_layer3_distill.pkl
python df_hier_visualization.py --image cat.jpg --df_basis df_basis_resnet50_layer3_kernel_D200.pkl

python eval_unified_cam.py --model resnet50 --class_id 10 \
    --methods grad_cam eigen_cam dcam \
    --lambda_sweep 0.0 0.25 0.5 0.75 1.0 \
    --bandwidth_sweep 0.4 0.8 1.6 \
    --D 50 --n_images 50
python csae_pca_baseline.py --cache_dir cache_activations \
  --cache_key resnet50_layer3_thresh0p95_samples50000_chunk100_gcmap1 \
  --D 0 --device cuda --save pca_baseline_resnet50_D0_model.pkl

python check_drop_csae_fixed.py --model efficientnet \
  --csae_model pca_baseline_efficientnet_D200_model.pkl \
  --norm_mode per_batch --debug_batches 3

python csae_pca_baseline.py \
  --cache_dir cache_activations \
  --cache_key resnet50_layer3_thresh0p95_samples50000_chunk100_gcmap1 \
  --D 700 --device cuda \
  --save pca_baseline_resnet50_D700_model.pkl
python check_drop_csae_fixed.py --model resnet50 \
  --csae_model pca_baseline_resnet50_D200_model.pkl \
  --norm_mode per_image --debug_batches 3

python csae_sb_anchor.py \
  --cache_dir cache_activations \
  --cache_key resnet50_layer3_thresh0p95_samples50000_chunk100_gcmap1 \
  --num_classes 1000 --D 256 --pool none \
  --verify_seeds 0 1 2 --device cuda \
  --save csae_sb_anchor_W0.npy

python csae_svd_anchor.py \
    --cache_dir cache_activations \
    --cache_key resnet50_layer3_thresh0p95_samples50000_chunk100_gcmap1 \
    --D 500 --verify_seeds 0 1 2 --device cuda

python csae_stable.py --ica_anchor csae_svd_anchor_W0.npy     --anchor_mode subspace --lambda_anchor 0.01 --model_seed 0 --epochs 5
python check_drop_csae_fixed.py     --model resnet50     --csae_model imagenet1k_csae_stable_resnet50_subspace_la0.01_seed42-0_model.pkl     --norm_mode per_image --debug_batches 3

python check_drop_csae_fixed.py     --model resnet50     --csae_model imagenet1k_csae_stable_resnet50_subspace_la0.01_seed42-0_model.pth     --norm_mode per_image --debug_batches 3

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