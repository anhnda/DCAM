python csae_svd_anchor.py \
    --cache_dir cache_activations \
    --cache_key resnet50_layer3_thresh0p85_samples50000_chunk100_gcmap1 \
    --D 200 --verify_seeds 0 1 2 --device cuda
    
python csae_stable.py --ica_anchor csae_svd_anchor_W0.npy     --anchor_mode subspace --lambda_anchor 0.01 --model_seed 0 --epochs 5

python csae_stable.py --ica_anchor csae_ica_anchor_W0.npy \
       --anchor_mode subspace --lambda_anchor 1.0 --model_seed 0