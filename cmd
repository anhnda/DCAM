python run_dcam_full.py --model resnet50 \
    --lambda_anchor 0.0 \
    --lambda_l1 0.0 \
    --ridge 1e-3 \
    --lr 1e-3 \
    --epochs 5
for la in 0.001 0.003 0.01 0.03 0.1; do
    python run_dcam_full.py --model resnet50 \
        --lambda_anchor $la \
        --lambda_l1 0.0 \
        --ridge 1e-3 \
        --lr 1e-3 \
        --epochs 5 \
        --model_suffix "_la${la}"
done