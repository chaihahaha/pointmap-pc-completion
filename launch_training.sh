#!/bin/bash
# ===================================================================
# PointSea & PointMap 30-Epoch Training Launcher
# Run this in YOUR terminal (not through opencode):
#
#   bash ~/SVDFormer_PointSea/launch_training.sh
#
# Or to run both sequentially:
#   bash ~/SVDFormer_PointSea/launch_training.sh
# ===================================================================
set -e
cd /home/ordhktvu3862/SVDFormer_PointSea
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointattn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

echo "=== PointSea 30-epoch training ==="
echo "Log: /tmp/pointsea_train.log"
python -u train_unified.py \
    --method pointsea \
    --epochs 30 \
    --batch_size 2 \
    --output_dir pointsea_30ep \
    2>&1 | tee /tmp/pointsea_train.log
echo "PointSea done (exit=$?)"

echo ""
echo "=== PointMap 30-epoch training ==="
echo "Log: /tmp/pointmap_train.log"
python -u train_unified.py \
    --method pointmap \
    --epochs 30 \
    --batch_size 2 \
    --output_dir pointmap_30ep \
    2>&1 | tee /tmp/pointmap_train.log
echo "PointMap done (exit=$?)"

echo ""
echo "=== All training complete! ==="
echo "PointSea checkpoint: pointsea_30ep_*/checkpoints/ckpt-best.pth"
echo "PointMap checkpoint: pointmap_30ep_*/checkpoints/ckpt-best.pth"
