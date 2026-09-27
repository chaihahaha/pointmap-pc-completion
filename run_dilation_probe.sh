#!/bin/bash
# Dilation 方法 (独立模型 class PointSea_dilated)，与 pointsea/pointmap 完全隔离。
# dense_pointmap 投影 + dilated 模型，dilation=2/4 各 5 epoch 快速对比。
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointattn
export CUDA_VISIBLE_DEVICES=0
LOGDIR=~/SVDFormer_PointSea/logs
mkdir -p "$LOGDIR"
cd ~/SVDFormer_PointSea
log(){ echo "[$(date +%m-%d %H:%M:%S)] $*" | tee -a "$LOGDIR/dilation_all.log"; }

for DIL in 2 4; do
  log "=== [isolated dilated class] dilation=$DIL, dense_pointmap, 5 epoch ==="
  python -u train_unified.py --method dense_pointmap --model dilated --dilation $DIL \
      --epochs 5 --batch_size 12 --output_dir dil${DIL}_iso 2>&1 | tee "$LOGDIR/dil${DIL}_iso.log"
  log "dilation=$DIL done"
done
log "=== ALL DONE ==="
