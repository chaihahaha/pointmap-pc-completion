#!/bin/bash
# 4x-smaller model comparison. Loss = original PointSea get_loss (三层对称L1 CD，无单向partial匹配，
# 与原版 Completion3D / GRNet-ShapeNetCompletion 协议完全一致)。
# 优先训练 Dense PointMap（更易出 bug），再训练 PointSea。
# 日志写入 ~/SVDFormer_PointSea/logs/ 以跨重启保留。
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointattn
export CUDA_VISIBLE_DEVICES=0
LOGDIR=~/SVDFormer_PointSea/logs
mkdir -p "$LOGDIR"
log(){ echo "[$(date +%m-%d %H:%M:%S)] $*" | tee -a "$LOGDIR/comp_all.log"; }

cd ~/SVDFormer_PointSea

log "=== Phase 1: Dense PointMap-small (4x smaller, bz=4, get_loss_PM) ==="
python -u train_unified.py --method dense_pointmap --model small --epochs 30 --batch_size 12 --output_dir densemap_small 2>&1 | tee "$LOGDIR/densemap_small.log"
log "Dense PointMap-small done"

log "=== Phase 2: PointSea-small (depth map, bz=4, get_loss_PM) ==="
python -u train_unified.py --method pointsea --model small --epochs 30 --batch_size 12 --output_dir pointsea_small 2>&1 | tee "$LOGDIR/pointsea_small.log"
log "PointSea-small done"

log "=== ALL DONE ==="
