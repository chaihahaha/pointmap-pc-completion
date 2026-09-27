#!/bin/bash
# Train Geom PointMap (PCA 切平面确定性扩充) 作为第三个对照方法。
# 与已有对比: PointSea(depth) 与 DensePointMap(高斯) 均已训练完成。
# 使用相同设置: 5M 小模型, bz=12, merge=256, 30 epoch, get_loss(原版Completion3D)。
# 日志持久化到 ~/SVDFormer_PointSea/logs/。
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointattn
export CUDA_VISIBLE_DEVICES=0
LOGDIR=~/SVDFormer_PointSea/logs
mkdir -p "$LOGDIR"

cd ~/SVDFormer_PointSea
rm -rf geommap_small_*

log(){ echo "[$(date +%m-%d %H:%M:%S)] $*" | tee -a "$LOGDIR/geom_all.log"; }

log "=== Geom PointMap-small (PCA 切平面, bz=12, merge=256) ==="
python -u train_unified.py --method geom_pointmap --model small --epochs 30 --batch_size 12 --output_dir geommap_small 2>&1 | tee "$LOGDIR/geommap_small.log"
log "Geom PointMap-small done"
log "=== ALL DONE ==="
