#!/bin/bash
# Sequential 30-epoch training: PointSea → PointMap → SymmCompletion
# All tested: 0 NaN, no OOM
set -e
cd /home/ordhktvu3862/SVDFormer_PointSea
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointattn
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ===== PointSea (bz=3, Adam/lr=1e-4) =====
log "=== PointSea 30 epochs (bz=3) ==="
rm -rf pointsea_30ep_*
python -u train_unified.py --method pointsea --epochs 30 --batch_size 3 --output_dir pointsea_30ep 2>&1 | tee /tmp/pointsea_train.log
log "PointSea done"

# ===== PointMap (bz=3, Adam/lr=1e-4) =====
log "=== PointMap 30 epochs (bz=3) ==="
rm -rf pointmap_30ep_*
python -u train_unified.py --method pointmap --epochs 30 --batch_size 3 --output_dir pointmap_30ep 2>&1 | tee /tmp/pointmap_train.log
log "PointMap done"

# ===== SymmCompletion (bz=6, AdamW/lr=5e-4) =====
log "=== SymmCompletion 30 epochs (bz=6) ==="
cd ~/SymmCompletion
python -u main.py --config cfgs/PCN_models/SymmCompletion_Comp3D_30ep.yaml --exp_name symmcompletion_30ep --num_workers 2 2>&1 | tee /tmp/symmcompletion_train.log
log "SymmCompletion done"

log "=== ALL TRAINING COMPLETE ==="
