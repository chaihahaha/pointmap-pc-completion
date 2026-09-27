#!/bin/bash
# Runs PointMap after PointSea finishes
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointattn
export CUDA_VISIBLE_DEVICES=0
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

while tmux has-session -t train 2>/dev/null; do sleep 120; done
sleep 10

log "Starting PointMap (bz=2)..."
cd ~/SVDFormer_PointSea
python -u train_unified.py --method pointmap --epochs 30 --batch_size 2 --output_dir pointmap_30ep 2>&1 | tee /tmp/pointmap_train.log
log "PointMap done"

sleep 10
log "Starting SymmCompletion (bz=4)..."
cd ~/SymmCompletion
python -u main.py --config cfgs/PCN_models/SymmCompletion_Comp3D_30ep.yaml --exp_name symmcompletion_30ep --num_workers 2 2>&1 | tee /tmp/symmcompletion_train.log
log "ALL DONE"
