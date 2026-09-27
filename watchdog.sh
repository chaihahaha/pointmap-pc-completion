#!/bin/bash
set -e
CONDA_PATH=~/miniconda3
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate pointattn
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$CONDA_PATH/envs/pointattn/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
rm -rf ~/.cache/torch_extensions/py310* 2>/dev/null || true

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "Waiting for PointSea (tmux:pointsea) to finish..."
while tmux has-session -t pointsea 2>/dev/null; do sleep 30; done
sleep 5

# === PointMap ===
log "Starting PointMap training (30 epochs)..."
cd ~/SVDFormer_PointSea
python -u train_unified.py --method pointmap --epochs 30 --batch_size 2 --output_dir pointmap_30ep 2>&1 | tee /tmp/pointmap_train.log
log "PointMap done (exit=$?)."
sleep 5

# === SymmCompletion ===
log "Starting SymmCompletion training (30 epochs)..."
cd ~/SymmCompletion
python -u main.py --config cfgs/PCN_models/SymmCompletion_Comp3D_30ep.yaml --exp_name symmcompletion_30ep --num_workers 2 2>&1 | tee /tmp/symmcompletion_train.log
log "SymmCompletion done (exit=$?)."
log "=== ALL DONE ==="
