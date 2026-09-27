#!/bin/bash
# ===================================================================
# Master training launcher for:
#   1. PointSea  (30 epochs, batch=2)  ~52h
#   2. PointMap  (30 epochs, batch=2)  ~52h
#   3. SymmCompletion (30 epochs, batch=2)  ~TBD
#
# Total estimated time: ~120+ hours
#
# Usage: bash ~/SVDFormer_PointSea/launch_all.sh
# ===================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_PATH=~/miniconda3
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate pointattn

export CUDA_VISIBLE_DEVICES=0
TORCH_LIB=$CONDA_PATH/envs/pointattn/lib/python3.10/site-packages/torch/lib
export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH

# Clean JIT cache
rm -rf ~/.cache/torch_extensions/py310* 2>/dev/null || true

# ==============================
# 1. PointSea (30 epochs)
# ==============================
echo "================================================"
echo " PHASE 1: PointSea Training (30 epochs)"
echo " Started: $(date)"
echo "================================================"
cd ~/SVDFormer_PointSea
python -u train_unified.py \
    --method pointsea \
    --epochs 30 \
    --batch_size 2 \
    --output_dir pointsea_30ep \
    2>&1 | tee /tmp/pointsea_train.log
echo "PointSea finished at $(date) (exit=$?)"

# ==============================
# 2. PointMap (30 epochs)
# ==============================
echo ""
echo "================================================"
echo " PHASE 2: PointMap Training (30 epochs)"
echo " Started: $(date)"
echo "================================================"
cd ~/SVDFormer_PointSea
python -u train_unified.py \
    --method pointmap \
    --epochs 30 \
    --batch_size 2 \
    --output_dir pointmap_30ep \
    2>&1 | tee /tmp/pointmap_train.log
echo "PointMap finished at $(date) (exit=$?)"

# ==============================
# 3. SymmCompletion (30 epochs)
# ==============================
echo ""
echo "================================================"
echo " PHASE 3: SymmCompletion Training (30 epochs)"
echo " Started: $(date)"
echo "================================================"
cd ~/SymmCompletion
python -u main.py \
    --config cfgs/PCN_models/SymmCompletion_Comp3D_30ep.yaml \
    --exp_name symmcompletion_30ep \
    --num_workers 2 \
    2>&1 | tee /tmp/symmcompletion_train.log
echo "SymmCompletion finished at $(date) (exit=$?)"

echo ""
echo "================================================"
echo " ALL TRAINING COMPLETED!"
echo " Finished: $(date)"
echo "================================================"
