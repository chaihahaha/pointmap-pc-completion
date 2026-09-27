@echo off
call "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONPATH=E:\source\compl_pc\SVDFormer_PointSea\pointnet2_ops_lib
set PYTORCH_ALLOC_CONF=expandable_segments:True

cd /d E:\source\compl_pc\SVDFormer_PointSea

E:\miniconda3\envs\tchoc\python.exe -c "from config_completion3d import cfg; cfg.TRAIN.N_EPOCHS=1; cfg.TRAIN.SAVE_FREQ=1; from core.train_pcn import train_net; train_net(cfg)"
