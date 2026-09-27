#!/usr/bin/env python3
"""Standalone training launcher that runs both PointSea and PointMap sequentially"""
import subprocess
import sys
import os

os.chdir('/home/ordhktvu3862/SVDFormer_PointSea')

cmds = [
    ['python', '-u', 'train_unified.py', '--method', 'pointsea', '--epochs', '30', '--batch_size', '2', '--output_dir', 'pointsea_30ep'],
    ['python', '-u', 'train_unified.py', '--method', 'pointmap', '--epochs', '30', '--batch_size', '2', '--output_dir', 'pointmap_30ep'],
]

env = os.environ.copy()
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

for cmd in cmds:
    name = cmd[cmd.index('--method') + 1]
    log_path = f'/tmp/{name}_train.log'
    print(f'Starting {name} training, log: {log_path}')
    with open(log_path, 'w') as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        proc.wait()
        print(f'{name} finished with code {proc.returncode}')

print('All training completed!')
