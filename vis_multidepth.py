import sys, os, h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_PointSea.mv_utils_multidepth import PCViews_MultiDepth

with h5py.File('/home/ordhktvu3862/shapenet/test/partial/all/0000.h5') as f:
    pts_raw = f['data'][:]
pts = torch.from_numpy(pts_raw).float().cuda().unsqueeze(0)
centroid = pts.mean(dim=1, keepdim=True)
pts = (pts - centroid) / pts.abs().max()

render = PCViews_MultiDepth()
pm, mask = render.get_multi_pointmap(pts)
print(f'pointmap {pm.shape}, mask {mask.shape}')
# view0: near_xyz (0,1,2), far_xyz (3,4,5)
near = pm[0, :3].cpu().numpy()
far = pm[0, 3:].cpu().numpy()
mk = mask[0, 0].cpu().numpy()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes[0,0].imshow(near.transpose(1,2,0), origin='lower'); axes[0,0].set_title('NEAR RGB (x,y,z)'); axes[0,0].axis('off')
axes[0,1].imshow(near[2], cmap='plasma', origin='lower'); axes[0,1].set_title('NEAR z'); axes[0,1].axis('off')
axes[0,2].imshow(mk, cmap='gray', origin='lower', vmin=0, vmax=1); axes[0,2].set_title('Mask'); axes[0,2].axis('off')
axes[1,0].imshow(far.transpose(1,2,0), origin='lower'); axes[1,0].set_title('FAR RGB (x,y,z)'); axes[1,0].axis('off')
axes[1,1].imshow(far[2], cmap='plasma', origin='lower'); axes[1,1].set_title('FAR z'); axes[1,1].axis('off')
# depth difference
diff = far[2] - near[2]
axes[1,2].imshow(diff, cmap='inferno', origin='lower'); axes[1,2].set_title('FAR z - NEAR z (depth span)'); axes[1,2].axis('off')
plt.tight_layout()
plt.savefig('/tmp/multidepth_view0.png', dpi=110)
print('保存 /tmp/multidepth_view0.png')
