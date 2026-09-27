import sys, os, h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_PointSea.mv_utils_geompointmap import PCViews_GeomPointMap

# 测试集样本
with h5py.File('/home/ordhktvu3862/shapenet/test/partial/all/0000.h5') as f:
    pts_raw = f['data'][:]
pts = torch.from_numpy(pts_raw).float().cuda().unsqueeze(0)
centroid = pts.mean(dim=1, keepdim=True)
pts = (pts - centroid) / pts.abs().max()

render = PCViews_GeomPointMap(TRANS=-0.7)
pm, mask = render.get_dense_pointmap(pts)
pm0 = pm[0].cpu().numpy()
mk0 = mask[0, 0].cpu().numpy()

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes[0,0].imshow(pm0.transpose(1,2,0), origin='lower')
axes[0,0].set_title('PointMap RGB (x,y,z) PCA-expanded'); axes[0,0].axis('off')
axes[0,1].imshow(mk0, cmap='gray', origin='lower', vmin=0, vmax=1)
axes[0,1].set_title('Mask (1=original)'); axes[0,1].axis('off')
axes[1,0].imshow(pm0[0], cmap='viridis', origin='lower')
axes[1,0].set_title('X coordinate'); axes[1,0].axis('off')
axes[1,1].imshow(pm0[1], cmap='plasma', origin='lower')
axes[1,1].set_title('Y coordinate'); axes[1,1].axis('off')
plt.tight_layout()
plt.savefig('/tmp/geom_pointmap_view0.png', dpi=120)
print('保存 /tmp/geom_pointmap_view0.png')

# 3视图
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
for vi in range(3):
    axes2[vi].imshow(pm[vi].cpu().numpy().transpose(1,2,0), origin='lower')
    axes2[vi].set_title(f'View {vi}'); axes2[vi].axis('off')
plt.tight_layout()
plt.savefig('/tmp/geom_pointmap_allviews.png', dpi=120)
print('保存 /tmp/geom_pointmap_allviews.png')
