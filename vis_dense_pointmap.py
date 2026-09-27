import sys, os, h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_PointSea.mv_utils_densepointmap import DensePCViews_PointMap

# Load a sample from the test set (partial only, 2048 points)
sample_path = '/home/ordhktvu3862/shapenet/test/partial/all/0000.h5'
with h5py.File(sample_path) as f:
    pts_raw = f['data'][:]  # [2048, 3]
pts = torch.from_numpy(pts_raw).float().cuda().unsqueeze(0)  # [1, 2048, 3]

# Center + normalize to unit sphere (standard preprocessing)
centroid = pts.mean(dim=1, keepdim=True)
pts = pts - centroid
m = pts.abs().max()
pts = pts / m

print(f'Input: {pts.shape}, range: {pts.min().item():.3f}..{pts.max().item():.3f}')

render = DensePCViews_PointMap(TRANS=-0.7)
pointmap, mask = render.get_dense_pointmap(pts)
print(f'pointmap: {pointmap.shape}  mask: {mask.shape}')

# Check why test set has an issue - dump stats
pm = pointmap[0].cpu().numpy()  # [3, H, W], view 0
mk = mask[0, 0].cpu().numpy()    # [H, W], view 0

print(f'View0 pointmap range: min={pm.min():.3f} max={pm.max():.3f}')
print(f'View0 mask: orig_px={ (mk>0.5).sum() } total_filled={ (np.abs(pm).sum(0)>0.1).sum() }')

# Visualize view 0
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# pointmap as RGB (x,y,z -> rgb)
rgb = pm.transpose(1, 2, 0)  # [H, W, 3]
axes[0, 0].imshow(rgb, origin='lower')
axes[0, 0].set_title('PointMap RGB (x,y,z)')
axes[0, 0].axis('off')

# spatial mask (orig=1)
axes[0, 1].imshow(mk, cmap='gray', origin='lower', vmin=0, vmax=1)
axes[0, 1].set_title('Mask (1=original)')
axes[0, 1].axis('off')

# x channel
axes[1, 0].imshow(pm[0], cmap='viridis', origin='lower')
axes[1, 0].set_title('X coordinate')
axes[1, 0].axis('off')

# y channel
axes[1, 1].imshow(pm[1], cmap='plasma', origin='lower')
axes[1, 1].set_title('Y coordinate')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('/tmp/dense_pointmap_view0.png', dpi=120)
print('Saved /tmp/dense_pointmap_view0.png')

# Also visualize all 3 views of RGB
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
for vi in range(3):
    pv = pointmap[vi].cpu().numpy().transpose(1, 2, 0)
    axes2[vi].imshow(pv, origin='lower')
    axes2[vi].set_title(f'View {vi} PointMap')
    axes2[vi].axis('off')
plt.tight_layout()
plt.savefig('/tmp/dense_pointmap_allviews.png', dpi=120)
print('Saved /tmp/dense_pointmap_allviews.png')
