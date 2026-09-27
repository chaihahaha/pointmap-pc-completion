import sys, os, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_completion3d import cfg
cfg.DATASETS.COMPLETION3D.ROOT = '/home/ordhktvu3862/shapenet'

import utils.data_loaders, utils.helpers
from models_PointSea.PointSea_small import Model as SmallModel
from models_PointSea.mv_utils_zs import PCViews_Real
from models_PointSea.mv_utils_densepointmap import DensePCViews_PointMap
from models_PointSea.mv_utils_geompointmap import PCViews_GeomPointMap
from utils.loss_utils import calc_cd, calc_dcd


def write_ply(path, pts):
    """Minimal ASCII PLY writer for a point cloud [N,3]."""
    pts = np.asarray(pts, dtype=np.float64)
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('end_header\n')
        for p in pts:
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')


def build_render(method):
    if method == 'pointsea':
        return PCViews_Real(TRANS=-cfg.NETWORK.view_distance)
    elif method == 'dense_pointmap':
        return DensePCViews_PointMap(TRANS=-cfg.NETWORK.view_distance)
    elif method == 'geom_pointmap':
        return PCViews_GeomPointMap(TRANS=-cfg.NETWORK.view_distance)
    raise ValueError(method)


def make_view(method, render, partial):
    if method == 'pointsea':
        img = render.get_img(partial)
        return torch.cat([img, torch.ones(img.size(0), 1, img.size(2), img.size(3),
                                          device=partial.device)], dim=1)
    else:
        pm, mask = render.get_dense_pointmap(partial)
        return torch.cat([pm, mask], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='0 = all val samples')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = SmallModel(cfg).cuda()
    ckpt = torch.load(args.weights)
    model.load_state_dict(ckpt['model'])
    model.eval()
    render = build_render(args.method)

    loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    ds = loader.get_dataset(utils.data_loaders.DatasetSubset.TEST)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=2,
                                     collate_fn=utils.data_loaders.collate_fn,
                                     shuffle=False)

    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    from tqdm import tqdm
    with torch.no_grad():
        for idx, (tid, mid, data) in enumerate(tqdm(dl, total=n, mininterval=5.0)):
            if args.limit and idx >= args.limit:
                break
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)
            partial = data['partial_cloud']
            gt = data['gtcloud']
            view = make_view(args.method, render, partial)
            pred = model(partial, view)[-1]  # finest [1, M, 3]

            name = f'{idx:04d}_{tid[0]}_{mid[0]}'
            write_ply(os.path.join(args.out, name + '_input.ply'), partial[0].cpu().numpy())
            write_ply(os.path.join(args.out, name + '_pred.ply'), pred[0].cpu().numpy())
            write_ply(os.path.join(args.out, name + '_gt.ply'), gt[0].cpu().numpy())

    print(f'PLY 已保存到 {args.out}  ({n} 个样本)')


if __name__ == '__main__':
    main()
