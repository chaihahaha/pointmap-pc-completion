import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from config_completion3d import cfg

cfg.DATASETS.COMPLETION3D.ROOT = '/home/ordhktvu3862/shapenet'

import utils.data_loaders
import utils.helpers
from utils.loss_utils import calc_cd, calc_dcd
from utils.average_meter import AverageMeter


def eval_pointattn(ckpt_path):
    sys.path.insert(0, os.path.expanduser('~/PointAttN'))
    sys.path.insert(0, os.path.expanduser('~/PointAttN/models'))
    from PointAttN import Model as PointAttNModel
    from pn2_utils import furthest_point_sample, gather_points
    from model_utils import calc_cd as pattn_calc_cd

    class Args:
        pass
    args = Args()
    args.dataset = 'c3d'

    net = PointAttNModel(args)
    net = torch.nn.DataParallel(net)
    net.cuda()
    ckpt = torch.load(ckpt_path)
    net.module.load_state_dict(ckpt['net_state_dict'])
    net.eval()

    def infer(partial):
        B, _, _ = partial.shape
        inputs = partial.transpose(1, 2).contiguous()
        result = net(inputs, torch.zeros(B, 2048, 3).cuda(), is_training=False)
        return [result['out1'], result['out2'], result['out2']]

    loader = _get_val_loader()
    return _run_eval(infer, loader, 'PointAttN')


def eval_pointsea(ckpt_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models_PointSea.PointSea import Model
    from models_PointSea.mv_utils_zs import PCViews_Real

    model = Model(cfg).cuda()
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    model.eval()
    render = PCViews_Real(TRANS=-cfg.NETWORK.view_distance)

    def infer(partial):
        partial_view = render.get_img(partial)
        return model(partial, partial_view)

    loader = _get_val_loader()
    return _run_eval(infer, loader, 'PointSea')


def eval_pointmap(ckpt_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models_PointSea.PointSea import Model
    from models_PointSea.mv_utils_pointmap import PCViews_PointMap

    model = Model(cfg).cuda()
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    model.eval()
    render = PCViews_PointMap(TRANS=-cfg.NETWORK.view_distance)

    def infer(partial):
        partial_view = render.get_pointmap(partial)
        return model(partial, partial_view)

    loader = _get_val_loader()
    return _run_eval(infer, loader, 'PointMap')


def _get_val_loader():
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    dataset = dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TEST)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=2,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=False)
    return loader


def _run_eval(infer_fn, loader, name):
    category_metrics = {}
    overall = AverageMeter(['CD', 'DCD', 'F1'])

    with tqdm(loader, desc=name) as t:
        for idx, (taxonomy_id, model_id, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else str(taxonomy_id[0].item())

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']

                pcds_pred = infer_fn(partial)
                final_pred = pcds_pred[-1]

                cdl1, cdl2, f1 = calc_cd(final_pred, gt, calc_f1=True)
                dcd, _, _ = calc_dcd(final_pred, gt)

                cd = cdl1.mean().item() * 1e3
                dcd = dcd.mean().item()
                f1 = f1.mean().item()

                _metrics = [cd, dcd, f1]
                overall.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(['CD', 'DCD', 'F1'])
                category_metrics[taxonomy_id].update(_metrics)

                t.set_postfix(CD=f'{cd:.4f}', DCD=f'{dcd:.4f}', F1=f'{f1:.4f}')

    print(f'\n{"="*60}')
    print(f'  {name} Evaluation Results')
    print(f'{"="*60}')
    print(f'{"Category":<15} {"#Samples":<10} {"CD":<12} {"DCD":<12} {"F1":<12}')
    print(f'{"-"*60}')
    for cat in sorted(category_metrics.keys()):
        cm = category_metrics[cat]
        print(f'{cat:<15} {cm.count(0):<10} {cm.avg(0):<12.4f} {cm.avg(1):<12.4f} {cm.avg(2):<12.4f}')
    print(f'{"-"*60}')
    print(f'{"Overall":<15} {"":<10} {overall.avg(0):<12.4f} {overall.avg(1):<12.4f} {overall.avg(2):<12.4f}')
    print(f'{"="*60}\n')

    return {'cd': overall.avg(0), 'dcd': overall.avg(1), 'f1': overall.avg(2), 'per_category': {}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['pointattn', 'pointsea', 'pointmap', 'all'])
    parser.add_argument('--pointattn_ckpt', type=str, default=None)
    parser.add_argument('--pointsea_ckpt', type=str, default=None)
    parser.add_argument('--pointmap_ckpt', type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    results = {}

    if args.method in ['pointattn', 'all']:
        ckpt = args.pointattn_ckpt or '~/PointAttN/log/PointAttN_cd_debug_c3d/best_cd_p_network.pth'
        results['PointAttN'] = eval_pointattn(os.path.expanduser(ckpt))

    if args.method in ['pointsea', 'all']:
        if not args.pointsea_ckpt:
            raise ValueError('--pointsea_ckpt required')
        results['PointSea'] = eval_pointsea(args.pointsea_ckpt)

    if args.method in ['pointmap', 'all']:
        if not args.pointmap_ckpt:
            raise ValueError('--pointmap_ckpt required')
        results['PointMap'] = eval_pointmap(args.pointmap_ckpt)

    if len(results) > 1:
        print(f'\n{"="*60}')
        print(f'  Final Comparison')
        print(f'{"="*60}')
        print(f'{"Method":<15} {"CD":<12} {"DCD":<12} {"F1":<12}')
        print(f'{"-"*60}')
        for name, res in results.items():
            print(f'{name:<15} {res["cd"]:<12.4f} {res["dcd"]:<12.4f} {res["f1"]:<12.4f}')
        print(f'{"="*60}')
