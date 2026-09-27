import logging
import os
import sys
import torch
import argparse
import utils.data_loaders
import utils.helpers
from datetime import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import *
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import *
from models_PointSea.PointSea import Model
from models_PointSea.mv_utils_zs import PCViews_Real
from models_PointSea.mv_utils_pointmap import PCViews_PointMap
from models_PointSea.mv_utils_densepointmap import DensePCViews_PointMap
from models_PointSea.mv_utils_geompointmap import PCViews_GeomPointMap
from models_PointSea.PointSea_small import Model as SmallModel
from models_PointSea.PointSea_dilated import Model as DilatedModel

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def train_net(cfg, method='pointsea', model_type='full'):
    torch.backends.cudnn.benchmark = True

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TRAIN),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=True,
        drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TEST),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS // 2,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=False)

    # Set up folders
    out_root = f'{cfg.DIR.OUT_PATH}_{method}'
    output_dir = os.path.join(out_root, '%s', datetime.now().isoformat().replace(':', '-'))
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    os.makedirs(cfg.DIR.CHECKPOINTS, exist_ok=True)

    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    if model_type == 'small':
        model = SmallModel(cfg)
    elif model_type == 'dilated':
        model = DilatedModel(cfg)
    else:
        model = Model(cfg)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.TRAIN.LEARNING_RATE,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        betas=cfg.TRAIN.BETAS)

    scheduler_steplr = MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    lr_scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
        after_scheduler=scheduler_steplr)

    init_epoch = 0
    best_metrics = float('inf')
    steps = 0
    BestEpoch = 0

    if method in ('dense_pointmap','geom_pointmap'):
        render = DensePCViews_PointMap(TRANS=-cfg.NETWORK.view_distance) if method=='dense_pointmap' else PCViews_GeomPointMap(TRANS=-cfg.NETWORK.view_distance)
    elif method == 'pointmap':
        render = PCViews_PointMap(TRANS=-cfg.NETWORK.view_distance)
    else:
        render = PCViews_Real(TRANS=-cfg.NETWORK.view_distance)

    logging.info(f'Using projection method: {method}')

    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        model.train()

        total_cd_pc = torch.tensor(0.0, device='cuda')
        total_cd_p1 = torch.tensor(0.0, device='cuda')
        total_cd_p2 = torch.tensor(0.0, device='cuda')

        batch_end_time = time()
        n_batches = len(train_data_loader)
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        # tqdm 低频刷新(mininterval=10s): 避免每 batch 刷屏拖累性能
        pbar = tqdm(total=n_batches, desc=f'Epoch {epoch_idx}/{cfg.TRAIN.N_EPOCHS}',
                    mininterval=10.0, ncols=100)
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader):
            data_time.update(time() - batch_end_time)
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)
            partial = data['partial_cloud']
            gt = data['gtcloud']

            if method in ('dense_pointmap', 'geom_pointmap'):
                partial_view, pmask = render.get_dense_pointmap(partial)
                partial_view = torch.cat([partial_view, pmask], dim=1)
            elif method == 'pointmap':
                partial_view = render.get_pointmap(partial)
            else:
                _img = render.get_img(partial)
                partial_view = torch.cat(
                    [_img,
                     torch.ones(_img.size(0), 1, _img.size(2), _img.size(3),
                                device=partial.device)], dim=1)

            pcds_pred = model(partial, partial_view)

            has_nan_out = False
            for p in pcds_pred:
                if torch.isnan(p).any() or torch.isinf(p).any():
                    has_nan_out = True
                    break
            if has_nan_out:
                logging.warning(f'NaN/Inf in output at batch {batch_idx}, resetting BN stats')
                for m in model.modules():
                    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                        m.reset_running_stats()
                continue

            loss_total, losses = get_loss(pcds_pred, gt, sqrt=True)

            if torch.isnan(loss_total) or torch.isinf(loss_total):
                logging.warning(f'NaN/Inf loss at batch {batch_idx}, skipping')
                continue

            optimizer.zero_grad()
            loss_total.backward()

            # Check gradients before clipping (clip_grad_norm_ propagates NaN to ALL params)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logging.warning(f'NaN/Inf gradient norm at batch {batch_idx}, skipping')
                optimizer.zero_grad()
                continue

            optimizer.step()

            if steps % 50 == 0:
                for name, param in model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        logging.error(f'NaN/Inf in weights at {name}, reinit model')
                        model = Model(cfg).cuda()
                        optimizer = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.TRAIN.WEIGHT_DECAY, betas=cfg.TRAIN.BETAS)
                        scheduler_steplr = MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
                        lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS, after_scheduler=scheduler_steplr)
                        break

            total_cd_pc += losses[0] * 1e3
            total_cd_p1 += losses[1] * 1e3
            total_cd_p2 += losses[2] * 1e3
            n_itr = (epoch_idx - 1) * n_batches + batch_idx

            # 每 batch 记录到 tensorboard（正式日志，开销小）
            if train_writer:
                train_writer.add_scalar('Loss/Batch/cd_pc', losses[0].item() * 1e3, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p1', losses[1].item() * 1e3, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p2', losses[2].item() * 1e3, n_itr)
                train_writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], n_itr)
                train_writer.add_scalar('Train/grad_norm', grad_norm.item(), n_itr)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            pbar.update(1)
            # 低频更新进度描述(postfix), 每50 batch
            if batch_idx % 50 == 0 or (batch_idx + 1) == n_batches:
                cd_pc_cur = losses[0].item() * 1e3
                cd_p1_cur = losses[1].item() * 1e3
                cd_p2_cur = losses[2].item() * 1e3
                pbar.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_pc_cur, cd_p1_cur, cd_p2_cur]])

            if steps <= cfg.TRAIN.WARMUP_STEPS:
                lr_scheduler.step()
                steps += 1

        pbar.close()
        avg_cdc = total_cd_pc.item() / n_batches
        avg_cd1 = total_cd_p1.item() / n_batches
        avg_cd2 = total_cd_p2.item() / n_batches

        lr_scheduler.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time,
             ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2]]))

        cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model, method=method)
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                BestEpoch = epoch_idx
                file_name = 'ckpt-best.pth'
            else:
                file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)
        logging.info('Best Performance: Epoch %d -- CD %.4f' % (BestEpoch, best_metrics))

    train_writer.close()
    val_writer.close()


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None, method='pointsea', model_type='full'):
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TEST),
            batch_size=1,
            num_workers=cfg.CONST.NUM_WORKERS // 2,
            collate_fn=utils.data_loaders.collate_fn,
            pin_memory=True,
            shuffle=False)

    if model is None:
        if model_type == 'small':
            model = SmallModel(cfg)
        elif model_type == 'dilated':
            model = DilatedModel(cfg)
        else:
            model = Model(cfg)
        if torch.cuda.is_available():
            model = model.cuda()
        logging.info('Recovering from %s ...' % cfg.CONST.WEIGHTS)
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])

    model.eval()

    if method in ('dense_pointmap','geom_pointmap'):
        render = DensePCViews_PointMap(TRANS=-cfg.NETWORK.view_distance) if method=='dense_pointmap' else PCViews_GeomPointMap(TRANS=-cfg.NETWORK.view_distance)
    elif method == 'pointmap':
        render = PCViews_PointMap(TRANS=-cfg.NETWORK.view_distance)
    else:
        render = PCViews_Real(TRANS=-cfg.NETWORK.view_distance)

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['CD', 'DCD', 'F1'])
    test_metrics = AverageMeter(['CD', 'DCD', 'F1'])
    category_metrics = dict()

    with tqdm(test_data_loader, mininterval=10.0, ncols=100) as t:
        for model_idx, (taxonomy_id, model_id, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']

                if method in ('dense_pointmap','geom_pointmap'):
                    partial_view, pmask = render.get_dense_pointmap(partial)
                    partial_view = torch.cat([partial_view, pmask], dim=1)
                elif method == 'pointmap':
                    partial_view = render.get_pointmap(partial)
                else:
                    _img = render.get_img(partial)
                    partial_view = torch.cat(
                        [_img,
                         torch.ones(_img.size(0), 1, _img.size(2), _img.size(3),
                                    device=partial.device)], dim=1)

                pcds_pred = model(partial.contiguous(), partial_view)
                cdl1, cdl2, f1 = calc_cd(pcds_pred[-1], gt, calc_f1=True)
                dcd, _, _ = calc_dcd(pcds_pred[-1], gt)

                cd = cdl1.mean().item() * 1e3
                dcd = dcd.mean().item()
                f1 = f1.mean().item()

                _metrics = [cd, dcd, f1]
                test_losses.update([cd, dcd, f1])
                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(['CD', 'DCD', 'F1'])
                category_metrics[taxonomy_id].update(_metrics)

                if model_idx % 50 == 0:
                    t.set_postfix_str(
                        'Test[%d/%d] Tax=%s Metrics=%s' %
                        (model_idx + 1, n_samples, taxonomy_id,
                         ['%.4f' % m for m in _metrics]))

    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/cd', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/dcd', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/f1', test_losses.avg(2), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='pointsea',
                        choices=['pointsea', 'pointmap', 'dense_pointmap', 'geom_pointmap'])
    parser.add_argument('--model', type=str, default='full', choices=['full', 'small', 'dilated'])
    parser.add_argument('--dilation', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    from config_completion3d import cfg

    cfg.DATASETS.COMPLETION3D.ROOT = '/home/ordhktvu3862/shapenet'
    cfg.CONST.NUM_WORKERS = 2

    if args.epochs is not None:
        cfg.TRAIN.N_EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.output_dir is not None:
        cfg.DIR.OUT_PATH = args.output_dir
    if args.dilation is not None:
        cfg.NETWORK.DILATION = args.dilation
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    import numpy as np
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logging.basicConfig(
        format='[%(levelname)s] %(asctime)s %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)])

    print('cuda available:', torch.cuda.is_available())
    print(f'Using method: {args.method}')
    print(f'Training for {cfg.TRAIN.N_EPOCHS} epochs, batch size {cfg.TRAIN.BATCH_SIZE}')
    print(f'Dataset root: {cfg.DATASETS.COMPLETION3D.ROOT}')

    if args.test:
        if cfg.CONST.WEIGHTS is None:
            raise Exception('Please specify --weights for testing')
        test_net(cfg, method=args.method, model_type=args.model)
    else:
        train_net(cfg, method=args.method, model_type=args.model)
