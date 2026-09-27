import sys, os, time, torch
sys.path.insert(0, '.')

from config_completion3d import cfg
cfg.DATASETS.COMPLETION3D.ROOT = '/home/ordhktvu3862/shapenet'
cfg.TRAIN.LEARNING_RATE = 0.0001

import utils.data_loaders, utils.helpers
from utils.loss_utils import get_loss

def test_method(name, ModelClass, render, batch_size):
    print(f'\n{"="*60}')
    print(f'  Testing {name} (batch_size={batch_size})')
    print(f'{"="*60}')
    
    model = ModelClass(cfg).cuda().train()
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  Params: {params_M:.1f}M')
    
    loader = utils.data_loaders.Completion3DDataLoader(cfg)
    ds = loader.get_dataset(utils.data_loaders.DatasetSubset.TRAIN)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    
    nan_out, nan_loss, nan_weight, oom = 0, 0, 0, False
    times = []
    mems = []
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    for idx, (tid, mid, data) in enumerate(dl):
        if idx >= 10:
            break
        
        try:
            t0 = time.time()
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)
            partial = data['partial_cloud']
            gt = data['gtcloud']
            
            pv = render(partial)
            out = model(partial, pv)
            
            for o in out:
                if torch.isnan(o).any() or torch.isinf(o).any():
                    nan_out += 1
                    break
            else:
                loss_total, losses = get_loss(out, gt, sqrt=True)
                
                if torch.isnan(loss_total) or torch.isinf(loss_total):
                    nan_loss += 1
                else:
                    opt.zero_grad()
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    opt.step()
            
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            times.append(elapsed)
            mems.append(torch.cuda.max_memory_allocated() / 1024**3)
            
        except torch.cuda.OutOfMemoryError:
            oom = True
            times.append(float('inf'))
            print(f'  OOM at batch {idx}!')
            break
    
    if times:
        avg_time = sum(t for t in times if t != float('inf')) / len([t for t in times if t != float('inf')])
        peak_mem = max(mems) if mems else 0
    else:
        avg_time, peak_mem = 0, 0
    
    status = 'OK' if not oom and nan_out == 0 and nan_loss == 0 and nan_weight == 0 else 'FAIL'
    
    print(f'  Result: {status}')
    print(f'  NaN(output): {nan_out}, NaN(loss): {nan_loss}, NaN(weight): {nan_weight}')
    print(f'  OOM: {oom}')
    print(f'  Avg batch time: {avg_time:.3f}s ({1/avg_time:.1f} it/s)')
    print(f'  Peak GPU mem: {peak_mem:.2f} GB')
    if not oom:
        hours_per_epoch = avg_time * 28974 / batch_size / 3600
        print(f'  Est per epoch: {hours_per_epoch:.1f}h')
        print(f'  Est 30 epochs: {hours_per_epoch * 30:.1f}h')
    
    return status == 'OK', avg_time, peak_mem


if __name__ == '__main__':
    # ---- PointSea ----
    from models_PointSea.PointSea import Model as PSeaModel
    from models_PointSea.mv_utils_zs import PCViews_Real
    r1 = PCViews_Real(TRANS=-cfg.NETWORK.view_distance)
    ok1, t1, m1 = test_method('PointSea', PSeaModel, lambda x: r1.get_img(x), batch_size=1)
    
    # ---- PointMap ----
    from models_PointSea.mv_utils_pointmap import PCViews_PointMap
    r2 = PCViews_PointMap(TRANS=-cfg.NETWORK.view_distance)
    ok2, t2, m2 = test_method('PointMap', PSeaModel, lambda x: r2.get_pointmap(x), batch_size=1)
    
    # ---- Try bz=2 for PointSea (faster if fits) ----
    r3 = PCViews_Real(TRANS=-cfg.NETWORK.view_distance)
    ok3, t3, m3 = test_method('PointSea bz=2 (check OOM)', PSeaModel, lambda x: r3.get_img(x), batch_size=2)
    
    # ---- SymmCompletion ----
    sys.path.insert(0, os.path.expanduser('~/SymmCompletion'))
    from models.SymmCompletion import SymmCompletion as SymModel
    from easydict import EasyDict as edict
    scfg = edict()
    scfg.model = edict()
    scfg.model.up_factors = '2, 8'
    scfg.model.include_input = False
    scfg.model.coarse_up_factor = 2

    class SymWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = SymModel(scfg.model)
        def forward(self, partial, _):
            return self.m(partial)
    
    class DummyRender:
        def __init__(self):
            pass
        def __call__(self, x):
            return x  # not used

    ok4, t4, m4 = test_method('SymmCompletion', lambda cfg: SymWrapper(), 
                               DummyRender().__call__, batch_size=1)
    
    print(f'\n{"="*60}')
    print(f'  FINAL SUMMARY')
    print(f'{"="*60}')
    for name, ok, t, m in [
        ('PointSea bz=1', ok1, t1, m1),
        ('PointMap bz=1', ok2, t2, m2),
        ('PointSea bz=2', ok3, t3, m3),
        ('SymmCompletion bz=1', ok4, t4, m4),
    ]:
        print(f'  {name:<20s} status={"OK" if ok else "FAIL":<5s} time={t:.3f}s/it  mem={m:.2f}GB')
