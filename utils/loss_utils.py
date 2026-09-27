import torch
import torch.nn
from pytorch3d.loss import chamfer_distance as pyt3d_cd
from pytorch3d.ops import knn_points
from models.model_utils import fps_subsample
from metrics.CD.fscore import fscore


def chamfer(p1, p2):
    """L2 Chamfer distance"""
    loss, _ = pyt3d_cd(p1, p2, norm=2, single_directional=False)
    return loss


def chamfer_sqrt(p1, p2):
    """L1 Chamfer distance (sqrt of squared)"""
    loss, _ = pyt3d_cd(p1, p2, norm=1, single_directional=False)
    return loss


def chamfer_single_side(pcd1, pcd2):
    """One-sided L2: pcd1 -> pcd2"""
    loss, _ = pyt3d_cd(pcd1, pcd2, norm=2, single_directional=True)
    return loss


def chamfer_single_side_sqrt(pcd1, pcd2):
    """One-sided L1: pcd1 -> pcd2"""
    loss, _ = pyt3d_cd(pcd1, pcd2, norm=1, single_directional=True)
    return loss


def get_loss(pcds_pred, gt, sqrt=True, alpha1=1, alpha2=1):
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2 = pcds_pred

    gt_1 = fps_subsample(gt, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt)

    loss_all = cdc + alpha1 * cd1 + alpha2 * cd2
    losses = [cdc, cd1, cd2]
    return loss_all, losses


def get_loss_PM(pcds_pred, partial, gt, sqrt=True):
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2 = pcds_pred

    gt_1 = fps_subsample(gt, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt)
    partial_matching = PM(partial, P2)

    loss_all = cdc + cd1 + cd2 + partial_matching
    losses = [cdc, cd1, cd2]
    return loss_all, losses


def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
    dist1, dist2, idx1, idx2 = _chamfer_raw(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0), dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res


def _chamfer_raw(x, y):
    knn_xy = knn_points(x, y, K=1, return_sorted=False)
    knn_yx = knn_points(y, x, K=1, return_sorted=False)
    d1 = knn_xy.dists[..., 0]
    d2 = knn_yx.dists[..., 0]
    i1 = knn_xy.idx[..., 0]
    i2 = knn_yx.idx[..., 0]
    return d1, d2, i1, i2


def calc_dcd(x, gt, alpha=1000, n_lambda=1, return_raw=False, non_reg=False):
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    loss = (loss1 + loss2) / 2

    res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res
