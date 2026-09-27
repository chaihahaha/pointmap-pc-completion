from torch import nn
from torch.autograd import Function
import torch


class chamfer_3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        B, N, _ = xyz1.size()
        _, M, _ = xyz2.size()

        dist = torch.cdist(xyz1, xyz2)

        dist1, idx1 = torch.min(dist, dim=2)
        dist2, idx2 = torch.min(dist, dim=1)
        dist1 = torch.clamp(dist1, min=0)
        dist2 = torch.clamp(dist2, min=0)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1.int(), idx2.int()

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        B, N, _ = xyz1.shape
        _, M, _ = xyz2.shape

        gradxyz1 = torch.zeros_like(xyz1)
        gradxyz2 = torch.zeros_like(xyz2)

        idx1_exp = idx1.unsqueeze(-1).expand(B, N, 3)
        nn_xyz2 = xyz2.gather(1, idx1_exp)
        gradxyz1 = 2.0 * (xyz1 - nn_xyz2) * graddist1.unsqueeze(-1)

        idx2_exp = idx2.unsqueeze(-1).expand(B, M, 3)
        nn_xyz1 = xyz1.gather(1, idx2_exp)
        gradxyz2 = 2.0 * (xyz2 - nn_xyz1) * graddist2.unsqueeze(-1)

        return gradxyz1, gradxyz2


class chamfer_3DDist(nn.Module):
    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_3DFunction.apply(input1, input2)
