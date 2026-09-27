import torch
import numpy as np
from models_PointSea.mv_utils_zs import (
    euler2mat, points2grid, Grid2Image, params
)


class PCViews_PointMap:
    """Same as PCViews_Real but stores full (x,y,z) coordinates instead of just z.
    
    The key difference: instead of one depth grid, we create 3 grids for (x,y,z),
    process each through the same Grid2Image pipeline (which handles sparsity via
    MaxPool3d+Gaussian smoothing+depth collapse), and stack into 3-channel image.
    """

    def __init__(self, TRANS=-0.7):
        _views = np.asarray([
            [[0 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[1 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0, -np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        ])
        _views_bias = np.asarray([
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        ])
        self.num_views = _views.shape[0]

        angle = torch.tensor(_views[:, 0, :]).float().cuda()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        angle2 = torch.tensor(_views_bias[:, 0, :]).float().cuda()
        self.rot_mat2 = euler2mat(angle2).transpose(1, 2)

        self.translation = torch.tensor(_views[:, 1, :]).float().cuda()
        self.translation = self.translation.unsqueeze(1)

        self.grid2image = Grid2Image().cuda()

    def get_pointmap(self, points):
        b, _, _ = points.shape
        v = self.translation.shape[0]

        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            rot_mat2=self.rot_mat2.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))
        # _points is in camera coords: [b*v, N, 3]

        # Method: create 3 grids for (x, y, z) by temporarily swapping each
        # coordinate into the z-position that points2grid scatters.
        img_channels = []
        for coord_idx in range(3):
            _pts = _points.clone()
            _pts[:, :, 2] = _points[:, :, coord_idx]  # move coord to z-position
            grid = points2grid(_pts, params['resolution'], params['depth'])
            img = self.grid2image(grid)  # [bv, 3, H, W], channels are identical
            img_channels.append(img[:, 0:1, :, :])   # take first channel only

        pointmap = torch.cat(img_channels, dim=1)  # [bv, 3, H, W]
        return pointmap

    @staticmethod
    def point_transform(points, rot_mat, rot_mat2, translation):
        rot_mat = rot_mat.to(points.device)
        rot_mat2 = rot_mat2.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = torch.matmul(points, rot_mat2)
        points = points - translation
        return points
