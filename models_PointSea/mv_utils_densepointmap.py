import torch
import numpy as np
from models_PointSea.mv_utils_zs import euler2mat

resolution = 224


class DensePCViews_PointMap:
    """PointMap with Gaussian-mixture density augmentation.

    Algorithm:
      1. Transform sparse input points to 3 camera views.
      2. For each view, compute the projection window from the ORIGINAL
         sparse points only (so sparse points fill the whole HxW frame).
      3. Sample `H*W` expansion points from a fixed-radius Gaussian mixture
         centered at the original points (radius chosen so the augmented
         cloud never exceeds ~3x the original sparse range, well under 10x).
      4. Project original + expansion points into the HxW grid.
         - pixel value = normalized (x, y, z) coordinate of the point that
           landed there
         - mask = 1 if that pixel was hit by an ORIGINAL point
                 = 0 if it was hit only by expansion points (or empty)
      Returns: (pointmap [B, 3, H, W], mask [B, 1, H, W]).
    """

    def __init__(self, TRANS=-0.7, sigma_factor=0.02):
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
        self.sigma_factor = sigma_factor

        angle = torch.tensor(_views[:, 0, :]).float().cuda()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        angle2 = torch.tensor(_views_bias[:, 0, :]).float().cuda()
        self.rot_mat2 = euler2mat(angle2).transpose(1, 2)
        self.translation = torch.tensor(_views[:, 1, :]).float().cuda()
        self.translation = self.translation.unsqueeze(1)

    def get_dense_pointmap(self, points):
        """points: [B, N, 3] -> (pointmap, mask) each [B*views, 3, H, W]."""
        b, _, _ = points.shape
        v = self.num_views
        H = W = resolution
        N = points.shape[1]
        M = H * W  # roughly the number of pixels

        orig_v = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            rot_mat2=self.rot_mat2.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))
        bv = orig_v.shape[0]  # B * views

        # ---- window normalization from ORIGINAL points only ----
        pmin = orig_v.min(dim=1)[0]            # [bv, 3]
        pmax = orig_v.max(dim=1)[0]
        prange = pmax - pmin
        prange = torch.clamp(prange, min=1e-6)  # [bv, 3]

        # ------- sample expansion points (Gaussian mixture) -------
        max_range = prange.max(dim=-1, keepdim=True)[0]   # [bv, 1]
        sigma = self.sigma_factor * max_range             # [bv, 1]

        comp = torch.randint(0, N, (bv, M), device=points.device)
        centers = orig_v.gather(1, comp.unsqueeze(-1).expand(bv, M, 3))
        expansion = centers + sigma.unsqueeze(-1) * torch.randn(
            bv, M, 3, device=points.device)

        # original points + expansion points
        all_pts = torch.cat([orig_v, expansion], dim=1)   # [bv, N+M, 3]
        is_orig = torch.cat([
            torch.ones(bv, N, device=points.device),
            torch.zeros(bv, M, device=points.device)], dim=1)  # [bv, N+M]

        # ------- normalize positions to the window -------
        x_norm = (all_pts[:, :, 0] - pmin[:, 0:1]) / prange[:, 0:1]
        y_norm = (all_pts[:, :, 1] - pmin[:, 1:2]) / prange[:, 1:2]
        z_norm = (all_pts[:, :, 2] - pmin[:, 2:3]) / prange[:, 2:3]

        col = (x_norm * (W - 1)).long()
        row = (y_norm * (H - 1)).long()
        valid = (col >= 0) & (col < W) & (row >= 0) & (row < H)  # [bv, N+M]
        flat_idx = (row.clamp(0, H - 1) * W + col.clamp(0, W - 1)).long()

        pointmap = torch.zeros(bv, 3, H * W, device=points.device)
        mask = torch.zeros(bv, 1, H * W, device=points.device)

        for i in range(bv):
            # --- original points first (mask = 1) ---
            o_valid = valid[i, :N]
            o_idx = flat_idx[i, :N][o_valid]
            o_coord = torch.stack(
                [x_norm[i, :N][o_valid], y_norm[i, :N][o_valid],
                 z_norm[i, :N][o_valid]], dim=0)  # [3, k]
            pointmap[i, :, o_idx] = o_coord
            mask[i, 0, o_idx] = 1.0

            # --- expansion points fill only empty pixels (mask stays 0) ---
            e_valid = valid[i, N:]
            e_idx = flat_idx[i, N:][e_valid]
            e_coord = torch.stack(
                [x_norm[i, N:][e_valid], y_norm[i, N:][e_valid],
                 z_norm[i, N:][e_valid]], dim=0)
            filled = mask[i, 0] > 0          # [H*W]
            keep = ~filled[e_idx]            # only target empty pixels
            if keep.any():
                pointmap[i, :, e_idx[keep]] = e_coord[:, keep]
                # mask remains 0

        pointmap = pointmap.view(bv, 3, H, W)
        mask = mask.view(bv, 1, H, W)

        # clip coords to [0,1] for image compatibility
        pointmap = torch.clamp(pointmap, 0.0, 1.0)
        return pointmap, mask

    @staticmethod
    def point_transform(points, rot_mat, rot_mat2, translation):
        rot_mat = rot_mat.to(points.device)
        rot_mat2 = rot_mat2.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = torch.matmul(points, rot_mat2)
        points = points - translation
        return points
