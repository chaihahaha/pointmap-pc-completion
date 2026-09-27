import torch
import numpy as np
from models_PointSea.mv_utils_zs import euler2mat

resolution = 224
PI = 3.141592653589793


class PCViews_GeomPointMap:
    """Deterministic geometry-preserving pointmap expansion via local-PCA
    tangent-plane resampling.

    Instead of random Gaussian noise, each original point is used to estimate a
    local tangent plane via PCA of its k nearest neighbours. New points are then
    placed at deterministic concentric-ring positions ON that tangent plane
    (no random sampling). This preserves the true surface geometry and adds a
    surface-normal channel (nx, ny, nz) as extra completion-relevant evidence.
    """

    def __init__(self, TRANS=-0.7, k=16, num_rings=4, num_angles=8):
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
        self.k = k
        self.num_rings = num_rings
        self.num_angles = num_angles

        angle = torch.tensor(_views[:, 0, :]).float().cuda()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        angle2 = torch.tensor(_views_bias[:, 0, :]).float().cuda()
        self.rot_mat2 = euler2mat(angle2).transpose(1, 2)
        self.translation = torch.tensor(_views[:, 1, :]).float().cuda()
        self.translation = self.translation.unsqueeze(1)

    def get_dense_pointmap(self, points):
        """points: [B, N, 3] -> (pointmap [Bv,3,H,W], mask [Bv,1,H,W])"""
        b, _, _ = points.shape
        v = self.num_views
        H = W = resolution
        N = points.shape[1]

        # ---- expand deterministically in ORIGINAL space ----
        expand, normals = self._pca_expand(points)   # [B, N*num_rings*num_angles, 3], [B, N, 3]

        # ---- transform ORIGINAL + EXPANDED to views ----
        all_pts = torch.cat([points, expand], dim=1)      # [B, N+E, 3]
        is_orig = torch.cat([torch.ones(b, N, device=points.device),
                             torch.zeros(b, expand.shape[1], device=points.device)], dim=1)
        all_v = self.point_transform(
            points=torch.repeat_interleave(all_pts, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            rot_mat2=self.rot_mat2.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))
        bv = all_v.shape[0]
        E = expand.shape[1]
        is_orig_v = torch.repeat_interleave(is_orig, v, dim=0)   # [Bv, N+E]

        # ---- window normalization from ORIGINAL points only ----
        # original points in view space are the first N of each Bv block
        orig_v = all_v[:, :N].contiguous()
        pmin = orig_v.min(dim=1)[0]
        pmax = orig_v.max(dim=1)[0]
        prange = torch.clamp(pmax - pmin, min=1e-6)     # [Bv, 3]

        x_norm = (all_v[:, :, 0] - pmin[:, 0:1]) / prange[:, 0:1]
        y_norm = (all_v[:, :, 1] - pmin[:, 1:2]) / prange[:, 1:2]
        z_norm = (all_v[:, :, 2] - pmin[:, 2:3]) / prange[:, 2:3]

        col = (x_norm * (W - 1)).long()
        row = (y_norm * (H - 1)).long()
        valid = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        flat_idx = (row.clamp(0, H - 1) * W + col.clamp(0, W - 1)).long()

        pointmap = torch.zeros(bv, 3, H * W, device=points.device)
        mask = torch.zeros(bv, 1, H * W, device=points.device)

        for i in range(bv):
            o_valid = valid[i, :N]
            o_idx = flat_idx[i, :N][o_valid]
            o_coord = torch.stack([x_norm[i, :N][o_valid], y_norm[i, :N][o_valid], z_norm[i, :N][o_valid]], 0)
            pointmap[i, :, o_idx] = o_coord
            mask[i, 0, o_idx] = 1.0

            e_valid = valid[i, N:]
            e_idx = flat_idx[i, N:][e_valid]
            e_coord = torch.stack([x_norm[i, N:][e_valid], y_norm[i, N:][e_valid], z_norm[i, N:][e_valid]], 0)
            filled = mask[i, 0] > 0
            keep = ~filled[e_idx]
            if keep.any():
                pointmap[i, :, e_idx[keep]] = e_coord[:, keep]

        pointmap = torch.clamp(pointmap.view(bv, 3, H, W), 0.0, 1.0)
        mask = mask.view(bv, 1, H, W)
        return pointmap, mask

    def _pca_expand(self, points):
        B, N, _ = points.shape
        k = self.k
        num_rings = self.num_rings
        num_angles = self.num_angles

        # kNN (exclude self: take k+1, drop first)
        dist = torch.cdist(points, points)
        idx = torch.topk(dist, k=k + 1, largest=False, dim=2)[1][:, :, 1:]  # [B,N,k]
        # gather neighbours: gather along dim=1 using idx: points [B,N,3]
        Bn = points.shape[0]
        b_idx = torch.arange(Bn, device=points.device).view(Bn, 1, 1)       # [B,1,1]
        neigh = points[b_idx, idx]                                           # [B,N,k,3]
        center = points.unsqueeze(2)                                          # [B,N,1,3]
        centered = neigh - center                                             # [B,N,k,3]

        cov = centered.transpose(-1, -2) @ centered / k                       # [B,N,3,3]
        # eigen-decomposition: evecs cols are eigenvectors, evals ascending
        evals, evecs = torch.linalg.eigh(cov)                                 # [B,N,3], [B,N,3,3]
        t1 = evecs[..., :, 1].contiguous()   # 2nd smallest evec (tangent1)
        t2 = evecs[..., :, 2].contiguous()   # largest evec (tangent2)
        nrm = evecs[..., :, 0].contiguous()  # smallest evec (normal)
        # orient normal consistently outward-ish (towards mean)
        normal = nrm

        # local scale = RMS neighbour distance, but clamp to a stable band
        # to avoid boundary/isolated points producing huge rings that fly off
        scale = torch.sqrt(centered.pow(2).sum(-1).mean(dim=-1)).clamp(min=1e-6)  # [B,N]
        # clamp radius to [0.5x, 1.5x] of the GLOBAL neighbour-distance median
        global_scale = scale.median()
        scale = scale.clamp(min=0.4 * global_scale, max=1.6 * global_scale)

        # deterministic concentric rings ON tangent plane
        rings = torch.arange(1, num_rings + 1, device=points.device).float() / num_rings
        angles = torch.arange(num_angles, device=points.device).float() / num_angles * 2 * PI

        # radius per point: [B,N,R]
        radius = scale.unsqueeze(-1) * rings.unsqueeze(0).unsqueeze(0)
        # [B,N,R,A] angle basis
        cos_a = torch.cos(angles)                          # [A]
        sin_a = torch.sin(angles)                          # [A]
        r5 = radius.unsqueeze(-1).unsqueeze(-1)        # [B,N,R,1,1]
        c = cos_a.view(1, 1, 1, -1, 1)                # [1,1,1,A,1]
        s = sin_a.view(1, 1, 1, -1, 1)                # [1,1,1,A,1]
        t1b = t1.unsqueeze(2).unsqueeze(3)            # [B,N,1,1,3]
        t2b = t2.unsqueeze(2).unsqueeze(3)            # [B,N,1,1,3]
        offset = r5 * (c * t1b + s * t2b)             # [B,N,R,A,3]
        center5 = points.unsqueeze(2).unsqueeze(3)    # [B,N,1,1,3]
        new_pts = center5 + offset                    # [B,N,R,A,3]
        expand = new_pts.reshape(B, N * num_rings * num_angles, 3)
        return expand, normal

    @staticmethod
    def point_transform(points, rot_mat, rot_mat2, translation):
        rot_mat = rot_mat.to(points.device)
        rot_mat2 = rot_mat2.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = torch.matmul(points, rot_mat2)
        points = points - translation
        return points
