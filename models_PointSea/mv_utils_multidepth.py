import torch
import numpy as np

resolution = 224
PI = 3.141592653589793


class PCViews_MultiDepth:
    """Multi-depth pointmap with adaptive PCA viewpoint selection.

    Key ideas:
      (1) For each projected pixel, keep BOTH the closest and farthest 3D
          point along the viewing ray -> [H, W, 3, 2] == 6 channels
          (near_xyz, far_xyz).  This preserves front/back layer info.
      (2) Before projection, select viewpoints whose direction makes the point
          cloud collapse to low rank (most collinear / coincident projected
          points).  Such views look along flat planes / straight lines and
          reveal the pre-corruption structure.  We search candidate directions
          (spanning the PCA principal frame plus spherical grid) and pick the
          ones with the best collinearity score.
    """

    def __init__(self, k=16, num_rings=4, num_angles=8,
                 num_candidates=64, num_views=3):
        self.k = k
        self.num_rings = num_rings
        self.num_angles = num_angles
        self.num_candidates = num_candidates
        self.num_views = num_views

    # ---------- viewpoint selection (traditional, collinearity-based) ----------
    def select_views(self, points):
        """points: [B,N,3] -> view_dirs: [B, num_views, 3]."""
        B, N, _ = points.shape
        centroid = points.mean(dim=1, keepdim=True)
        centered = points - centroid
        cov = centered.transpose(1, 2) @ centered / N          # [B,3,3]
        evals, evecs = torch.linalg.eigh(cov)                  # ascending
        axes = evecs.permute(0, 2, 1)                          # [B,3,3]

        # candidate directions: 3 PCA axes + spherical grid
        cands = torch.cat([axes,                       # [B,3,3]
                           axes[:, [0, 1], :],         # combos
                           -axes[:, [1, 2], :],
                           -axes[:, [0, 2], :]], dim=1)  # [B,9,3]
        # add spherical samples
        n_sph = max(8, self.num_candidates)
        phis = torch.linspace(0, PI, n_sph, device=points.device)
        thetas = torch.linspace(0, 2 * PI, n_sph, device=points.device)[:-1]
        sph = torch.stack([
            torch.sin(phis).view(-1, 1) * torch.cos(thetas).view(1, -1),
            torch.sin(phis).view(-1, 1) * torch.sin(thetas).view(1, -1),
            torch.cos(phis).view(-1, 1).expand(-1, thetas.numel()),
        ], dim=-1).reshape(-1, 3)                                  # [S,3]
        sph = sph.repeat(B, 1, 1)                                 # [B,S,3]
        all_cands = torch.cat([cands, sph], dim=1)                # [B,C,3]

        # normalize
        all_cands = all_cands / (all_cands.norm(dim=-1, keepdim=True) + 1e-8)

        best_dirs = []
        for b in range(B):
            best = []
            centered_b = centered[b]                               # [N,3]
            for di in range(all_cands.shape[1]):
                d = all_cands[b, di]
                idn = torch.eye(3, device=points.device)
                P = idn - d.unsqueeze(-1) @ d.unsqueeze(0)         # 3x3 projector
                proj2d = centered_b @ P.T                          # [N,3]
                cov2 = proj2d.transpose(0, 1) @ proj2d            # [3,3]
                ev = torch.linalg.eigvalsh(cov2)                  # ascending
                sv = ev.flip(0)                                   # descending
                score = (sv[0] + 1e-8) / (sv[1] + 1e-8)           # high = collinear
                best.append((score.item(), d.detach().clone()))
            best.sort(key=lambda t: t[0], reverse=True)
            picked = []
            for sc, d in best:
                if all((d - p).norm() > 0.35 for p in picked):
                    picked.append(d)
                if len(picked) >= self.num_views:
                    break
            while len(picked) < self.num_views:
                picked.append(best[len(picked)][1])
            best_dirs.append(torch.stack(picked))
        return torch.stack(best_dirs)                              # [B, num_views, 3]

    # ---------- PCA expansion (geometry-preserving) ----------
    def _pca_expand(self, points):
        B, N, _ = points.shape
        k = self.k
        num_rings = self.num_rings
        num_angles = self.num_angles
        dist = torch.cdist(points, points)
        idx = torch.topk(dist, k=k + 1, largest=False, dim=2)[1][:, :, 1:]
        b_idx = torch.arange(B, device=points.device).view(B, 1, 1)
        neigh = points[b_idx, idx]
        centered = neigh - points.unsqueeze(2)
        cov = centered.transpose(-1, -2) @ centered / k
        evals, evecs = torch.linalg.eigh(cov)
        t1 = evecs[..., :, 1].contiguous()
        t2 = evecs[..., :, 2].contiguous()
        scale = torch.sqrt(centered.pow(2).sum(-1).mean(dim=-1)).clamp(min=1e-6)
        gs = scale.median()
        scale = scale.clamp(min=0.4 * gs, max=1.6 * gs)
        rings = torch.arange(1, num_rings + 1, device=points.device).float() / num_rings
        angles = torch.arange(num_angles, device=points.device).float() / num_angles * 2 * PI
        radius = scale.unsqueeze(-1) * rings
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        r5 = radius.unsqueeze(-1).unsqueeze(-1)
        c = cos_a.view(1, 1, 1, -1, 1)
        s = sin_a.view(1, 1, 1, -1, 1)
        t1b = t1.unsqueeze(2).unsqueeze(3)
        t2b = t2.unsqueeze(2).unsqueeze(3)
        offset = r5 * (c * t1b + s * t2b)
        center5 = points.unsqueeze(2).unsqueeze(3)
        new_pts = center5 + offset
        expand = new_pts.reshape(B, N * num_rings * num_angles, 3)
        return expand

    # ---------- project with multi-depth ----------
    def get_multi_pointmap(self, points):
        """points: [B,N,3] -> (pointmap [Bv,6,H,W], mask [Bv,1,H,W])"""
        B, N, _ = points.shape
        H = W = resolution
        expand = self._pca_expand(points)         # [B, E, 3]
        all_pts = torch.cat([points, expand], 1)  # [B, N+E, 3]
        is_orig = torch.cat(
            [torch.ones(B, N, device=points.device),
             torch.zeros(B, expand.shape[1], device=points.device)], 1)

        view_dirs = self.select_views(points)     # [B, v, 3]
        v = view_dirs.shape[1]
        bv = B * v

        pms = []
        mks = []
        for b in range(B):
            for vi in range(v):
                d = view_dirs[b, vi]
                d = d / (d.norm() + 1e-8)
                ez = torch.tensor([0.0, 0.0, 1.0], device=points.device)
                vv = ez - d * (ez @ d)
                if vv.norm() < 1e-6:
                    R = torch.eye(3, device=points.device).flip(0).flip(1)
                else:
                    vv = vv / vv.norm()
                    cross = torch.linalg.cross(d, vv)
                    R = torch.stack([vv, cross, d], dim=0)
                pts_v = all_pts[b] @ R.T
                pm, mk = self._project_view(pts_v, is_orig[b], H, W)
                pms.append(pm)
                mks.append(mk)
        pointmap = torch.stack(pms).view(bv, 6, H, W)
        mask = torch.stack(mks).view(bv, 1, H, W)
        return pointmap, mask

    def _project_view(self, pts_v, iso, H, W):
        pmin = pts_v.min(dim=0)[0]
        pmax = pts_v.max(dim=0)[0]
        prange = torch.clamp(pmax - pmin, min=1e-6)
        x_norm = (pts_v[:, 0] - pmin[0]) / prange[0]
        y_norm = (pts_v[:, 1] - pmin[1]) / prange[1]
        z_norm = (pts_v[:, 2] - pmin[2]) / prange[2]
        z = pts_v[:, 2]

        col = (x_norm * (W - 1)).long().clamp(0, W - 1)
        row = (y_norm * (H - 1)).long().clamp(0, H - 1)
        flat = row * W + col
        coords = torch.stack([x_norm, y_norm, z_norm], 0)  # [3, M]

        n_pix = H * W
        near = torch.zeros(3, n_pix, device=pts_v.device)
        far = torch.zeros(3, n_pix, device=pts_v.device)
        mask = torch.zeros(n_pix, device=pts_v.device)

        # nearest (min z) and farthest (max z) per pixel via scatter argmin/argmax
        for c in range(3):
            vc = coords[c]
            # min z -> store coord of argmin
            near[c] = self._scatter_first(vc, flat, z, n_pix, mode='min')
            far[c] = self._scatter_first(vc, flat, z, n_pix, mode='max')
        # mask: 1 if the NEAREST point at that pixel is an ORIGINAL point
        near_orig = self._scatter_first(iso.float(), flat, z, n_pix, mode='min')
        mask[flat[iso.bool()]] = 1.0
        near_mask = near_orig > 0.5
        mask[near_mask] = 1.0
        # also mark pixels that got any original point as original
        oidx = flat[iso.bool()]
        mask[oidx] = 1.0

        pointmap = torch.cat([near, far], 0).view(6, H, W)
        return pointmap, mask.view(1, H, W)

    @staticmethod
    def _scatter_first(values, flat, z, n_pix, mode='min'):
        """For each pixel, take the value of the point with min (or max) z."""
        z_sign = z if mode == 'min' else -z
        # tie-break: value from the argmin-z point
        # scatter minimises z_sign; recover the chosen point's value via a key
        key = z_sign.float()
        # we need value at the argmin point. Use scatter with key+epsilon*value trick:
        # not exact. Instead compute per-pixel: gather using argmin.
        # Simpler correct approach via sorting.
        order = torch.argsort(key)
        key_s = key[order].contiguous()
        flat_s = flat[order].contiguous()
        val_s = values[order].contiguous()
        # first occurrence of each pixel = min key point
        pos = torch.zeros(n_pix, dtype=torch.long, device=values.device)
        seen = torch.zeros(n_pix, dtype=torch.bool, device=values.device)
        # vectorised: first index where each pixel appears in sorted order
        uniq, first_idx = torch.unique(flat_s, return_inverse=True)
        out = torch.zeros(n_pix, device=values.device)
        # first_idx gives, for each sorted position, its unique-group; we want the FIRST
        # sorted position of each group. Use scatter_min on positions.
        arange = torch.arange(flat_s.numel(), device=values.device)
        first_pos = torch.full((n_pix,), 10**9, dtype=torch.long, device=values.device)
        first_pos.scatter_reduce_(0, flat_s, arange, reduce='amin', include_self=False)
        valid = first_pos < 10**9
        out[valid] = val_s[first_pos[valid].clamp(max=val_s.numel() - 1)]
        return out
