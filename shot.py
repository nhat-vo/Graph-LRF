import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform
from sklearn.neighbors import NearestNeighbors
from pytorch3d.ops import estimate_pointcloud_local_coord_frames

# correction term for numerical stability
EPS = 1e-5


class SHOTTransform(BaseTransform):
    def forward(self, data):
        pos = data.pos
        shot = SHOTDescriptor()(pos, False)
        data.shot = shot
        return data


class SHOTDescriptor(nn.Module):
    def __init__(
        self, radius=0.1, local_bins=10, radial_bins=2, elevation_bins=2, azimuth_bins=4
    ):
        super().__init__()
        self.radius = radius
        self.local_bins = local_bins
        self.spatial_bins = [radial_bins, elevation_bins, azimuth_bins]
        self.total_bins = local_bins * radial_bins * elevation_bins * azimuth_bins

    def forward(self, points, return_lrfs=False):
        # calculate neighbors
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors.fit(points)
        nbh_dist, nbh_idx = neighbors.kneighbors(points, return_distance=True)
        nbh_dist = [torch.tensor(dist, dtype=torch.float) for dist in nbh_dist]

        # remove self
        # nbh_idx = [idx[1:] for idx in nbh_idx]
        # nbh_dist = [torch.tensor(dist[1:], dtype=torch.float) for dist in nbh_dist]

        # calculate LRF and normals
        lrfs = self._compute_lrfs(points, nbh_dist, nbh_idx)
        normals = lrfs[:, 2]  # normal is the z-axis
        # lrfs = estimate_pointcloud_local_coord_frames(points[None], 5)
        # normals = lrfs[:0]

        # calculate SHOT descriptor
        # shot = self._compute_desc(points, nbh_idx, lrfs, normals)
        shot = 0

        return shot if not return_lrfs else (shot, lrfs)

    def _calc_axis(self, point, nbh, eigvecs, axis):
        """Correct the axis direction"""
        x_plus = eigvecs.T[:, axis]

        count = 0
        for pt in nbh:
            if (pt - point).dot(x_plus) >= -EPS:
                count += 1

        x_axis = x_plus if count >= len(nbh) / 2 else -x_plus
        return x_axis

    def _compute_lrfs(self, points: torch.Tensor, nbh_dist, nbh_idx) -> torch.Tensor:
        """Compute the Local Reference Frames for each point in the point cloud."""

        lrfs = []
        for i, point in enumerate(points):
            nbh = points[nbh_idx[i]] - point
            weighted_cov = torch.cov(nbh.T, correction=0)
            # nbh -= nbh.mean(dim=1, keepdim=True)
            # weighted_cov = nbh.T @ nbh
            # print(weighted_cov)

            eigvals, eigvecs = torch.linalg.eigh(weighted_cov)
            order = torch.argsort(eigvals, descending=True)
            print(order)

            x_axis = self._calc_axis(point, nbh, eigvecs, order[0])
            z_axis = self._calc_axis(point, nbh, eigvecs, order[2])
            y_axis = z_axis.cross(x_axis)
            lrfs.append(torch.stack([x_axis, y_axis, z_axis]))

        return torch.stack(lrfs)

    def _compute_desc(self, points, nbh_idx, lrfs, normals):
        descriptors = torch.zeros(
            (points.shape[0], self.total_bins), dtype=torch.float32
        )

        # precalculate cosines
        norm = torch.linalg.norm(normals, axis=1)
        cos = torch.inner(normals, normals) / torch.outer(norm, norm)

        for i, point in enumerate(points):
            # project neighbors onto LRF
            nbh = points[nbh_idx[i]]
            if nbh.shape[0] == 0:
                continue
            if nbh.shape[0] == 1:
                print("WARN: only 1 neighbor")
            proj = torch.inner(nbh - point, lrfs[i])

            for j, pt_prj in enumerate(proj):
                local_desc = int(self.local_bins * (cos[i, j] + 1) / 2)
                local_desc = max(0, min(local_desc, self.local_bins - 1))

                rad_desc = 1 if pt_prj.norm() >= (self.radius / 2) else 0
                ele_desc = 1 if (pt_prj[2] >= 0) else 0
                azi_desc_1 = (
                    1 if pt_prj[0] >= 0 or (pt_prj[0] == 0 and pt_prj[1] < 0) else 0
                )
                azi_desc_2 = (
                    1 if pt_prj[0] <= 0 or (pt_prj[0] == 0 and pt_prj[1] > 0) else 0
                )
                spatial_desc = rad_desc * 8 + ele_desc * 4 + azi_desc_1 * 2 + azi_desc_2

                descriptors[i, spatial_desc * self.local_bins + local_desc] += 1

            # descriptors[i] /= torch.sum(descriptors[i])
        return descriptors
