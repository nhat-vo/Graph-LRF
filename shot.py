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
        shot = SHOTDescriptor().forward(pos.cuda(), False)
        data.shot = shot
        return data


class SHOTDescriptor:
    def __init__(
        self,
        n_neighbors=5,
        local_bins=10,
        radial_bins=2,
        elevation_bins=2,
        azimuth_bins=4,
    ):
        super().__init__()
        self.n_neighbors = 5
        self.local_bins = local_bins
        self.spatial_bins = [radial_bins, elevation_bins, azimuth_bins]
        self.total_bins = local_bins * radial_bins * elevation_bins * azimuth_bins

    def forward(self, points, return_lrfs=False):
        # calculate neighbors
        n_neighbors = min(self.n_neighbors, points.shape[0] - 1)
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors.fit(points)
        nbh_dist, nbh_idx = neighbors.kneighbors(points, return_distance=True)
        nbh_dist = [torch.tensor(dist, dtype=torch.float) for dist in nbh_dist]

        # remove self
        # nbh_idx = [idx[1:] for idx in nbh_idx]
        # nbh_dist = [torch.tensor(dist[1:], dtype=torch.float) for dist in nbh_dist]

        # calculate LRF and normals
        curvature, lrfs = estimate_pointcloud_local_coord_frames(
            points[None], n_neighbors
        )
        lrfs = lrfs[0]
        normals = lrfs[..., 0]

        # calculate SHOT descriptor
        shot = self._compute_desc(points, nbh_idx, lrfs, normals)

        return shot if not return_lrfs else (shot, lrfs)

    def _compute_desc(self, points, nbh_idx, lrfs, normals):
        descriptors = torch.zeros(
            (points.shape[0], self.total_bins), dtype=torch.float32
        )

        cos = torch.inner(normals, normals)

        for i, point in enumerate(points):
            # project neighbors onto LRF
            nbh = points[nbh_idx[i]]
            proj = (nbh - point) @ lrfs[i]

            for j, pt_prj in enumerate(proj):
                local_desc = int(self.local_bins * (cos[i, j] + 1) / 2)
                local_desc = max(0, min(local_desc, self.local_bins - 1))

                rad_desc = 1 if pt_prj.norm() >= (2 / 2) else 0
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
