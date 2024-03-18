from math import e
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import unbatch
from pytorch3d.ops import estimate_pointcloud_local_coord_frames, knn_points
from pytorch3d.structures import Pointclouds
from torch_scatter import scatter_add

# correction term for numerical stability
EPS = 1e-5


class SHOTTransform(BaseTransform):
    def forward(self, data):
        pos = data.pos
        shot = SHOTDescriptor().forward(pos.cuda(), False)
        data.shot = shot
        return data


class SHOTDescriptor(nn.Module):
    def __init__(
        self,
        n_neighbors=5,
        local_bins=10,
        radial_bins=2,
        elevation_bins=2,
        azimuth_bins=2,
    ):
        super().__init__()
        self.n_neighbors = 5
        self.local_bins = local_bins
        self.spatial_bins = [radial_bins, elevation_bins, azimuth_bins]
        self.total_bins = local_bins * radial_bins * elevation_bins * azimuth_bins

    def forward(self, points, batch, return_lrfs=False):
        point_clouds = Pointclouds(points=list(unbatch(points, batch)))
        point_clouds_padded = point_clouds.points_padded()
        point_clouds_length = point_clouds.num_points_per_cloud()
        n_neighbors = self.n_neighbors

        # calculate LRF and normals
        curvature, lrfs = estimate_pointcloud_local_coord_frames(
            point_clouds, n_neighbors
        )  # lrfs: (B, N_max, 3, 3)
        normals = lrfs[..., 0]  # (B, N_max, 3)

        # calculate neighbors
        nbh_dist2, nbh_idx, nbh = knn_points(
            point_clouds_padded,
            point_clouds_padded,
            point_clouds_length,
            point_clouds_length,
            K=n_neighbors,
            return_nn=True,
        )  # _, (B, N_max, K), (B, N_max, K, 3)

        # remove self
        # nbh_idx = [idx[1:] for idx in nbh_idx]
        # nbh_dist = [torch.tensor(dist[1:], dtype=torch.float) for dist in nbh_dist]

        descriptors = []
        for b in range(lrfs.shape[0]):
            N = point_clouds_length[b]

            # project to LRF
            nbh_proj = torch.bmm(
                nbh[b, :N], lrfs[b, :N]
            )  # (N, K, 3) @ (N, 3, 3) -> (N, K, 3)

            upper_x = nbh_proj[:, :, 0] >= 0  # (N, K)
            upper_y = nbh_proj[:, :, 1] >= 0  # (N, K)
            upper_z = nbh_proj[:, :, 2] >= 0  # (N, K)
            spatial_id = (upper_x << 2) + (upper_y << 1) + (upper_z)

            # nbh_idx[b, :N]: (N, K)
            # normals[b, nbh_idx[b, :N]]: (N, K, 3)
            cos = torch.sum(
                normals[b, :N, None] * normals[b, nbh_idx[b, :N]], dim=-1
            )  # (N, 1, 3) * (N, K, 3) -> (N, K, 3) -> (N, K)
            normal_id = torch.floor(
                self.local_bins * (cos + 1) / 2
            )  # [-1, 1] -> [0, 1] -> [0, local_bins]
            normal_id = torch.clamp(normal_id, 0, self.local_bins - 1)
            bin_id = spatial_id * self.local_bins + normal_id  # (N, K)
            descriptor = scatter_add(
                torch.ones_like(bin_id, dtype=torch.float),
                bin_id.type(torch.int64),
                dim=-1,
                dim_size=self.total_bins,
            )  # (N, n_bins)
            descriptors.append(descriptor)

        shot = torch.cat(descriptors, dim=0)  # (sum(N), n_bins)

        return shot if not return_lrfs else (shot, lrfs)

    def _compute_desc(self, points, nbh_idx, lrfs, normals):
        descriptors = torch.zeros(
            (points.shape[0], self.total_bins), dtype=torch.float32
        )

        cos = torch.inner(normals, normals)

        for batch in points:
            for i, point in enumerate(batch):
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
                    spatial_desc = (
                        rad_desc * 8 + ele_desc * 4 + azi_desc_1 * 2 + azi_desc_2
                    )

                    descriptors[i, spatial_desc * self.local_bins + local_desc] += 1

                # descriptors[i] /= torch.sum(descriptors[i])
        return descriptors
