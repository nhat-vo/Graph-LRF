import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import unbatch
from pytorch3d.ops import estimate_pointcloud_local_coord_frames, knn_points
from pytorch3d.structures import Pointclouds
from torch_scatter import scatter_add

# correction term for numerical stability
EPS = 1e-5


class LRFDescriptor(nn.Module):
    def __init__(self, n_neighbors=5):
        super().__init__()
        self.n_neighbors = n_neighbors

    def forward(self, points, batch):
        point_clouds = Pointclouds(points=list(unbatch(points, batch)))
        point_clouds_padded = point_clouds.points_padded()  # (B, N_max, 3)
        point_clouds_length = point_clouds.num_points_per_cloud()
        n_neighbors = self.n_neighbors

        # calculate LRF and normals
        curvature, lrfs = estimate_pointcloud_local_coord_frames(
            point_clouds, n_neighbors
        )  # lrfs: (B, N_max, 3, 3)

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

        # project to LRF
        nbh_proj = (
            nbh - point_clouds_padded.unsqueeze(2)
        ) @ lrfs  # (B, N_max, K, 3) @ (B, N_max, 3, 3) -> (B, N_max, K, 3)

        return lrfs, nbh_proj, nbh_idx, point_clouds_length
