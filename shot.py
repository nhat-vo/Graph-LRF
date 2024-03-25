import torch
from lrf import LRFDescriptor

# correction term for numerical stability
EPS = 1e-5


class SHOTDescriptor(LRFDescriptor):
    def __init__(
        self,
        n_neighbors=5,
        local_bins=10,
        radial_bins=2,
        elevation_bins=2,
        azimuth_bins=2,
    ):
        super().__init__(n_neighbors=n_neighbors)
        self.local_bins = local_bins
        self.spatial_bins = [radial_bins, elevation_bins, azimuth_bins]
        self.total_bins = local_bins * radial_bins * elevation_bins * azimuth_bins

    def forward(self, points, batch, return_lrfs=False):
        lrfs, nbh_proj, nbh_idx, point_clouds_length = super().forward(points, batch)

        partition = nbh_proj >= 0  # (B, N_max, K, 3)
        spatial_id = (
            (partition[..., 0] << 2) + (partition[..., 1] << 1) + (partition[..., 2])
        )  # (B, N_max, K)

        B, N_max, K = nbh_idx.shape
        normals = lrfs[..., 0]  # (B, N_max, 3)
        normals_expand = normals.unsqueeze(2).expand(B, N_max, K, 3)
        nbh_idx_expand = nbh_idx.unsqueeze(-1).expand(B, N_max, K, 3)
        nbh_normals = torch.gather(
            normals_expand, 1, nbh_idx_expand
        )  # need to gather along N_max

        cos = torch.sum(
            normals[:, :, None] * nbh_normals, dim=-1
        )  # (B, N, 1, 3) * (B, N_max, K, 3) -> (B, N_max, K, 3) -> (B, N, K)

        normal_id = torch.floor(
            self.local_bins * (cos + 1) / 2
        )  # [-1, 1] -> [0, 1] -> [0, local_bins]
        normal_id = torch.clamp(normal_id, 0, self.local_bins - 1)

        bin_id = spatial_id * self.local_bins + normal_id  # (B, N_max, K)
        descriptor = torch.zeros([B, N_max, self.total_bins]).to(lrfs)
        descriptor = descriptor.scatter_add_(
            -1, bin_id.long(), torch.ones_like(bin_id).to(descriptor)
        )  # (B, N_max, n_bins)

        # we were using padded tensor for calculation, so here we convert them back
        shot = [descriptor[i, :pclen] for i, pclen in enumerate(point_clouds_length)]
        shot = torch.cat(shot, dim=0)
        return shot if not return_lrfs else (shot, lrfs)
