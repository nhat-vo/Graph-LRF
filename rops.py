import torch
from lrf import LRFDescriptor

# correction term for numerical stability
EPS = 1e-5


class ROPSDescriptor(LRFDescriptor):
    def __init__(self, n_neighbors=5, n_angles=2):
        super().__init__(n_neighbors=n_neighbors)
        self.n_angles = n_angles
        self.total_bins = n_angles

        angles = torch.linspace(0, 2 * torch.pi, n_angles + 1)[:-1]
        self.rots = torch.cat(
            [self._compute_rotations(a) for a in angles], dim=0
        )  # (n_angles*3, 3, 3)

    def _compute_rotations(self, angle):
        c = torch.cos(angle)
        s = torch.sin(angle)
        return torch.tensor(
            [
                [[1, 0, 0], [0, c, -s], [0, s, c]],
                [[c, 0, s], [0, 1, 0], [-s, 0, c]],
                [[c, -s, 0], [s, c, 0], [0, 0, 1]],
            ]
        )

    def _compute_statistics_2d(self, nbh_rot_xy):
        # nbh_rot_xy (B, N_max, n_angles*3, K, 2)

        # (B, N_max, n_angles*3)
        mu_11 = (nbh_rot_xy[..., 0] * nbh_rot_xy[..., 1]).mean(dim=-1)
        mu_22 = (nbh_rot_xy[..., 0] ** 2 * nbh_rot_xy[..., 1] ** 2).mean(dim=-1)
        mu_12 = (nbh_rot_xy[..., 0] * nbh_rot_xy[..., 1] ** 2).mean(dim=-1)
        mu_21 = (nbh_rot_xy[..., 0] ** 2 * nbh_rot_xy[..., 1]).mean(dim=-1)

        return torch.stack(
            [mu_11, mu_22, mu_12, mu_21], dim=-1
        )  # (B, N_max, n_angles*3, 4)

    def _compute_statistics(self, nbh_rot):
        # nbh_rot (B, N_max, n_angles*3, K, 3)

        return torch.cat(
            [
                self._compute_statistics_2d(nbh_rot[..., [0, 1]]),
                self._compute_statistics_2d(nbh_rot[..., [0, 2]]),
                self._compute_statistics_2d(nbh_rot[..., [1, 2]]),
            ],
            dim=-1,
        )  # (B, N_max, n_angles*3, 4*3)

    def forward(self, points, batch, return_lrfs=False):
        (
            lrfs,  # (B, N_max, 3, 3)
            nbh_proj,  # (B, N_max, K, 3)
            nbh_idx,  # (B, N_max, K)
            point_clouds_length,  # (B,)
        ) = super().forward(points, batch)

        # there is supposed to be self.rots.transpose here, but it doesn't matter
        nbh_rot = nbh_proj[:, :, None] @ self.rots.to(
            nbh_proj
        )  # (B, N_max, 1, K, 3) @ (n_angles*3, 3, 3) -> (B, N_max, n_angles*3, K, 3)

        stats = self._compute_statistics(nbh_rot)  # (B, N_max, n_angles*3, 4*3)
        B, N_max, n_angles3, _ = stats.shape
        stats = stats.view(B, N_max, -1)

        shot = [stats[i, :pclen] for i, pclen in enumerate(point_clouds_length)]
        shot = torch.cat(shot, dim=0)
        return shot if not return_lrfs else (shot, lrfs)
