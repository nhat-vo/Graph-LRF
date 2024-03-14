import torch_geometric.transforms as T
import torch


class PosAsX(T.BaseTransform):
    def forward(self, data):
        if data.x is None:
            data.x = data.pos
        else:
            pos = data.pos.repeat(1, 1, data.x.shape[0])
            data.x = torch.cat([data.x, pos], dim=1)
        return data


class SetTarget(T.BaseTransform):
    def __init__(self, target_idx):
        self.target = target_idx if isinstance(target_idx, list) else [target_idx]

    def forward(self, data):
        data.y = data.y[..., self.target]
        return data
