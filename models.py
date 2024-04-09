import torch
from torch import nn
from torch.nn import functional as F

import torch_geometric.nn as gnn


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0, skip=0):
        super().__init__()
        self.gconvs = torch.nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(1, len(dims)):
            self.gconvs.append(gnn.GCNConv(dims[i - 1], dims[i]))

        self.logit = nn.Linear(dims[-1], output_dim)
        self.dropout = dropout
        self.skip = skip

    def forward(self, x, edge_index, batch):
        for i, l in enumerate(self.gconvs):
            prev_x = x
            x = l(x, edge_index)
            x = F.relu(x)

            if self.skip != 0 and i % self.skip == 0:
                x += prev_x

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = gnn.global_mean_pool(x, batch)
        x = self.logit(x)
        return x


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0, skip=0):
        super().__init__()
        self.gconvs = torch.nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(1, len(dims)):
            self.gconvs.append(gnn.GATv2Conv(dims[i - 1], dims[i]))

        self.logit = nn.Linear(dims[-1], output_dim)
        self.dropout = dropout
        self.skip = skip

    def forward(self, x, edge_index, batch):
        for i, l in enumerate(self.gconvs):
            prev_x = x
            x = l(x, edge_index)
            x = F.relu(x)

            if self.skip != 0 and i % self.skip == 0:
                x += prev_x

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = gnn.global_mean_pool(x, batch)
        x = self.logit(x)
        return x
