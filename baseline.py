import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as G


def calculate_cov3D(points: torch.Tensor):
    """Calculate the 3D Covariance matrix of a point cloud."""
