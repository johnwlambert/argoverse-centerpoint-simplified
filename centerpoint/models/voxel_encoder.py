

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VoxelFeatureExtractorV3(nn.Module):
    def __init__(
        self,
        num_input_features: int = 4,
        norm_cfg = None,
        name: str = "VoxelFeatureExtractorV3"
    ) -> None:
        """
        """
        super(VoxelFeatureExtractorV3, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(
        self,
        features: torch.Tensor,
        num_voxels: torch.Tensor,
        coors=None
    ) -> torch.Tensor:
        """
        Args:
            features: tensor of shape [360000, 10, 5], zero-padded where less
                than 10 points per voxel
            num_voxels: tensor of shape [360000,] representing #pts / voxel

        Returns:
            points_mean: tensor of shape [360000, 5] representing mean of
                all points within the voxel
        """
        points_mean = features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)

        return points_mean.contiguous()