

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VoxelFeatureExtractorV3(nn.Module):
    def __init__(
        self, num_input_features=4, norm_cfg=None, name="VoxelFeatureExtractorV3"
    ):
        super(VoxelFeatureExtractorV3, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        points_mean = features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)

        return points_mean.contiguous()