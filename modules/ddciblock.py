import torch
import torch.nn as nn
import torch.nn.functional as F

from .DDCINet.csca import CSCA
from .CMAFF.dem import DEM


class DDCIBlock(nn.Module):
    def __init__(self, channels, num_heads, spatial_reduction, channel_reduction):
        super().__init__()
        self.csca = CSCA(
            channels=channels,
            num_heads=num_heads,
            spatial_reduction=spatial_reduction,
            channel_reduction=channel_reduction
        )
        self.dem = DEM(
            channels=channels, channel_reduction=channel_reduction
        )
    
    def forward(self, opt_feat, sar_feat):
        opt_att, sar_att = self.csca(opt_feat, sar_feat)
        fused = self.dem(opt_att, sar_att)
        return fused