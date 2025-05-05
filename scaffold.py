# ddcinet_fusion_scaffold

import torch
import torch.nn as nn
import torch.nn.functional as F

class CSCA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # MSSR for multiscale spatial reduction
        self.sr_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=3, padding=d, dilation=d)
            for d in [1, 2, 3]
        ])
        self.sr_fuse = nn.Conv2d(in_channels // 8 * 3, in_channels, kernel_size=1)

        # Cross-attention
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # Channel-Gated FFN
        self.channel_expand = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)
        self.relu = nn.ReLU()
        self.channel_reduce = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # MSSR: multiscale spatial reduction
        mssr_feats = torch.cat([conv(x1) for conv in self.sr_convs], dim=1)
        reduced_x1 = self.sr_fuse(mssr_feats)

        # Cross-attention
        q = self.query_conv(reduced_x1).flatten(2)
        k = self.key_conv(x2).flatten(2)
        v = self.value_conv(x2).flatten(2)
        attn = self.softmax(torch.bmm(q.transpose(1, 2), k))
        attended = torch.bmm(v, attn.transpose(1, 2)).view_as(x1)

        # CG-FFN
        fused = x1 + attended
        gate = self.sigmoid(self.channel_reduce(self.relu(self.channel_expand(fused))))
        return fused * gate + x1


class CMDF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Static filter
        self.static_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # Dynamic spatial filter
        self.dynamic_spatial = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )

        # Dynamic channel filter
        self.dynamic_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x_cat = torch.cat([x1, x2], dim=1)
        spatial_out = self.dynamic_spatial(x_cat)
        channel_gate = self.dynamic_channel(x_cat)
        static_out = self.static_conv(x1)
        return static_out + spatial_out * channel_gate


class DDCIModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.csca = CSCA(in_channels)
        self.cmdf = CMDF(in_channels)

    def forward(self, x1, x2):
        fused = self.csca(x1, x2)
        out = self.cmdf(fused, x2)
        return out
