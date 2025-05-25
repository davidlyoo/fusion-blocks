import torch
import torch.nn as nn
import torch.nn.functional as F


class DEM(nn.Module):
    def __init__(self, channels, channel_reduction=4):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)  # [B, C, H, W] → [B, C, 1, 1]
        self.gmp = nn.AdaptiveMaxPool2d(1)  # [B, C, H, W] → [B, C, 1, 1]

        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels, channels // channel_reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // channel_reduction, channels, kernel_size=1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, f_o, f_s):
        f_d = f_o - f_s
        
        s1 = self.gap(f_d)
        s2 = self.gmp(f_d)
        
        z1 = self.shared_conv(s1)
        z2 = self.shared_conv(s2)

        m_dm = self.sigmoid(z1 + z2)

        f_o_enhanced = f_o * (1 + m_dm)
        f_s_enhanced = f_s * (1 + m_dm)
        
        f_dm = f_o_enhanced + f_s_enhanced
        
        return f_dm