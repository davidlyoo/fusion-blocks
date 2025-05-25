import torch
import torch.nn as nn
import torch.nn.functional as F


class DFFModule(nn.Module):
    def __init__(self, in_channels, channel_reduction=4):
        super(DFFModule, self).__init__()
        
        self.conv_gx = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv_gy = nn.Conv2d(in_channels, 1, kernel_size=1)
        
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // r, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, y):
        B, C, H, W = x.shape

        # Global context extraction
        gx = self.conv_gx(x)                          # (B, 1, H, W)
        gx = gx.view(B, -1)                           # (B, HW)
        gx = F.softmax(gx, dim=1)
        gx = gx.view(B, -1, 1, 1)                     # (B, HW, 1, 1)
        
        gy = self.conv_gy(y)
        gy = gy.view(B, -1)
        gy = F.softmax(gy, dim=1)
        gy = gy.view(B, -1, 1, 1)
        
        x_reshaped = x.view(B, C, -1)    # (B, C, HW)
        y_reshaped = y.view(B, C, -1)
        
        context_x = torch.matmul(x_reshaped, gx)
        context_y = torch.matmul(y_reshaped, gy)
        
        s = context_x + context_y
        s = s.view(B, C, 1, 1)
        
        # Channel-wise attention weight
        a = self.fuse(s)
        
        z = a * x + (1 - a) * y
        
        return z