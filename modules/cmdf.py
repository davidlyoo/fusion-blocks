import torch
import torch.nn as nn
import torch.nn.functional as F


class CMDF(nn.Module):
    def __init__(self, in_channels, kernel_size=3, r=4):
        super(CMDF, self).__init__()
        c_half = in_channels // 2
        k_square = kernel_size * kernel_size
        self.k_square = k_square
        
        # Static depth-wise conv filter
        self.static_kernel = nn.Conv2d(c_half, c_half, kernel_size=kernel_size,
                                     padding=kernel_size // 2, groups=c_half)
        
        # Spatial-wise kernel
        self.spatial_kernel = nn.Conv2d(in_channels, k_square, kernel_size=1)
        
        # Channel-wise kernel
        self.channel_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // r, in_channels * k_square // 2, kernel_size=1),
        )
        
        self.out_proj = nn.Conv2d(in_channels, c_half, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x2, y2):
        B, C_half, H, W = x2.shape                                   # (C/2, H, W)
        
        x_static = self.static_kernel(x2)                
        fused = torch.cat([x_static, y2], dim=1)                     # (C, H, W)
        
        k_spatial = self.spatial_kernel(fused)                       # (K², H, W)
        k_channel = self.channel_kernel(fused)                       # (C/2 * K², 1, 1)
        
        # Reshape (with broadcast)
        k_spatial = k_spatial.unsqueeze(1)                           # (1, K², H, W)
        k_channel = k_channel.view(B, C_half, self.k_square, 1, 1)   # (C/2, K², 1, 1)
        k_channel = k_channel.expand(-1, -1, -1, H, W)               # (C/2, K², H, W)
        
        # Dynamic filter
        k_dynamic = k_spatial + k_channel
        
        x2_unfold = F.unfold(x2, kernel_size=3, padding=1)                  # (C/2 * 9, H * W)
        x2_unfold = x2_unfold.view(B, C_half, 9, H, W)                      # (C/2, 9, H, W)
        x_dynamic = torch.einsum('bckhw,bckhw->bchw', x2_unfold, k_dynamic) # (C/2, H, W)
        
        out = torch.cat([x_static, x_dynamic], dim=1)
        out = self.out_proj(out)
        
        return out, y2