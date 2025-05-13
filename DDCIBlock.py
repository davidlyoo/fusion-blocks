import torch
import torch.nn as nn
import torch.nn.functional as F


class MSSR(nn.Module):
    def __init__(self, in_channels, r=4): # r: spatial reduction factor - paper X
        super().__init__()
        reduced_channels = in_channels // 8
        
        # 4-branch depth-wise separable convolution blocks
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1,
                      padding=0, dilation=1, groups=in_channels).
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=1, dilation=1, groups=in_channels).
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=2, dilation=2, groups=in_channels).
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=3, dilation=3, groups=in_channels).
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        )        
        
        self.merge = nn.Conv2d(reduced_channels * 4, in_channels, kernel_size=1)
        
        self.dw_conv = nn.Conv2d(in_channels, in_channels, dilation=1,
                                 kernel_size=r,stride=r, groups=in_channels)
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        multiscale = self.merge(torch.cat([b1, b2, b3, b4], dim=1)) # (B, C, H, W)
        
        residual = x + multiscale
        
        reduced = self.dw_conv(residual)
        out = self.final_conv(reduced)
        return out


class CSCA(nn.Module):
    def __init__(self, in_channels, num_heads, spatial_reduction, channel_reduction):
        super().__init__()
        assert (in_channels // 2) % num_heads == 0, "in_channels must be divisible by 2 * num_heads"
        
        self.num_heads = num_heads
        self.head_dim = (in_channels // 2) // self.num_heads
        self.mssr = MSSR(in_channels, stride=spatial_reduction)
        
        # Q, K, V projection
        self.q_proj = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1)
        self.kv_proj = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        
        # Attention output projection
        self.attn_proj = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1)
        
        # Channel-gated FFN
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_fc1 = nn.Conv2d(in_channels, in_channels // channel_reduction, kernel_size=1)
        self.gate_fc2 = nn.Conv2d(in_channels // channel_reduction, in_channels // 2, kernel_size=1)
        
        # Output projection
        self.linear1 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.linear2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def _multihead_attn(self, Q, K, V, B, C_half, H, W):
        HW = H * W
        
        Q = Q.view(B, self.num_heads, self.head_dim, HW).permute(0, 1, 3, 2)  # [B, num_heads, HW, d]
        K = K.view(B, self.num_heads, self.head_dim, HW).permute(0, 1, 3, 2)
        V = V.view(B, self.num_heads, self.head_dim, HW).permute(0, 1, 3, 2)
            
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # [B, num_heads, HW, d]
        
        out = out.permute(0, 2, 1, 3).reshape(B, HW, C_half).permute(0, 2, 1).view(B, C_half, H, W)
        return out
    
    def forward(self, x1, y1):
        B, C_half, H, W = x1.shape
        C = C_half * 2
        
        # Forward 1: Q=y1, KV=x1
        Q1 = self.q_proj(y1).view(B, C_half, -1)
        KV1 = self.kv_proj(self.mssr(x1)).view(B, C, -1)
        K1, V1 = KV1.chunk(2, dim=1)
        x1_sca = self.attn_proj(self._multihead_attn(Q1, K1, V1, B, C_half, H, W))
        x1_sca = x1_sca + x1
        
        # Forward 2: Q=x1, KV=y1
        Q2 = self.q_proj(x1).view(B, C_half, -1)
        KV2 = self.kv_proj(self.mssr(y1)).view(B, C, -1)
        K2, V2 = KV2.chunk(2, dim=1)
        y1_sca = self.attn_proj(self._multihead_attn(Q2, K2, V2, B, C_half, H, W))
        y1_sca = y1_sca + y1
        
        # Channel-Gated FFN
        z_x = torch.cat([x1_sca, y1], dim=1)
        z_y = torch.cat([y1_sca, x1], dim=1)

        g_x = self.sigmoid(self.gate_fc2(self.relu(self.gate_fc1(self.avg_pool(z_x)))))
        g_y = self.sigmoid(self.gate_fc2(self.relu(self.gate_fc1(self.avg_pool(z_y)))))

        x_out = self.linear2(self.relu(self.linear1(x1_sca))) * g_x
        y_out = self.linear2(self.relu(self.linear1(y1_sca))) * g_y

        return x_out, y_out


class CMDF(nn.Module):
    def __init__(self, in_channels, kernel_size=3, channel_reduction):
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
            nn.Conv2d(in_channels, in_channels // channel_reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // channel_reduction, in_channels * k_square // 2, kernel_size=1),
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


class DDCIModule(nn.Module):
    def __init__(self, in_channels, num_heads, spatial_reduction, channel_reduction):
        super().__init__()
        self.input_proj_x = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.input_proj_y = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.csca = CSCA(
            in_channels=in_channels,
            num_heads=num_heads,
            spatial_reduction=spatial_reduction,
            channel_reduction=channel_reduction
            )
        self.cmdf = CMDF(in_channels=in_channels, channel_reduction=channel_reduction)
        
        self.output_proj_x = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_proj_y = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x, y):
        # Input projection
        x = self.input_proj_x(x)
        y = self.input_proj_y(y)
        
        # Channel-wise split
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        y1, y2 = torch.chunk(y, chunks=2, dim=1)
        
        # Feature interaction
        x1_prime, y1_prime = self.csca(x1, y1)
        x2_prime, y2_prime = self.cmdf(x2, y2)
        
        # Concatenate
        x_cat = torch.cat([x1_prime, x2_prime], dim=1)
        y_cat = torch.cat([y1_prime, y2_prime], dim=1)
        
        # Output projection
        x_out = self.output_proj_x(x_cat)
        y_out = self.output_proj_y(y_cat)
        
        return x_out, y_out


class DFFModule(nn.Module):
    def __init__(self, in_channels, channel_reduction=4):
        super().__init__()
        
        self.conv_gx = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv_gy = nn.Conv2d(in_channels, 1, kernel_size=1)
        
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // channel_reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // channel_reduction, in_channels, kernel_size=1),
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


class DDCIBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, num_heads=4, spatial_reduction=16, channel_reduction=4):
        super().__init__()
        self.ddci = DDCIModule(
            in_channels=in_channels,
            num_heads=num_heads,
            spatial_reduction=spatial_reduction,
            channel_reduction=channel_reduction
            )
        self.dff = DFFModule(in_channels=in_channels, channel_reduction=channel_reduction)
    
    def forward(self, opt_feat, sar_feat):
        fused_opt, fused_sar = self.ddci(opt_feat, sar_feat)
        fused = self.dff(fused_opt, fused_sar)
        return fused