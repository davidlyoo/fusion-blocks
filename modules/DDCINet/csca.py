import torch
import torch.nn as nn
import torch.nn.functional as F


class MSSR(nn.Module):
    def __init__(self, channels, spatial_reduction): # r: spatial reduction factor - paper X
        super().__init__()
        reduced_channels = channels // 4
        r = spatial_reduction
        
        # 4-branch depth-wise separable convolution blocks
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1,
                      padding=0, dilation=1, groups=channels),
            nn.Conv2d(channels, reduced_channels, kernel_size=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=1, dilation=1, groups=channels),
            nn.Conv2d(channels, reduced_channels, kernel_size=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=2, dilation=2, groups=channels),
            nn.Conv2d(channels, reduced_channels, kernel_size=1)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=3, dilation=3, groups=channels),
            nn.Conv2d(channels, reduced_channels, kernel_size=1)
        )        
        
        self.merge = nn.Conv2d(reduced_channels * 4, channels, kernel_size=1)
        
        self.dw_conv = nn.Conv2d(channels, channels, dilation=1,
                                 kernel_size=r,stride=r, groups=channels)
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
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
    def __init__(self, channels, num_heads=4, spatial_reduction=4, channel_reduction=4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.mssr = MSSR(channels, spatial_reduction=spatial_reduction)

        # Q, K, V projection
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.kv_proj = nn.Conv2d(channels, channels * 2, kernel_size=1)

        # Attention output projection
        self.attn_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # Channel-gated FFN
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_fc1 = nn.Conv2d(channels * 2, channels // channel_reduction, kernel_size=1)
        self.gate_fc2 = nn.Conv2d(channels // channel_reduction, channels, kernel_size=1)

        # Output projection
        self.linear1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.linear2 = nn.Conv2d(channels * 2, channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def _multihead_attn(self, Q, K, V, B, C, H, W):
        HW_q = H * W
        H_k, W_k = K.shape[2], K.shape[3]
        HW_k = H_k * W_k

        # print(f"[DEBUG] Q shape: {Q.shape}")
        # print(f"[DEBUG] K shape: {K.shape}, V shape: {V.shape}")
        # print(f"[DEBUG] HW_q = {HW_q}, HW_k = {HW_k}")
        # print(f"[DEBUG] K.numel() = {K.numel()} (Expected = {B*self.num_heads*self.head_dim*HW_k})")

        Q = Q.view(B, self.num_heads, self.head_dim, HW_q).permute(0, 1, 3, 2)
        K = K.view(B, self.num_heads, self.head_dim, HW_k).permute(0, 1, 3, 2)
        V = V.view(B, self.num_heads, self.head_dim, HW_k).permute(0, 1, 3, 2)

        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ V

        out = out.permute(0, 2, 1, 3).reshape(B, HW_q, C).permute(0, 2, 1).view(B, C, H, W)
        return out


    def forward(self, x1, y1):
        B, C, H, W = x1.shape

        # Forward 1: Q from y1, KV from x1
        Q1 = self.q_proj(y1)
        KV1 = self.kv_proj(self.mssr(x1))
        K1, V1 = KV1.chunk(2, dim=1)
        x1_sca = self.attn_proj(self._multihead_attn(Q1, K1, V1, B, C, H, W)) + x1

        # Forward 2: Q from x1, KV from y1
        Q2 = self.q_proj(x1)
        KV2 = self.kv_proj(self.mssr(y1))
        K2, V2 = KV2.chunk(2, dim=1)
        y1_sca = self.attn_proj(self._multihead_attn(Q2, K2, V2, B, C, H, W)) + y1
        
        # Channel-Gated FFN
        z_x = torch.cat([x1_sca, y1], dim=1)
        z_y = torch.cat([y1_sca, x1], dim=1)

        g_x = self.sigmoid(self.gate_fc2(self.relu(self.gate_fc1(self.avg_pool(z_x)))))
        g_y = self.sigmoid(self.gate_fc2(self.relu(self.gate_fc1(self.avg_pool(z_y)))))

        x_out = self.linear2(self.relu(self.linear1(x1_sca))) * g_x
        y_out = self.linear2(self.relu(self.linear1(y1_sca))) * g_y

        return x_out, y_out
    
        # # Q projection from y1
        # Q = self.q_proj(y1)                              # (B, C/2, H, W)
        # Q = Q.view(B, C_half, -1).permute(0, 2, 1)       # (B, HW, C/2)
        
        # # K, V from x1 via MSSR
        # kv = self.kv_proj(self.mssr(x1))                 # (B, C, H, W)
        # kv = kv.view(B, C_half * 2, -1).permute(0, 2, 1) # (B, HW, C)
        # K, V = kv.chunk(2, dim=2)                        # (B, HW, C/2) each
        
        # # Attention
        # attn = torch.matmul(Q, K.transpose(-2, -1)) / (C_half ** 0.5) # (B, HW, HW)
        # attn = torch.softmax(attn, dim=-1)
        # attn_out = torch.matmul(attn, V)
        # attn_out = attn_out.permute(0, 2, 1).view(B, C_half, H, W)    # (B, C/2, H, W)
        # x_sca = self.attn_proj(attn_out)
        
        # # Channel-gated FFN
        # y_sca = y1
        # z = torch.cat([x_sca, y_sca], dim=1)             # (B, C, H, W)
        # gap = self.avg_pool(z)
        # g = self.gate_fc2(self.relu(self.gate_fc1(gap))) # (B, C/2, 1, 1)
        # g = self.sigmoid(g)
        
        # # Final projection
        # x_out = self.linear2(self.relu(self.linear1(x_sca))) * g
        # y_out = y_sca
        
        # return x_out, y_out