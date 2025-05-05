import torch
import torch.nn as nn
import torch.nn.functional as F


class MSSR(nn.Module):
    def __init__(self, in_channels, r=4): # r: spatial reduction factor - paper X
        super(MSSR, self).__init__()
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
    def __init__(self, in_channels, r=4):
        super(CSCA, self).__init__()
        self.mssr = MSSR(in_channels, stride=r)
        
        # Q, K, V projection
        self.q_proj = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1)
        self.kv_proj = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        
        # Attention output projection
        self.attn_proj = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1)
        
        # Channel-gated FFN
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_fc1 = nn.Conv2d(in_channels, in_channels // r, kernel_size=1)
        self.gate_fc2 = nn.Conv2d(in_channels // r, in_channels // 2, kernel_size=1)
        
        # Output projection
        self.linear1 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.linear2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x1, y1):
        B, C_half, H, W = x1.shape
        
        # Forward 1: Q=y1, KV=x1
        Q1 = self.q_proj(y1).view(B, C_half, -1).permute(0, 2, 1)
        kv1 = self.kv_proj(self.mssr(x1)).view(B, C_half * 2, -1).permute(0, 2, 1)
        K1, V1 = kv1.chunk(2, dim=2)
    
        attn1 = torch.softmax(torch.matmul(Q1, K1.transpose(-2, -1)) / (C_half ** 0.5), dim=-1)
        attn_out1 = torch.matmul(attn1, V1).permute(0, 2, 1).view(B, C_half, H, W)
        x1_sca = self.attn_proj(attn_out1)
        
        # Forward 2: Q=x1, KV=y1
        Q2 = self.q_proj(x1).view(B, C_half, -1).permute(0, 2, 1)
        kv2 = self.kv_proj(self.mssr(y1)).view(B, C_half * 2, -1).permute(0, 2, 1)
        K2, V2 = kv2.chunk(2, dim=2)

        attn2 = torch.softmax(torch.matmul(Q2, K2.transpose(-2, -1)) / (C_half ** 0.5), dim=-1)
        attn_out2 = torch.matmul(attn2, V2).permute(0, 2, 1).view(B, C_half, H, W)
        y1_sca = self.attn_proj(attn_out2)
        
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