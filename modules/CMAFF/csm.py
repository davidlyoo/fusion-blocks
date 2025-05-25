import torch
import torch.nn as nn


class CSM(nn.Module):
    def __init__(self, channels, reduction = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        
        self.fc_o = nn.Linear(channels // reduction, channels * 2, bias=False)
        self.fc_s = nn.Linear(channels // reduction, channels * 2, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()
        
    def forward(self, f_o, f_s):
        # Element-wise Summation
        f_cm = f_o + f_s
        
        # Global Average Pooling
        f_gap = self.avg_pool(f_cm).view(f_cm.size(0), -1)
        
        # fc -> relu -> fc
        shared = self.relu(self.fc1(f_gap))
        
        # Softmax
        z = self.softmax(s.view(s.size(0), 2, -1))
        z1 = z[:, 0, :].unsqueeze(2).unsqueeze(3)
        z2 = z[:, 1, :].unsqueeze(2).unsqueeze(3)
        
        f_o_weighted = f_o * z1
        f_s_weighted = f_s * z2
        
        output = f_o_weighted + f_s_weighted
        return output