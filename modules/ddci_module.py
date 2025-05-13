import torch
import torch.nn as nn


class CSCA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dummy = nn.Identity()
    
    def forward(self, x1, x2):
        return self.dummy(x1), self.dummy(x2)


class CMDF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dummy = nn.Identity()

    def forward(self, x1, x2):
        return self.dummy(x1), self.dummy(x2)


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