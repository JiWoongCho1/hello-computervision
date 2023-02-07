import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_dim, reduction_ratio = 16):
        super().__init__()

        self.avg_pool = nn.AdatptiveAvgPool2d((1,1))
        self.max_pool = nn.AdatptiveMaxPool2d((1,1))

        self.fc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // reduction_ratio, bias = False),
            nn.ReLU(),
            nn.Conv2d(in_dim // reduction_ratio, in_dim, bias = False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding = kernel_size //2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1)
        max_out, _ = torch.max(x, dim = 1)
        x = torch.cat([avg_out, max_out], dim = 1)
        x = self.conv(x)
        return self.sigmoid(x)

