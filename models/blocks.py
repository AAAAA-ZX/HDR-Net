import torch
import torch.nn as nn
from torch.nn import functional as F

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5 * self.scale

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return x + out * self.scale
class MSRB(nn.Module):
    """
    多尺度残差块（Multi-Scale Residual Block）
    提取模糊核、噪声分布等多尺度退化特征
    """
    def __init__(self, nf=64):
        super().__init__()
        self.conv3 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.scale = nn.Parameter(torch.zeros(1))  # 残差缩放因子

    def forward(self, x):
        x3 = self.lrelu(self.conv3(x))
        x5 = self.lrelu(self.conv5(x))
        return x + (x3 + x5) * self.scale  # 多尺度特征融合


class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, degradation_dim=64, kernel_size=3, padding=1):
        super().__init__()
        self.weight_generator = nn.Sequential(
            nn.Linear(degradation_dim, 256),  # 输入维度=64
            nn.ReLU(),
            nn.Linear(256, in_channels * out_channels * kernel_size * kernel_size)
        )

    def forward(self, x, degradation_features):
        B = x.size(0)
        weights = self.weight_generator(degradation_features).view(
            B, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        output = []
        for i in range(B):
            output.append(F.conv2d(x[i:i+1], weights[i]))
        return torch.cat(output, dim=0)

class DRRDB(nn.Module):
    def __init__(self, nf, gc=32, degradation_dim=64):  # 确保维度为64
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = DynamicConv2d(gc, gc, degradation_dim=degradation_dim, kernel_size=3)
        self.conv3 = nn.Conv2d(gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, degradation_features):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.conv2(x1, degradation_features)  # 传递64维参数
        x3 = self.lrelu(self.conv3(x2))
        return x + x3 * self.scale