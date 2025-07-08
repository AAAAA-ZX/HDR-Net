import torch
import torch.nn as nn
from models.blocks import ResidualDenseBlock, RRDB, MSRB
from torch.nn import functional as F

class GateModule(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv = nn.Conv2d(channels*2, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, msrb_out, rrdb_out):
        fused = torch.cat([msrb_out, rrdb_out], dim=1)  # [B, 128, H, W]
        weights = self.sigmoid(self.conv(fused))  # [B, 2, H, W]
        w_ms = weights[:, 0:1].expand(-1, 64, -1, -1)  # 显式指定64通道
        w_rr = weights[:, 1:2].expand(-1, 64, -1, -1)
        return w_ms * msrb_out + w_rr * rrdb_out

class DegradationEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, 64, 3, 1, 1)
        self.msrb_blocks = nn.Sequential(*[MSRB() for _ in range(3)])
        self.rrdb_blocks = nn.Sequential(*[RRDB(64) for _ in range(3)])
        self.gate = GateModule(channels=64)
        self.fc = nn.Sequential(
            nn.Linear(64, 256),  # 修正输入维度
            nn.ReLU(),
            nn.Linear(256, 64),  # 输出64维
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_conv(x)
        msrb_out = self.msrb_blocks(x)
        rrdb_out = self.rrdb_blocks(x)
        fused = self.gate(msrb_out, rrdb_out)
        pooled = F.adaptive_avg_pool2d(fused, 1).view(x.size(0), -1)  # [B, 64]
        return self.fc(pooled)  # 输出[B, 64]