import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .blocks import MSRB, DRRDB

class Generator(nn.Module):
    def __init__(self, nf=32, nb=16, degradation_dim=64):  # 统一为64
        super().__init__()
        self.degradation_proj = nn.Sequential(
            nn.Linear(degradation_dim, 256),  # 输入64 → 输出256
            nn.ReLU(),
            nn.Linear(256, nf)  # 输出nf（32）维
        )
        self.drrdb_blocks = nn.ModuleList([
            MSRB(nf),
            *[DRRDB(nf, degradation_dim=degradation_dim) for _ in range(nb-1)]
        ])
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, nf*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(nf, 3, 3, 1, 1)
        )

    def forward(self, x, degradation_features):
        x = self.initial_conv(x)
        d = self.degradation_proj(degradation_features)  # [B, 32]
        for block in self.drrdb_blocks:
            if isinstance(block, MSRB):
                x = block(x)
            else:
                x = checkpoint(block, x, d)  # 传递d参数
        return self.upsample(x)