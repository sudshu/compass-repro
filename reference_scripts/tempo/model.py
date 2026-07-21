"""Small U-Net for Stage-2B CONUS training.

Mirrors the GEOS-CF model: 4-level encoder/decoder, GroupNorm + GELU, base=48
channels. Input: 2 channels (TEMPO NO2, GOES WV radiance). Output: 3 channels
(U10, V10, PBLH).
"""
from __future__ import annotations
import torch
import torch.nn as nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
    )


class UNet2to3(nn.Module):
    def __init__(self, c_in: int = 2, c_out: int = 3, base: int = 48, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        chs = [c_in] + [base * (2 ** i) for i in range(depth)]
        for i in range(depth):
            self.enc.append(conv_block(chs[i], chs[i + 1]))
            self.pool.append(nn.MaxPool2d(2))
        self.bottom = conv_block(chs[-1], chs[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        chs_dec = [chs[-1] * 2] + [base * (2 ** i) for i in range(depth - 1, -1, -1)]
        for i in range(depth):
            self.up.append(nn.ConvTranspose2d(chs_dec[i], chs_dec[i + 1], 2, stride=2))
            self.dec.append(conv_block(chs_dec[i + 1] * 2, chs_dec[i + 1]))
        self.out_conv = nn.Conv2d(base, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.enc, self.pool):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottom(x)
        for up, dec, sk in zip(self.up, self.dec, skips[::-1]):
            x = up(x)
            x = torch.cat([x, sk], dim=1)
            x = dec(x)
        return self.out_conv(x)
