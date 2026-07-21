"""Model architectures, verbatim from the analysis repository.

- ``UNetWindPBLH`` : GEOS-CF / CAMS global model
  (paper configuration: in_channels=10, out_channels=5, base_channels=48,
  depth=4, dropout_rate=0.05 -> ~17.46 M parameters).
  Source: ml_wind_pblh/ml_wind_pblh/model.py

- ``UNet2to3`` : TEMPO -> HRRR model
  (paper configuration: c_in=12, c_out=3, base=64, depth=3
  -> 7,708,291 parameters).
  Source: stage2b_train/model.py

Both are pure 2D CNNs (MPS/CUDA/CPU portable). Checkpoints are saved as
``{"epoch": int, "model": state_dict}``; use :func:`load_checkpoint`.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# GEOS-CF / CAMS model                                                         #
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetWindPBLH(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        enc_channels = [base_channels * (2**i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_c = in_channels
        for ch in enc_channels:
            self.encoders.append(ConvBlock(prev_c, ch, dropout_rate=dropout_rate))
            self.pools.append(nn.MaxPool2d(2))
            prev_c = ch

        bottleneck_c = enc_channels[-1] * 2
        self.bottleneck = ConvBlock(enc_channels[-1], bottleneck_c, dropout_rate=dropout_rate)

        dec_in_channels = [bottleneck_c] + [enc_channels[-i] for i in range(1, depth)]
        dec_out_channels = [enc_channels[-i] for i in range(1, depth + 1)]

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for up_in, up_out in zip(dec_in_channels, dec_out_channels):
            self.upconvs.append(nn.ConvTranspose2d(up_in, up_out, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(up_out * 2, up_out, dropout_rate=dropout_rate))

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        out = x
        for enc, pool in zip(self.encoders, self.pools):
            out = enc(out)
            skips.append(out)
            out = pool(out)

        out = self.bottleneck(out)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            out = upconv(out)
            if out.shape[-2:] != skip.shape[-2:]:
                out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = torch.cat([skip, out], dim=1)
            out = decoder(out)

        return self.final_conv(out)


# --------------------------------------------------------------------------- #
# TEMPO -> HRRR model                                                          #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Checkpoint loading                                                           #
# --------------------------------------------------------------------------- #
def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> nn.Module:
    """Load a ``{"epoch": ..., "model": state_dict}`` (or bare state_dict) checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
