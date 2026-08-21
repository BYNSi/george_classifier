from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, drop: float = 0.0):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = x + residual
        return self.act(x)


class SaintGeorgeNet(nn.Module):
    def __init__(self, num_classes: int = 1, width: int = 16, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, width, 3, 1),
            ConvBNAct(width, width, 3, 1),
        )
        self.stage1 = nn.Sequential(ResidualBlock(width, width, 1, dropout), ResidualBlock(width, width, 1, dropout))
        self.stage2 = nn.Sequential(
            ResidualBlock(width, width * 2, 2, dropout),
            ResidualBlock(width * 2, width * 2, 1, dropout),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(width * 2, width * 4, 2, dropout),
            ResidualBlock(width * 4, width * 4, 1, dropout),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(width * 4, width * 8, 2, dropout),
            ResidualBlock(width * 8, width * 8, 1, dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 8, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)


def build_model(width: int = 16) -> nn.Module:
    return SaintGeorgeNet(width=width)
