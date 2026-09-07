"""
AgroGrow Corn-Net Architecture Module — Model 4.
Upgraded CornNet with MSCAB blocks at both bottleneck and decoder levels 3 & 4
for finer kernel boundary segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CBAM-style).
    Learns channel importance using Global Avg and Max pooling.
    """
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction_ratio, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.mlp(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (CBAM-style).
    Highlights regions of interest across space.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class MSCAB(nn.Module):
    """
    Multi-Scale Context Attention Block (MSCAB).
    Parallel dilated convolutions + CBAM attention + residual connection.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        branch_ch = max(in_channels // 4, 8)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True)
        )
        self.reconstruct = nn.Sequential(
            nn.Conv2d(branch_ch * 4, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)
        )
        self.channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attn = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi = torch.cat([self.branch1(x), self.branch2(x),
                           self.branch3(x), self.branch4(x)], dim=1)
        out = self.spatial_attn(self.channel_attn(self.reconstruct(multi)))
        return out + x


class DoubleConv(nn.Module):
    """(Conv2D → BN → ReLU) × 2 with optional dropout."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CornNet(nn.Module):
    """
    Corn-Net Model 4.
    U-Net encoder-decoder with MSCAB at bottleneck AND decoder levels 4 & 3
    for improved fine-grained kernel boundary segmentation.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # ── Bottleneck + MSCAB ────────────────────────────────
        self.bottleneck = DoubleConv(512, 512, dropout=0.2)
        self.mscab_bot  = MSCAB(512)

        # ── Decoder ───────────────────────────────────────────
        self.up4   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4  = DoubleConv(256 + 512, 256)   # up4 (256) + skip enc4 (512)
        self.mscab4 = MSCAB(256)                  # extra attention at level 4

        self.up3   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3  = DoubleConv(128 + 256, 128)   # up3 (128) + skip enc3 (256)
        self.mscab3 = MSCAB(128)                  # extra attention at level 3

        self.up2   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2  = DoubleConv(64 + 128, 64)     # up2 (64)  + skip enc2 (128)

        self.up1   = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1  = DoubleConv(64 + 64, 64)      # up1 (64)  + skip enc1 (64)

        # ── Output ───────────────────────────────────────────
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x);  p1 = self.pool1(x1)
        x2 = self.enc2(p1); p2 = self.pool2(x2)
        x3 = self.enc3(p2); p3 = self.pool3(x3)
        x4 = self.enc4(p3); p4 = self.pool4(x4)

        # Bottleneck
        b = self.mscab_bot(self.bottleneck(p4))

        # Decoder
        u4 = self.up4(b)
        d4 = self.mscab4(self.dec4(torch.cat([u4, x4], dim=1)))

        u3 = self.up3(d4)
        d3 = self.mscab3(self.dec3(torch.cat([u3, x3], dim=1)))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, x1], dim=1))

        return self.out_conv(d1)


if __name__ == "__main__":
    model = CornNet(num_classes=4)
    dummy = torch.randn(2, 3, 512, 512)
    out = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")
