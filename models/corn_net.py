"""
AgroGrow Corn-Net Architecture Module.
Implements the custom Corn-Net semantic segmentation network from scratch in PyTorch,
featuring U-Net style skip connections and a bottleneck Multi-Scale Context Attention Block (MSCAB).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CBAM-style).
    Learns spatial importance across channels using Global Avg and Max pooling.
    """
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.mlp(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (CBAM-style).
    Highlights regions of interest across space using 7x7 convolution on pooled channel dimensions.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention

class MSCAB(nn.Module):
    """
    Multi-Scale Context Attention Block (MSCAB).
    Captures multi-scale contextual features using parallel dilated convolutions
    and adaptively refines them with channel and spatial attention.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.out_channels = in_channels
        branch_channels = in_channels // 4
        
        # Branch 1: 1x1 Convolution (local details)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 3x3 Convolution with dilation=1
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 3x3 Convolution with dilation=2
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3 Convolution with dilation=4
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Reconstruction layer to project aggregated features
        self.reconstruct = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention modules
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Concatenate scale branches
        multiscale = torch.cat([b1, b2, b3, b4], dim=1)
        projected = self.reconstruct(multiscale)
        
        # Sequence of Attention
        refined = self.channel_attention(projected)
        refined = self.spatial_attention(refined)
        
        # Residual skip connection
        return refined + x

class DoubleConv(nn.Module):
    """
    Standard building block: (Conv2D -> BatchNorm2D -> ReLU) * 2
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CornNet(nn.Module):
    """
    Corn-Net semantic segmentation architecture.
    Combines an encoder-decoder backbone with MSCAB bottleneck attention.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()
        
        # Encoder (Downsampling)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck bridge with Multi-Scale Context Attention Block (MSCAB)
        self.bottleneck = DoubleConv(512, 512)
        self.mscab = MSCAB(512)
        
        # Decoder (Upsampling & Skip Connections)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(768, 256)  # 256 (from up4) + 512 (skip from enc4)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(384, 128)  # 128 (from up3) + 256 (skip from enc3)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(192, 64)   # 64 (from up2) + 128 (skip from enc2)
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)   # 64 (from up1) + 64 (skip from enc1)
        
        # Final Output classification layer
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)
        p1 = self.pool1(x1)
        
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        
        x3 = self.enc3(p2)
        p3 = self.pool3(x3)
        
        x4 = self.enc4(p3)
        p4 = self.pool4(x4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        b = self.mscab(b)
        
        # Decoder with skip connections
        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, x4], dim=1))
        
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, x3], dim=1))
        
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))
        
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, x1], dim=1))
        
        # Output Logits
        logits = self.out_conv(d1)
        return logits

if __name__ == "__main__":
    # Standard check
    model = CornNet(num_classes=4)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
