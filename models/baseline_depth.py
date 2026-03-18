"""
Baseline depth-only UNet model.

Supports an optional pretrained ResNet encoder (resnet18/34/50) via --backbone.
Default backbone is resnet18. Use --backbone none for the original scratch ConvNet.
"""
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from . import ModelOutput, register
from .base import DepthModelBase
from ._blocks import ConvBlock, _RESNET_CHANNELS, _RESNET_WEIGHTS, _RESNET_FN


# ---------------------------------------------------------------------------
# Private nn.Module implementations
# ---------------------------------------------------------------------------

class _DepthUNet(nn.Module):
    """Scratch ConvNet UNet for depth estimation: RGB -> per-pixel depth."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        b = base_channels

        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b,     b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(b * 8, b * 16)

        self.up4  = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(b * 16, b * 8)
        self.up3  = nn.ConvTranspose2d(b * 8,  b * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(b * 8,  b * 4)
        self.up2  = nn.ConvTranspose2d(b * 4,  b * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(b * 4,  b * 2)
        self.up1  = nn.ConvTranspose2d(b * 2,  b,     kernel_size=2, stride=2)
        self.dec1 = ConvBlock(b * 2,  b)

        self.depth_head = nn.Conv2d(b, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _up_cat(self, up, x, skip):
        x = up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward_single(self, x):
        e1  = self.enc1(x)
        e2  = self.enc2(self.pool(e1))
        e3  = self.enc3(self.pool(e2))
        e4  = self.enc4(self.pool(e3))
        bot = self.bottleneck(self.pool(e4))

        d4 = self.dec4(self._up_cat(self.up4, bot, e4))
        d3 = self.dec3(self._up_cat(self.up3, d4,  e3))
        d2 = self.dec2(self._up_cat(self.up2, d3,  e2))
        d1 = self.dec1(self._up_cat(self.up1, d2,  e1))

        return self.depth_head(d1)

    def forward(self, x):
        """x: (B, S, C, 3, H, W) -> depth (B, S, C, 1, H, W)"""
        b, s, c = x.shape[:3]
        x_flat = rearrange(x, 'b s c ch h w -> (b s c) ch h w')
        depth  = self.forward_single(x_flat)
        return rearrange(depth, '(b s c) 1 h w -> b s c 1 h w', b=b, s=s, c=c)


class _ResNetDepthUNet(nn.Module):
    """Pretrained ResNet encoder + lightweight decoder for depth estimation."""

    _DEC = (256, 128, 64, 32)

    def __init__(self, backbone: str = "resnet18"):
        super().__init__()

        encoder = _RESNET_FN[backbone](weights=_RESNET_WEIGHTS[backbone])
        stem_ch, l1_ch, l2_ch, l3_ch, l4_ch = _RESNET_CHANNELS[backbone]
        d = self._DEC

        self.stem   = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool   = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.up4  = nn.ConvTranspose2d(l4_ch, d[0], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(d[0] + l3_ch, d[0])
        self.up3  = nn.ConvTranspose2d(d[0],  d[1], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(d[1] + l2_ch, d[1])
        self.up2  = nn.ConvTranspose2d(d[1],  d[2], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(d[2] + l1_ch, d[2])
        self.up1  = nn.ConvTranspose2d(d[2],  d[3], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(d[3] + stem_ch, d[3])

        self.depth_head = nn.Conv2d(d[3], 1, kernel_size=1)

        self._init_decoder()

    def _init_decoder(self):
        decoder = [self.up4, self.dec4, self.up3, self.dec3,
                   self.up2, self.dec2, self.up1, self.dec1, self.depth_head]
        for module in decoder:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _up_cat(self, up, x, skip):
        x = up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward_single(self, x):
        s  = self.stem(x)
        e1 = self.layer1(self.pool(s))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        d4 = self.dec4(self._up_cat(self.up4, e4, e3))
        d3 = self.dec3(self._up_cat(self.up3, d4, e2))
        d2 = self.dec2(self._up_cat(self.up2, d3, e1))
        d1 = self.dec1(self._up_cat(self.up1, d2, s))

        return F.interpolate(self.depth_head(d1), scale_factor=2,
                             mode="bilinear", align_corners=False)

    def forward(self, x):
        """x: (B, S, C, 3, H, W) -> depth (B, S, C, 1, H, W)"""
        b, s, c = x.shape[:3]
        x_flat = rearrange(x, 'b s c ch h w -> (b s c) ch h w')
        depth  = self.forward_single(x_flat)
        return rearrange(depth, '(b s c) 1 h w -> b s c 1 h w', b=b, s=s, c=c)


# ---------------------------------------------------------------------------
# Registered wrapper
# ---------------------------------------------------------------------------

@register("baseline_depth")
class BaselineDepth(DepthModelBase):
    produces_semantic = False
    is_stateful       = False
    state_keys        = ()

    def __init__(self, backbone: str = "resnet18", base_channels: int = 64):
        super().__init__()
        if backbone == "none":
            self.net = _DepthUNet(base_channels=base_channels)
        else:
            self.net = _ResNetDepthUNet(backbone=backbone)

    def forward(self, x, state=None) -> ModelOutput:
        return ModelOutput(depth=self.net(x))

    @classmethod
    def add_model_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--backbone", type=str, default="resnet18",
                            choices=["none", "resnet18", "resnet34", "resnet50"],
                            help="Encoder backbone. 'none' = scratch ConvNet UNet")
        parser.add_argument("--base-channels", type=int, default=64,
                            help="Base channels for scratch ConvNet (ignored when backbone != none)")

    @classmethod
    def from_args(cls, args: Namespace):
        return cls(backbone=args.backbone, base_channels=args.base_channels)
