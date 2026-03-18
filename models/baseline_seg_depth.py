"""
Baseline joint depth + semantic segmentation UNet model.

Single UNet backbone with two output heads: depth and semantic.
"""
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from . import ModelOutput, register
from .base import DepthModelBase
from ._blocks import ConvBlock

NUM_CLASSES = 23


# ---------------------------------------------------------------------------
# Private nn.Module implementation
# ---------------------------------------------------------------------------

class _BaselineSegDepthUNet(nn.Module):
    """
    UNet with shared encoder+decoder and two output heads:
      depth_head    : (N, 1,          H, W)
      semantic_head : (N, NUM_CLASSES, H, W)
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_classes: int = NUM_CLASSES):
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

        self.depth_head    = nn.Conv2d(b, 1,           kernel_size=1)
        self.semantic_head = nn.Conv2d(b, num_classes, kernel_size=1)

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

        return self.depth_head(d1), self.semantic_head(d1)

    def forward(self, x):
        """x: (B, S, C, 3, H, W) -> depth (B,S,C,1,H,W), sem (B,S,C,NUM_CLS,H,W)"""
        b, s, c = x.shape[:3]
        x_flat = rearrange(x, 'b s c ch h w -> (b s c) ch h w')
        depth, sem = self.forward_single(x_flat)
        depth = rearrange(depth, '(b s c) 1 h w -> b s c 1 h w',    b=b, s=s, c=c)
        sem   = rearrange(sem,   '(b s c) cls h w -> b s c cls h w', b=b, s=s, c=c)
        return depth, sem


# ---------------------------------------------------------------------------
# Registered wrapper
# ---------------------------------------------------------------------------

@register("baseline_seg_depth")
class BaselineSegDepth(DepthModelBase):
    produces_semantic = True
    is_stateful       = False
    state_keys        = ()

    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.net = _BaselineSegDepthUNet(base_channels=base_channels)

    def forward(self, x, state=None) -> ModelOutput:
        depth, sem = self.net(x)
        return ModelOutput(depth=depth, semantic=sem)

    @classmethod
    def add_model_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--base-channels", type=int, default=64)

    @classmethod
    def from_args(cls, args: Namespace):
        return cls(base_channels=args.base_channels)
