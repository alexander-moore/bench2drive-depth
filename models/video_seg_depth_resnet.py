"""
ResNet + LSTM video joint depth + semantic segmentation model.

Architecture:
  Encoder : ResNet-18/34/50 (pretrained ImageNet, fine-tunable)
  Temporal: LSTM at the layer4 bottleneck
  Decoder : UNet with skip connections from all 5 ResNet stages
  Heads   : depth_head (1ch) + semantic_head (23ch)

Input  : (B, S, C, 3, H, W)
Outputs: depth (B, S, C, 1, H, W), semantic (B, S, C, 23, H, W)
"""
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from . import ModelOutput, register
from .base import DepthModelBase
from ._blocks import ConvBlock, _RESNET_CHANNELS, _RESNET_WEIGHTS, _RESNET_FN

NUM_CLASSES = 23


# ---------------------------------------------------------------------------
# Private nn.Module implementation
# ---------------------------------------------------------------------------

class _ResNetVideoSegDepthUNet(nn.Module):
    """ResNet encoder (pretrained) + LSTM bottleneck + UNet decoder."""

    _DEC = (256, 128, 64, 32)

    def __init__(self, backbone: str = "resnet18", num_classes: int = NUM_CLASSES,
                 lstm_hidden: int = 512):
        super().__init__()

        stem_ch, l1_ch, l2_ch, l3_ch, l4_ch = _RESNET_CHANNELS[backbone]
        self.l4_ch = l4_ch
        d = self._DEC

        encoder = _RESNET_FN[backbone](weights=_RESNET_WEIGHTS[backbone])
        self.stem   = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool   = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.temporal_lstm = nn.LSTM(l4_ch, lstm_hidden, batch_first=True)

        self.up4  = nn.ConvTranspose2d(lstm_hidden, d[0], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(d[0] + l3_ch, d[0])
        self.up3  = nn.ConvTranspose2d(d[0], d[1], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(d[1] + l2_ch, d[1])
        self.up2  = nn.ConvTranspose2d(d[1], d[2], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(d[2] + l1_ch, d[2])
        self.up1  = nn.ConvTranspose2d(d[2], d[3], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(d[3] + stem_ch, d[3])

        self.depth_head    = nn.Conv2d(d[3], 1,           kernel_size=1)
        self.semantic_head = nn.Conv2d(d[3], num_classes, kernel_size=1)

        self._init_decoder()

    def _init_decoder(self):
        decoder_modules = [self.up4, self.dec4, self.up3, self.dec3,
                           self.up2, self.dec2, self.up1, self.dec1,
                           self.depth_head, self.semantic_head]
        for module in decoder_modules:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _encode(self, x):
        s  = self.stem(x)
        e1 = self.layer1(self.pool(s))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return s, e1, e2, e3, e4

    def _apply_lstm(self, bot, B, S, C):
        l4_ch, h, w = bot.shape[1], bot.shape[2], bot.shape[3]
        x = bot.view(B, S, C, l4_ch, h, w)
        x = x.permute(0, 2, 3, 4, 1, 5).contiguous()
        x = x.view(B * C * h * w, S, l4_ch)

        lstm_out, _ = self.temporal_lstm(x)

        lstm_hidden = lstm_out.shape[-1]
        x = lstm_out.view(B, C, h, w, S, lstm_hidden)
        x = x.permute(0, 4, 1, 5, 2, 3).contiguous()
        x = x.view(B * S * C, lstm_hidden, h, w)
        return x

    def _decode(self, bot_lstm, e3, e2, e1, stem):
        def up_cat(up, x, skip):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
            return torch.cat([x, skip], dim=1)

        d4 = self.dec4(up_cat(self.up4, bot_lstm, e3))
        d3 = self.dec3(up_cat(self.up3, d4,       e2))
        d2 = self.dec2(up_cat(self.up2, d3,       e1))
        d1 = self.dec1(up_cat(self.up1, d2,       stem))

        depth = F.interpolate(self.depth_head(d1),    scale_factor=2,
                              mode="bilinear", align_corners=False)
        sem   = F.interpolate(self.semantic_head(d1), scale_factor=2,
                              mode="bilinear", align_corners=False)
        return depth, sem

    def forward(self, x):
        """x: (B, S, C, 3, H, W) -> depth (B,S,C,1,H,W), sem (B,S,C,NUM_CLASSES,H,W)"""
        B, S, C = x.shape[:3]
        x_flat = rearrange(x, 'b s c ch h w -> (b s c) ch h w')

        stem, e1, e2, e3, e4 = self._encode(x_flat)
        bot_lstm = self._apply_lstm(e4, B, S, C)
        depth, sem = self._decode(bot_lstm, e3, e2, e1, stem)

        depth = rearrange(depth, '(b s c) 1 h w -> b s c 1 h w',    b=B, s=S, c=C)
        sem   = rearrange(sem,   '(b s c) cls h w -> b s c cls h w', b=B, s=S, c=C)
        return depth, sem


# ---------------------------------------------------------------------------
# Registered wrapper
# ---------------------------------------------------------------------------

@register("video_seg_depth_resnet")
class VideoSegDepthResNet(DepthModelBase):
    produces_semantic = True
    is_stateful       = False   # LSTM state is internal
    state_keys        = ()

    def __init__(self, backbone: str = "resnet18", lstm_hidden: int = 512):
        super().__init__()
        self.net = _ResNetVideoSegDepthUNet(backbone=backbone, lstm_hidden=lstm_hidden)

    def forward(self, x, state=None) -> ModelOutput:
        depth, sem = self.net(x)
        return ModelOutput(depth=depth, semantic=sem)

    @classmethod
    def add_model_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--backbone", type=str, default="resnet18",
                            choices=["resnet18", "resnet34", "resnet50"])
        parser.add_argument("--lstm-hidden", type=int, default=512)

    @classmethod
    def from_args(cls, args: Namespace):
        return cls(backbone=args.backbone, lstm_hidden=args.lstm_hidden)
