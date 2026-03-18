"""
TinyViT + LSTM video joint depth + semantic segmentation model.

Architecture:
  Encoder : TinyViT-21M (pretrained, frozen)
  Temporal: LSTM at the bottleneck
  Decoder : ConvBlock UNet decoder

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
from ._blocks import ConvBlock
from ._tinyvit import TinyViTEncoder

NUM_CLASSES = 23


# ---------------------------------------------------------------------------
# Private nn.Module implementation
# ---------------------------------------------------------------------------

class _VideoSegDepthUNet(nn.Module):
    """TinyViT-21M encoder + LSTM bottleneck + UNet decoder."""

    def __init__(self, num_classes: int = NUM_CLASSES, lstm_hidden: int = 512,
                 img_h: int = 224, img_w: int = 224):
        super().__init__()

        self.encoder = TinyViTEncoder(img_h=img_h, img_w=img_w)

        # Temporal LSTM at bottleneck (dim=576 from TinyViT layer[3])
        self.temporal_lstm = nn.LSTM(576, lstm_hidden, batch_first=True)

        # Decoder channel sizes tuned to TinyViT skip dims
        self.up4  = nn.ConvTranspose2d(lstm_hidden, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512 + 384, 512)

        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256 + 192, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 + 96, 128)

        # H/4 → H/2, no skip
        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 64)

        # H/2 → H, no skip
        self.up0  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(32, 32)

        self.depth_head    = nn.Conv2d(32, 1,           kernel_size=1)
        self.semantic_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def _apply_lstm(self, bot, B, S, C):
        """Apply LSTM across the sequence dimension at each spatial location.

        Args:
            bot: (B*S*C, 576, h, w)

        Returns:
            (B*S*C, lstm_hidden, h, w)
        """
        _, _, h, w = bot.shape
        x = bot.view(B, S, C, 576, h, w)
        x = x.permute(0, 2, 3, 4, 1, 5).contiguous()
        x = x.view(B * C * h * w, S, 576)

        lstm_out, _ = self.temporal_lstm(x)

        lstm_hidden = lstm_out.shape[-1]
        x = lstm_out.view(B, C, h, w, S, lstm_hidden)
        x = x.permute(0, 4, 1, 5, 2, 3).contiguous()
        x = x.view(B * S * C, lstm_hidden, h, w)
        return x

    def _decode(self, bot_lstm, skip2, skip1, skip0):
        def up_cat(up, x, skip):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
            return torch.cat([x, skip], dim=1)

        d4 = self.dec4(up_cat(self.up4, bot_lstm, skip2))
        d3 = self.dec3(up_cat(self.up3, d4,       skip1))
        d2 = self.dec2(up_cat(self.up2, d3,       skip0))

        d1 = self.dec1(self.up1(d2))
        d0 = self.dec0(self.up0(d1))

        return self.depth_head(d0), self.semantic_head(d0)

    def forward(self, x):
        """x: (B, S, C, 3, H, W) -> depth (B,S,C,1,H,W), sem (B,S,C,NUM_CLASSES,H,W)"""
        B, S, C = x.shape[:3]
        x_flat = rearrange(x, 'b s c ch h w -> (b s c) ch h w')

        with torch.no_grad():
            skip0, skip1, skip2, bot = self.encoder(x_flat)

        bot_lstm = self._apply_lstm(bot, B, S, C)
        depth, sem = self._decode(bot_lstm, skip2, skip1, skip0)

        depth = rearrange(depth, '(b s c) 1 h w -> b s c 1 h w',    b=B, s=S, c=C)
        sem   = rearrange(sem,   '(b s c) cls h w -> b s c cls h w', b=B, s=S, c=C)
        return depth, sem


# ---------------------------------------------------------------------------
# Registered wrapper
# ---------------------------------------------------------------------------

@register("video_seg_depth")
class VideoSegDepth(DepthModelBase):
    produces_semantic = True
    is_stateful       = False   # LSTM state is internal to each forward call
    state_keys        = ()

    def __init__(self, lstm_hidden: int = 512, img_h: int = 224, img_w: int = 224):
        super().__init__()
        self.net = _VideoSegDepthUNet(lstm_hidden=lstm_hidden, img_h=img_h, img_w=img_w)

    def forward(self, x, state=None) -> ModelOutput:
        depth, sem = self.net(x)
        return ModelOutput(depth=depth, semantic=sem)

    @classmethod
    def add_model_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--lstm-hidden", type=int, default=512)
        # TinyViT requires square 224×224 input
        parser.set_defaults(img_h=224, img_w=224)

    @classmethod
    def from_args(cls, args: Namespace):
        return cls(
            lstm_hidden=args.lstm_hidden,
            img_h=args.img_h,
            img_w=args.img_w,
        )
