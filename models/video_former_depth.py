"""
VideoFormerDepth — TinyViT + Depth Token Transformer for Streaming Depth Estimation.

Architecture:
  Encoder : TinyViT-21M (pretrained, frozen) → 4 multi-scale feature maps
  Decoder : Transformer with depth tokens (self-attn + DPT cross-attn + FFN)
  Head    : CNN progressive upsampler (token grid → full-res depth)

Temporal streaming: depth tokens carry over frame-to-frame.
  Frame 0: tokens ← token_init (learnable parameter)
  Frame t: tokens ← enriched tokens from frame t-1

Input  : (B, S, C, 3, H, W)
Output : depth (B, S, C, 1, H, W)
"""
import math
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from einops import rearrange

from . import ModelOutput, register
from .base import DepthModelBase
from ._blocks import make_2d_sincos_pos_enc
from ._tinyvit import TinyViTEncoder
from ._transformer import DepthDecoderLayer, TokenCNNHead


# ---------------------------------------------------------------------------
# Private nn.Module implementation
# ---------------------------------------------------------------------------

class _VideoFormerDepthNet(nn.Module):
    """
    TinyViT encoder (frozen) + depth-token transformer decoder.

    forward(x, prev_tokens=None):
        x           : (B, 3, H, W)   — one frame
        prev_tokens : (B, N_q, D) or None

    Returns:
        depth  : (B, 1, H, W)
        tokens : (B, N_q, D)  — enriched tokens for next frame
    """

    def __init__(
        self,
        token_stride: int = 8,
        token_dim: int = 256,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        img_h: int = 224,
        img_w: int = 224,
    ):
        super().__init__()
        self.token_stride = token_stride
        self.token_dim    = token_dim
        self.H_q = img_h // token_stride
        self.W_q = img_w // token_stride
        self.N_q = self.H_q * self.W_q

        self.encoder = TinyViTEncoder(img_h=img_h, img_w=img_w)

        # Project TinyViT-21M channel sizes [96, 192, 384, 576] → token_dim
        for lvl, enc_dim in enumerate([96, 192, 384, 576]):
            self.add_module(f"enc_proj_{lvl}", nn.Linear(enc_dim, token_dim))

        # Fixed 2D sinusoidal position encodings
        self.register_buffer("token_pos_enc",
            make_2d_sincos_pos_enc(self.H_q, self.W_q, token_dim).unsqueeze(0))

        H4, W4 = img_h // 4, img_w // 4
        enc_grids = [(H4, W4), (H4 // 2, W4 // 2), (H4 // 4, W4 // 4), (H4 // 8, W4 // 8)]
        for lvl, (H, W) in enumerate(enc_grids):
            self.register_buffer(f"enc_pos_{lvl}",
                make_2d_sincos_pos_enc(H, W, token_dim).unsqueeze(0))

        # Learnable depth token initialisation
        self.token_init = nn.Parameter(torch.zeros(1, self.N_q, token_dim))
        nn.init.trunc_normal_(self.token_init, std=0.02)

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            DepthDecoderLayer(token_dim, num_heads)
            for _ in range(num_decoder_layers)
        ])

        # CNN depth head
        self.depth_head = TokenCNNHead(token_dim, token_stride, out_channels=1)

    def forward(self, x: torch.Tensor, prev_tokens=None):
        """
        Args:
            x           : (B, 3, H, W)
            prev_tokens : (B, N_q, D) or None

        Returns:
            depth  : (B, 1, H, W)
            tokens : (B, N_q, D)
        """
        B = x.shape[0]

        with torch.no_grad():
            skip0, skip1, skip2, bot = self.encoder(x)

        enc_feats = [
            self.enc_proj_0(skip0.flatten(2).transpose(1, 2)),
            self.enc_proj_1(skip1.flatten(2).transpose(1, 2)),
            self.enc_proj_2(skip2.flatten(2).transpose(1, 2)),
            self.enc_proj_3(bot  .flatten(2).transpose(1, 2)),
        ]
        enc_pos = [getattr(self, f"enc_pos_{i}") for i in range(4)]

        tokens = self.token_init.expand(B, -1, -1) if prev_tokens is None else prev_tokens
        for layer in self.decoder_layers:
            tokens = layer(tokens, enc_feats, self.token_pos_enc, enc_pos)

        tokens_spatial = tokens.transpose(1, 2).view(
            B, self.token_dim, self.H_q, self.W_q)
        depth = self.depth_head(tokens_spatial)

        return depth, tokens


# ---------------------------------------------------------------------------
# Registered wrapper
# ---------------------------------------------------------------------------

@register("video_former_depth")
class VideoFormerDepth(DepthModelBase):
    produces_semantic = False
    is_stateful       = True
    state_keys        = ("depth_tokens",)

    def __init__(
        self,
        token_stride: int = 8,
        token_dim: int = 256,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        img_h: int = 224,
        img_w: int = 224,
    ):
        super().__init__()
        self.net = _VideoFormerDepthNet(
            token_stride=token_stride,
            token_dim=token_dim,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            img_h=img_h,
            img_w=img_w,
        )

    def forward(self, x, state=None) -> ModelOutput:
        """
        Per-frame sequential loop; tokens carry over across frames.

        Args:
            x     : (B, S, C, 3, H, W)
            state : {"depth_tokens": (B*C, N_q, D)} or None

        Returns:
            ModelOutput with depth (B, S, C, 1, H, W) and state dict
        """
        B, S, C = x.shape[:3]
        prev = state["depth_tokens"] if state is not None else None
        depth_frames = []

        for s in range(S):
            x_s = rearrange(x[:, s], 'b c ch h w -> (b c) ch h w')
            d_s, prev = self.net(x_s, prev)
            depth_frames.append(
                rearrange(d_s, '(b c) 1 h w -> b c 1 h w', b=B, c=C))

        return ModelOutput(
            depth=torch.stack(depth_frames, dim=1),   # (B, S, C, 1, H, W)
            state={"depth_tokens": prev},
        )

    @classmethod
    def add_model_args(cls, parser: ArgumentParser) -> None:
        # TinyViT requires square 224×224 input
        parser.set_defaults(img_h=224, img_w=224)
        parser.add_argument("--token-stride", type=int, default=8,
                            help="Depth token grid stride (token grid = H/stride × W/stride)")
        parser.add_argument("--token-dim", type=int, default=256,
                            help="Depth token feature dimension")
        parser.add_argument("--num-decoder-layers", type=int, default=6)
        parser.add_argument("--num-heads", type=int, default=8)

    @classmethod
    def from_args(cls, args: Namespace):
        return cls(
            token_stride=args.token_stride,
            token_dim=args.token_dim,
            num_decoder_layers=args.num_decoder_layers,
            num_heads=args.num_heads,
            img_h=args.img_h,
            img_w=args.img_w,
        )
