"""
VideoFormerSegDepth — TinyViT + Parallel Depth & Segmentation Token Transformers.

Architecture:
  Encoder : TinyViT-21M (pretrained, frozen) → 4 multi-scale feature maps
  Two parallel token stacks (depth + seg) share the encoder features but have
  separate decoder layer stacks and CNN output heads.

Temporal streaming: both depth tokens and seg tokens carry over frame-to-frame.

Input  : (B, S, C, 3, H, W)
Output : depth (B, S, C, 1, H, W), semantic (B, S, C, NUM_CLASSES, H, W)
"""
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from einops import rearrange

from . import ModelOutput, register
from .base import DepthModelBase
from ._blocks import make_2d_sincos_pos_enc
from ._tinyvit import TinyViTEncoder
from ._transformer import DepthDecoderLayer, TokenCNNHead

NUM_CLASSES = 23


# ---------------------------------------------------------------------------
# Private nn.Module implementation
# ---------------------------------------------------------------------------

class _VideoFormerSegDepthNet(nn.Module):
    """
    TinyViT encoder (frozen) + parallel depth-token and seg-token decoders.

    forward(x, prev_depth_tokens=None, prev_seg_tokens=None):
        x               : (B, 3, H, W)
        prev_depth_tokens : (B, N_q, D) or None
        prev_seg_tokens   : (B, N_q, D) or None

    Returns:
        depth        : (B, 1, H, W)
        sem          : (B, NUM_CLASSES, H, W)
        depth_tokens : (B, N_q, D)
        seg_tokens   : (B, N_q, D)
    """

    def __init__(
        self,
        token_stride: int = 8,
        token_dim: int = 256,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_classes: int = NUM_CLASSES,
        img_h: int = 224,
        img_w: int = 224,
    ):
        super().__init__()
        self.token_stride = token_stride
        self.token_dim    = token_dim
        self.num_classes  = num_classes
        self.H_q = img_h // token_stride
        self.W_q = img_w // token_stride
        self.N_q = self.H_q * self.W_q

        self.encoder = TinyViTEncoder(img_h=img_h, img_w=img_w)

        # Project encoder levels → token_dim (shared by both decoders)
        for lvl, enc_dim in enumerate([96, 192, 384, 576]):
            self.add_module(f"enc_proj_{lvl}", nn.Linear(enc_dim, token_dim))

        # Fixed 2D sinusoidal position encodings (shared)
        self.register_buffer("token_pos_enc",
            make_2d_sincos_pos_enc(self.H_q, self.W_q, token_dim).unsqueeze(0))

        H4, W4 = img_h // 4, img_w // 4
        enc_grids = [(H4, W4), (H4 // 2, W4 // 2), (H4 // 4, W4 // 4), (H4 // 8, W4 // 8)]
        for lvl, (H, W) in enumerate(enc_grids):
            self.register_buffer(f"enc_pos_{lvl}",
                make_2d_sincos_pos_enc(H, W, token_dim).unsqueeze(0))

        # Learnable token initialisations
        self.depth_token_init = nn.Parameter(torch.zeros(1, self.N_q, token_dim))
        nn.init.trunc_normal_(self.depth_token_init, std=0.02)
        self.seg_token_init = nn.Parameter(torch.zeros(1, self.N_q, token_dim))
        nn.init.trunc_normal_(self.seg_token_init, std=0.02)

        # Separate decoder layer stacks
        self.depth_decoder_layers = nn.ModuleList([
            DepthDecoderLayer(token_dim, num_heads)
            for _ in range(num_decoder_layers)
        ])
        self.seg_decoder_layers = nn.ModuleList([
            DepthDecoderLayer(token_dim, num_heads)
            for _ in range(num_decoder_layers)
        ])

        # Output heads
        self.depth_head = TokenCNNHead(token_dim, token_stride, out_channels=1)
        self.seg_head   = TokenCNNHead(token_dim, token_stride, out_channels=num_classes)

    def forward(self, x: torch.Tensor, prev_depth_tokens=None, prev_seg_tokens=None):
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

        # Depth branch
        depth_tokens = self.depth_token_init.expand(B, -1, -1) \
                       if prev_depth_tokens is None else prev_depth_tokens
        for layer in self.depth_decoder_layers:
            depth_tokens = layer(depth_tokens, enc_feats, self.token_pos_enc, enc_pos)

        depth_spatial = depth_tokens.transpose(1, 2).view(
            B, self.token_dim, self.H_q, self.W_q)
        depth = self.depth_head(depth_spatial)

        # Seg branch
        seg_tokens = self.seg_token_init.expand(B, -1, -1) \
                     if prev_seg_tokens is None else prev_seg_tokens
        for layer in self.seg_decoder_layers:
            seg_tokens = layer(seg_tokens, enc_feats, self.token_pos_enc, enc_pos)

        seg_spatial = seg_tokens.transpose(1, 2).view(
            B, self.token_dim, self.H_q, self.W_q)
        sem = self.seg_head(seg_spatial)

        return depth, sem, depth_tokens, seg_tokens


# ---------------------------------------------------------------------------
# Registered wrapper
# ---------------------------------------------------------------------------

@register("video_former_seg_depth")
class VideoFormerSegDepth(DepthModelBase):
    produces_semantic = True
    is_stateful       = True
    state_keys        = ("depth_tokens", "seg_tokens")

    def __init__(
        self,
        token_stride: int = 8,
        token_dim: int = 256,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_classes: int = NUM_CLASSES,
        img_h: int = 224,
        img_w: int = 224,
    ):
        super().__init__()
        self.net = _VideoFormerSegDepthNet(
            token_stride=token_stride,
            token_dim=token_dim,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            img_h=img_h,
            img_w=img_w,
        )

    def forward(self, x, state=None) -> ModelOutput:
        """
        Per-frame sequential loop; both token sets carry over across frames.

        Args:
            x     : (B, S, C, 3, H, W)
            state : {"depth_tokens": ..., "seg_tokens": ...} or None

        Returns:
            ModelOutput with depth, semantic, and state dict
        """
        B, S, C = x.shape[:3]
        prev_depth = state["depth_tokens"] if state is not None else None
        prev_seg   = state["seg_tokens"]   if state is not None else None
        depth_frames, sem_frames = [], []

        for s in range(S):
            x_s = rearrange(x[:, s], 'b c ch h w -> (b c) ch h w')
            d_s, sem_s, prev_depth, prev_seg = self.net(x_s, prev_depth, prev_seg)
            depth_frames.append(
                rearrange(d_s,   '(b c) 1 h w -> b c 1 h w',    b=B, c=C))
            sem_frames.append(
                rearrange(sem_s, '(b c) cls h w -> b c cls h w', b=B, c=C))

        return ModelOutput(
            depth=torch.stack(depth_frames, dim=1),    # (B, S, C, 1, H, W)
            semantic=torch.stack(sem_frames, dim=1),   # (B, S, C, NUM_CLASSES, H, W)
            state={"depth_tokens": prev_depth, "seg_tokens": prev_seg},
        )

    @classmethod
    def add_model_args(cls, parser: ArgumentParser) -> None:
        # TinyViT requires square 224×224 input
        parser.set_defaults(img_h=224, img_w=224)
        parser.add_argument("--token-stride", type=int, default=8)
        parser.add_argument("--token-dim", type=int, default=256)
        parser.add_argument("--num-decoder-layers", type=int, default=6)
        parser.add_argument("--num-heads", type=int, default=8)
        parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)

    @classmethod
    def from_args(cls, args: Namespace):
        return cls(
            token_stride=args.token_stride,
            token_dim=args.token_dim,
            num_decoder_layers=args.num_decoder_layers,
            num_heads=args.num_heads,
            num_classes=args.num_classes,
            img_h=args.img_h,
            img_w=args.img_w,
        )
