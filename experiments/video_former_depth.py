"""
VideoFormerDepth — TinyViT + Depth Token Transformer for Streaming Depth Estimation
=====================================================================================

Architecture
------------
  Encoder : TinyViT-21M (pretrained, frozen) → 4 multi-scale feature maps

  Depth tokens form a 2D spatial grid (H/token_stride × W/token_stride) with
  sinusoidal 2D position encodings.  They act as a spatial working memory that
  is updated each frame via a transformer decoder.

  Each decoder layer (pre-norm) applies:
    1. Self-attention among depth tokens   (tokens communicate spatially)
    2. DPT-style cross-attention           (see note below)
    3. FFN (Linear → GELU → Linear)

  CNN depth head: token grid → progressive ConvTranspose2d → full-resolution depth

Streaming / temporal mechanism
-------------------------------
  Frame 0 : tokens  ←  token_init  (learnable parameter)
  Frame t : tokens  ←  enriched tokens from frame t-1  (NO projection to scalar depth;
                         the full token feature vector is passed forward)

  At inference over a T-frame video, tokens accumulate temporal context; earlier
  frames influence later depth estimates with O(1) memory per camera.

  --single-frame  (pre-training flag)
    Every frame independently uses token_init — no carry-over.
    Trains the single-frame depth quality before adding temporal streaming.
    Recommended workflow:  pre-train with --single-frame, then fine-tune without.

──────────────────────────────────────────────────────────────────────────────────
DPT-style cross-attention  [design note]
──────────────────────────────────────────────────────────────────────────────────
  Inspired by Ranftl et al. "Vision Transformers for Dense Prediction" (ICCV 2021).
  DPT reassembles internal ViT features from multiple layers and fuses them via a
  feature pyramid.  Here we adapt the multi-scale cross-attention aspect only:

    • The 4 TinyViT encoder scales (skip0–skip2 + bottleneck) are each projected
      to `token_dim` and kept as SEPARATE key-value sets.
    • Depth tokens run a SEPARATE nn.MultiheadAttention for each scale.
    • The 4 cross-attention outputs are SUMMED before adding the residual.

  This lets tokens gather coarse (bottleneck, 576ch, H/32) and fine (skip0, 96ch,
  H/4) information independently, without concatenating into one huge KV sequence.

  Future research directions
  ──────────────────────────
  • Learned per-scale scalar weights (α_i) instead of uniform sum
  • Concatenation + linear projection of scale outputs (richer, costlier)
  • Coarse-to-fine ordering: bottleneck first, then progressively finer skips
  • FPN-style feature merge before a single cross-attention
  • Original DPT "reassemble + fusion block" feature merging

──────────────────────────────────────────────────────────────────────────────────
CNN depth head  [design note]
──────────────────────────────────────────────────────────────────────────────────
  token_grid (B, token_dim, H_q, W_q)
    → ConvTranspose2d(×2) + ConvBlock   ×  log2(token_stride)  stages
    → Conv2d(→ 1)

  Future research directions
  ──────────────────────────
  • Bilinear upsample + Conv (fewer checkerboard artifacts than ConvTranspose)
  • Add encoder skip connections to the head (U-Net style)
  • SegFormer MLP head: linear projection per token + single bilinear upsample

Input  : (B, S, C, 3, H, W)   — B batch, S sequence frames, C cameras
Output : depth  (B, S, C, 1, H, W)
"""
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, "/workspace/TinyViT")
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from einops import rearrange
import argparse
from datetime import datetime
import torchvision.transforms.functional as TF

from dataset import Bench2DriveDataset, CAMERA_NAMES
from visualization import collect_viz_clip, save_depth_image, save_depth_video
from config import DATA_ROOT, LOG_ROOT, CHECKPOINT_ROOT


# ---------------------------------------------------------------------------
# 2-D sinusoidal position encoding
# ---------------------------------------------------------------------------

def make_2d_sincos_pos_enc(H: int, W: int, dim: int) -> torch.Tensor:
    """
    Fixed 2D sinusoidal position encoding.

    Allocates dim/2 dimensions to row positions and dim/2 to column positions,
    each encoded with the standard 1D sin/cos scheme at geometrically-spaced
    frequencies (following Vaswani et al. 2017).

    Args:
        H, W : spatial grid size
        dim  : total encoding dimension (must be divisible by 4)

    Returns:
        (H * W, dim) float tensor
    """
    assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
    half = dim // 2  # half for rows, half for cols

    def sincos_1d(n: int, d: int) -> torch.Tensor:
        pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)        # (n, 1)
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d)
        )                                                               # (d/2,)
        enc = torch.zeros(n, d)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div)
        return enc  # (n, d)

    row_enc = sincos_1d(H, half)  # (H, half)
    col_enc = sincos_1d(W, half)  # (W, half)

    pe = torch.cat([
        row_enc.unsqueeze(1).expand(H, W, half),   # row info broadcast over W
        col_enc.unsqueeze(0).expand(H, W, half),   # col info broadcast over H
    ], dim=-1)                                      # (H, W, dim)
    return pe.reshape(H * W, dim)


# ---------------------------------------------------------------------------
# Transformer decoder layer  (DPT-style multi-scale cross-attention)
# ---------------------------------------------------------------------------

class DepthDecoderLayer(nn.Module):
    """
    One pre-norm transformer decoder layer.

    Self-attention (tokens attend to each other) followed by DPT-style
    cross-attention (tokens attend separately to each of 4 encoder scales,
    outputs summed) followed by a position-wise FFN.

    Pre-norm layout:
        tokens = tokens + SelfAttn  ( LayerNorm(tokens) + token_pos )
        tokens = tokens + Σ_k CrossAttn_k( LayerNorm(tokens) + token_pos,
                                            enc_feats[k]     + enc_pos[k],
                                            enc_feats[k] )
        tokens = tokens + FFN( LayerNorm(tokens) )
    """

    NUM_ENC_LEVELS = 4

    def __init__(self, token_dim: int, num_heads: int, ffn_dim: int = None):
        super().__init__()
        ffn_dim = ffn_dim or token_dim * 4

        self.norm1     = nn.LayerNorm(token_dim)
        self.self_attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)

        self.norm2       = nn.LayerNorm(token_dim)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
            for _ in range(self.NUM_ENC_LEVELS)
        ])

        self.norm3 = nn.LayerNorm(token_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(token_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, token_dim),
        )

    @staticmethod
    def _add_pos(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return t + pos

    def forward(
        self,
        tokens:    torch.Tensor,       # (B, N_q, D)
        enc_feats: list,               # 4 × (B, N_k, D)  — projected encoder features
        token_pos: torch.Tensor,       # (1, N_q, D)
        enc_pos:   list,               # 4 × (1, N_k, D)
    ) -> torch.Tensor:

        # 1. Self-attention (pre-norm; pos added to Q and K, not V)
        normed = self.norm1(tokens)
        q = self._add_pos(normed, token_pos)
        tokens = tokens + self.self_attn(q, q, normed)[0]

        # 2. DPT-style cross-attention (pre-norm; sum over 4 encoder levels)
        normed = self.norm2(tokens)
        q = self._add_pos(normed, token_pos)
        cross_sum = torch.zeros_like(tokens)
        for i, ca in enumerate(self.cross_attns):
            k = self._add_pos(enc_feats[i], enc_pos[i])
            cross_sum = cross_sum + ca(q, k, enc_feats[i])[0]
        tokens = tokens + cross_sum

        # 3. FFN (pre-norm)
        tokens = tokens + self.ffn(self.norm3(tokens))

        return tokens


# ---------------------------------------------------------------------------
# CNN depth head  (token grid → full-resolution depth)
# ---------------------------------------------------------------------------

class DepthCNNHead(nn.Module):
    """
    Progressive upsampler: (B, token_dim, H_q, W_q) → (B, 1, H, W).

    Uses log2(token_stride) ConvTranspose2d stages, each doubling spatial
    resolution and halving channel count (floor at 32).

    # TODO(future): compare with bilinear upsample + Conv (fewer artifacts)
    # TODO(future): add encoder skip connections (U-Net style)
    # TODO(future): benchmark SegFormer MLP head as an alternative
    """

    def __init__(self, token_dim: int, token_stride: int):
        super().__init__()
        assert (token_stride & (token_stride - 1)) == 0, \
            "token_stride must be a power of 2"
        n_ups = int(math.log2(token_stride))

        # Channel schedule: halve at each stage, floor at 32
        dims = [max(32, token_dim >> i) for i in range(n_ups + 1)]

        layers = []
        for i in range(n_ups):
            layers += [
                nn.ConvTranspose2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                nn.BatchNorm2d(dims[i + 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(dims[i + 1], dims[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[i + 1]),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(dims[-1], 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class VideoFormerDepth(nn.Module):
    """
    TinyViT encoder (frozen) + depth-token transformer decoder.

    forward(x, prev_tokens=None):
        x           : (B, 3, H, W)   — one frame; B = batch_size × num_cameras
        prev_tokens : (B, N_q, D) or None  (None → use learnable token_init)

    Returns:
        depth  : (B, 1, H, W)
        tokens : (B, N_q, D)  — enriched tokens; pass as prev_tokens next frame
    """

    def __init__(
        self,
        token_stride: int = 8,
        token_dim: int = 256,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        img_h: int = 224,
        img_w: int = 224,
        img_size: int = 224,
        debug_shapes: bool = False,
    ):
        super().__init__()
        self.token_stride = token_stride
        self.token_dim    = token_dim
        self.H_q = img_h // token_stride
        self.W_q = img_w // token_stride
        self.N_q = self.H_q * self.W_q
        self.debug_shapes   = debug_shapes
        self._shapes_logged = False

        # ── Encoder: TinyViT-21M (pretrained, frozen) ──────────────────────
        from models.tiny_vit import tiny_vit_21m_224
        try:
            backbone = tiny_vit_21m_224(pretrained=True, img_size=img_size)
        except Exception:
            backbone = tiny_vit_21m_224(pretrained=False, img_size=img_size)

        self.patch_embed = backbone.patch_embed
        self.enc_layers  = backbone.layers
        for p in list(self.patch_embed.parameters()) + list(self.enc_layers.parameters()):
            p.requires_grad_(False)

        # ── Project encoder levels → token_dim ─────────────────────────────
        # TinyViT-21M channel sizes: [96, 192, 384, 576]
        for lvl, enc_dim in enumerate([96, 192, 384, 576]):
            self.add_module(f"enc_proj_{lvl}", nn.Linear(enc_dim, token_dim))

        # ── Fixed 2D sinusoidal position encodings (registered as buffers) ──
        #
        # Depth token grid: (H/token_stride) × (W/token_stride)
        # For non-square data, the token grid naturally follows the aspect ratio
        # (e.g., 450×800 with token_stride=8 → 56×100 tokens).
        self.register_buffer("token_pos_enc",
            make_2d_sincos_pos_enc(self.H_q, self.W_q, token_dim).unsqueeze(0))

        # Encoder feature grids (one per TinyViT scale)
        H4, W4 = img_h // 4, img_w // 4
        enc_grids = [(H4, W4), (H4 // 2, W4 // 2), (H4 // 4, W4 // 4), (H4 // 8, W4 // 8)]
        for lvl, (H, W) in enumerate(enc_grids):
            self.register_buffer(f"enc_pos_{lvl}",
                make_2d_sincos_pos_enc(H, W, token_dim).unsqueeze(0))

        # ── Learnable depth token initialisation ───────────────────────────
        # Shape (1, N_q, token_dim) — expanded to (B, N_q, token_dim) at runtime.
        # In single-frame pre-training mode this is the only token source.
        self.token_init = nn.Parameter(torch.zeros(1, self.N_q, token_dim))
        nn.init.trunc_normal_(self.token_init, std=0.02)

        # ── Transformer decoder layers ──────────────────────────────────────
        self.decoder_layers = nn.ModuleList([
            DepthDecoderLayer(token_dim, num_heads)
            for _ in range(num_decoder_layers)
        ])

        # ── CNN depth head ──────────────────────────────────────────────────
        self.depth_head = DepthCNNHead(token_dim, token_stride)

    # ------------------------------------------------------------------
    # TinyViT encoder  (identical to video_seg_depth.py)
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor):
        """Run frozen TinyViT encoder; return 4 spatial feature maps.

        Args:
            x: (N, 3, H, W)

        Returns:
            skip0      : (N,  96, H/4,  W/4)
            skip1      : (N, 192, H/8,  W/8)
            skip2      : (N, 384, H/16, W/16)
            bottleneck : (N, 576, H/32, W/32)
        """
        x = self.patch_embed(x)          # (N, 96, H/4, W/4)
        H4, W4 = x.shape[-2], x.shape[-1]

        for blk in self.enc_layers[0].blocks:
            x = blk(x)
        skip0 = x
        x = self.enc_layers[0].downsample(x)    # → tokens (N, H/8·W/8, 192)

        for blk in self.enc_layers[1].blocks:
            x = blk(x)
        skip1_tok = x
        x = self.enc_layers[1].downsample(x)    # → tokens (N, H/16·W/16, 384)

        for blk in self.enc_layers[2].blocks:
            x = blk(x)
        skip2_tok = x
        x = self.enc_layers[2].downsample(x)    # → tokens (N, H/32·W/32, 576)

        for blk in self.enc_layers[3].blocks:
            x = blk(x)
        bot_tok = x                             # (N, H/32·W/32, 576)

        N = x.shape[0]
        H8,  W8  = H4 // 2, W4 // 2
        H16, W16 = H4 // 4, W4 // 4
        H32, W32 = H4 // 8, W4 // 8

        skip1 = skip1_tok.view(N, H8,  W8,  192).permute(0, 3, 1, 2).contiguous()
        skip2 = skip2_tok.view(N, H16, W16, 384).permute(0, 3, 1, 2).contiguous()
        bot   = bot_tok  .view(N, H32, W32, 576).permute(0, 3, 1, 2).contiguous()

        return skip0, skip1, skip2, bot

    # ------------------------------------------------------------------
    # Debug shape logging  (fires once when debug_shapes=True)
    # ------------------------------------------------------------------

    def _log_shapes(self, x, skip0, skip1, skip2, bot,
                    enc_feats, tokens_init, tokens_out, tokens_spatial, depth):
        lines = [
            "=" * 64,
            "VideoFormerDepth — tensor shapes (first forward pass)",
            "=" * 64,
            f"  input              (B,3,H,W)        : {tuple(x.shape)}",
            "  --- encoder (frozen TinyViT) ---",
            f"  skip0              (N, 96,H/4,W/4)  : {tuple(skip0.shape)}",
            f"  skip1              (N,192,H/8,W/8)  : {tuple(skip1.shape)}",
            f"  skip2              (N,384,H/16,W/16): {tuple(skip2.shape)}",
            f"  bottleneck         (N,576,H/32,W/32): {tuple(bot.shape)}",
            "  --- projected encoder features (per level) ---",
            f"  enc_feats[0] skip0 (N, N0, token_dim): {tuple(enc_feats[0].shape)}",
            f"  enc_feats[1] skip1 (N, N1, token_dim): {tuple(enc_feats[1].shape)}",
            f"  enc_feats[2] skip2 (N, N2, token_dim): {tuple(enc_feats[2].shape)}",
            f"  enc_feats[3] bot   (N, N3, token_dim): {tuple(enc_feats[3].shape)}",
            "  --- depth tokens ---",
            f"  tokens (init/prev) (B, N_q, D)      : {tuple(tokens_init.shape)}",
            f"  tokens (after dec) (B, N_q, D)      : {tuple(tokens_out.shape)}",
            f"  tokens_spatial     (B, D, H_q, W_q) : {tuple(tokens_spatial.shape)}",
            "  --- depth output ---",
            f"  depth              (B, 1, H, W)     : {tuple(depth.shape)}",
            "=" * 64,
        ]
        table = "\n".join(lines)
        print(table, flush=True)
        self._shapes_table = table   # picked up by Lightning module for TensorBoard

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, prev_tokens=None):
        """
        Args:
            x           : (B, 3, H, W)  — one frame; B = batch × cameras
            prev_tokens : (B, N_q, D) or None

        Returns:
            depth  : (B, 1, H, W)
            tokens : (B, N_q, D)   — enriched; use as prev_tokens next frame
        """
        B = x.shape[0]

        with torch.no_grad():
            skip0, skip1, skip2, bot = self._encode(x)

        # Project encoder features to token_dim, flatten spatial → sequence
        enc_feats = [
            self.enc_proj_0(skip0.flatten(2).transpose(1, 2)),   # (B, N0, D)
            self.enc_proj_1(skip1.flatten(2).transpose(1, 2)),   # (B, N1, D)
            self.enc_proj_2(skip2.flatten(2).transpose(1, 2)),   # (B, N2, D)
            self.enc_proj_3(bot  .flatten(2).transpose(1, 2)),   # (B, N3, D)
        ]
        enc_pos = [getattr(self, f"enc_pos_{i}") for i in range(4)]

        # Initialise depth tokens
        tokens_in = self.token_init.expand(B, -1, -1) if prev_tokens is None \
                    else prev_tokens

        # Run transformer decoder layers
        tokens = tokens_in
        for layer in self.decoder_layers:
            tokens = layer(tokens, enc_feats, self.token_pos_enc, enc_pos)

        # Reshape enriched tokens to spatial grid and decode to depth
        tokens_spatial = tokens.transpose(1, 2).view(
            B, self.token_dim, self.H_q, self.W_q)
        depth = self.depth_head(tokens_spatial)    # (B, 1, H, W)

        if self.debug_shapes and not self._shapes_logged:
            self._log_shapes(x, skip0, skip1, skip2, bot,
                             enc_feats, tokens_in, tokens, tokens_spatial, depth)
            self._shapes_logged = True

        return depth, tokens


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class VideoFormerDepthModule(pl.LightningModule):
    """
    Lightning wrapper for VideoFormerDepth.

    Training modes
    --------------
    single_frame=True  (pre-training):
        Every frame uses token_init; no temporal carry-over.
        All B×S×C frames are processed in one batched forward pass.

    single_frame=False  (streaming / default):
        Frames are processed in sequence s=0..S-1; tokens from frame s are
        passed as prev_tokens to frame s+1.

    Visualisation
    -------------
    save_best_video  runs the model in STREAMING mode over the fixed viz clip so
    that the video shows the temporal benefit of the depth token memory.
    """

    def __init__(
        self,
        token_stride: int = 8,
        token_dim: int = 256,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        learning_rate: float = 1e-4,
        depth_loss_fn: str = "l1",
        single_frame: bool = False,
        cli_command: str = "",
        viz_rgb=None,
        viz_depth=None,
        img_h: int = 224,
        img_w: int = 224,
        img_size: int = 224,
        debug_shapes: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["viz_rgb", "viz_depth"])

        self.model        = VideoFormerDepth(
            token_stride=token_stride, token_dim=token_dim,
            num_decoder_layers=num_decoder_layers, num_heads=num_heads,
            img_h=img_h, img_w=img_w, img_size=img_size,
            debug_shapes=debug_shapes,
        )
        self.learning_rate = learning_rate
        self.single_frame  = single_frame
        self.cli_command   = cli_command
        self.best_val_loss = float("inf")

        self._viz_rgb   = viz_rgb    # (1, T, cams, 3, H, W)
        self._viz_depth = viz_depth  # (1, T, cams, 1, H, W) or None

        if depth_loss_fn == "l1":
            self.depth_loss_fn = nn.L1Loss()
        elif depth_loss_fn == "mse":
            self.depth_loss_fn = nn.MSELoss()
        else:
            self.depth_loss_fn = nn.SmoothL1Loss()

    # ------------------------------------------------------------------
    def on_train_start(self):
        if self.cli_command:
            self.logger.experiment.add_text("cli_command", self.cli_command,
                                            global_step=0)
        mode = "single-frame (pre-training)" if self.single_frame else "streaming"
        self.logger.experiment.add_text("training_mode", mode, global_step=0)
        if self.single_frame:
            print("[VideoFormerDepth] pre-training mode: single-frame (no token carry-over)")

    # ------------------------------------------------------------------
    def forward(self, x):
        """x: (B, S, C, 3, H, W) → depth (B, S, C, 1, H, W)  [streaming]"""
        B, S, C = x.shape[:3]
        prev_tokens = None
        depth_frames = []
        for s in range(S):
            x_s = rearrange(x[:, s], 'b c ch h w -> (b c) ch h w')
            depth_s, prev_tokens = self.model(x_s, prev_tokens)
            depth_frames.append(
                rearrange(depth_s, '(b c) 1 h w -> b c 1 h w', b=B, c=C))
        return torch.stack(depth_frames, dim=1)   # (B, S, C, 1, H, W)

    # ------------------------------------------------------------------
    def _step(self, batch, log_shapes: bool = False):
        rgb      = batch["rgb"]    # (B, S, C, 3, H, W)
        depth_gt = batch["depth"]  # (B, S, C, 1, H, W)
        B, S, C = rgb.shape[:3]

        if self.single_frame:
            # Pre-training: all frames independent, batched forward
            x_flat    = rearrange(rgb, 'b s c ch h w -> (b s c) ch h w')
            depth_flat, _ = self.model(x_flat, prev_tokens=None)
            depth_pred = rearrange(depth_flat,
                                   '(b s c) 1 h w -> b s c 1 h w', b=B, s=S, c=C)
        else:
            # Streaming: sequential frame processing, tokens passed forward
            prev_tokens = None
            depth_frames = []
            for s in range(S):
                x_s = rearrange(rgb[:, s], 'b c ch h w -> (b c) ch h w')
                depth_s, prev_tokens = self.model(x_s, prev_tokens)
                depth_frames.append(
                    rearrange(depth_s, '(b c) 1 h w -> b c 1 h w', b=B, c=C))
            depth_pred = torch.stack(depth_frames, dim=1)   # (B, S, C, 1, H, W)

        if log_shapes and hasattr(self.model, "_shapes_table"):
            self.logger.experiment.add_text(
                "debug/shapes", self.model._shapes_table, global_step=0)
            del self.model._shapes_table

        loss = self.depth_loss_fn(depth_pred, depth_gt)
        return loss, depth_pred

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        log_shapes = (batch_idx == 0 and self.current_epoch == 0)
        loss, _ = self._step(batch, log_shapes=log_shapes)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, depth_pred = self._step(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        mse = F.mse_loss(depth_pred, batch["depth"])
        self.log("val/mse", mse, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            log_dir = Path(self.trainer.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            save_depth_image(
                batch["rgb"], depth_pred, batch["depth"],
                log_dir / f"validation_epoch_{self.current_epoch:04d}.png",
            )
        return loss

    def on_validation_epoch_end(self):
        val_mse = self.trainer.callback_metrics.get("val/mse")
        if val_mse is None:
            return
        if val_mse.item() < self.best_val_loss:
            self.best_val_loss = val_mse.item()
            self.save_best_video()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def save_best_video(self):
        """Run the model in STREAMING mode over the fixed viz clip."""
        if self._viz_rgb is None or self._viz_depth is None:
            return

        T    = self._viz_rgb.shape[1]
        cams = self._viz_rgb.shape[2]
        prev_tokens = None
        pred_frames = []

        for t in range(T):
            # frame: (1, 1, cams, 3, H, W) — one time-step from the viz clip
            frame = self._viz_rgb[:, t:t + 1].to(self.device)
            B = 1
            x_t = rearrange(frame[:, 0], 'b c ch h w -> (b c) ch h w')  # (cams, 3, H, W)
            depth_t, prev_tokens = self.model(x_t, prev_tokens)
            pred_frames.append(
                rearrange(depth_t, '(b c) 1 h w -> b 1 c 1 h w', b=B, c=cams).cpu())

        viz_pred = torch.cat(pred_frames, dim=1)  # (1, T, cams, 1, H, W)

        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        save_depth_video(self._viz_rgb, viz_pred, self._viz_depth,
                         log_dir / "best_depth.mp4")

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# Resize transform  (identical to other experiments)
# ---------------------------------------------------------------------------

def make_resize_transform(img_h: int, img_w: int):
    def transform(sample):
        def resize_tensor(t, mode):
            shape = t.shape
            h, w  = shape[-2], shape[-1]
            if h == img_h and w == img_w:
                return t
            flat = t.reshape(-1, 1, h, w).float()
            flat = TF.resize(flat, [img_h, img_w],
                             interpolation=TF.InterpolationMode.NEAREST
                             if mode == "nearest" else TF.InterpolationMode.BILINEAR,
                             antialias=(mode == "bilinear"))
            return flat.reshape(*shape[:-2], img_h, img_w).to(t.dtype)

        out = dict(sample)
        out["rgb"] = resize_tensor(sample["rgb"], "bilinear")
        if "depth"          in sample: out["depth"]          = resize_tensor(sample["depth"],          "nearest")
        if "instance_class" in sample: out["instance_class"] = resize_tensor(sample["instance_class"], "nearest")
        if "instance_id"    in sample: out["instance_id"]    = resize_tensor(sample["instance_id"],    "nearest")
        return out
    return transform


def _resolved_config(parser: argparse.ArgumentParser,
                     args: argparse.Namespace) -> str:
    """Return a human-readable block with the fully-resolved CLI invocation."""
    script  = Path(sys.argv[0]).name
    invoked = " ".join(sys.argv)
    parts   = [f"python {script}"]
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        val  = getattr(args, action.dest)
        flag = action.option_strings[-1]
        if isinstance(action, argparse._StoreTrueAction):
            if val:
                parts.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            if not val:
                parts.append(flag)
        else:
            parts.append(f"{flag} {val}")
    reproducible = " \\\n  ".join(parts)
    col  = max(len(a.option_strings[-1]) for a in parser._actions
               if a.option_strings and a.dest != "help") + 2
    rows = []
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        val  = getattr(args, action.dest)
        flag = action.option_strings[-1]
        tag  = "  *" if val != action.default else ""
        rows.append(f"  {flag:<{col}} {val}{tag}")
    sep = "=" * 64
    return "\n".join([sep, f"{script} — resolved configuration",
                      f"invoked : {invoked}", sep, reproducible, sep,
                      "all flags  (* = non-default):", *rows, sep])


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    data_root: str = str(DATA_ROOT),
    max_epochs: int = 100,
    batch_size: int = 2,
    num_workers: int = 16,
    prefetch_factor: int = 2,
    token_stride: int = 8,
    token_dim: int = 256,
    num_decoder_layers: int = 6,
    num_heads: int = 8,
    learning_rate: float = 1e-4,
    depth_loss_fn: str = "l1",
    single_frame: bool = False,
    devices: int = 1,
    accelerator: str = "auto",
    gradient_clip_val: float = 1.0,
    log_dir: str = str(LOG_ROOT / "video_former_depth"),
    checkpoint_dir: str = str(CHECKPOINT_ROOT),
    patience: int = 10,
    trial_name: str = None,
    sequence_length: int = 2,
    img_h: int = 224,
    img_w: int = 224,
    precision: str = "32",
    val_check_interval: float = 0.5,
    limit_val_batches: float = 0.2,
    debug: bool = False,
    cli_command: str = "",
):
    transform = make_resize_transform(img_h, img_w) if (img_h > 0 and img_w > 0) else None
    train_dataset = Bench2DriveDataset(
        data_root, split="train", sequence_length=sequence_length,
        load_depth_as_label=True, load_instance=False, transform=transform)
    val_dataset = Bench2DriveDataset(
        data_root, split="val", sequence_length=sequence_length,
        load_depth_as_label=True, load_instance=False, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers,
        prefetch_factor=prefetch_factor, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers,
        prefetch_factor=prefetch_factor, shuffle=False, pin_memory=True)

    torch.set_float32_matmul_precision("medium")

    if debug:
        limit_train_batches = 0.01
        limit_val_batches   = 0.01
        debug_shapes        = True
        print("[debug] limit_train_batches=0.01, limit_val_batches=0.01, debug_shapes=True")
    else:
        limit_train_batches = 1.0
        debug_shapes        = False

    viz_rgb, viz_depth = collect_viz_clip(val_dataset, n_frames=16)

    img_size = img_h   # TinyViT uses square img_size
    model = VideoFormerDepthModule(
        token_stride=token_stride, token_dim=token_dim,
        num_decoder_layers=num_decoder_layers, num_heads=num_heads,
        learning_rate=learning_rate, depth_loss_fn=depth_loss_fn,
        single_frame=single_frame, cli_command=cli_command,
        viz_rgb=viz_rgb, viz_depth=viz_depth,
        img_h=img_h, img_w=img_w, img_size=img_size,
        debug_shapes=debug_shapes,
    )

    log_base_dir = Path(log_dir)
    if trial_name is None:
        existing   = list(log_base_dir.glob("trial_*"))
        trial_name = f"trial_{len(existing) + 1:05d}"

    trial_log_dir = log_base_dir / trial_name
    trial_log_dir.mkdir(parents=True, exist_ok=True)

    print(cli_command, flush=True)
    (trial_log_dir / "command.sh").write_text(cli_command + "\n")

    logger = TensorBoardLogger(save_dir=str(trial_log_dir), name="video_former_depth")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        accelerator=accelerator,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        val_check_interval=val_check_interval,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        callbacks=[
            ModelCheckpoint(dirpath=checkpoint_dir, filename="best-video-former-depth",
                            monitor="val/mse", mode="min", save_top_k=1,
                            auto_insert_metric_name=False),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val/mse", mode="min", patience=patience, verbose=True),
        ],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VideoFormerDepth: TinyViT + depth-token transformer (streaming)")
    parser.add_argument("--data-root",          type=str,   default=str(DATA_ROOT))
    parser.add_argument("--max-epochs",         type=int,   default=100)
    parser.add_argument("--batch-size",         type=int,   default=2)
    parser.add_argument("--num-workers",        type=int,   default=16)
    parser.add_argument("--prefetch-factor",    type=int,   default=2)
    parser.add_argument("--token-stride",       type=int,   default=8,
                        help="Depth token grid stride (token grid = H/stride × W/stride)")
    parser.add_argument("--token-dim",          type=int,   default=256,
                        help="Depth token feature dimension")
    parser.add_argument("--num-decoder-layers", type=int,   default=6)
    parser.add_argument("--num-heads",          type=int,   default=8)
    parser.add_argument("--learning-rate",      type=float, default=1e-4)
    parser.add_argument("--depth-loss-fn",      type=str,   default="l1",
                        choices=["l1", "mse", "smooth_l1"])
    parser.add_argument("--single-frame",       action="store_true",
                        help="Pre-training mode: no temporal token carry-over")
    parser.add_argument("--devices",            type=int,   default=1)
    parser.add_argument("--accelerator",        type=str,   default="auto")
    parser.add_argument("--gradient-clip-val",  type=float, default=1.0)
    parser.add_argument("--log-dir",            type=str,
                        default=str(LOG_ROOT / "video_former_depth"))
    parser.add_argument("--checkpoint-dir",     type=str,   default=str(CHECKPOINT_ROOT))
    parser.add_argument("--patience",           type=int,   default=10)
    parser.add_argument("--trial-name",         type=str,   default=None)
    parser.add_argument("--sequence-length",    type=int,   default=2)
    parser.add_argument("--img-h",              type=int,   default=224,
                        help="Input height — must be compatible with TinyViT window sizes")
    parser.add_argument("--img-w",              type=int,   default=224)
    parser.add_argument("--precision",          type=str,   default="32")
    parser.add_argument("--val-check-interval", type=float, default=0.5)
    parser.add_argument("--limit-val-batches",  type=float, default=0.2)
    parser.add_argument("--debug",              action="store_true",
                        help="1%% data, 1%% val, print tensor shapes")

    args = parser.parse_args()
    cli_command = _resolved_config(parser, args)

    train(
        data_root=args.data_root,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        token_stride=args.token_stride,
        token_dim=args.token_dim,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        learning_rate=args.learning_rate,
        depth_loss_fn=args.depth_loss_fn,
        single_frame=args.single_frame,
        devices=args.devices,
        accelerator=args.accelerator,
        gradient_clip_val=args.gradient_clip_val,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        patience=args.patience,
        trial_name=args.trial_name,
        sequence_length=args.sequence_length,
        img_h=args.img_h,
        img_w=args.img_w,
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        debug=args.debug,
        cli_command=cli_command,
    )
