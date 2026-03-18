"""
VideoFormerSegDepth — TinyViT + Parallel Depth & Segmentation Token Transformers
==================================================================================

Architecture
------------
  Encoder : TinyViT-21M (pretrained, frozen) → 4 multi-scale feature maps

  Two parallel token stacks share the same encoder features but have separate
  learnable token parameters, decoder layer stacks, and CNN output heads:

    • Depth tokens  → TokenCNNHead(out_channels=1)    → (B, 1, H, W)
    • Seg tokens    → TokenCNNHead(out_channels=NUM_CLS) → (B, NUM_CLS, H, W)

  Encoder projections (enc_proj_0..3) and position encodings are shared
  between the two decoder stacks.

Streaming / temporal mechanism
-------------------------------
  Both depth tokens and seg tokens carry over between frames independently.
  Frame 0: each uses its own learnable *_token_init parameter.
  Frame t: the enriched tokens from frame t-1 are passed forward.

Input  : (B, S, C, 3, H, W)   — B batch, S sequence frames, C cameras
Output : depth (B, S, C, 1, H, W),  sem (B, S, C, NUM_CLASSES, H, W)
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

from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

from dataset import Bench2DriveDataset, CAMERA_NAMES
from visualization import collect_viz_clip_joint, JointVizMixin, CARLA_CLASS_NAMES, save_joint_video
from config import DATA_ROOT, LOG_ROOT, CHECKPOINT_ROOT
from losses import SILogLoss, DiceLoss, abs_rel

NUM_CLASSES = 23  # CARLA semantic classes 0-22


# ---------------------------------------------------------------------------
# 2-D sinusoidal position encoding
# ---------------------------------------------------------------------------

def make_2d_sincos_pos_enc(H: int, W: int, dim: int) -> torch.Tensor:
    assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
    half = dim // 2

    def sincos_1d(n: int, d: int) -> torch.Tensor:
        pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d)
        )
        enc = torch.zeros(n, d)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div)
        return enc

    row_enc = sincos_1d(H, half)
    col_enc = sincos_1d(W, half)

    pe = torch.cat([
        row_enc.unsqueeze(1).expand(H, W, half),
        col_enc.unsqueeze(0).expand(H, W, half),
    ], dim=-1)
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
        tokens:    torch.Tensor,
        enc_feats: list,
        token_pos: torch.Tensor,
        enc_pos:   list,
    ) -> torch.Tensor:

        normed = self.norm1(tokens)
        q = self._add_pos(normed, token_pos)
        tokens = tokens + self.self_attn(q, q, normed)[0]

        normed = self.norm2(tokens)
        q = self._add_pos(normed, token_pos)
        cross_sum = torch.zeros_like(tokens)
        for i, ca in enumerate(self.cross_attns):
            k = self._add_pos(enc_feats[i], enc_pos[i])
            cross_sum = cross_sum + ca(q, k, enc_feats[i])[0]
        tokens = tokens + cross_sum

        tokens = tokens + self.ffn(self.norm3(tokens))

        return tokens


# ---------------------------------------------------------------------------
# Generalised CNN head  (token grid → full-resolution output)
# ---------------------------------------------------------------------------

class TokenCNNHead(nn.Module):
    """
    Progressive upsampler: (B, token_dim, H_q, W_q) → (B, out_channels, H, W).

    Uses log2(token_stride) ConvTranspose2d stages, each doubling spatial
    resolution and halving channel count (floor at 32).
    """

    def __init__(self, token_dim: int, token_stride: int, out_channels: int = 1):
        super().__init__()
        assert (token_stride & (token_stride - 1)) == 0, \
            "token_stride must be a power of 2"
        n_ups = int(math.log2(token_stride))

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
        layers.append(nn.Conv2d(dims[-1], out_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class VideoFormerSegDepth(nn.Module):
    """
    TinyViT encoder (frozen) + parallel depth-token and seg-token transformer decoders.

    forward(x, prev_depth_tokens=None, prev_seg_tokens=None):
        x               : (B, 3, H, W)   — one frame; B = batch_size × num_cameras
        prev_depth_tokens : (B, N_q, D) or None
        prev_seg_tokens   : (B, N_q, D) or None

    Returns:
        depth      : (B, 1, H, W)
        sem        : (B, NUM_CLASSES, H, W)
        depth_tokens : (B, N_q, D)   — enriched; pass as prev_depth_tokens next frame
        seg_tokens   : (B, N_q, D)   — enriched; pass as prev_seg_tokens next frame
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
        img_size: int = 224,
        debug_shapes: bool = False,
    ):
        super().__init__()
        self.token_stride = token_stride
        self.token_dim    = token_dim
        self.num_classes  = num_classes
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

        # ── Project encoder levels → token_dim (shared by both decoders) ───
        for lvl, enc_dim in enumerate([96, 192, 384, 576]):
            self.add_module(f"enc_proj_{lvl}", nn.Linear(enc_dim, token_dim))

        # ── Fixed 2D sinusoidal position encodings (shared) ─────────────────
        self.register_buffer("token_pos_enc",
            make_2d_sincos_pos_enc(self.H_q, self.W_q, token_dim).unsqueeze(0))

        H4, W4 = img_h // 4, img_w // 4
        enc_grids = [(H4, W4), (H4 // 2, W4 // 2), (H4 // 4, W4 // 4), (H4 // 8, W4 // 8)]
        for lvl, (H, W) in enumerate(enc_grids):
            self.register_buffer(f"enc_pos_{lvl}",
                make_2d_sincos_pos_enc(H, W, token_dim).unsqueeze(0))

        # ── Learnable depth token initialisation ────────────────────────────
        self.depth_token_init = nn.Parameter(torch.zeros(1, self.N_q, token_dim))
        nn.init.trunc_normal_(self.depth_token_init, std=0.02)

        # ── Learnable seg token initialisation ──────────────────────────────
        self.seg_token_init = nn.Parameter(torch.zeros(1, self.N_q, token_dim))
        nn.init.trunc_normal_(self.seg_token_init, std=0.02)

        # ── Depth transformer decoder layers ────────────────────────────────
        self.depth_decoder_layers = nn.ModuleList([
            DepthDecoderLayer(token_dim, num_heads)
            for _ in range(num_decoder_layers)
        ])

        # ── Seg transformer decoder layers (separate weights) ───────────────
        self.seg_decoder_layers = nn.ModuleList([
            DepthDecoderLayer(token_dim, num_heads)
            for _ in range(num_decoder_layers)
        ])

        # ── Output heads ────────────────────────────────────────────────────
        self.depth_head = TokenCNNHead(token_dim, token_stride, out_channels=1)
        self.seg_head   = TokenCNNHead(token_dim, token_stride, out_channels=num_classes)

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor):
        """Run frozen TinyViT encoder; return 4 spatial feature maps."""
        x = self.patch_embed(x)
        H4, W4 = x.shape[-2], x.shape[-1]

        for blk in self.enc_layers[0].blocks:
            x = blk(x)
        skip0 = x
        x = self.enc_layers[0].downsample(x)

        for blk in self.enc_layers[1].blocks:
            x = blk(x)
        skip1_tok = x
        x = self.enc_layers[1].downsample(x)

        for blk in self.enc_layers[2].blocks:
            x = blk(x)
        skip2_tok = x
        x = self.enc_layers[2].downsample(x)

        for blk in self.enc_layers[3].blocks:
            x = blk(x)
        bot_tok = x

        N = x.shape[0]
        H8,  W8  = H4 // 2, W4 // 2
        H16, W16 = H4 // 4, W4 // 4
        H32, W32 = H4 // 8, W4 // 8

        skip1 = skip1_tok.view(N, H8,  W8,  192).permute(0, 3, 1, 2).contiguous()
        skip2 = skip2_tok.view(N, H16, W16, 384).permute(0, 3, 1, 2).contiguous()
        bot   = bot_tok  .view(N, H32, W32, 576).permute(0, 3, 1, 2).contiguous()

        return skip0, skip1, skip2, bot

    # ------------------------------------------------------------------
    def _log_shapes(self, x, depth, sem):
        lines = [
            "=" * 64,
            "VideoFormerSegDepth — tensor shapes (first forward pass)",
            "=" * 64,
            f"  input  (B,3,H,W)               : {tuple(x.shape)}",
            f"  depth  (B,1,H,W)               : {tuple(depth.shape)}",
            f"  sem    (B,NUM_CLASSES,H,W)      : {tuple(sem.shape)}",
            "=" * 64,
        ]
        table = "\n".join(lines)
        print(table, flush=True)
        self._shapes_table = table

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, prev_depth_tokens=None, prev_seg_tokens=None):
        """
        Args:
            x               : (B, 3, H, W)
            prev_depth_tokens : (B, N_q, D) or None
            prev_seg_tokens   : (B, N_q, D) or None

        Returns:
            depth        : (B, 1, H, W)
            sem          : (B, NUM_CLASSES, H, W)
            depth_tokens : (B, N_q, D)
            seg_tokens   : (B, N_q, D)
        """
        B = x.shape[0]

        with torch.no_grad():
            skip0, skip1, skip2, bot = self._encode(x)

        # Project encoder features to token_dim (shared)
        enc_feats = [
            self.enc_proj_0(skip0.flatten(2).transpose(1, 2)),
            self.enc_proj_1(skip1.flatten(2).transpose(1, 2)),
            self.enc_proj_2(skip2.flatten(2).transpose(1, 2)),
            self.enc_proj_3(bot  .flatten(2).transpose(1, 2)),
        ]
        enc_pos = [getattr(self, f"enc_pos_{i}") for i in range(4)]

        # ── Depth branch ────────────────────────────────────────────────────
        depth_tokens = self.depth_token_init.expand(B, -1, -1) \
                       if prev_depth_tokens is None else prev_depth_tokens
        for layer in self.depth_decoder_layers:
            depth_tokens = layer(depth_tokens, enc_feats, self.token_pos_enc, enc_pos)

        depth_spatial = depth_tokens.transpose(1, 2).view(
            B, self.token_dim, self.H_q, self.W_q)
        depth = self.depth_head(depth_spatial)  # (B, 1, H, W)

        # ── Seg branch ──────────────────────────────────────────────────────
        seg_tokens = self.seg_token_init.expand(B, -1, -1) \
                     if prev_seg_tokens is None else prev_seg_tokens
        for layer in self.seg_decoder_layers:
            seg_tokens = layer(seg_tokens, enc_feats, self.token_pos_enc, enc_pos)

        seg_spatial = seg_tokens.transpose(1, 2).view(
            B, self.token_dim, self.H_q, self.W_q)
        sem = self.seg_head(seg_spatial)  # (B, NUM_CLASSES, H, W)

        if self.debug_shapes and not self._shapes_logged:
            self._log_shapes(x, depth, sem)
            self._shapes_logged = True

        return depth, sem, depth_tokens, seg_tokens


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class VideoFormerSegDepthModule(JointVizMixin, pl.LightningModule):
    """
    Lightning wrapper for VideoFormerSegDepth.

    Inherits JointVizMixin for joint depth+seg visualisation.
    Overrides save_best_video to use streaming token carry-over.

    forward(x) returns (depth, sem) — compatible with JointVizMixin's
    save_best_val_image and save_train_image (frame-by-frame, no carry-over).
    """

    def __init__(
        self,
        token_stride: int = 8,
        token_dim: int = 256,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_classes: int = NUM_CLASSES,
        learning_rate: float = 1e-4,
        depth_loss_fn: str = "silog",
        depth_weight: float = 1.0,
        sem_weight: float = 1.0,
        single_frame: bool = False,
        cli_command: str = "",
        viz_rgb=None,
        viz_depth=None,
        viz_sem=None,
        train_viz_rgb=None,
        train_viz_depth=None,
        train_viz_sem=None,
        img_h: int = 224,
        img_w: int = 224,
        img_size: int = 224,
        debug_shapes: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "viz_rgb", "viz_depth", "viz_sem",
            "train_viz_rgb", "train_viz_depth", "train_viz_sem",
        ])

        self.model = VideoFormerSegDepth(
            token_stride=token_stride, token_dim=token_dim,
            num_decoder_layers=num_decoder_layers, num_heads=num_heads,
            num_classes=num_classes, img_h=img_h, img_w=img_w, img_size=img_size,
            debug_shapes=debug_shapes,
        )
        self.learning_rate = learning_rate
        self.depth_weight  = depth_weight
        self.sem_weight    = sem_weight
        self.single_frame  = single_frame
        self.cli_command   = cli_command
        self.best_val_loss = float("inf")
        self.num_classes   = num_classes

        self.setup_viz(viz_rgb, viz_depth, viz_sem)
        self.setup_train_viz(train_viz_rgb, train_viz_depth, train_viz_sem)

        if depth_loss_fn == "smooth_l1":
            self.depth_loss_fn = nn.SmoothL1Loss()
        elif depth_loss_fn == "silog":
            self.depth_loss_fn = SILogLoss()
        else:
            self.depth_loss_fn = nn.L1Loss()
        self.silog_metric      = SILogLoss()
        self.dice_loss         = DiceLoss()
        self.val_miou          = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
        self.val_macc          = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_iou_per_class = MulticlassJaccardIndex(num_classes=num_classes, average="none")

    # ------------------------------------------------------------------
    def on_train_start(self):
        if self.cli_command:
            self.logger.experiment.add_text("cli_command", self.cli_command, global_step=0)
        mode = "single-frame (pre-training)" if self.single_frame else "streaming"
        self.logger.experiment.add_text("training_mode", mode, global_step=0)

    # ------------------------------------------------------------------
    def forward(self, x):
        """x: (B, S, C, 3, H, W) → (depth (B,S,C,1,H,W), sem (B,S,C,NUM_CLS,H,W))

        Frame-by-frame without token carry-over (compatible with JointVizMixin).
        """
        B, S, C = x.shape[:3]
        depth_frames, sem_frames = [], []
        for s in range(S):
            x_s = rearrange(x[:, s], 'b c ch h w -> (b c) ch h w')
            depth_s, sem_s, _, _ = self.model(x_s, None, None)
            depth_frames.append(
                rearrange(depth_s, '(b c) 1 h w -> b c 1 h w', b=B, c=C))
            sem_frames.append(
                rearrange(sem_s, '(b c) cls h w -> b c cls h w', b=B, c=C))
        depth = torch.stack(depth_frames, dim=1)  # (B, S, C, 1, H, W)
        sem   = torch.stack(sem_frames,   dim=1)  # (B, S, C, NUM_CLS, H, W)
        return depth, sem

    # ------------------------------------------------------------------
    def _step(self, batch, log_shapes: bool = False):
        rgb      = batch["rgb"]    # (B, S, C, 3, H, W)
        depth_gt = batch["depth"]  # (B, S, C, 1, H, W)
        gt_sem   = batch.get("instance_class", None)
        B, S, C = rgb.shape[:3]

        if self.single_frame:
            x_flat = rearrange(rgb, 'b s c ch h w -> (b s c) ch h w')
            depth_flat, sem_flat, _, _ = self.model(x_flat, None, None)
            depth_pred = rearrange(depth_flat, '(b s c) 1 h w -> b s c 1 h w',
                                   b=B, s=S, c=C)
            sem_pred   = rearrange(sem_flat, '(b s c) cls h w -> b s c cls h w',
                                   b=B, s=S, c=C)
        else:
            prev_depth_tokens = None
            prev_seg_tokens   = None
            depth_frames, sem_frames = [], []
            for s in range(S):
                x_s = rearrange(rgb[:, s], 'b c ch h w -> (b c) ch h w')
                depth_s, sem_s, prev_depth_tokens, prev_seg_tokens = \
                    self.model(x_s, prev_depth_tokens, prev_seg_tokens)
                depth_frames.append(
                    rearrange(depth_s, '(b c) 1 h w -> b c 1 h w', b=B, c=C))
                sem_frames.append(
                    rearrange(sem_s, '(b c) cls h w -> b c cls h w', b=B, c=C))
            depth_pred = torch.stack(depth_frames, dim=1)
            sem_pred   = torch.stack(sem_frames,   dim=1)

        if log_shapes and hasattr(self.model, "_shapes_table"):
            self.logger.experiment.add_text(
                "debug/shapes", self.model._shapes_table, global_step=0)
            del self.model._shapes_table

        l_depth = self.depth_loss_fn(depth_pred, depth_gt)
        loss    = self.depth_weight * l_depth
        l_sem   = rgb.new_zeros(1).squeeze()

        if gt_sem is not None and self.sem_weight > 0:
            sem_flat = rearrange(sem_pred, 'b s c cls h w -> (b s c) cls h w')
            gt_flat  = rearrange(gt_sem,  'b s c 1 h w -> (b s c) h w').long()
            gt_flat  = gt_flat.clamp(0, self.num_classes - 1)
            l_sem    = F.cross_entropy(sem_flat, gt_flat) + \
                       0.5 * self.dice_loss(sem_flat, gt_flat)
            loss     = loss + self.sem_weight * l_sem

        return loss, l_depth, l_sem, depth_pred, sem_pred

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        log_shapes = (batch_idx == 0 and self.current_epoch == 0)
        loss, l_depth, l_sem, _, _ = self._step(batch, log_shapes=log_shapes)
        self.log("train/loss",       loss,    prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/loss_depth", l_depth,                on_step=False, on_epoch=True)
        self.log("train/loss_sem",   l_sem,                  on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, l_depth, l_sem, depth_pred, sem_pred = self._step(batch)
        gt = batch["depth"]
        self.log("val/loss",       loss,                                   prog_bar=True, sync_dist=True)
        self.log("val/loss_depth", l_depth,                                               sync_dist=True)
        self.log("val/loss_sem",   l_sem,                                                 sync_dist=True)
        self.log("val/silog",      self.silog_metric(depth_pred, gt),                     sync_dist=True)
        self.log("val/abs_rel",    abs_rel(depth_pred, gt),                prog_bar=True, sync_dist=True)
        self.log("val/mse",        F.mse_loss(depth_pred, gt),                            sync_dist=True)

        gt_sem = batch.get("instance_class")
        if gt_sem is not None:
            sem_flat = rearrange(sem_pred, 'b s c cls h w -> (b s c) cls h w')
            gt_flat  = rearrange(gt_sem,  'b s c 1 h w -> (b s c) h w').long()
            gt_flat  = gt_flat.clamp(0, self.num_classes - 1)
            preds    = sem_flat.argmax(dim=1)
            self.val_miou(preds, gt_flat)
            self.val_macc(preds, gt_flat)
            self.val_iou_per_class(preds, gt_flat)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/miou", self.val_miou.compute(), prog_bar=True)
        self.log("val/macc", self.val_macc.compute())
        for c, iou_val in enumerate(self.val_iou_per_class.compute()):
            self.log(f"val/iou_{CARLA_CLASS_NAMES[c]}", iou_val)
        self.val_miou.reset()
        self.val_macc.reset()
        self.val_iou_per_class.reset()

        epoch_val_loss = self.trainer.callback_metrics.get("val/abs_rel")
        if epoch_val_loss is None:
            return
        if epoch_val_loss.item() < self.best_val_loss:
            self.best_val_loss = epoch_val_loss.item()
            self.save_best_val_image()
            self.save_best_video()

    def on_train_epoch_end(self):
        self.save_train_image()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def save_best_video(self):
        """Run model in STREAMING mode with token carry-over over the viz clip."""
        if getattr(self, "_viz_rgb", None) is None:
            return

        T    = self._viz_rgb.shape[1]
        cams = self._viz_rgb.shape[2]
        B    = 1
        prev_depth_tokens = None
        prev_seg_tokens   = None
        depth_frames, sem_frames = [], []

        for t in range(T):
            frame = self._viz_rgb[:, t].to(self.device)       # (1, cams, 3, H, W)
            x_t   = rearrange(frame, 'b c ch h w -> (b c) ch h w')  # (cams, 3, H, W)
            depth_t, sem_t, prev_depth_tokens, prev_seg_tokens = \
                self.model(x_t, prev_depth_tokens, prev_seg_tokens)
            depth_frames.append(
                rearrange(depth_t, '(b c) 1 h w -> b 1 c 1 h w', b=B, c=cams).cpu())
            sem_frames.append(
                rearrange(sem_t, '(b c) cls h w -> b 1 c cls h w', b=B, c=cams).cpu())

        depth_vid = torch.cat(depth_frames, dim=1)  # (1, T, cams, 1, H, W)
        sem_vid   = torch.cat(sem_frames,   dim=1)  # (1, T, cams, NUM_CLS, H, W)

        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        save_joint_video(
            self._viz_rgb,
            depth_vid, self._viz_depth,
            sem_vid,   self._viz_sem,
            log_dir / "best_joint.mp4",
        )

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# Resize transform
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
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 1e-4,
    depth_loss_fn: str = "silog",
    depth_weight: float = 1.0,
    sem_weight: float = 1.0,
    single_frame: bool = False,
    devices: int = 1,
    accelerator: str = "auto",
    gradient_clip_val: float = 1.0,
    log_dir: str = str(LOG_ROOT / "video_former_seg_depth"),
    checkpoint_dir: str = str(CHECKPOINT_ROOT),
    patience: int = 10,
    trial_name: str = None,
    sequence_length: int = 2,
    img_h: int = 224,
    img_w: int = 224,
    precision: str = "32",
    val_check_every_n_epochs: int = 5,
    limit_val_batches: float = 0.1,
    limit_train_batches: int = 500,
    debug: bool = False,
    cli_command: str = "",
):
    transform = make_resize_transform(img_h, img_w) if (img_h > 0 and img_w > 0) else None
    train_dataset = Bench2DriveDataset(
        data_root, split="train", sequence_length=sequence_length,
        load_depth_as_label=True, load_instance=True, transform=transform)
    val_dataset = Bench2DriveDataset(
        data_root, split="val", sequence_length=sequence_length,
        load_depth_as_label=True, load_instance=True, transform=transform)

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
        debug_shapes        = False

    viz_rgb, viz_depth, viz_sem, _ = collect_viz_clip_joint(val_dataset,   n_frames=16)
    train_viz_rgb, train_viz_depth, train_viz_sem, _ = collect_viz_clip_joint(train_dataset, n_frames=16)

    img_size = img_h  # TinyViT uses square img_size
    model = VideoFormerSegDepthModule(
        token_stride=token_stride, token_dim=token_dim,
        num_decoder_layers=num_decoder_layers, num_heads=num_heads,
        num_classes=num_classes,
        learning_rate=learning_rate, depth_loss_fn=depth_loss_fn,
        depth_weight=depth_weight, sem_weight=sem_weight,
        single_frame=single_frame, cli_command=cli_command,
        viz_rgb=viz_rgb, viz_depth=viz_depth, viz_sem=viz_sem,
        train_viz_rgb=train_viz_rgb, train_viz_depth=train_viz_depth, train_viz_sem=train_viz_sem,
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

    logger = TensorBoardLogger(save_dir=str(trial_log_dir), name="video_former_seg_depth")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        accelerator=accelerator,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        check_val_every_n_epoch=val_check_every_n_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        callbacks=[
            ModelCheckpoint(dirpath=checkpoint_dir, filename="best-video-former-seg-depth",
                            monitor="val/abs_rel", mode="min", save_top_k=1,
                            auto_insert_metric_name=False),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val/abs_rel", mode="min", patience=patience, verbose=True),
        ],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VideoFormerSegDepth: TinyViT + parallel depth & seg token transformers (streaming)")
    parser.add_argument("--data-root",           type=str,   default=str(DATA_ROOT))
    parser.add_argument("--max-epochs",          type=int,   default=100)
    parser.add_argument("--batch-size",          type=int,   default=2)
    parser.add_argument("--num-workers",         type=int,   default=16)
    parser.add_argument("--prefetch-factor",     type=int,   default=2)
    parser.add_argument("--token-stride",        type=int,   default=8)
    parser.add_argument("--token-dim",           type=int,   default=256)
    parser.add_argument("--num-decoder-layers",  type=int,   default=6)
    parser.add_argument("--num-heads",           type=int,   default=8)
    parser.add_argument("--num-classes",         type=int,   default=NUM_CLASSES)
    parser.add_argument("--learning-rate",       type=float, default=1e-4)
    parser.add_argument("--depth-loss-fn",       type=str,   default="silog",
                        choices=["l1", "smooth_l1", "silog"])
    parser.add_argument("--depth-weight",        type=float, default=1.0)
    parser.add_argument("--sem-weight",          type=float, default=1.0)
    parser.add_argument("--single-frame",        action="store_true",
                        help="Pre-training mode: no temporal token carry-over")
    parser.add_argument("--devices",             type=int,   default=1)
    parser.add_argument("--accelerator",         type=str,   default="auto")
    parser.add_argument("--gradient-clip-val",   type=float, default=1.0)
    parser.add_argument("--log-dir",             type=str,
                        default=str(LOG_ROOT / "video_former_seg_depth"))
    parser.add_argument("--checkpoint-dir",      type=str,   default=str(CHECKPOINT_ROOT))
    parser.add_argument("--patience",            type=int,   default=10)
    parser.add_argument("--trial-name",          type=str,   default=None)
    parser.add_argument("--sequence-length",     type=int,   default=2)
    parser.add_argument("--img-h",               type=int,   default=224)
    parser.add_argument("--img-w",               type=int,   default=224)
    parser.add_argument("--precision",           type=str,   default="32")
    parser.add_argument("--val-check-interval",  type=int,   default=5)
    parser.add_argument("--limit-val-batches",   type=float, default=0.1)
    parser.add_argument("--limit-train-batches", type=int,   default=500)
    parser.add_argument("--debug",               action="store_true",
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
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        depth_loss_fn=args.depth_loss_fn,
        depth_weight=args.depth_weight,
        sem_weight=args.sem_weight,
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
        val_check_every_n_epochs=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        limit_train_batches=args.limit_train_batches,
        debug=args.debug,
        cli_command=cli_command,
    )
