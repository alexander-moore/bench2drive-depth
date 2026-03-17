"""
TinyViT + LSTM Video Joint Depth + Semantic Segmentation
=========================================================
Architecture:
  Encoder : TinyViT-21M (pretrained, frozen) — 4-stage hierarchical ViT
  Temporal: LSTM at the bottleneck — allows information to flow across S frames
  Decoder : ConvBlock UNet decoder adapted for TinyViT skip-channel sizes

Input :  (B, S, C, 3, H, W)   — B batch, S sequence frames, C cameras
Outputs: depth (B,S,C,1,H,W) and semantic logits (B,S,C,23,H,W)

Requirements
  pip install timm yacs termcolor
  git clone https://github.com/wkcn/TinyViT.git /workspace/TinyViT

Image size must be compatible with TinyViT window sizes:
  Recommended: 224×224 (pass --img-h 224 --img-w 224)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
# PROJECT_ROOT first so our config.py takes priority over TinyViT's config.py
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
from visualization import collect_viz_clip_joint, JointVizMixin
from config import DATA_ROOT, LOG_ROOT, CHECKPOINT_ROOT

NUM_CLASSES = 23  # CARLA semantic classes 0-22


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VideoSegDepthUNet(nn.Module):
    """
    TinyViT-21M encoder + LSTM bottleneck + UNet decoder.

    forward(x) expects x: (B, S, C, 3, H, W)
    Returns:
        depth : (B, S, C, 1, H, W)
        sem   : (B, S, C, NUM_CLASSES, H, W)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, lstm_hidden: int = 512,
                 img_size: int = 224, debug_shapes: bool = False):
        super().__init__()
        self.debug_shapes   = debug_shapes
        self._shapes_logged = False   # fire only on the first forward pass

        # ---- Encoder: TinyViT-21M (pretrained, frozen) --------------------
        from models.tiny_vit import tiny_vit_21m_224
        try:
            backbone = tiny_vit_21m_224(pretrained=True, img_size=img_size)
        except Exception:
            backbone = tiny_vit_21m_224(pretrained=False, img_size=img_size)

        self.patch_embed = backbone.patch_embed      # stride-4 conv → (B, 96, H/4, W/4)
        self.enc_layers  = backbone.layers           # 4 stages

        # Freeze encoder
        for p in list(self.patch_embed.parameters()) + list(self.enc_layers.parameters()):
            p.requires_grad_(False)

        # ---- Temporal LSTM at bottleneck (dim=576 from layer[3]) ----------
        self.temporal_lstm = nn.LSTM(576, lstm_hidden, batch_first=True)

        # ---- Decoder (channel sizes tuned to TinyViT skip dims) -----------
        # up4 receives lstm_hidden from bottleneck, then cat with skip2 (384 ch)
        self.up4  = nn.ConvTranspose2d(lstm_hidden, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512 + 384, 512)

        # up3 receives 512, cat with skip1 (192 ch)
        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256 + 192, 256)

        # up2 receives 256, cat with skip0 (96 ch)
        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 + 96, 128)

        # up1: H/4 → H/2, no skip (TinyViT starts at H/4)
        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 64)

        # up0: H/2 → H, no skip
        self.up0  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(32, 32)

        self.depth_head    = nn.Conv2d(32, 1,           kernel_size=1)
        self.semantic_head = nn.Conv2d(32, num_classes, kernel_size=1)

    # ------------------------------------------------------------------
    # Encoder helpers
    # ------------------------------------------------------------------

    def _encode(self, x):
        """Run TinyViT encoder while capturing skip features.

        Args:
            x: (N, 3, H, W)  where N = B*S*C

        Returns:
            skip0      : (N,  96, H/4,  W/4)   spatial feature map
            skip1      : (N, 192, H/8,  W/8)   spatial
            skip2      : (N, 384, H/16, W/16)  spatial
            bottleneck : (N, 576, H/32, W/32)  spatial
        """
        # patch embed: stride-4 conv → spatial feature map
        x = self.patch_embed(x)                      # (N, 96, H4, W4)
        H4, W4 = x.shape[-2], x.shape[-1]

        # ---- layer 0: ConvLayer (MBConv blocks work on spatial tensors) ---
        for blk in self.enc_layers[0].blocks:
            x = blk(x)
        skip0 = x                                    # (N, 96, H4, W4)
        x = self.enc_layers[0].downsample(x)         # (N, H4/2*W4/2, 192) tokens

        # ---- layer 1: BasicLayer (window attention on tokens) -------------
        for blk in self.enc_layers[1].blocks:
            x = blk(x)
        skip1_tokens = x                             # (N, H8*W8, 192)
        x = self.enc_layers[1].downsample(x)         # (N, H16*W16, 384) tokens

        # ---- layer 2 ------------------------------------------------------
        for blk in self.enc_layers[2].blocks:
            x = blk(x)
        skip2_tokens = x                             # (N, H16*W16, 384)
        x = self.enc_layers[2].downsample(x)         # (N, H32*W32, 576) tokens

        # ---- layer 3 (no downsample) --------------------------------------
        for blk in self.enc_layers[3].blocks:
            x = blk(x)
        bot_tokens = x                               # (N, H32*W32, 576)

        # Reshape tokens → spatial feature maps
        N = x.shape[0]
        H8,  W8  = H4 // 2, W4 // 2
        H16, W16 = H4 // 4, W4 // 4
        H32, W32 = H4 // 8, W4 // 8

        skip1 = skip1_tokens.view(N, H8,  W8,  192).permute(0, 3, 1, 2).contiguous()
        skip2 = skip2_tokens.view(N, H16, W16, 384).permute(0, 3, 1, 2).contiguous()
        bot   = bot_tokens.view(  N, H32, W32, 576).permute(0, 3, 1, 2).contiguous()

        return skip0, skip1, skip2, bot

    # ------------------------------------------------------------------
    # LSTM temporal mixing
    # ------------------------------------------------------------------

    def _apply_lstm(self, bot, B, S, C):
        """Apply LSTM across the sequence dimension at each spatial location.

        Args:
            bot: (B*S*C, 576, h, w)
            B, S, C: original batch / sequence / camera dims

        Returns:
            (B*S*C, lstm_hidden, h, w)
        """
        _, _, h, w = bot.shape

        # (B*S*C, 576, h, w) → (B, S, C, 576, h, w)
        x = bot.view(B, S, C, 576, h, w)

        # Expose spatial locations in the batch axis so LSTM sees each
        # (B, C, h, w) location independently across S frames.
        # → (B, C, h, w, S, 576) → (B*C*h*w, S, 576)
        x = x.permute(0, 2, 3, 4, 1, 5).contiguous()
        x = x.view(B * C * h * w, S, 576)

        lstm_out, _ = self.temporal_lstm(x)          # (B*C*h*w, S, lstm_hidden)

        lstm_hidden = lstm_out.shape[-1]
        # → (B, C, h, w, S, lstm_hidden) → (B, S, C, lstm_hidden, h, w)
        x = lstm_out.view(B, C, h, w, S, lstm_hidden)
        x = x.permute(0, 4, 1, 5, 2, 3).contiguous()

        # → (B*S*C, lstm_hidden, h, w)
        x = x.view(B * S * C, lstm_hidden, h, w)
        return x

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def _decode(self, bot_lstm, skip2, skip1, skip0):
        """UNet decoder from bottleneck → full resolution.

        All inputs have leading dim N = B*S*C.
        Returns: depth (N, 1, H, W), sem (N, num_classes, H, W)
        """
        def up_cat(up, x, skip):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
            return torch.cat([x, skip], dim=1)

        d4 = self.dec4(up_cat(self.up4, bot_lstm, skip2))   # H/16
        d3 = self.dec3(up_cat(self.up3, d4,       skip1))   # H/8
        d2 = self.dec2(up_cat(self.up2, d3,       skip0))   # H/4

        d1 = self.dec1(self.up1(d2))   # H/2
        d0 = self.dec0(self.up0(d1))   # H

        return self.depth_head(d0), self.semantic_head(d0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _log_shapes(self, x_input, x_flat, skip0, skip1, skip2, bot,
                    bot_lstm, depth, sem):
        """Print a shape table to stdout and store it for TensorBoard logging."""
        lines = [
            "=" * 60,
            "VideoSegDepthUNet — tensor shapes (first forward pass)",
            "=" * 60,
            f"  input           (B,S,C,3,H,W)   : {tuple(x_input.shape)}",
            f"  x_flat          (N,3,H,W)        : {tuple(x_flat.shape)}",
            "  --- encoder (frozen TinyViT) ---",
            f"  skip0           (N, 96,H/4,W/4)  : {tuple(skip0.shape)}",
            f"  skip1           (N,192,H/8,W/8)  : {tuple(skip1.shape)}",
            f"  skip2           (N,384,H/16,W/16): {tuple(skip2.shape)}",
            f"  bottleneck      (N,576,H/32,W/32): {tuple(bot.shape)}",
            "  --- lstm ---",
            f"  bot_lstm        (N,hid,H/32,W/32): {tuple(bot_lstm.shape)}",
            "  --- decoder outputs (pre-reshape) ---",
            f"  depth (flat)    (N,1,H,W)        : {tuple(depth.shape)}",
            f"  sem   (flat)    (N,cls,H,W)      : {tuple(sem.shape)}",
            "=" * 60,
        ]
        table = "\n".join(lines)
        print(table, flush=True)
        self._shapes_table = table   # picked up by the Lightning module for TensorBoard

    def forward(self, x):
        """
        Args:
            x: (B, S, C, 3, H, W)
        Returns:
            depth : (B, S, C, 1, H, W)
            sem   : (B, S, C, NUM_CLASSES, H, W)
        """
        B, S, C = x.shape[:3]
        N = B * S * C

        # Flatten all frames into the batch dimension
        x_flat = rearrange(x, 'b s c ch h w -> (b s c) ch h w')

        # Encode (frozen TinyViT)
        with torch.no_grad():
            skip0, skip1, skip2, bot = self._encode(x_flat)

        # Temporal LSTM at bottleneck
        bot_lstm = self._apply_lstm(bot, B, S, C)      # (N, lstm_hidden, h, w)

        # Decode
        depth, sem = self._decode(bot_lstm, skip2, skip1, skip0)

        if self.debug_shapes and not self._shapes_logged:
            self._log_shapes(x, x_flat, skip0, skip1, skip2, bot, bot_lstm, depth, sem)
            self._shapes_logged = True

        # Reshape outputs back to (B, S, C, *, H, W)
        depth = rearrange(depth, '(b s c) 1 h w -> b s c 1 h w',   b=B, s=S, c=C)
        sem   = rearrange(sem,   '(b s c) cls h w -> b s c cls h w', b=B, s=S, c=C)
        return depth, sem


# ---------------------------------------------------------------------------
# Lightning module (mirrors BaselineSegDepthModule with model swapped)
# ---------------------------------------------------------------------------

class VideoSegDepthModule(JointVizMixin, pl.LightningModule):
    def __init__(
        self,
        lstm_hidden: int = 512,
        learning_rate: float = 1e-4,
        depth_loss_fn: str = "l1",
        depth_weight: float = 1.0,
        sem_weight: float = 1.0,
        cli_command: str = "",
        viz_rgb=None,
        viz_depth=None,
        viz_sem=None,
        img_size: int = 224,
        debug_shapes: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["viz_rgb", "viz_depth", "viz_sem"])

        self.model         = VideoSegDepthUNet(num_classes=NUM_CLASSES,
                                               lstm_hidden=lstm_hidden,
                                               img_size=img_size,
                                               debug_shapes=debug_shapes)
        self.learning_rate = learning_rate
        self.depth_weight  = depth_weight
        self.sem_weight    = sem_weight
        self.best_val_loss = float("inf")
        self.cli_command   = cli_command
        self.setup_viz(viz_rgb, viz_depth, viz_sem)

        if depth_loss_fn == "l1":
            self.depth_loss_fn = nn.L1Loss()
        elif depth_loss_fn == "mse":
            self.depth_loss_fn = nn.MSELoss()
        else:
            self.depth_loss_fn = nn.SmoothL1Loss()

    def on_train_start(self):
        if self.cli_command:
            self.logger.experiment.add_text("cli_command", self.cli_command,
                                            global_step=0)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, log_shapes: bool = False):
        rgb      = batch["rgb"]
        depth_gt = batch["depth"]
        gt_sem   = batch.get("instance_class", None)

        depth_pred, sem_pred = self.model(rgb)

        # After the very first forward the model stores a shapes table; log it.
        if log_shapes and hasattr(self.model, "_shapes_table"):
            self.logger.experiment.add_text(
                "debug/shapes", self.model._shapes_table, global_step=0)
            del self.model._shapes_table   # only log once

        l_depth = self.depth_loss_fn(depth_pred, depth_gt)
        loss    = self.depth_weight * l_depth
        l_sem   = rgb.new_zeros(1).squeeze()

        if gt_sem is not None and self.sem_weight > 0:
            sem_flat = rearrange(sem_pred, 'b s c cls h w -> (b s c) cls h w')
            gt_flat  = rearrange(gt_sem,  'b s c 1 h w -> (b s c) h w').long()
            gt_flat  = gt_flat.clamp(0, NUM_CLASSES - 1)
            l_sem    = F.cross_entropy(sem_flat, gt_flat)
            loss     = loss + self.sem_weight * l_sem

        return loss, l_depth, l_sem, depth_pred, sem_pred

    def training_step(self, batch, batch_idx):
        log_shapes = (batch_idx == 0 and self.current_epoch == 0)
        loss, l_depth, l_sem, _, _ = self._step(batch, log_shapes=log_shapes)
        self.log("train/loss",       loss,    prog_bar=True)
        self.log("train/loss_depth", l_depth)
        self.log("train/loss_sem",   l_sem)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, l_depth, l_sem, depth_pred, sem_pred = self._step(batch)
        self.log("val/loss",       loss,    prog_bar=True, sync_dist=True)
        self.log("val/loss_depth", l_depth, sync_dist=True)
        self.log("val/loss_sem",   l_sem,   sync_dist=True)
        mse = F.mse_loss(depth_pred, batch["depth"])
        self.log("val/mse", mse, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self.save_validation_image(
                batch["rgb"], depth_pred, batch["depth"],
                sem_pred, batch.get("instance_class", None),
            )

        return loss

    def on_validation_epoch_end(self):
        epoch_val_mse = self.trainer.callback_metrics.get("val/mse")
        if epoch_val_mse is None:
            return
        val_mse = epoch_val_mse.item()
        if val_mse < self.best_val_loss:
            self.best_val_loss = val_mse
            self.save_best_video()

    def configure_optimizers(self):
        # Only train unfrozen parameters (LSTM + decoder)
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer  = torch.optim.AdamW(trainable, lr=self.learning_rate,
                                       weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=100)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# Resize transform (identical to baseline)
# ---------------------------------------------------------------------------

def make_resize_transform(img_h: int, img_w: int):
    def transform(sample):
        def resize_tensor(t, mode):
            shape = t.shape
            h, w = shape[-2], shape[-1]
            if h == img_h and w == img_w:
                return t
            flat = t.reshape(-1, 1, h, w).float()
            flat = TF.resize(flat, [img_h, img_w],
                             interpolation=TF.InterpolationMode.NEAREST if mode == "nearest"
                             else TF.InterpolationMode.BILINEAR,
                             antialias=(mode == "bilinear"))
            return flat.reshape(*shape[:-2], img_h, img_w).to(t.dtype)

        out = dict(sample)
        out["rgb"] = resize_tensor(sample["rgb"], "bilinear")
        if "depth"          in sample: out["depth"]          = resize_tensor(sample["depth"],          "nearest")
        if "instance_class" in sample: out["instance_class"] = resize_tensor(sample["instance_class"], "nearest")
        if "instance_id"    in sample: out["instance_id"]    = resize_tensor(sample["instance_id"],    "nearest")
        return out
    return transform


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _resolved_config(parser: argparse.ArgumentParser,
                     args: argparse.Namespace) -> str:
    """Return a human-readable block showing every flag and its resolved value.

    The output is valid shell — copy-paste it to reproduce the run exactly.
    Non-default values are marked with  *  so changes from defaults are obvious.
    """
    script  = Path(sys.argv[0]).name
    invoked = " ".join(sys.argv)

    parts = [f"python {script}"]
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
    return "\n".join([
        sep,
        f"{script} — resolved configuration",
        f"invoked : {invoked}",
        sep,
        reproducible,
        sep,
        "all flags  (* = non-default):",
        *rows,
        sep,
    ])


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    data_root: str = str(DATA_ROOT),
    max_epochs: int = 100,
    batch_size: int = 2,
    num_workers: int = 16,
    prefetch_factor: int = 2,
    lstm_hidden: int = 512,
    learning_rate: float = 1e-4,
    depth_loss_fn: str = "l1",
    depth_weight: float = 1.0,
    sem_weight: float = 1.0,
    devices: int = 1,
    accelerator: str = "auto",
    gradient_clip_val: float = 1.0,
    log_dir: str = str(LOG_ROOT / "video_seg_depth"),
    checkpoint_dir: str = str(CHECKPOINT_ROOT),
    patience: int = 10,
    trial_name: str = None,
    sequence_length: int = 4,
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

    # --debug: 1 % of data, shape logging enabled
    if debug:
        limit_train_batches = 0.01
        limit_val_batches   = 0.01
        debug_shapes        = True
        print("[debug] limit_train_batches=0.01, limit_val_batches=0.01, debug_shapes=True")
    else:
        limit_train_batches = 1.0
        debug_shapes        = False

    viz_rgb, viz_depth, viz_sem, _ = collect_viz_clip_joint(val_dataset, n_frames=16)

    img_size = img_h  # TinyViT uses square img_size; assume img_h == img_w
    model = VideoSegDepthModule(
        lstm_hidden=lstm_hidden,
        learning_rate=learning_rate,
        depth_loss_fn=depth_loss_fn,
        depth_weight=depth_weight,
        sem_weight=sem_weight,
        cli_command=cli_command,
        viz_rgb=viz_rgb,
        viz_depth=viz_depth,
        viz_sem=viz_sem,
        img_size=img_size,
        debug_shapes=debug_shapes,
    )

    log_base_dir = Path(log_dir)
    if trial_name is None:
        existing = list(log_base_dir.glob("trial_*"))
        trial_name = f"trial_{len(existing) + 1:05d}"

    trial_log_dir = log_base_dir / trial_name
    trial_log_dir.mkdir(parents=True, exist_ok=True)

    print(cli_command, flush=True)
    (trial_log_dir / "command.sh").write_text(cli_command + "\n")

    logger = TensorBoardLogger(save_dir=str(trial_log_dir), name="video_seg_depth")

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
            ModelCheckpoint(dirpath=checkpoint_dir, filename="best-video-seg-depth",
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
        description="TinyViT + LSTM joint depth + semantic segmentation")
    parser.add_argument("--data-root",         type=str,   default=str(DATA_ROOT))
    parser.add_argument("--max-epochs",        type=int,   default=100)
    parser.add_argument("--batch-size",        type=int,   default=2)
    parser.add_argument("--num-workers",       type=int,   default=16)
    parser.add_argument("--prefetch-factor",   type=int,   default=2)
    parser.add_argument("--lstm-hidden",       type=int,   default=512)
    parser.add_argument("--learning-rate",     type=float, default=1e-4)
    parser.add_argument("--depth-loss-fn",     type=str,   default="l1",
                        choices=["l1", "mse", "smooth_l1"])
    parser.add_argument("--depth-weight",      type=float, default=1.0)
    parser.add_argument("--sem-weight",        type=float, default=1.0)
    parser.add_argument("--devices",           type=int,   default=1)
    parser.add_argument("--accelerator",       type=str,   default="auto")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--log-dir",           type=str,
                        default=str(LOG_ROOT / "video_seg_depth"))
    parser.add_argument("--checkpoint-dir",    type=str,
                        default=str(CHECKPOINT_ROOT))
    parser.add_argument("--patience",          type=int,   default=10)
    parser.add_argument("--trial-name",        type=str,   default=None)
    parser.add_argument("--sequence-length",   type=int,   default=4)
    parser.add_argument("--img-h",             type=int,   default=224,
                        help="Image height (must match TinyViT img_size, e.g. 224)")
    parser.add_argument("--img-w",             type=int,   default=224,
                        help="Image width (must match TinyViT img_size, e.g. 224)")
    parser.add_argument("--precision",         type=str,   default="32")
    parser.add_argument("--val-check-interval", type=float, default=0.5)
    parser.add_argument("--limit-val-batches",  type=float, default=0.2)
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 1%% data, 1%% val, print tensor shapes")

    args = parser.parse_args()
    cli_command = _resolved_config(parser, args)

    train(
        data_root=args.data_root,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        lstm_hidden=args.lstm_hidden,
        learning_rate=args.learning_rate,
        depth_loss_fn=args.depth_loss_fn,
        depth_weight=args.depth_weight,
        sem_weight=args.sem_weight,
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
