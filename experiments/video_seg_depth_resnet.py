"""
ResNet Video Seg+Depth — ResNet encoder + LSTM + UNet decoder (depth + seg)
===========================================================================
Architecture
------------
  Encoder : ResNet-18/34/50 (pretrained ImageNet, fine-tunable)
              stem   → H/2,  64ch
              layer1 → H/4,  64/256ch
              layer2 → H/8,  128/512ch
              layer3 → H/16, 256/1024ch
              layer4 → H/32, 512/2048ch
  Temporal : LSTM at the layer4 bottleneck — information flows across S frames
  Decoder  : UNet with skip connections from all 5 ResNet stages (including stem)
  Heads    : depth_head (1ch) + semantic_head (23ch)

Compared to video_seg_depth.py (TinyViT encoder):
  + Fine-tunable encoder (not frozen)
  + Extra stem skip at H/2 → one more decoder stage with real skip data
  + No img_size constraint (works with any resolution)
  - Smaller receptive field than TinyViT attention layers

Input  : (B, S, C, 3, H, W)   — B batch, S sequence frames, C cameras
Outputs: depth (B,S,C,1,H,W) and semantic logits (B,S,C,23,H,W)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
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
import torchvision.models as tvm
import torchvision.transforms.functional as TF

from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

from dataset import Bench2DriveDataset, CAMERA_NAMES
from visualization import collect_viz_clip_joint, JointVizMixin, CARLA_CLASS_NAMES
from config import DATA_ROOT, LOG_ROOT, CHECKPOINT_ROOT
from losses import SILogLoss, DiceLoss, abs_rel

NUM_CLASSES = 23  # CARLA semantic classes 0-22

# Encoder output channels per stage: (stem, layer1, layer2, layer3, layer4)
_RESNET_CHANNELS = {
    "resnet18": (64,  64,  128,  256,  512),
    "resnet34": (64,  64,  128,  256,  512),
    "resnet50": (64, 256,  512, 1024, 2048),
}
_RESNET_WEIGHTS = {
    "resnet18": tvm.ResNet18_Weights.DEFAULT,
    "resnet34": tvm.ResNet34_Weights.DEFAULT,
    "resnet50": tvm.ResNet50_Weights.DEFAULT,
}
_RESNET_FN = {
    "resnet18": tvm.resnet18,
    "resnet34": tvm.resnet34,
    "resnet50": tvm.resnet50,
}


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

class ResNetVideoSegDepthUNet(nn.Module):
    """
    ResNet encoder (pretrained) + LSTM bottleneck + UNet decoder.

    forward(x) expects x: (B, S, C, 3, H, W)
    Returns:
        depth : (B, S, C, 1, H, W)
        sem   : (B, S, C, NUM_CLASSES, H, W)
    """

    # Fixed decoder widths — independent of backbone
    _DEC = (256, 128, 64, 32)

    def __init__(self, backbone: str = "resnet18", num_classes: int = NUM_CLASSES,
                 lstm_hidden: int = 512, debug_shapes: bool = False):
        super().__init__()
        self.debug_shapes   = debug_shapes
        self._shapes_logged = False

        stem_ch, l1_ch, l2_ch, l3_ch, l4_ch = _RESNET_CHANNELS[backbone]
        self.l4_ch = l4_ch
        d = self._DEC

        # ---- Encoder: pretrained ResNet (fine-tunable) --------------------
        encoder = _RESNET_FN[backbone](weights=_RESNET_WEIGHTS[backbone])
        self.stem   = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # H/2, 64ch
        self.pool   = encoder.maxpool                                            # H/4
        self.layer1 = encoder.layer1   # H/4,  l1_ch
        self.layer2 = encoder.layer2   # H/8,  l2_ch
        self.layer3 = encoder.layer3   # H/16, l3_ch
        self.layer4 = encoder.layer4   # H/32, l4_ch

        # ---- Temporal LSTM at bottleneck ----------------------------------
        self.temporal_lstm = nn.LSTM(l4_ch, lstm_hidden, batch_first=True)

        # ---- Decoder ------------------------------------------------------
        # up4: lstm_hidden → d0, cat with layer3 (l3_ch)
        self.up4  = nn.ConvTranspose2d(lstm_hidden, d[0], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(d[0] + l3_ch, d[0])

        # up3: d0 → d1, cat with layer2 (l2_ch)
        self.up3  = nn.ConvTranspose2d(d[0], d[1], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(d[1] + l2_ch, d[1])

        # up2: d1 → d2, cat with layer1 (l1_ch)
        self.up2  = nn.ConvTranspose2d(d[1], d[2], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(d[2] + l1_ch, d[2])

        # up1: d2 → d3, cat with stem (stem_ch=64)
        self.up1  = nn.ConvTranspose2d(d[2], d[3], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(d[3] + stem_ch, d[3])

        # Final bilinear ×2: H/2 → H (no skip above stem)
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

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _encode(self, x):
        """ResNet encoder, capturing all skip features.

        Args:
            x: (N, 3, H, W)  where N = B*S*C

        Returns:
            stem  : (N,  64,  H/2,  W/2)
            e1    : (N, l1_ch, H/4,  W/4)
            e2    : (N, l2_ch, H/8,  W/8)
            e3    : (N, l3_ch, H/16, W/16)
            e4    : (N, l4_ch, H/32, W/32)
        """
        s  = self.stem(x)
        e1 = self.layer1(self.pool(s))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return s, e1, e2, e3, e4

    # ------------------------------------------------------------------
    # LSTM temporal mixing
    # ------------------------------------------------------------------

    def _apply_lstm(self, bot, B, S, C):
        """Apply LSTM across the sequence dimension at each spatial location.

        Args:
            bot: (B*S*C, l4_ch, h, w)

        Returns:
            (B*S*C, lstm_hidden, h, w)
        """
        l4_ch, h, w = bot.shape[1], bot.shape[2], bot.shape[3]

        # → (B, S, C, l4_ch, h, w) → (B*C*h*w, S, l4_ch)
        x = bot.view(B, S, C, l4_ch, h, w)
        x = x.permute(0, 2, 3, 4, 1, 5).contiguous()
        x = x.view(B * C * h * w, S, l4_ch)

        lstm_out, _ = self.temporal_lstm(x)   # (B*C*h*w, S, lstm_hidden)

        lstm_hidden = lstm_out.shape[-1]
        # → (B, C, h, w, S, lstm_hidden) → (B*S*C, lstm_hidden, h, w)
        x = lstm_out.view(B, C, h, w, S, lstm_hidden)
        x = x.permute(0, 4, 1, 5, 2, 3).contiguous()
        x = x.view(B * S * C, lstm_hidden, h, w)
        return x

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def _decode(self, bot_lstm, e3, e2, e1, stem):
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

        d4 = self.dec4(up_cat(self.up4, bot_lstm, e3))  # H/16
        d3 = self.dec3(up_cat(self.up3, d4,       e2))  # H/8
        d2 = self.dec2(up_cat(self.up2, d3,       e1))  # H/4
        d1 = self.dec1(up_cat(self.up1, d2,       stem))  # H/2

        # Bilinear upsample H/2 → H
        depth = F.interpolate(self.depth_head(d1),    scale_factor=2,
                              mode="bilinear", align_corners=False)
        sem   = F.interpolate(self.semantic_head(d1), scale_factor=2,
                              mode="bilinear", align_corners=False)
        return depth, sem

    # ------------------------------------------------------------------
    # Shape logging (fires once when debug_shapes=True)
    # ------------------------------------------------------------------

    def _log_shapes(self, x_input, x_flat, stem, e1, e2, e3, e4, bot_lstm, depth, sem):
        lines = [
            "=" * 64,
            "ResNetVideoSegDepthUNet — tensor shapes (first forward pass)",
            "=" * 64,
            f"  input      (B,S,C,3,H,W)      : {tuple(x_input.shape)}",
            f"  x_flat     (N,3,H,W)           : {tuple(x_flat.shape)}",
            "  --- encoder (ResNet, fine-tunable) ---",
            f"  stem       (N, 64, H/2, W/2)   : {tuple(stem.shape)}",
            f"  e1         (N,l1,  H/4, W/4)   : {tuple(e1.shape)}",
            f"  e2         (N,l2,  H/8, W/8)   : {tuple(e2.shape)}",
            f"  e3         (N,l3, H/16,W/16)   : {tuple(e3.shape)}",
            f"  e4/bot     (N,l4, H/32,W/32)   : {tuple(e4.shape)}",
            "  --- lstm ---",
            f"  bot_lstm   (N,hid,H/32,W/32)   : {tuple(bot_lstm.shape)}",
            "  --- decoder outputs (pre-reshape) ---",
            f"  depth      (N,1,H,W)           : {tuple(depth.shape)}",
            f"  sem        (N,cls,H,W)         : {tuple(sem.shape)}",
            "=" * 64,
        ]
        table = "\n".join(lines)
        print(table, flush=True)
        self._shapes_table = table

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x: (B, S, C, 3, H, W)
        Returns:
            depth : (B, S, C, 1, H, W)
            sem   : (B, S, C, NUM_CLASSES, H, W)
        """
        B, S, C = x.shape[:3]
        x_flat = rearrange(x, 'b s c ch h w -> (b s c) ch h w')

        stem, e1, e2, e3, e4 = self._encode(x_flat)
        bot_lstm = self._apply_lstm(e4, B, S, C)
        depth, sem = self._decode(bot_lstm, e3, e2, e1, stem)

        if self.debug_shapes and not self._shapes_logged:
            self._log_shapes(x, x_flat, stem, e1, e2, e3, e4, bot_lstm, depth, sem)
            self._shapes_logged = True

        depth = rearrange(depth, '(b s c) 1 h w -> b s c 1 h w',    b=B, s=S, c=C)
        sem   = rearrange(sem,   '(b s c) cls h w -> b s c cls h w', b=B, s=S, c=C)
        return depth, sem


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class ResNetVideoSegDepthModule(JointVizMixin, pl.LightningModule):
    def __init__(
        self,
        backbone: str = "resnet18",
        lstm_hidden: int = 512,
        learning_rate: float = 1e-4,
        depth_loss_fn: str = "silog",
        depth_weight: float = 1.0,
        sem_weight: float = 1.0,
        cli_command: str = "",
        viz_rgb=None,
        viz_depth=None,
        viz_sem=None,
        train_viz_rgb=None,
        train_viz_depth=None,
        train_viz_sem=None,
        debug_shapes: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["viz_rgb", "viz_depth", "viz_sem",
                                          "train_viz_rgb", "train_viz_depth", "train_viz_sem"])

        self.model         = ResNetVideoSegDepthUNet(backbone=backbone,
                                                     lstm_hidden=lstm_hidden,
                                                     debug_shapes=debug_shapes)
        self.learning_rate = learning_rate
        self.depth_weight  = depth_weight
        self.sem_weight    = sem_weight
        self.best_val_loss = float("inf")
        self.cli_command   = cli_command
        self.setup_viz(viz_rgb, viz_depth, viz_sem)
        self.setup_train_viz(train_viz_rgb, train_viz_depth, train_viz_sem)

        if depth_loss_fn == "smooth_l1":
            self.depth_loss_fn = nn.SmoothL1Loss()
        elif depth_loss_fn == "silog":
            self.depth_loss_fn = SILogLoss()
        else:
            self.depth_loss_fn = nn.L1Loss()
        self.silog_metric   = SILogLoss()
        self.dice_loss      = DiceLoss()
        self.val_miou       = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="macro")
        self.val_macc       = MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro")
        self.val_iou_per_class = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="none")

    def on_train_start(self):
        if self.cli_command:
            self.logger.experiment.add_text("cli_command", self.cli_command, global_step=0)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, log_shapes: bool = False):
        rgb      = batch["rgb"]
        depth_gt = batch["depth"]
        gt_sem   = batch.get("instance_class", None)

        depth_pred, sem_pred = self.model(rgb)

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
            gt_flat  = gt_flat.clamp(0, NUM_CLASSES - 1)
            l_sem    = F.cross_entropy(sem_flat, gt_flat) + 0.5 * self.dice_loss(sem_flat, gt_flat)
            loss     = loss + self.sem_weight * l_sem

        return loss, l_depth, l_sem, depth_pred, sem_pred

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
            gt_flat  = rearrange(gt_sem,  'b s c 1 h w -> (b s c) h w').long().clamp(0, NUM_CLASSES - 1)
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
        val_loss = epoch_val_loss.item()
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_best_val_image()
            self.save_best_video()

    def on_train_epoch_end(self):
        self.save_train_image()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
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
    backbone: str = "resnet18",
    lstm_hidden: int = 512,
    learning_rate: float = 1e-4,
    depth_loss_fn: str = "silog",
    depth_weight: float = 1.0,
    sem_weight: float = 1.0,
    devices: int = 1,
    accelerator: str = "auto",
    gradient_clip_val: float = 1.0,
    log_dir: str = str(LOG_ROOT / "video_seg_depth_resnet"),
    checkpoint_dir: str = str(CHECKPOINT_ROOT),
    patience: int = 10,
    trial_name: str = None,
    sequence_length: int = 4,
    img_h: int = 0,
    img_w: int = 0,
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
        debug_shapes = False

    viz_rgb, viz_depth, viz_sem, _ = collect_viz_clip_joint(val_dataset, n_frames=16)
    train_viz_rgb, train_viz_depth, train_viz_sem, _ = collect_viz_clip_joint(train_dataset, n_frames=16)

    model = ResNetVideoSegDepthModule(
        backbone=backbone,
        lstm_hidden=lstm_hidden,
        learning_rate=learning_rate,
        depth_loss_fn=depth_loss_fn,
        depth_weight=depth_weight,
        sem_weight=sem_weight,
        cli_command=cli_command,
        viz_rgb=viz_rgb,
        viz_depth=viz_depth,
        viz_sem=viz_sem,
        train_viz_rgb=train_viz_rgb,
        train_viz_depth=train_viz_depth,
        train_viz_sem=train_viz_sem,
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

    logger = TensorBoardLogger(save_dir=str(trial_log_dir), name="video_seg_depth_resnet")

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
            ModelCheckpoint(dirpath=checkpoint_dir, filename="best-video-seg-depth-resnet",
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
        description="ResNet Video Seg+Depth: ResNet encoder + LSTM + UNet decoder")
    parser.add_argument("--data-root",           type=str,   default=str(DATA_ROOT))
    parser.add_argument("--max-epochs",          type=int,   default=100)
    parser.add_argument("--batch-size",          type=int,   default=2)
    parser.add_argument("--num-workers",         type=int,   default=16)
    parser.add_argument("--prefetch-factor",     type=int,   default=2)
    parser.add_argument("--backbone",            type=str,   default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--lstm-hidden",         type=int,   default=512)
    parser.add_argument("--learning-rate",       type=float, default=1e-4)
    parser.add_argument("--depth-loss-fn",       type=str,   default="silog",
                        choices=["l1", "smooth_l1", "silog"])
    parser.add_argument("--depth-weight",        type=float, default=1.0)
    parser.add_argument("--sem-weight",          type=float, default=1.0)
    parser.add_argument("--devices",             type=int,   default=1)
    parser.add_argument("--accelerator",         type=str,   default="auto")
    parser.add_argument("--gradient-clip-val",   type=float, default=1.0)
    parser.add_argument("--log-dir",             type=str,
                        default=str(LOG_ROOT / "video_seg_depth_resnet"))
    parser.add_argument("--checkpoint-dir",      type=str,   default=str(CHECKPOINT_ROOT))
    parser.add_argument("--patience",            type=int,   default=10)
    parser.add_argument("--trial-name",          type=str,   default=None)
    parser.add_argument("--sequence-length",     type=int,   default=4)
    parser.add_argument("--img-h",               type=int,   default=0,
                        help="Resize images to this height (0 = no resize)")
    parser.add_argument("--img-w",               type=int,   default=0,
                        help="Resize images to this width (0 = no resize)")
    parser.add_argument("--precision",           type=str,   default="32")
    parser.add_argument("--val-check-interval",  type=int,   default=5)
    parser.add_argument("--limit-val-batches",   type=float, default=0.1)
    parser.add_argument("--limit-train-batches", type=int,   default=500,
                        help="Steps per epoch")
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
        backbone=args.backbone,
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
        val_check_every_n_epochs=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        limit_train_batches=args.limit_train_batches,
        debug=args.debug,
        cli_command=cli_command,
    )
