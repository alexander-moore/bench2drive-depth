"""
Baseline Joint Depth + Semantic Segmentation UNet
===================================================
Single UNet backbone with two output heads:
  1. depth_head    — per-pixel depth  (L1 loss against depth labels)
  2. semantic_head — per-pixel class logits (cross-entropy against instance_class labels)

Both heads are trained simultaneously. Visualisations show depth and semantic
segmentation side by side, with GT overlaid when available.
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
from datetime import datetime
import torchvision.transforms.functional as TF

from dataset import Bench2DriveDataset, CAMERA_NAMES
from visualization import collect_viz_clip_joint, JointVizMixin
from config import DATA_ROOT, LOG_ROOT, CHECKPOINT_ROOT

NUM_CLASSES = 23  # CARLA semantic classes 0-22


# ---------------------------------------------------------------------------
# Model
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


class BaselineSegDepthUNet(nn.Module):
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
        depth = rearrange(depth, '(b s c) 1 h w -> b s c 1 h w',   b=b, s=s, c=c)
        sem   = rearrange(sem,   '(b s c) cls h w -> b s c cls h w', b=b, s=s, c=c)
        return depth, sem


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class BaselineSegDepthModule(JointVizMixin, pl.LightningModule):
    def __init__(
        self,
        base_channels: int = 64,
        learning_rate: float = 1e-4,
        depth_loss_fn: str = "l1",
        depth_weight: float = 1.0,
        sem_weight: float = 1.0,
        cli_command: str = "",
        viz_rgb=None,
        viz_depth=None,
        viz_sem=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["viz_rgb", "viz_depth", "viz_sem"])

        self.model = BaselineSegDepthUNet(base_channels=base_channels)
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
            self.logger.experiment.add_text("cli_command", self.cli_command, global_step=0)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        rgb      = batch["rgb"]
        depth_gt = batch["depth"]
        gt_sem   = batch.get("instance_class", None)

        depth_pred, sem_pred = self.model(rgb)

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
        loss, l_depth, l_sem, _, _ = self._step(batch)
        self.log("train/loss",       loss,    prog_bar=True)
        self.log("train/loss_depth", l_depth)
        self.log("train/loss_sem",   l_sem)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, l_depth, l_sem, depth_pred, sem_pred = self._step(batch)
        self.log("val/loss",       loss,    prog_bar=True, sync_dist=True)
        self.log("val/loss_depth", l_depth, sync_dist=True)
        self.log("val/loss_sem",   l_sem,   sync_dist=True)

        if batch_idx == 0:
            self.save_validation_image(
                batch["rgb"], depth_pred, batch["depth"],
                sem_pred, batch.get("instance_class", None),
            )

        return loss

    def on_validation_epoch_end(self):
        epoch_val_loss = self.trainer.callback_metrics.get("val/loss")
        if epoch_val_loss is None:
            return
        val_loss = epoch_val_loss.item()
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_best_video()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


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

def make_resize_transform(img_h: int, img_w: int):
    """Returns a transform that resizes all spatial tensors in a sample dict."""
    def transform(sample):
        def resize_tensor(t, mode):
            # t: (..., H, W) — flatten leading dims, resize, restore
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
        out["rgb"]   = resize_tensor(sample["rgb"],   "bilinear")
        if "depth"          in sample: out["depth"]          = resize_tensor(sample["depth"],          "nearest")
        if "instance_class" in sample: out["instance_class"] = resize_tensor(sample["instance_class"], "nearest")
        if "instance_id"    in sample: out["instance_id"]    = resize_tensor(sample["instance_id"],    "nearest")
        return out
    return transform


def train(
    data_root: str = str(DATA_ROOT),
    max_epochs: int = 100,
    batch_size: int = 4,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    base_channels: int = 64,
    learning_rate: float = 1e-4,
    depth_loss_fn: str = "l1",
    depth_weight: float = 1.0,
    sem_weight: float = 1.0,
    devices: int = 1,
    accelerator: str = "auto",
    gradient_clip_val: float = 1.0,
    log_dir: str = str(LOG_ROOT / "baseline_seg_depth"),
    checkpoint_dir: str = str(CHECKPOINT_ROOT),
    patience: int = 10,
    trial_name: str = None,
    sequence_length: int = 1,
    img_h: int = 0,
    img_w: int = 0,
    precision: str = "32",
    val_check_interval: float = 0.5,
    limit_val_batches: float = 0.2,
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

    viz_rgb, viz_depth, viz_sem, _ = collect_viz_clip_joint(val_dataset, n_frames=16)

    model = BaselineSegDepthModule(
        base_channels=base_channels,
        learning_rate=learning_rate,
        depth_loss_fn=depth_loss_fn,
        depth_weight=depth_weight,
        sem_weight=sem_weight,
        cli_command=cli_command,
        viz_rgb=viz_rgb,
        viz_depth=viz_depth,
        viz_sem=viz_sem,
    )

    log_base_dir = Path(log_dir)
    if trial_name is None:
        existing = list(log_base_dir.glob("trial_*"))
        trial_name = f"trial_{len(existing) + 1:05d}"

    trial_log_dir = log_base_dir / trial_name
    trial_log_dir.mkdir(parents=True, exist_ok=True)

    print(cli_command, flush=True)
    (trial_log_dir / "command.sh").write_text(cli_command + "\n")

    logger = TensorBoardLogger(save_dir=str(trial_log_dir), name="baseline_seg_depth")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        accelerator=accelerator,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        callbacks=[
            ModelCheckpoint(dirpath=checkpoint_dir, filename="best-baseline-seg-depth",
                            monitor="val/loss", mode="min", save_top_k=1,
                            auto_insert_metric_name=False),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val/loss", mode="min", patience=patience, verbose=True),
        ],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline joint depth + semantic segmentation UNet")
    parser.add_argument("--data-root",         type=str,   default=str(DATA_ROOT))
    parser.add_argument("--max-epochs",        type=int,   default=100)
    parser.add_argument("--batch-size",        type=int,   default=4)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--prefetch-factor",   type=int,   default=2)
    parser.add_argument("--base-channels",     type=int,   default=64)
    parser.add_argument("--learning-rate",     type=float, default=1e-4)
    parser.add_argument("--depth-loss-fn",     type=str,   default="l1",
                        choices=["l1", "mse", "smooth_l1"])
    parser.add_argument("--depth-weight",      type=float, default=1.0)
    parser.add_argument("--sem-weight",        type=float, default=1.0)
    parser.add_argument("--devices",           type=int,   default=1)
    parser.add_argument("--accelerator",       type=str,   default="auto")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--log-dir",           type=str,
                        default=str(LOG_ROOT / "baseline_seg_depth"))
    parser.add_argument("--checkpoint-dir",    type=str,
                        default=str(CHECKPOINT_ROOT))
    parser.add_argument("--patience",          type=int,   default=10)
    parser.add_argument("--trial-name",        type=str,   default=None)
    parser.add_argument("--sequence-length",   type=int,   default=1)
    parser.add_argument("--img-h",             type=int,   default=0,
                        help="Resize images to this height (0 = no resize)")
    parser.add_argument("--img-w",             type=int,   default=0,
                        help="Resize images to this width (0 = no resize)")
    parser.add_argument("--precision",         type=str,   default="32",
                        help="Training precision: 32, 16-mixed, bf16-mixed")
    parser.add_argument("--val-check-interval", type=float, default=0.5,
                        help="Run validation every N epochs (or fraction thereof)")
    parser.add_argument("--limit-val-batches",  type=float, default=0.2,
                        help="Fraction of val batches to use per validation check")

    args = parser.parse_args()
    cli_command = _resolved_config(parser, args)

    train(
        data_root=args.data_root,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        base_channels=args.base_channels,
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
        cli_command=cli_command,
    )
