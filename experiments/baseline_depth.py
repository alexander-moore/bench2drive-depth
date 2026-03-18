"""
Baseline Depth-Only UNet
========================
Simple ConvNet UNet trained for per-pixel depth estimation.
Single output head predicting depth (L1 loss against depth labels).

Supports an optional pretrained ResNet encoder (resnet18/34/50) via --backbone.
Default backbone is resnet18. Use --backbone none for the original scratch ConvNet.
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
import torchvision.transforms.functional as TF
import torchvision.models as tvm

from dataset import Bench2DriveDataset
from visualization import collect_viz_clip, DepthVizMixin
from config import DATA_ROOT, LOG_ROOT, CHECKPOINT_ROOT


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


class DepthUNet(nn.Module):
    """Scratch ConvNet UNet for depth estimation: RGB -> per-pixel depth."""
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
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

        self.depth_head = nn.Conv2d(b, 1, kernel_size=1)

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

        return self.depth_head(d1)

    def forward(self, x):
        """x: (B, S, C, 3, H, W) -> depth (B, S, C, 1, H, W)"""
        b, s, c = x.shape[:3]
        x_flat = rearrange(x, 'b s c ch h w -> (b s c) ch h w')
        depth  = self.forward_single(x_flat)
        return rearrange(depth, '(b s c) 1 h w -> b s c 1 h w', b=b, s=s, c=c)


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


class ResNetDepthUNet(nn.Module):
    """
    Pretrained ResNet encoder + lightweight decoder for depth estimation.

    Encoder skip connections:
      stem   -> H/2,  64ch
      layer1 -> H/4,  64/256ch
      layer2 -> H/8,  128/512ch
      layer3 -> H/16, 256/1024ch
      layer4 -> H/32, 512/2048ch

    Decoder upsamples back to the original resolution with skip connections
    from each encoder stage. Only the decoder is randomly initialised;
    the encoder carries pretrained ImageNet weights.
    """
    # Fixed decoder widths — independent of backbone size
    _DEC = (256, 128, 64, 32)

    def __init__(self, backbone: str = "resnet18"):
        super().__init__()

        encoder = _RESNET_FN[backbone](weights=_RESNET_WEIGHTS[backbone])
        stem_ch, l1_ch, l2_ch, l3_ch, l4_ch = _RESNET_CHANNELS[backbone]
        d = self._DEC

        # Encoder stages (frozen weights loaded from torchvision)
        self.stem   = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # -> H/2
        self.pool   = encoder.maxpool                                            # -> H/4
        self.layer1 = encoder.layer1   # -> H/4
        self.layer2 = encoder.layer2   # -> H/8
        self.layer3 = encoder.layer3   # -> H/16
        self.layer4 = encoder.layer4   # -> H/32

        # Decoder
        self.up4  = nn.ConvTranspose2d(l4_ch, d[0], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(d[0] + l3_ch, d[0])
        self.up3  = nn.ConvTranspose2d(d[0],  d[1], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(d[1] + l2_ch, d[1])
        self.up2  = nn.ConvTranspose2d(d[1],  d[2], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(d[2] + l1_ch, d[2])
        self.up1  = nn.ConvTranspose2d(d[2],  d[3], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(d[3] + stem_ch, d[3])

        self.depth_head = nn.Conv2d(d[3], 1, kernel_size=1)

        self._init_decoder()

    def _init_decoder(self):
        decoder = [self.up4, self.dec4, self.up3, self.dec3,
                   self.up2, self.dec2, self.up1, self.dec1, self.depth_head]
        for module in decoder:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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
        s  = self.stem(x)              # H/2
        e1 = self.layer1(self.pool(s)) # H/4
        e2 = self.layer2(e1)           # H/8
        e3 = self.layer3(e2)           # H/16
        e4 = self.layer4(e3)           # H/32

        d4 = self.dec4(self._up_cat(self.up4, e4, e3))
        d3 = self.dec3(self._up_cat(self.up3, d4, e2))
        d2 = self.dec2(self._up_cat(self.up2, d3, e1))
        d1 = self.dec1(self._up_cat(self.up1, d2, s))

        # Final upsample H/2 -> H
        return F.interpolate(self.depth_head(d1), scale_factor=2,
                             mode="bilinear", align_corners=False)

    def forward(self, x):
        """x: (B, S, C, 3, H, W) -> depth (B, S, C, 1, H, W)"""
        b, s, c = x.shape[:3]
        x_flat = rearrange(x, 'b s c ch h w -> (b s c) ch h w')
        depth  = self.forward_single(x_flat)
        return rearrange(depth, '(b s c) 1 h w -> b s c 1 h w', b=b, s=s, c=c)


def build_model(backbone: str, base_channels: int) -> nn.Module:
    if backbone == "none":
        return DepthUNet(base_channels=base_channels)
    return ResNetDepthUNet(backbone=backbone)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class BaselineDepthModule(DepthVizMixin, pl.LightningModule):
    def __init__(
        self,
        backbone: str = "resnet18",
        base_channels: int = 64,
        learning_rate: float = 1e-4,
        depth_loss_fn: str = "l1",
        cli_command: str = "",
        viz_rgb=None,
        viz_depth=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["viz_rgb", "viz_depth"])

        self.model         = build_model(backbone, base_channels)
        self.learning_rate = learning_rate
        self.best_val_loss = float("inf")
        self.cli_command   = cli_command
        self.setup_viz(viz_rgb, viz_depth)

        self.depth_loss_fn = nn.SmoothL1Loss() if depth_loss_fn == "smooth_l1" else nn.L1Loss()

    def on_train_start(self):
        if self.cli_command:
            self.logger.experiment.add_text("cli_command", self.cli_command, global_step=0)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        depth_pred = self.model(batch["rgb"])
        loss       = self.depth_loss_fn(depth_pred, batch["depth"])
        return loss, depth_pred

    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, depth_pred = self._step(batch)
        mse = F.mse_loss(depth_pred, batch["depth"])
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/mse",  mse,  prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        val_mse = self.trainer.callback_metrics.get("val/mse")
        if val_mse is None:
            return
        if val_mse.item() < self.best_val_loss:
            self.best_val_loss = val_mse.item()
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
    def transform(sample):
        def resize_tensor(t, mode):
            shape = t.shape
            h, w  = shape[-2], shape[-1]
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
        if "depth" in sample:
            out["depth"] = resize_tensor(sample["depth"], "nearest")
        return out
    return transform


def train(
    data_root: str = str(DATA_ROOT),
    max_epochs: int = 100,
    batch_size: int = 4,
    num_workers: int = 16,
    prefetch_factor: int = 2,
    backbone: str = "resnet18",
    base_channels: int = 64,
    learning_rate: float = 1e-4,
    depth_loss_fn: str = "l1",
    devices: int = 1,
    accelerator: str = "auto",
    gradient_clip_val: float = 1.0,
    log_dir: str = str(LOG_ROOT / "baseline_depth"),
    checkpoint_dir: str = str(CHECKPOINT_ROOT),
    patience: int = 10,
    trial_name: str = None,
    sequence_length: int = 1,
    img_h: int = 0,
    img_w: int = 0,
    precision: str = "32",
    val_check_interval: float = 5,
    limit_val_batches: float = 0.1,
    limit_train_batches: int = 500,
    cli_command: str = "",
):
    transform = make_resize_transform(img_h, img_w) if (img_h > 0 and img_w > 0) else None
    train_dataset = Bench2DriveDataset(
        data_root, split="train", sequence_length=sequence_length,
        load_depth_as_label=True, transform=transform)
    val_dataset = Bench2DriveDataset(
        data_root, split="val", sequence_length=sequence_length,
        load_depth_as_label=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers,
        prefetch_factor=prefetch_factor, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers,
        prefetch_factor=prefetch_factor, shuffle=False, pin_memory=True)

    torch.set_float32_matmul_precision("medium")

    viz_rgb, viz_depth = collect_viz_clip(val_dataset, n_frames=16)

    model = BaselineDepthModule(
        backbone=backbone,
        base_channels=base_channels,
        learning_rate=learning_rate,
        depth_loss_fn=depth_loss_fn,
        cli_command=cli_command,
        viz_rgb=viz_rgb,
        viz_depth=viz_depth,
    )

    log_base_dir = Path(log_dir)
    if trial_name is None:
        existing   = list(log_base_dir.glob("trial_*"))
        trial_name = f"trial_{len(existing) + 1:05d}"

    trial_log_dir = log_base_dir / trial_name
    trial_log_dir.mkdir(parents=True, exist_ok=True)

    print(cli_command, flush=True)
    (trial_log_dir / "command.sh").write_text(cli_command + "\n")

    logger = TensorBoardLogger(save_dir=str(trial_log_dir), name="baseline_depth")

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
            ModelCheckpoint(dirpath=checkpoint_dir, filename="best-baseline-depth",
                            monitor="val/mse", mode="min", save_top_k=1,
                            auto_insert_metric_name=False),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val/mse", mode="min", patience=patience, verbose=True),
        ],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline depth-only UNet")
    parser.add_argument("--data-root",           type=str,   default=str(DATA_ROOT))
    parser.add_argument("--max-epochs",          type=int,   default=100)
    parser.add_argument("--batch-size",          type=int,   default=4)
    parser.add_argument("--num-workers",         type=int,   default=16)
    parser.add_argument("--prefetch-factor",     type=int,   default=2)
    parser.add_argument("--backbone",            type=str,   default="resnet18",
                        choices=["none", "resnet18", "resnet34", "resnet50"],
                        help="Encoder backbone. 'none' = scratch ConvNet UNet")
    parser.add_argument("--base-channels",       type=int,   default=64,
                        help="Base channels for scratch ConvNet (ignored when backbone != none)")
    parser.add_argument("--learning-rate",       type=float, default=1e-4)
    parser.add_argument("--depth-loss-fn",       type=str,   default="l1",
                        choices=["l1", "smooth_l1"])
    parser.add_argument("--devices",             type=int,   default=1)
    parser.add_argument("--accelerator",         type=str,   default="auto")
    parser.add_argument("--gradient-clip-val",   type=float, default=1.0)
    parser.add_argument("--log-dir",             type=str,
                        default=str(LOG_ROOT / "baseline_depth"))
    parser.add_argument("--checkpoint-dir",      type=str,   default=str(CHECKPOINT_ROOT))
    parser.add_argument("--patience",            type=int,   default=10)
    parser.add_argument("--trial-name",          type=str,   default=None)
    parser.add_argument("--sequence-length",     type=int,   default=1)
    parser.add_argument("--img-h",               type=int,   default=0,
                        help="Resize images to this height (0 = no resize)")
    parser.add_argument("--img-w",               type=int,   default=0,
                        help="Resize images to this width (0 = no resize)")
    parser.add_argument("--precision",           type=str,   default="32",
                        help="Training precision: 32, 16-mixed, bf16-mixed")
    parser.add_argument("--val-check-interval",  type=float, default=5,
                        help="Run validation every N epochs")
    parser.add_argument("--limit-val-batches",   type=float, default=0.1,
                        help="Fraction of val batches per validation check")
    parser.add_argument("--limit-train-batches", type=int,   default=500,
                        help="Steps per epoch")

    args = parser.parse_args()
    cli_command = _resolved_config(parser, args)

    train(
        data_root=args.data_root,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        backbone=args.backbone,
        base_channels=args.base_channels,
        learning_rate=args.learning_rate,
        depth_loss_fn=args.depth_loss_fn,
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
        limit_train_batches=args.limit_train_batches,
        cli_command=cli_command,
    )
