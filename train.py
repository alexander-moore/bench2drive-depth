"""
Shared launch script for all depth estimation models.

Usage:
    python train.py --model baseline_depth --backbone resnet18 --max-epochs 100
    python train.py --model video_former_depth --token-stride 8 --single-frame
    python train.py --model video_former_seg_depth --depth-weight 1.0 --sem-weight 0.5
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from config import DATA_ROOT, LOG_ROOT, CHECKPOINT_ROOT, DASHCAM_PATH
from dataset import Bench2DriveDataset
from visualization import collect_viz_clip, collect_viz_clip_joint, load_dashcam_frames

import models as model_registry
from module import DepthModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolved_config(parser: argparse.ArgumentParser,
                     args: argparse.Namespace) -> str:
    """Return a human-readable block showing every flag and its resolved value."""
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


def make_resize_transform(img_h: int, img_w: int):
    """Returns a transform that resizes all spatial tensors in a sample dict."""
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
        if "depth"          in sample: out["depth"]          = resize_tensor(sample["depth"],          "nearest")
        if "instance_class" in sample: out["instance_class"] = resize_tensor(sample["instance_class"], "nearest")
        if "instance_id"    in sample: out["instance_id"]    = resize_tensor(sample["instance_id"],    "nearest")
        return out
    return transform


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add all common (non-model-specific) CLI flags."""
    parser.add_argument("--data-root",            type=str,   default=str(DATA_ROOT))
    parser.add_argument("--max-epochs",           type=int,   default=100)
    parser.add_argument("--batch-size",           type=int,   default=4)
    parser.add_argument("--num-workers",          type=int,   default=16)
    parser.add_argument("--prefetch-factor",      type=int,   default=2)
    parser.add_argument("--learning-rate",        type=float, default=1e-4)
    parser.add_argument("--devices",              type=int,   default=1)
    parser.add_argument("--accelerator",          type=str,   default="auto")
    parser.add_argument("--gradient-clip-val",    type=float, default=1.0)
    parser.add_argument("--log-dir",              type=str,
                        default=str(LOG_ROOT / "depth"))
    parser.add_argument("--checkpoint-dir",       type=str,   default=str(CHECKPOINT_ROOT))
    parser.add_argument("--patience",             type=int,   default=10)
    parser.add_argument("--trial-name",           type=str,   default=None)
    parser.add_argument("--sequence-length",      type=int,   default=1)
    parser.add_argument("--precision",            type=str,   default="32",
                        help="Training precision: 32, 16-mixed, bf16-mixed")
    parser.add_argument("--val-check-interval",   type=int,   default=5,
                        help="Run validation every N epochs")
    parser.add_argument("--limit-val-batches",    type=float, default=0.1,
                        help="Fraction of val batches per validation check")
    parser.add_argument("--limit-train-batches",  type=int,   default=500,
                        help="Steps per epoch")
    parser.add_argument("--depth-loss-fn",        type=str,   default="silog",
                        choices=["l1", "smooth_l1", "silog"])
    parser.add_argument("--depth-weight",         type=float, default=1.0)
    parser.add_argument("--sem-weight",           type=float, default=1.0)
    parser.add_argument("--img-h",                type=int,   default=0,
                        help="Resize images to this height (0 = no resize)")
    parser.add_argument("--img-w",                type=int,   default=0,
                        help="Resize images to this width (0 = no resize)")
    parser.add_argument("--single-frame",         action="store_true",
                        help="Single-frame pre-training mode (use --sequence-length 1)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Pass 1: parse only --model to determine which class to load --------
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model", required=True,
                            choices=model_registry.list_models())
    pre_args, _ = pre_parser.parse_known_args()

    model_cls = model_registry.get_model_class(pre_args.model)

    # --- Pass 2: full parser with common + model-specific args --------------
    parser = argparse.ArgumentParser(
        description=f"Train {pre_args.model} depth estimation model")
    parser.add_argument("--model", type=str, required=True)
    add_common_args(parser)
    model_cls.add_model_args(parser)

    args = parser.parse_args()
    cli_command = _resolved_config(parser, args)

    # --- Datasets -----------------------------------------------------------
    load_instance = model_cls.produces_semantic
    transform     = make_resize_transform(args.img_h, args.img_w) \
                    if (args.img_h > 0 and args.img_w > 0) else None

    train_dataset = Bench2DriveDataset(
        args.data_root, split="train", sequence_length=args.sequence_length,
        load_depth_as_label=True, load_instance=load_instance, transform=transform)
    val_dataset = Bench2DriveDataset(
        args.data_root, split="val", sequence_length=args.sequence_length,
        load_depth_as_label=True, load_instance=load_instance, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor, shuffle=False, pin_memory=True)

    torch.set_float32_matmul_precision("medium")

    # --- Visualization clips ------------------------------------------------
    if model_cls.produces_semantic:
        viz_rgb, viz_depth, viz_sem, _ = collect_viz_clip_joint(val_dataset, n_frames=16)
    else:
        viz_rgb, viz_depth = collect_viz_clip(val_dataset, n_frames=16)
        viz_sem = None

    target_h, target_w = viz_rgb.shape[-2], viz_rgb.shape[-1]
    dashcam_rgb = load_dashcam_frames(DASHCAM_PATH, n_frames=32,
                                      target_h=target_h, target_w=target_w)

    # --- Model + Lightning module -------------------------------------------
    model  = model_cls.from_args(args)
    module = DepthModule(
        model=model,
        learning_rate=args.learning_rate,
        depth_loss_fn=args.depth_loss_fn,
        depth_weight=args.depth_weight,
        sem_weight=args.sem_weight,
        single_frame=args.single_frame,
        cli_command=cli_command,
        viz_rgb=viz_rgb,
        viz_depth=viz_depth,
        viz_sem=viz_sem,
        dashcam_rgb=dashcam_rgb,
    )

    # --- Trial naming and logging -------------------------------------------
    log_base_dir = Path(args.log_dir)
    trial_name   = args.trial_name
    if trial_name is None:
        existing   = list(log_base_dir.glob("trial_*"))
        trial_name = f"trial_{len(existing) + 1:05d}"

    trial_log_dir = log_base_dir / trial_name
    trial_log_dir.mkdir(parents=True, exist_ok=True)

    print(cli_command, flush=True)
    (trial_log_dir / "command.sh").write_text(cli_command + "\n")

    logger = TensorBoardLogger(save_dir=str(trial_log_dir), name=pre_args.model)

    # --- Trainer ------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=args.val_check_interval,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        callbacks=[
            ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename=f"best-{pre_args.model}",
                monitor="val/abs_rel", mode="min", save_top_k=1,
                auto_insert_metric_name=False,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val/abs_rel", mode="min",
                          patience=args.patience, verbose=True),
        ],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
