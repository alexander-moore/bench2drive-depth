"""
Shared Lightning module for all depth (and joint depth+seg) models.

Branching is driven by model.produces_semantic and model.is_stateful —
no isinstance checks.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from einops import rearrange
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

from losses import SILogLoss, DiceLoss, abs_rel
from visualization import (
    save_depth_video, save_joint_video,
    save_depth_image, save_joint_image,
    save_dashcam_image, save_dashcam_video,
    CARLA_CLASS_NAMES,
)

NUM_CLASSES = 23


class DepthModule(pl.LightningModule):
    """
    Unified Lightning wrapper for all 6 model types.

    Handles:
      - depth-only models (produces_semantic=False)
      - joint depth+seg models (produces_semantic=True)
      - non-stateful models (is_stateful=False): one forward over full sequence
      - stateful models (is_stateful=True): per-frame loop with token carry-over
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        depth_loss_fn: str = "silog",
        depth_weight: float = 1.0,
        sem_weight: float = 1.0,
        single_frame: bool = False,
        cli_command: str = "",
        viz_rgb=None,
        viz_depth=None,
        viz_sem=None,
        dashcam_rgb=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "viz_rgb", "viz_depth", "viz_sem", "dashcam_rgb"])

        self.model         = model
        self.learning_rate = learning_rate
        self.depth_weight  = depth_weight
        self.sem_weight    = sem_weight
        self.single_frame  = single_frame
        self.cli_command   = cli_command
        self.best_val_loss = float("inf")

        # Visualization clips (kept on CPU)
        self._viz_rgb     = viz_rgb      # (1, T, cams, 3, H, W) or None
        self._viz_depth   = viz_depth    # (1, T, cams, 1, H, W) or None
        self._viz_sem     = viz_sem      # (1, T, cams, 1, H, W) int or None
        self._dashcam_rgb = dashcam_rgb  # (1, T, 1, 3, H, W) or None

        # Depth loss
        if depth_loss_fn == "smooth_l1":
            self.depth_loss_fn = nn.SmoothL1Loss()
        elif depth_loss_fn == "silog":
            self.depth_loss_fn = SILogLoss()
        else:
            self.depth_loss_fn = nn.L1Loss()
        self.silog_metric = SILogLoss()

        # Semantic metrics (only if the model produces semantic output)
        if model.produces_semantic:
            self.dice_loss         = DiceLoss()
            self.val_miou          = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="macro")
            self.val_macc          = MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro")
            self.val_iou_per_class = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="none")

    # ------------------------------------------------------------------

    def on_train_start(self):
        if self.cli_command and self.logger is not None:
            self.logger.experiment.add_text("cli_command", self.cli_command, global_step=0)
        if self.single_frame and self.logger is not None:
            self.logger.experiment.add_text(
                "training_mode", "single-frame (pre-training)", global_step=0)

    # ------------------------------------------------------------------

    def _step(self, batch):
        output = self.model(batch["rgb"], state=None)

        depth_loss = self.depth_loss_fn(output.depth, batch["depth"])
        loss = self.depth_weight * depth_loss
        sem_loss = batch["rgb"].new_zeros(1).squeeze()

        if self.model.produces_semantic:
            gt_sem = batch.get("instance_class")
            if gt_sem is not None and self.sem_weight > 0:
                sem_flat = output.semantic.flatten(0, 2)          # (B*S*C, cls, H, W)
                gt_flat  = gt_sem.flatten(0, 2)[:, 0].long()      # (B*S*C, H, W)
                gt_flat  = gt_flat.clamp(0, NUM_CLASSES - 1)
                ce       = F.cross_entropy(sem_flat, gt_flat)
                dice     = self.dice_loss(sem_flat, gt_flat)
                sem_loss = ce + 0.5 * dice
                loss     = loss + self.sem_weight * sem_loss

        return loss, depth_loss, sem_loss, output

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, depth_loss, sem_loss, _ = self._step(batch)
        self.log("train/loss",       loss,       prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/loss_depth", depth_loss,                on_step=False, on_epoch=True)
        if self.model.produces_semantic:
            self.log("train/loss_sem", sem_loss,                on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, depth_loss, sem_loss, output = self._step(batch)
        gt_depth = batch["depth"]

        self.log("val/loss",       loss,                                    prog_bar=True, sync_dist=True)
        self.log("val/loss_depth", depth_loss,                                             sync_dist=True)
        self.log("val/silog",      self.silog_metric(output.depth, gt_depth),              sync_dist=True)
        self.log("val/abs_rel",    abs_rel(output.depth, gt_depth),         prog_bar=True, sync_dist=True)
        self.log("val/mse",        F.mse_loss(output.depth, gt_depth),                    sync_dist=True)

        if self.model.produces_semantic:
            self.log("val/loss_sem", sem_loss, sync_dist=True)
            gt_sem = batch.get("instance_class")
            if gt_sem is not None and output.semantic is not None:
                sem_flat = output.semantic.flatten(0, 2)       # (B*S*C, cls, H, W)
                gt_flat  = gt_sem.flatten(0, 2)[:, 0].long().clamp(0, NUM_CLASSES - 1)
                preds    = sem_flat.argmax(dim=1)
                self.val_miou(preds, gt_flat)
                self.val_macc(preds, gt_flat)
                self.val_iou_per_class(preds, gt_flat)

        return loss

    def on_validation_epoch_end(self):
        if self.model.produces_semantic:
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
            self._save_best_val_image()
            self.save_best_video()
            self._save_dashcam_viz()

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _save_best_val_image(self):
        if self._viz_rgb is None:
            return
        frame = self._viz_rgb[:, :1].to(self.device)
        output = self.model(frame, state=None)
        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if self.model.produces_semantic:
            save_joint_image(
                frame.cpu(), output.depth.cpu(),
                self._viz_depth[:, :1] if self._viz_depth is not None else None,
                output.semantic.cpu(),
                self._viz_sem[:, :1] if self._viz_sem is not None else None,
                log_dir / "validation_best.png",
            )
        else:
            save_depth_image(
                frame.cpu(), output.depth.cpu(),
                self._viz_depth[:, :1] if self._viz_depth is not None else None,
                log_dir / "validation_best.png",
            )

    @torch.no_grad()
    def save_best_video(self):
        if self._viz_rgb is None or self._viz_depth is None:
            return

        T = self._viz_rgb.shape[1]
        depth_preds = []
        sem_preds   = []

        if self.model.is_stateful:
            state = None
            for t in range(T):
                x_t    = self._viz_rgb[:, t:t + 1].to(self.device)
                output = self.model(x_t, state=state)
                state  = output.state
                depth_preds.append(output.depth.cpu())
                if output.semantic is not None:
                    sem_preds.append(output.semantic.cpu())
        else:
            output = self.model(self._viz_rgb.to(self.device), state=None)
            depth_preds = [output.depth.cpu()]
            if output.semantic is not None:
                sem_preds = [output.semantic.cpu()]

        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        viz_depth = torch.cat(depth_preds, dim=1)  # (1, T, cams, 1, H, W)

        if self.model.produces_semantic and sem_preds:
            viz_sem = torch.cat(sem_preds, dim=1)  # (1, T, cams, NUM_CLS, H, W)
            save_joint_video(
                self._viz_rgb,
                viz_depth, self._viz_depth,
                viz_sem,   self._viz_sem,
                log_dir / "best_joint.mp4",
            )
        else:
            save_depth_video(
                self._viz_rgb, viz_depth, self._viz_depth,
                log_dir / "best_depth.mp4",
            )

    @torch.no_grad()
    def _save_dashcam_viz(self):
        if self._dashcam_rgb is None:
            return

        T           = self._dashcam_rgb.shape[1]
        depth_preds = []
        sem_preds   = []

        if self.model.is_stateful:
            state = None
            for t in range(T):
                x_t    = self._dashcam_rgb[:, t:t + 1].to(self.device)
                output = self.model(x_t, state=state)
                state  = output.state
                depth_preds.append(output.depth.cpu())
                if output.semantic is not None:
                    sem_preds.append(output.semantic.cpu())
        else:
            output = self.model(self._dashcam_rgb.to(self.device), state=None)
            depth_preds = [output.depth.cpu()]
            if output.semantic is not None:
                sem_preds = [output.semantic.cpu()]

        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        depth_pred = torch.cat(depth_preds, dim=1)
        sem_pred   = torch.cat(sem_preds,   dim=1) if sem_preds else None

        save_dashcam_image(
            self._dashcam_rgb, depth_pred, sem_pred,
            log_dir / "dashcam_best.png",
        )
        save_dashcam_video(
            self._dashcam_rgb, depth_pred, sem_pred,
            log_dir / "dashcam_best.mp4",
        )

    # ------------------------------------------------------------------

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt   = torch.optim.AdamW(trainable, lr=self.learning_rate, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
