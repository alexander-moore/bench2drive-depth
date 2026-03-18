"""
AdversarialDepthModule — Lightning module that adds an ImageDiscriminator
on top of any DepthModelBase.

Optimisation strategy:
  - automatic_optimization=False (we step gen & disc manually each batch)
  - Generator  : AdamW, cosine annealing (same as DepthModule)
  - Discriminator: Adam, constant LR (discriminators prefer stable LR)

Adversarial schedule:
  Epochs 0 … adv_warmup_epochs-1  → reconstruction only (disc not updated,
                                     gen adv loss weight = 0)
  Epochs adv_warmup_epochs … end  → reconstruction + adversarial

Loss breakdown (generator step):
  L_gen = depth_weight * depth_loss
        + sem_weight   * (CE + 0.5*Dice)   [if produces_semantic]
        + adv_weight   * disc.loss_gen(...)  [after warmup]

Loss breakdown (discriminator step, after warmup):
  L_disc = 0.5 * (disc.loss_real(...) + disc.loss_fake(...))
         + r1_weight * disc.r1_penalty(...)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

from losses import SILogLoss, DiceLoss, abs_rel
from models._discriminator import ImageDiscriminator
from visualization import (
    save_depth_video, save_joint_video,
    save_depth_image, save_joint_image,
    save_dashcam_image, save_dashcam_video,
    CARLA_CLASS_NAMES,
)

NUM_CLASSES = 23


class AdversarialDepthModule(pl.LightningModule):
    """
    Unified adversarial training wrapper.

    Compatible with all 6 model types (stateful / non-stateful,
    depth-only / joint depth+seg).

    Args:
        model           : any DepthModelBase subclass
        learning_rate   : generator learning rate
        depth_loss_fn   : "silog" | "l1" | "smooth_l1"
        depth_weight    : weight for depth reconstruction loss
        sem_weight      : weight for segmentation loss (ignored if not produces_semantic)
        adv_weight      : weight for generator adversarial loss (after warmup)
        disc_lr         : discriminator learning rate
        disc_channels   : PatchGAN base channel width
        disc_mode       : "depth" | "semantic" | "both"
        adv_warmup_epochs : number of warmup epochs before adversarial loss is added
        r1_weight       : R1 gradient penalty weight (0 = disabled)
        single_frame    : single-frame pre-training flag (informational only)
        cli_command     : logged to TensorBoard on train start
        viz_rgb / viz_depth / viz_sem : optional visualisation clips
    """

    automatic_optimization = False

    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        depth_loss_fn: str = "silog",
        depth_weight: float = 1.0,
        sem_weight: float = 1.0,
        adv_weight: float = 0.1,
        disc_lr: float = 1e-4,
        disc_channels: int = 64,
        disc_mode: str = "both",
        adv_warmup_epochs: int = 10,
        r1_weight: float = 10.0,
        single_frame: bool = False,
        cli_command: str = "",
        viz_rgb=None,
        viz_depth=None,
        viz_sem=None,
        dashcam_rgb=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "viz_rgb", "viz_depth", "viz_sem", "dashcam_rgb"])

        self.model              = model
        self.learning_rate      = learning_rate
        self.depth_weight       = depth_weight
        self.sem_weight         = sem_weight
        self.adv_weight         = adv_weight
        self.disc_lr            = disc_lr
        self.disc_mode          = disc_mode
        self.adv_warmup_epochs  = adv_warmup_epochs
        self.r1_weight          = r1_weight
        self.single_frame       = single_frame
        self.cli_command        = cli_command
        self.best_val_loss      = float("inf")

        # Validate disc_mode vs model capabilities
        if disc_mode in ("semantic", "both") and not model.produces_semantic:
            raise ValueError(
                f"disc_mode={disc_mode!r} requires a model that produces semantic output, "
                f"but {type(model).__name__}.produces_semantic=False. "
                f"Use disc_mode='depth' for depth-only models."
            )

        # Visualisation clips (CPU)
        self._viz_rgb     = viz_rgb
        self._viz_depth   = viz_depth
        self._viz_sem     = viz_sem
        self._dashcam_rgb = dashcam_rgb  # (1, T, 1, 3, H, W) or None

        # Depth loss
        import torch.nn as nn
        if depth_loss_fn == "smooth_l1":
            self.depth_loss_fn = nn.SmoothL1Loss()
        elif depth_loss_fn == "silog":
            self.depth_loss_fn = SILogLoss()
        else:
            self.depth_loss_fn = nn.L1Loss()
        self.silog_metric = SILogLoss()

        # Semantic metrics
        if model.produces_semantic:
            self.dice_loss         = DiceLoss()
            self.val_miou          = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="macro")
            self.val_macc          = MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro")
            self.val_iou_per_class = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="none")

        # Discriminator
        # For semantic mode we pass logits (C channels); for depth it's 1 channel
        self.discriminator = ImageDiscriminator(
            disc_mode=disc_mode,
            base_channels=disc_channels,
            num_classes=NUM_CLASSES,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _adv_active(self) -> bool:
        """True once we're past the warmup phase."""
        return self.current_epoch >= self.adv_warmup_epochs

    # ------------------------------------------------------------------

    def on_train_start(self):
        if self.cli_command and self.logger is not None:
            self.logger.experiment.add_text("cli_command", self.cli_command, global_step=0)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _flatten_batch(self, t: torch.Tensor) -> torch.Tensor:
        """(B, S, C, ch, H, W) → (B*S*C, ch, H, W)"""
        return t.flatten(0, 2)

    def _disc_inputs(self, batch, output):
        """
        Extract flat (N, ch, H, W) tensors for discriminator.

        Returns:
            rgb_flat   : (N, 3, H, W)
            depth_real : (N, 1, H, W) or None
            depth_fake : (N, 1, H, W) or None
            sem_real   : (N, 1, H, W) int or None
            sem_fake   : (N, C, H, W) float or None
        """
        rgb_flat   = self._flatten_batch(batch["rgb"])
        depth_real = self._flatten_batch(batch["depth"])         if "depth" in batch else None
        depth_fake = self._flatten_batch(output.depth)
        sem_real   = None
        sem_fake   = None

        if self.disc_mode in ("semantic", "both") and self.model.produces_semantic:
            gt_sem   = batch.get("instance_class")
            sem_real = self._flatten_batch(gt_sem)[:, :1].long() if gt_sem is not None else None
            sem_fake = self._flatten_batch(output.semantic)       # logits, (N, C, H, W)

        return rgb_flat, depth_real, depth_fake, sem_real, sem_fake

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def _reconstruction_losses(self, batch, output):
        """Returns (total_recon_loss, depth_loss, sem_loss)."""
        depth_loss = self.depth_loss_fn(output.depth, batch["depth"])
        loss       = self.depth_weight * depth_loss
        sem_loss   = batch["rgb"].new_zeros(1).squeeze()

        if self.model.produces_semantic:
            gt_sem = batch.get("instance_class")
            if gt_sem is not None and self.sem_weight > 0:
                sem_flat = output.semantic.flatten(0, 2)
                gt_flat  = gt_sem.flatten(0, 2)[:, 0].long().clamp(0, NUM_CLASSES - 1)
                sem_loss = (F.cross_entropy(sem_flat, gt_flat)
                            + 0.5 * self.dice_loss(sem_flat, gt_flat))
                loss     = loss + self.sem_weight * sem_loss

        return loss, depth_loss, sem_loss

    def training_step(self, batch, batch_idx):
        gen_opt, disc_opt = self.optimizers()

        # ----------------------------------------------------------------
        # Forward pass
        # ----------------------------------------------------------------
        output = self.model(batch["rgb"], state=None)

        # ----------------------------------------------------------------
        # Generator step
        # ----------------------------------------------------------------
        gen_opt.zero_grad()

        recon_loss, depth_loss, sem_loss = self._reconstruction_losses(batch, output)
        gen_loss = recon_loss

        adv_gen_loss = batch["rgb"].new_zeros(1).squeeze()
        if self._adv_active:
            rgb_flat, _, depth_fake, _, sem_fake = self._disc_inputs(batch, output)
            adv_gen_loss = self.discriminator.loss_gen(
                rgb_flat,
                depth_fake if self.disc_mode in ("depth", "both") else None,
                sem_fake   if self.disc_mode in ("semantic", "both") else None,
            )
            gen_loss = gen_loss + self.adv_weight * adv_gen_loss

        self.manual_backward(gen_loss)
        self.clip_gradients(gen_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        gen_opt.step()

        # ----------------------------------------------------------------
        # Discriminator step (only after warmup)
        # ----------------------------------------------------------------
        disc_loss = batch["rgb"].new_zeros(1).squeeze()
        r1_loss   = batch["rgb"].new_zeros(1).squeeze()

        if self._adv_active:
            disc_opt.zero_grad()

            rgb_flat, depth_real, depth_fake, sem_real, sem_fake = self._disc_inputs(batch, output)

            d_real = self.discriminator.loss_real(
                rgb_flat,
                depth_real if self.disc_mode in ("depth", "both") else None,
                sem_real   if self.disc_mode in ("semantic", "both") else None,
            )
            d_fake = self.discriminator.loss_fake(
                rgb_flat,
                depth_fake if self.disc_mode in ("depth", "both") else None,
                sem_fake   if self.disc_mode in ("semantic", "both") else None,
            )
            disc_loss = 0.5 * (d_real + d_fake)

            if self.r1_weight > 0:
                r1_loss = self.r1_weight * self.discriminator.r1_penalty(
                    rgb_flat,
                    depth_real if self.disc_mode in ("depth", "both") else None,
                    sem_real   if self.disc_mode in ("semantic", "both") else None,
                )
                disc_loss = disc_loss + r1_loss

            self.manual_backward(disc_loss)
            disc_opt.step()

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        self.log("train/gen_loss",       gen_loss,     prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/loss_depth",     depth_loss,                  on_step=False, on_epoch=True)
        self.log("train/adv_gen_loss",   adv_gen_loss,                on_step=False, on_epoch=True)
        self.log("train/disc_loss",      disc_loss,                   on_step=False, on_epoch=True)
        self.log("train/r1_loss",        r1_loss,                     on_step=False, on_epoch=True)
        self.log("train/adv_active",     float(self._adv_active),     on_step=False, on_epoch=True)
        if self.model.produces_semantic:
            self.log("train/loss_sem",   sem_loss,                    on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Validation (reconstruction metrics only — no adversarial)
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        output = self.model(batch["rgb"], state=None)
        loss, depth_loss, sem_loss = self._reconstruction_losses(batch, output)
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
                sem_flat = output.semantic.flatten(0, 2)
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
    # Visualisation (identical to DepthModule)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _save_best_val_image(self):
        if self._viz_rgb is None:
            return
        frame  = self._viz_rgb[:, :1].to(self.device)
        output = self.model(frame, state=None)
        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if self.model.produces_semantic:
            save_joint_image(
                frame.cpu(), output.depth.cpu(),
                self._viz_depth[:, :1] if self._viz_depth is not None else None,
                output.semantic.cpu(),
                self._viz_sem[:, :1]   if self._viz_sem   is not None else None,
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

        T           = self._viz_rgb.shape[1]
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
        viz_depth = torch.cat(depth_preds, dim=1)

        if self.model.produces_semantic and sem_preds:
            viz_sem = torch.cat(sem_preds, dim=1)
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
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # Generator: all model params
        gen_params  = [p for p in self.model.parameters() if p.requires_grad]
        disc_params = list(self.discriminator.parameters())

        gen_opt  = torch.optim.AdamW(gen_params,  lr=self.learning_rate, weight_decay=1e-4)
        disc_opt = torch.optim.Adam( disc_params, lr=self.disc_lr,       betas=(0.5, 0.999))

        gen_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            gen_opt, T_max=self.trainer.max_epochs)

        return (
            [gen_opt, disc_opt],
            [{"scheduler": gen_sched, "interval": "epoch"}],
        )
