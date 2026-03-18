"""
PatchGAN ImageDiscriminator for adversarial depth / segmentation training.

Supports three input modes controlled by `disc_mode`:
  "depth"    : concat(rgb [3], depth_norm [1])                     →  4 input channels
  "semantic" : concat(rgb [3], semantic_onehot [NUM_CLASSES])       → 26 input channels
  "both"     : concat(rgb [3], depth_norm [1], semantic_onehot [N]) → 27 input channels

Architecture: 5-layer PatchGAN with spectral normalisation on every conv.
Produces a spatial grid of real/fake scores (no global pooling).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NUM_CLASSES = 23


def _spectral_conv(in_ch: int, out_ch: int, kernel_size: int = 4,
                   stride: int = 2, padding: int = 1, bias: bool = True) -> nn.Module:
    return nn.utils.spectral_norm(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
    )


class ImageDiscriminator(nn.Module):
    """
    PatchGAN discriminator with spectral normalisation.

    Args:
        disc_mode     : "depth" | "semantic" | "both"
        base_channels : feature width at first conv layer (default 64)
        num_classes   : number of semantic classes (default 23)
    """

    def __init__(
        self,
        disc_mode: str = "both",
        base_channels: int = 64,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        if disc_mode not in ("depth", "semantic", "both"):
            raise ValueError(f"disc_mode must be depth/semantic/both, got {disc_mode!r}")
        self.disc_mode   = disc_mode
        self.num_classes = num_classes

        # Input channel count
        in_ch = 3  # rgb always included
        if disc_mode in ("depth", "both"):
            in_ch += 1               # normalised depth
        if disc_mode in ("semantic", "both"):
            in_ch += num_classes     # one-hot semantic

        C = base_channels
        # PatchGAN — receptive field grows with depth; no BN on first layer
        self.net = nn.Sequential(
            # Layer 0: no norm
            _spectral_conv(in_ch, C, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 1
            _spectral_conv(C,     C * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            _spectral_conv(C * 2, C * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3 (stride 1 to preserve spatial resolution)
            _spectral_conv(C * 4, C * 8, 4, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # Output — 1-channel patch score grid
            _spectral_conv(C * 8, 1, 4, 1, 1),
        )

    # ------------------------------------------------------------------
    # Input construction helpers
    # ------------------------------------------------------------------

    def _normalise_depth(self, depth: Tensor) -> Tensor:
        """Per-sample log-normalise depth to [0, 1] for discriminator input."""
        B = depth.shape[0]
        d = torch.log1p(depth.flatten(1).clamp(min=0))
        d_min = d.min(dim=1, keepdim=True).values
        d_max = d.max(dim=1, keepdim=True).values
        d = (d - d_min) / (d_max - d_min + 1e-6)
        return d.view_as(depth)

    def _sem_onehot(self, sem: Tensor) -> Tensor:
        """
        Convert semantic map to one-hot float.

        sem : (N, 1, H, W)  int64  class indices
        Returns: (N, num_classes, H, W)  float
        """
        N, _, H, W = sem.shape
        idx = sem[:, 0].long().clamp(0, self.num_classes - 1)  # (N, H, W)
        oh  = F.one_hot(idx, self.num_classes)                 # (N, H, W, C)
        return oh.permute(0, 3, 1, 2).float()                  # (N, C, H, W)

    def _build_input(
        self,
        rgb: Tensor,             # (N, 3, H, W)
        depth: Tensor | None,    # (N, 1, H, W)  — float
        semantic: Tensor | None, # (N, 1, H, W)  — int indices  OR  (N, C, H, W) logits/onehot
    ) -> Tensor:
        parts = [rgb]

        if self.disc_mode in ("depth", "both"):
            assert depth is not None, "depth required for disc_mode={self.disc_mode!r}"
            parts.append(self._normalise_depth(depth))

        if self.disc_mode in ("semantic", "both"):
            assert semantic is not None, "semantic required for disc_mode={self.disc_mode!r}"
            if semantic.shape[1] == 1:
                parts.append(self._sem_onehot(semantic))
            else:
                # Already logits or one-hot (C channels) — use directly
                parts.append(semantic.float())

        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------

    def forward(
        self,
        rgb: Tensor,
        depth: Tensor | None = None,
        semantic: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            rgb      : (N, 3, H, W)
            depth    : (N, 1, H, W)   float depth map
            semantic : (N, 1, H, W)   int class map  OR  (N, C, H, W) logits

        Returns:
            patch_scores : (N, 1, H', W')  — real/fake logits per patch
        """
        x = self._build_input(rgb, depth, semantic)
        return self.net(x)

    # ------------------------------------------------------------------
    # Loss helpers (LSGAN)
    # ------------------------------------------------------------------

    def loss_real(self, rgb, depth=None, semantic=None) -> Tensor:
        """Discriminator loss on real samples: E[(D(real) - 1)²]."""
        scores = self.forward(rgb, depth, semantic)
        return F.mse_loss(scores, torch.ones_like(scores))

    def loss_fake(self, rgb, depth=None, semantic=None) -> Tensor:
        """Discriminator loss on fake samples (detached): E[D(fake)²]."""
        scores = self.forward(rgb, depth.detach() if depth is not None else None,
                              semantic.detach() if semantic is not None else None)
        return F.mse_loss(scores, torch.zeros_like(scores))

    def loss_gen(self, rgb, depth=None, semantic=None) -> Tensor:
        """Generator adversarial loss: E[(D(fake) - 1)²]  (no detach)."""
        scores = self.forward(rgb, depth, semantic)
        return F.mse_loss(scores, torch.ones_like(scores))

    def r1_penalty(self, rgb, depth=None, semantic=None) -> Tensor:
        """
        R1 gradient penalty on real samples.

        Computes ||∇_x D(x_real)||² and returns its mean.
        Caller is responsible for scaling by r1_weight / 2.
        """
        x = self._build_input(rgb, depth, semantic)
        x = x.requires_grad_(True)
        scores = self.net(x)
        grad, = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
        )
        return grad.pow(2).flatten(1).sum(dim=1).mean()
