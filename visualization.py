import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from pathlib import Path
from einops import rearrange


# ---------------------------------------------------------------------------
# E2E trajectory visualisation
# ---------------------------------------------------------------------------

def plot_trajectory(
    past_traj: np.ndarray,
    future_traj_gt: np.ndarray,
    future_traj_pred: np.ndarray | None = None,
    title: str = "",
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot ego trajectories in the ego coordinate frame (x=forward, y=left).

    Args:
        past_traj:        (T_past, 2)   — history including anchor frame
        future_traj_gt:   (T_future, 2) — ground-truth future waypoints
        future_traj_pred: (T_future, 2) — model prediction (optional)
        title:            plot title string
        save_path:        if given, saves figure to this path
        show:             if True, calls plt.show()

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 8))

    # Past trajectory — blue dots, fading older → newer
    T_past = len(past_traj)
    alphas = np.linspace(0.2, 1.0, T_past)
    for i, (pt, a) in enumerate(zip(past_traj, alphas)):
        ax.scatter(pt[1], pt[0], color="royalblue", alpha=a, s=18, zorder=3)
    ax.plot(past_traj[:, 1], past_traj[:, 0],
            color="royalblue", alpha=0.4, linewidth=1, zorder=2)

    # Ground-truth future — green dots
    T_fut = len(future_traj_gt)
    gt_alphas = np.linspace(0.4, 1.0, T_fut)
    for pt, a in zip(future_traj_gt, gt_alphas):
        ax.scatter(pt[1], pt[0], color="seagreen", alpha=a, s=18, zorder=3)
    ax.plot(future_traj_gt[:, 1], future_traj_gt[:, 0],
            color="seagreen", alpha=0.5, linewidth=1, zorder=2)

    # Predicted future — red/orange dots
    if future_traj_pred is not None:
        pred_alphas = np.linspace(0.4, 1.0, len(future_traj_pred))
        for pt, a in zip(future_traj_pred, pred_alphas):
            ax.scatter(pt[1], pt[0], color="tomato", alpha=a, s=18, zorder=3)
        ax.plot(future_traj_pred[:, 1], future_traj_pred[:, 0],
                color="tomato", alpha=0.5, linewidth=1, zorder=2)

    # Anchor (current position) marker
    ax.scatter(0, 0, color="black", s=80, marker="*", zorder=5, label="anchor")

    # Direction arrow for ego heading (forward = +x, so arrow along +x axis)
    ax.annotate("", xy=(2, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Legend proxy artists
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color="royalblue", marker="o", markersize=5, label="past"),
        Line2D([0], [0], color="seagreen",  marker="o", markersize=5, label="future (GT)"),
    ]
    if future_traj_pred is not None:
        legend_items.append(
            Line2D([0], [0], color="tomato", marker="o", markersize=5, label="future (pred)")
        )
    ax.legend(handles=legend_items, loc="upper right", fontsize=8)

    ax.set_xlabel("y  (left +)")
    ax.set_ylabel("x  (forward +)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=8, wrap=True)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_trajectory_batch(
    past_trajs: np.ndarray,
    future_trajs_gt: np.ndarray,
    future_trajs_pred: np.ndarray | None = None,
    max_cols: int = 4,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Grid plot of trajectory_plot for a batch of samples.

    Args:
        past_trajs:        (B, T_past, 2)
        future_trajs_gt:   (B, T_future, 2)
        future_trajs_pred: (B, T_future, 2) or None
        max_cols:          maximum number of columns in the grid
        save_path:         optional save path
        show:              call plt.show()
    """
    B = len(past_trajs)
    ncols = min(B, max_cols)
    nrows = (B + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 5))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        if idx >= B:
            ax.axis("off")
            continue

        past   = past_trajs[idx]
        gt     = future_trajs_gt[idx]
        pred   = future_trajs_pred[idx] if future_trajs_pred is not None else None

        T_past = len(past)
        for i, (pt, a) in enumerate(zip(past, np.linspace(0.2, 1.0, T_past))):
            ax.scatter(pt[1], pt[0], color="royalblue", alpha=a, s=12)
        ax.plot(past[:, 1], past[:, 0], color="royalblue", alpha=0.3, lw=1)

        for pt, a in zip(gt, np.linspace(0.4, 1.0, len(gt))):
            ax.scatter(pt[1], pt[0], color="seagreen", alpha=a, s=12)
        ax.plot(gt[:, 1], gt[:, 0], color="seagreen", alpha=0.4, lw=1)

        if pred is not None:
            for pt, a in zip(pred, np.linspace(0.4, 1.0, len(pred))):
                ax.scatter(pt[1], pt[0], color="tomato", alpha=a, s=12)
            ax.plot(pred[:, 1], pred[:, 0], color="tomato", alpha=0.4, lw=1)

        ax.scatter(0, 0, color="black", s=60, marker="*")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"sample {idx}", fontsize=7)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig

# CARLA semantic class ID -> name
CARLA_CLASS_NAMES = {
    0:  "unlabeled",
    1:  "building",
    2:  "fence",
    3:  "other",
    4:  "pedestrian",
    5:  "pole",
    6:  "roadline",
    7:  "road",
    8:  "sidewalk",
    9:  "vegetation",
    10: "vehicle",
    11: "wall",
    12: "trafficsign",
    13: "sky",
    14: "ground",
    15: "bridge",
    16: "railtrack",
    17: "guardrail",
    18: "trafficlight",
    19: "static",
    20: "dynamic",
    21: "water",
    22: "terrain",
}

# CARLA semantic class ID -> (R, G, B) colour
CARLA_SEMANTIC_COLORS = {
    0:  (0,   0,   0),    # Unlabeled
    1:  (70,  70,  70),   # Building
    2:  (190, 153, 153),  # Fence
    3:  (110, 190, 160),  # Other
    4:  (220, 20,  60),   # Pedestrian
    5:  (153, 153, 153),  # Pole
    6:  (157, 234, 50),   # RoadLine
    7:  (128, 64,  128),  # Road
    8:  (244, 35,  232),  # SideWalk
    9:  (107, 142, 35),   # Vegetation
    10: (0,   0,   142),  # Vehicle
    11: (102, 102, 156),  # Wall
    12: (220, 220, 0),    # TrafficSign
    13: (70,  130, 180),  # Sky
    14: (81,  0,   81),   # Ground
    15: (150, 100, 100),  # Bridge
    16: (230, 150, 140),  # RailTrack
    17: (180, 165, 180),  # GuardRail
    18: (250, 170, 30),   # TrafficLight
    19: (110, 190, 160),  # Static
    20: (170, 120, 50),   # Dynamic
    21: (45,  60,  150),  # Water
    22: (145, 170, 100),  # Terrain
}


def _colorize_semantic(class_map: np.ndarray) -> np.ndarray:
    """class_map: (H, W) int -> (H, W, 3) uint8 RGB."""
    out = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, color in CARLA_SEMANTIC_COLORS.items():
        mask = class_map == cls_id
        out[mask] = color
    return out


def _colorize_instances(instance_id_map: np.ndarray) -> np.ndarray:
    """instance_id_map: (H, W) int -> (H, W, 3) uint8 with a random color per instance ID."""
    out = np.zeros((*instance_id_map.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=42)
    for inst_id in np.unique(instance_id_map):
        if inst_id == 0:
            continue  # background stays black
        color = rng.integers(60, 255, size=3).astype(np.uint8)
        out[instance_id_map == inst_id] = color
    return out


def _ensure_6d(rgb, depth_pred, depth_gt):
    """Ensure tensors are (B, seq_len, num_cameras, C, H, W)."""
    if rgb.dim() == 5:
        rgb = rgb.unsqueeze(1)
        depth_pred = depth_pred.unsqueeze(1)
        depth_gt = depth_gt.unsqueeze(1)
    return rgb, depth_pred, depth_gt


def save_depth_image(rgb, depth_pred, depth_gt, save_path):
    """
    Save a matplotlib grid of RGB / predicted depth / GT depth.

    Args:
        rgb:        (B, [seq_len,] num_cameras, 3, H, W)
        depth_pred: (B, [seq_len,] num_cameras, 1, H, W)
        depth_gt:   (B, [seq_len,] num_cameras, 1, H, W)
        save_path:  str or Path
    """
    rgb, depth_pred, depth_gt = _ensure_6d(rgb, depth_pred, depth_gt)

    batch_size = rgb.shape[0]
    num_cameras = rgb.shape[2]
    num_rows = min(2, batch_size)

    fig, axes = plt.subplots(
        num_rows,
        num_cameras * 3,
        figsize=(num_cameras * 6, num_rows * 2),
    )

    if num_rows == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_rows):
        for j in range(num_cameras):
            rgb_img = rgb[i, 0, j].cpu().float()
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)

            pred = depth_pred[i, 0, j, 0].cpu().float()
            gt = depth_gt[i, 0, j, 0].cpu().float()

            axes[i, j * 3].imshow(rearrange(rgb_img, 'c h w -> h w c').numpy())
            axes[i, j * 3].set_title(f"RGB cam{j}")
            axes[i, j * 3].axis("off")

            axes[i, j * 3 + 1].imshow(np.log1p(pred.numpy()), cmap="viridis")
            axes[i, j * 3 + 1].set_title("Pred (log)")
            axes[i, j * 3 + 1].axis("off")

            axes[i, j * 3 + 2].imshow(np.log1p(gt.numpy()), cmap="viridis")
            axes[i, j * 3 + 2].set_title("GT (log)")
            axes[i, j * 3 + 2].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def save_depth_video(rgb, depth_pred, depth_gt, save_path, fps=5, scale=0.25):
    """
    Save an mp4 arranged as 3 rows: RGB | predicted depth | GT depth,
    each row containing all cameras side by side.

    Args:
        rgb:        (B, [seq_len,] num_cameras, 3, H, W)
        depth_pred: (B, [seq_len,] num_cameras, 1, H, W)
        depth_gt:   (B, [seq_len,] num_cameras, 1, H, W)
        save_path:  str or Path
        fps:        frames per second
        scale:      resize factor applied to each camera tile before concatenation
    """
    rgb, depth_pred, depth_gt = _ensure_6d(rgb, depth_pred, depth_gt)

    num_cameras = rgb.shape[2]
    seq_len = rgb.shape[1]
    H, W = rgb.shape[-2], rgb.shape[-1]
    th = max(2, int(H * scale) // 2 * 2)  # round down to even
    tw = max(2, int(W * scale) // 2 * 2)
    log_max_depth = np.log1p(depth_gt.max().item())

    def resize(arr):
        """arr: (H, W, 3) uint8 -> (th, tw, 3) uint8"""
        from PIL import Image as _Image
        return np.array(_Image.fromarray(arr).resize((tw, th), _Image.BILINEAR))

    frames = []
    for t in range(seq_len):
        rgb_row, pred_row, gt_row = [], [], []

        for j in range(num_cameras):
            img = rgb[0, t, j].cpu().float()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (rearrange(img, 'c h w -> h w c').numpy() * 255).astype(np.uint8)
            rgb_row.append(resize(img))

            pred = depth_pred[0, t, j, 0].cpu().numpy()
            pred_norm = (np.log1p(pred) / (log_max_depth + 1e-8) * 255).astype(np.uint8)
            pred_rgb = np.stack([pred_norm] * 3, axis=-1)
            pred_row.append(resize(pred_rgb))

            gt = depth_gt[0, t, j, 0].cpu().numpy()
            gt_norm = (np.log1p(gt) / (log_max_depth + 1e-8) * 255).astype(np.uint8)
            gt_rgb = np.stack([gt_norm] * 3, axis=-1)
            gt_row.append(resize(gt_rgb))

        frame = np.concatenate([
            np.concatenate(rgb_row, axis=1),
            np.concatenate(pred_row, axis=1),
            np.concatenate(gt_row, axis=1),
        ], axis=0)
        frames.append(frame)

    # torchvision.io.write_video expects (T, H, W, C) uint8
    video = torch.from_numpy(np.stack(frames, axis=0))
    torchvision.io.write_video(str(save_path), video, fps=fps)


def collect_viz_clip(dataset, n_frames: int = 16):
    """Load a fixed clip for video visualization.

    Picks n_frames consecutive samples from the longest clip in the dataset
    so the resulting video is temporally coherent.  Works with any dataset
    that stores ``.samples`` with ``clip_name`` keys and returns ``"rgb"``
    (and optionally ``"depth"``) tensors shaped (seq_len, num_cameras, C, H, W).

    Returns:
        viz_rgb   : (1, T, num_cameras, 3, H, W) float tensor on CPU
        viz_depth : (1, T, num_cameras, 1, H, W) float tensor on CPU, or None
    """
    from collections import defaultdict
    clips: dict = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        clips[s["clip_name"]].append(i)

    # Use the clip with the most frames so we get a full n_frames window
    indices = max(clips.values(), key=len)[:n_frames]

    rgb_list, depth_list = [], []
    for idx in indices:
        sample = dataset[idx]
        rgb_list.append(sample["rgb"][0])        # (num_cameras, 3, H, W)
        if "depth" in sample:
            depth_list.append(sample["depth"][0])  # (num_cameras, 1, H, W)

    viz_rgb   = torch.stack(rgb_list, dim=0).unsqueeze(0)  # (1, T, cams, 3, H, W)
    viz_depth = torch.stack(depth_list, dim=0).unsqueeze(0) if depth_list else None
    return viz_rgb, viz_depth


class DepthVizMixin:
    """Mixin for PyTorch Lightning depth-estimation modules.

    Provides three helpers that any depth model can call:

      setup_viz(viz_rgb, viz_depth)
          Call once in __init__ to register the fixed visualization clip.

      save_validation_image(rgb, depth_pred, depth_gt)
          Call at batch_idx == 0 inside validation_step to save a per-epoch
          image grid (RGB | pred | GT) for each camera.

      save_best_video()
          Call inside on_validation_epoch_end whenever a new best val loss is
          reached.  Runs the model forward on every frame of the fixed clip and
          writes best_depth.mp4 to the TensorBoard log directory.

    The host class must be a pl.LightningModule (provides self.device,
    self.trainer, self.current_epoch, and self() for inference).

    Forward signature expected by save_best_video:
        input  : (B=1, seq_len=1, num_cameras, C, H, W)
        output : (B=1, seq_len=1, num_cameras, 1, H, W)
    """

    def setup_viz(self, viz_rgb, viz_depth):
        """Store the fixed visualization clip (kept on CPU throughout training)."""
        self._viz_rgb   = viz_rgb    # (1, T, cams, 3, H, W) or None
        self._viz_depth = viz_depth  # (1, T, cams, 1, H, W) or None

    def save_validation_image(self, rgb, depth_pred, depth_gt):
        """Save a matplotlib image grid for the current validation epoch."""
        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        save_depth_image(
            rgb, depth_pred, depth_gt,
            log_dir / f"validation_epoch_{self.current_epoch:04d}.png",
        )

    @torch.no_grad()
    def save_best_video(self):
        """Run the model over the fixed clip and write best_depth.mp4."""
        if getattr(self, "_viz_rgb", None) is None:
            return
        if getattr(self, "_viz_depth", None) is None:
            return

        T = self._viz_rgb.shape[1]
        pred_frames = []
        for t in range(T):
            frame = self._viz_rgb[:, t:t+1].to(self.device)  # (1, 1, cams, 3, H, W)
            pred  = self(frame).cpu()                          # (1, 1, cams, 1, H, W)
            pred_frames.append(pred)

        viz_pred = torch.cat(pred_frames, dim=1)   # (1, T, cams, 1, H, W)

        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        save_depth_video(self._viz_rgb, viz_pred, self._viz_depth,
                         log_dir / "best_depth.mp4")


def collect_viz_clip_inst(dataset, n_frames: int = 16):
    """Load a fixed clip for instance segmentation video visualization.

    Like collect_viz_clip but also returns instance_class and instance_id.

    Returns:
        viz_rgb           : (1, T, num_cameras, 3, H, W) float tensor on CPU
        viz_instance_class: (1, T, num_cameras, 1, H, W) int32 tensor on CPU, or None
        viz_instance_id   : (1, T, num_cameras, 1, H, W) int32 tensor on CPU, or None
    """
    from collections import defaultdict
    clips: dict = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        clips[s["clip_name"]].append(i)

    indices = max(clips.values(), key=len)[:n_frames]

    rgb_list, cls_list, id_list = [], [], []
    for idx in indices:
        sample = dataset[idx]
        rgb_list.append(sample["rgb"][0])
        if "instance_class" in sample:
            cls_list.append(sample["instance_class"][0])
        if "instance_id" in sample:
            id_list.append(sample["instance_id"][0])

    viz_rgb  = torch.stack(rgb_list, dim=0).unsqueeze(0)
    viz_cls  = torch.stack(cls_list, dim=0).unsqueeze(0) if cls_list else None
    viz_id   = torch.stack(id_list,  dim=0).unsqueeze(0) if id_list  else None
    return viz_rgb, viz_cls, viz_id


def _embedding_to_rgb(emb: torch.Tensor) -> np.ndarray:
    """Project a (D, H, W) embedding tensor to (H, W, 3) uint8 via PCA.

    Uses the top-3 principal components so that pixels of different instances
    map to visually distinct colours.
    """
    D, H, W = emb.shape
    flat = rearrange(emb, 'd h w -> (h w) d').float()
    flat = flat - flat.mean(dim=0)
    try:
        _, _, V = torch.pca_lowrank(flat, q=min(3, D), niter=2)
        projected = flat @ V                  # (H*W, 3)
    except Exception:
        projected = flat[:, :3]
    projected = rearrange(projected, '(h w) c -> h w c', h=H, w=W)
    # Normalise each channel independently to [0, 255]
    lo = projected.amin(dim=(0, 1), keepdim=True)
    hi = projected.amax(dim=(0, 1), keepdim=True)
    projected = (projected - lo) / (hi - lo + 1e-8)
    # Pad to 3 channels if embedding dim < 3
    if projected.shape[-1] < 3:
        pad = torch.zeros(*projected.shape[:2], 3 - projected.shape[-1])
        projected = torch.cat([projected, pad], dim=-1)
    return (projected.numpy() * 255).astype(np.uint8)


def save_instance_seg_video(rgb, pred_semantic, pred_embedding, gt_semantic,
                            gt_instance_id, save_path, fps=5, scale=0.25):
    """
    Save an mp4 with 5 rows per frame, each row showing all cameras side by side:
      Row 1: RGB
      Row 2: Predicted semantic (colorized by class)
      Row 3: Predicted embedding (PCA → RGB, shows instance separation)
      Row 4: GT semantic (colorized)
      Row 5: GT instance IDs (colorized)

    Args:
        rgb:            (B, [seq_len,] num_cameras, 3, H, W) float [0,1]
        pred_semantic:  (B, [seq_len,] num_cameras, NUM_CLASSES, H, W) logits
        pred_embedding: (B, [seq_len,] num_cameras, D, H, W) float embeddings
        gt_semantic:    (B, [seq_len,] num_cameras, 1, H, W) int class indices
        gt_instance_id: (B, [seq_len,] num_cameras, 1, H, W) int instance IDs
        save_path:      str or Path
        fps:            frames per second
        scale:          resize factor applied to each camera tile
    """
    def _ensure_6d_inst(t):
        return t.unsqueeze(1) if t.dim() == 5 else t

    rgb           = _ensure_6d_inst(rgb)
    pred_semantic  = _ensure_6d_inst(pred_semantic)
    pred_embedding = _ensure_6d_inst(pred_embedding)
    gt_semantic   = _ensure_6d_inst(gt_semantic)
    gt_instance_id = _ensure_6d_inst(gt_instance_id)

    num_cameras = rgb.shape[2]
    seq_len     = rgb.shape[1]
    H, W = rgb.shape[-2], rgb.shape[-1]
    th = max(2, int(H * scale) // 2 * 2)
    tw = max(2, int(W * scale) // 2 * 2)

    def resize(arr):
        from PIL import Image as _Image
        return np.array(_Image.fromarray(arr).resize((tw, th), _Image.BILINEAR))

    frames = []
    for t in range(seq_len):
        rows = [[] for _ in range(5)]  # rgb, pred_sem, pred_emb, gt_sem, gt_inst

        for j in range(num_cameras):
            # RGB
            img = rgb[0, t, j].cpu().float()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            rows[0].append(resize((rearrange(img, 'c h w -> h w c').numpy() * 255).astype(np.uint8)))

            # Predicted semantic
            ps = pred_semantic[0, t, j].cpu()
            ps = ps.argmax(dim=0) if ps.shape[0] > 1 else ps[0]
            rows[1].append(resize(_colorize_semantic(ps.numpy())))

            # Predicted embedding → PCA RGB
            emb = pred_embedding[0, t, j].cpu()
            rows[2].append(resize(_embedding_to_rgb(emb)))

            # GT semantic
            gs = gt_semantic[0, t, j, 0].cpu().numpy()
            rows[3].append(resize(_colorize_semantic(gs.astype(int))))

            # GT instance IDs
            gi = gt_instance_id[0, t, j, 0].cpu().numpy()
            rows[4].append(resize(_colorize_instances(gi.astype(int))))

        frame = np.concatenate(
            [np.concatenate(row, axis=1) for row in rows], axis=0
        )
        frames.append(frame)

    video = torch.from_numpy(np.stack(frames, axis=0))
    torchvision.io.write_video(str(save_path), video, fps=fps)


class InstSegVizMixin:
    """Mixin for PyTorch Lightning instance-segmentation modules.

    Provides helpers analogous to DepthVizMixin:

      setup_viz(viz_rgb, viz_cls, viz_id)
          Call once in __init__ to register the fixed visualization clip.

      save_validation_image(rgb, pred_sem, gt_sem, gt_inst_id)
          Call at batch_idx == 0 inside validation_step.

      save_best_video()
          Call inside on_validation_epoch_end on new best val loss.

    The host class must be a pl.LightningModule with a forward signature:
        input  : (B=1, seq_len=1, num_cameras, 3, H, W)
        output : (pred_sem, pred_emb) where pred_sem has shape
                 (B=1, seq_len=1, num_cameras, NUM_CLASSES, H, W)
    """

    def setup_viz(self, viz_rgb, viz_cls, viz_id):
        self._viz_rgb = viz_rgb  # (1, T, cams, 3, H, W) or None
        self._viz_cls = viz_cls  # (1, T, cams, 1, H, W) int32 or None
        self._viz_id  = viz_id   # (1, T, cams, 1, H, W) int32 or None

    def save_validation_image(self, rgb, pred_sem, pred_emb, gt_sem, gt_inst_id):
        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        save_instance_seg_image(
            rgb, pred_sem, pred_emb, gt_sem, gt_inst_id,
            log_dir / f"validation_epoch_{self.current_epoch:04d}.png",
        )

    @torch.no_grad()
    def save_best_video(self):
        if getattr(self, "_viz_rgb", None) is None:
            return
        if getattr(self, "_viz_cls", None) is None:
            return

        T = self._viz_rgb.shape[1]
        pred_sem_frames, pred_emb_frames = [], []
        for t in range(T):
            frame = self._viz_rgb[:, t:t+1].to(self.device)
            pred_sem, pred_emb = self(frame)
            pred_sem_frames.append(pred_sem.cpu())
            pred_emb_frames.append(pred_emb.cpu())

        viz_pred_sem = torch.cat(pred_sem_frames, dim=1)
        viz_pred_emb = torch.cat(pred_emb_frames, dim=1)

        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        save_instance_seg_video(
            self._viz_rgb, viz_pred_sem, viz_pred_emb,
            self._viz_cls, self._viz_id,
            log_dir / "best_instance_seg.mp4",
        )


def save_instance_seg_image(
    rgb,
    pred_semantic,
    pred_embedding,
    gt_semantic,
    gt_instance_id,
    save_path,
):
    """
    Save a matplotlib grid for instance segmentation.

    Columns per camera: RGB | pred semantic | pred embedding (PCA) | GT semantic | GT instances

    Args:
        rgb:            (B, [seq_len,] num_cameras, 3, H, W)  float [0,1]
        pred_semantic:  (B, [seq_len,] num_cameras, C, H, W)  logits or class indices
        pred_embedding: (B, [seq_len,] num_cameras, D, H, W)  float embeddings
        gt_semantic:    (B, [seq_len,] num_cameras, 1, H, W)  int class indices
        gt_instance_id: (B, [seq_len,] num_cameras, 1, H, W)  int instance IDs
        save_path:      str or Path
    """
    def _ensure_6d_inst(t):
        return t.unsqueeze(1) if t.dim() == 5 else t

    rgb            = _ensure_6d_inst(rgb)
    pred_semantic  = _ensure_6d_inst(pred_semantic)
    pred_embedding = _ensure_6d_inst(pred_embedding)
    gt_semantic    = _ensure_6d_inst(gt_semantic)
    gt_instance_id = _ensure_6d_inst(gt_instance_id)

    batch_size  = rgb.shape[0]
    num_cameras = rgb.shape[2]
    num_rows    = min(2, batch_size)
    num_cols    = num_cameras * 5  # RGB | pred_sem | pred_emb | gt_sem | gt_inst

    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(num_cameras * 10, num_rows * 2))
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_rows):
        for j in range(num_cameras):
            col = j * 5

            # RGB
            img = rgb[i, 0, j].cpu().float()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            axes[i, col].imshow(rearrange(img, 'c h w -> h w c').numpy())
            axes[i, col].set_title(f"RGB cam{j}")
            axes[i, col].axis("off")

            # Predicted semantic
            ps = pred_semantic[i, 0, j].cpu()
            if ps.shape[0] > 1:
                ps = ps.argmax(dim=0, keepdim=True)
            axes[i, col + 1].imshow(_colorize_semantic(ps[0].numpy()))
            axes[i, col + 1].set_title("Pred sem")
            axes[i, col + 1].axis("off")

            # Predicted embedding → PCA RGB
            emb = pred_embedding[i, 0, j].cpu()
            axes[i, col + 2].imshow(_embedding_to_rgb(emb))
            axes[i, col + 2].set_title("Pred emb (PCA)")
            axes[i, col + 2].axis("off")

            # GT semantic
            gs = gt_semantic[i, 0, j, 0].cpu().numpy()
            axes[i, col + 3].imshow(_colorize_semantic(gs.astype(int)))
            axes[i, col + 3].set_title("GT sem")
            axes[i, col + 3].axis("off")

            # GT instance IDs
            gi = gt_instance_id[i, 0, j, 0].cpu().numpy()
            axes[i, col + 4].imshow(_colorize_instances(gi.astype(int)))
            axes[i, col + 4].set_title("GT inst")
            axes[i, col + 4].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Joint depth + semantic visualisation
# ---------------------------------------------------------------------------

def collect_viz_clip_joint(dataset, n_frames: int = 16):
    """Load a fixed clip that has both depth and semantic/instance labels.

    Returns:
        viz_rgb   : (1, T, cams, 3, H, W)
        viz_depth : (1, T, cams, 1, H, W) or None
        viz_sem   : (1, T, cams, 1, H, W) int32 semantic class IDs, or None
        viz_inst  : (1, T, cams, 1, H, W) int32 instance IDs, or None
    """
    from collections import defaultdict
    clips: dict = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        clips[s["clip_name"]].append(i)

    indices = max(clips.values(), key=len)[:n_frames]

    rgb_list, depth_list, sem_list, inst_list = [], [], [], []
    for idx in indices:
        sample = dataset[idx]
        rgb_list.append(sample["rgb"][0])
        if "depth" in sample:
            depth_list.append(sample["depth"][0])
        if "instance_class" in sample:
            sem_list.append(sample["instance_class"][0])
        if "instance_id" in sample:
            inst_list.append(sample["instance_id"][0])

    viz_rgb   = torch.stack(rgb_list,   dim=0).unsqueeze(0)
    viz_depth = torch.stack(depth_list, dim=0).unsqueeze(0) if depth_list else None
    viz_sem   = torch.stack(sem_list,   dim=0).unsqueeze(0) if sem_list   else None
    viz_inst  = torch.stack(inst_list,  dim=0).unsqueeze(0) if inst_list  else None
    return viz_rgb, viz_depth, viz_sem, viz_inst


def save_joint_image(rgb, depth_pred, depth_gt, sem_pred, sem_gt, save_path):
    """
    Save a matplotlib grid showing depth and semantic outputs side by side.

    Columns per camera: RGB | pred depth | GT depth | pred semantic | GT semantic

    Args:
        rgb:        (B, [S,] C, 3, H, W)
        depth_pred: (B, [S,] C, 1, H, W)
        depth_gt:   (B, [S,] C, 1, H, W)  or None
        sem_pred:   (B, [S,] C, NUM_CLS, H, W)  or None
        sem_gt:     (B, [S,] C, 1, H, W) int class IDs  or None
        save_path:  str or Path
    """
    def _e6(t):
        return t.unsqueeze(1) if t is not None and t.dim() == 5 else t

    rgb        = _e6(rgb)
    depth_pred = _e6(depth_pred)
    depth_gt   = _e6(depth_gt)
    sem_pred   = _e6(sem_pred)
    sem_gt     = _e6(sem_gt)

    num_cameras = rgb.shape[2]

    has_depth = depth_pred is not None and depth_gt is not None
    has_sem   = sem_pred   is not None and sem_gt   is not None
    n_cols    = 1 + (2 if has_depth else 0) + (2 if has_sem else 0)

    col_titles = ["RGB"]
    if has_depth:
        col_titles += ["Pred depth", "GT depth"]
    if has_sem:
        col_titles += ["Pred sem", "GT sem"]

    # rows = cameras, cols = modalities
    fig, axes = plt.subplots(num_cameras, n_cols,
                             figsize=(n_cols * 3, num_cameras * 2))
    if num_cameras == 1:
        axes = axes[np.newaxis, :]

    max_depth     = depth_gt.max().item() if has_depth else 1.0
    log_max_depth = np.log1p(max_depth)

    for j in range(num_cameras):
        c = 0

        # RGB
        img = rgb[0, 0, j].cpu().float()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[j, c].imshow(rearrange(img, 'c h w -> h w c').numpy())
        axes[j, c].set_ylabel(f"cam {j}", fontsize=8)
        axes[j, c].axis("off")
        c += 1

        if has_depth:
            axes[j, c].imshow(np.log1p(depth_pred[0, 0, j, 0].cpu().numpy()),
                              cmap="viridis", vmin=0, vmax=log_max_depth)
            axes[j, c].axis("off")
            c += 1
            axes[j, c].imshow(np.log1p(depth_gt[0, 0, j, 0].cpu().numpy()),
                              cmap="viridis", vmin=0, vmax=log_max_depth)
            axes[j, c].axis("off")
            c += 1

        if has_sem:
            ps = sem_pred[0, 0, j].cpu()
            ps = ps.argmax(dim=0) if ps.shape[0] > 1 else ps[0]
            axes[j, c].imshow(_colorize_semantic(ps.numpy()))
            axes[j, c].axis("off")
            c += 1
            gs = sem_gt[0, 0, j, 0].cpu().numpy().astype(int)
            axes[j, c].imshow(_colorize_semantic(gs))
            axes[j, c].axis("off")

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=9)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def save_joint_video(rgb, depth_pred, depth_gt, sem_pred, sem_gt,
                     save_path, fps=5, scale=0.25):
    """
    Save an mp4 with one row per output type, all cameras side by side:
      Row 1: RGB
      Row 2: Predicted depth (viridis colormap)
      Row 3: GT depth
      Row 4: Predicted semantic (colorized)
      Row 5: GT semantic (colorized)

    Any of depth or sem rows are skipped when the corresponding tensor is None.
    """
    def _e6(t):
        return t.unsqueeze(1) if t is not None and t.dim() == 5 else t

    rgb        = _e6(rgb)
    depth_pred = _e6(depth_pred)
    depth_gt   = _e6(depth_gt)
    sem_pred   = _e6(sem_pred)
    sem_gt     = _e6(sem_gt)

    num_cameras = rgb.shape[2]
    seq_len     = rgb.shape[1]
    H, W = rgb.shape[-2], rgb.shape[-1]
    th = max(2, int(H * scale) // 2 * 2)
    tw = max(2, int(W * scale) // 2 * 2)

    has_depth = depth_pred is not None and depth_gt is not None
    has_sem   = sem_pred   is not None and sem_gt   is not None
    max_depth = depth_gt.max().item() if has_depth else 1.0
    log_max_depth = np.log1p(max_depth)

    cmap_fn = plt.get_cmap("viridis")

    def resize(arr):
        from PIL import Image as _Image
        return np.array(_Image.fromarray(arr).resize((tw, th), _Image.BILINEAR))

    def depth_to_rgb(d_np):
        normed = np.clip(np.log1p(d_np) / (log_max_depth + 1e-8), 0, 1)
        return (cmap_fn(normed)[:, :, :3] * 255).astype(np.uint8)

    frames = []
    for t in range(seq_len):
        rows = []

        # RGB row
        rgb_row = []
        for j in range(num_cameras):
            img = rgb[0, t, j].cpu().float()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            rgb_row.append(resize((rearrange(img, 'c h w -> h w c').numpy() * 255).astype(np.uint8)))
        rows.append(np.concatenate(rgb_row, axis=1))

        if has_depth:
            pred_row, gt_row = [], []
            for j in range(num_cameras):
                pred_row.append(resize(depth_to_rgb(depth_pred[0, t, j, 0].cpu().numpy())))
                gt_row.append(resize(depth_to_rgb(depth_gt[0, t, j, 0].cpu().numpy())))
            rows.append(np.concatenate(pred_row, axis=1))
            rows.append(np.concatenate(gt_row,   axis=1))

        if has_sem:
            sem_pred_row, sem_gt_row = [], []
            for j in range(num_cameras):
                ps = sem_pred[0, t, j].cpu()
                ps = ps.argmax(dim=0) if ps.shape[0] > 1 else ps[0]
                sem_pred_row.append(resize(_colorize_semantic(ps.numpy())))
                gs = sem_gt[0, t, j, 0].cpu().numpy().astype(int)
                sem_gt_row.append(resize(_colorize_semantic(gs)))
            rows.append(np.concatenate(sem_pred_row, axis=1))
            rows.append(np.concatenate(sem_gt_row,   axis=1))

        frames.append(np.concatenate(rows, axis=0))

    video = torch.from_numpy(np.stack(frames, axis=0))
    torchvision.io.write_video(str(save_path), video, fps=fps)


class JointVizMixin:
    """Mixin for Lightning modules that predict both depth and semantic segmentation.

    Expects forward to return (depth, sem):
        depth : (B, seq, cams, 1, H, W)
        sem   : (B, seq, cams, NUM_CLS, H, W)
    """

    def setup_viz(self, viz_rgb, viz_depth, viz_sem):
        self._viz_rgb   = viz_rgb    # (1, T, cams, 3, H, W) or None
        self._viz_depth = viz_depth  # (1, T, cams, 1, H, W) or None
        self._viz_sem   = viz_sem    # (1, T, cams, 1, H, W) int or None

    def save_validation_image(self, rgb, depth_pred, depth_gt, sem_pred, sem_gt):
        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        save_joint_image(
            rgb, depth_pred, depth_gt, sem_pred, sem_gt,
            log_dir / f"validation_epoch_{self.current_epoch:04d}.png",
        )

    def setup_train_viz(self, viz_rgb, viz_depth, viz_sem):
        self._train_viz_rgb   = viz_rgb    # (1, T, cams, 3, H, W) or None
        self._train_viz_depth = viz_depth  # (1, T, cams, 1, H, W) or None
        self._train_viz_sem   = viz_sem    # (1, T, cams, 1, H, W) int or None

    @torch.no_grad()
    def save_train_image(self):
        if getattr(self, "_train_viz_rgb", None) is None:
            return
        frame = self._train_viz_rgb[:, :1].to(self.device)
        depth_pred, sem_pred = self(frame)
        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        save_joint_image(
            frame.cpu(),
            depth_pred.cpu(),
            self._train_viz_depth[:, :1] if self._train_viz_depth is not None else None,
            sem_pred.cpu(),
            self._train_viz_sem[:, :1] if self._train_viz_sem is not None else None,
            log_dir / "train_latest.png",
        )

    @torch.no_grad()
    def save_best_val_image(self):
        if getattr(self, "_viz_rgb", None) is None:
            return
        frame = self._viz_rgb[:, :1].to(self.device)
        depth_pred, sem_pred = self(frame)
        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        save_joint_image(
            frame.cpu(),
            depth_pred.cpu(),
            self._viz_depth[:, :1] if self._viz_depth is not None else None,
            sem_pred.cpu(),
            self._viz_sem[:, :1] if self._viz_sem is not None else None,
            log_dir / "validation_best.png",
        )

    @torch.no_grad()
    def save_best_video(self):
        if getattr(self, "_viz_rgb", None) is None:
            return
        T = self._viz_rgb.shape[1]
        depth_frames, sem_frames = [], []
        for t in range(T):
            frame = self._viz_rgb[:, t:t+1].to(self.device)
            depth, sem = self(frame)
            depth_frames.append(depth.cpu())
            sem_frames.append(sem.cpu())

        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        save_joint_video(
            self._viz_rgb,
            torch.cat(depth_frames, dim=1), self._viz_depth,
            torch.cat(sem_frames,   dim=1), self._viz_sem,
            log_dir / "best_joint.mp4",
        )


# ---------------------------------------------------------------------------
# Real-world dashcam visualisation (no ground truth)
# ---------------------------------------------------------------------------

def load_dashcam_frames(mp4_path, n_frames=32, target_h=None, target_w=None):
    """
    Load n_frames evenly sampled from a dashcam MP4 using PyAV.

    Returns:
        (1, T, 1, 3, H, W) float tensor [0, 1] on CPU.
        Returns None if the file doesn't exist or decoding fails.
    """
    mp4_path = Path(mp4_path)
    if not mp4_path.exists():
        return None

    try:
        import av as _av
        container = _av.open(str(mp4_path))
        stream    = container.streams.video[0]
        T_total   = stream.frames or 0

        # Decode all frames into a list of (H, W, 3) uint8 arrays
        all_frames = []
        for packet in container.demux(stream):
            for frame in packet.decode():
                all_frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
    except Exception:
        return None

    T_total = len(all_frames)
    if T_total == 0:
        return None

    indices = torch.linspace(0, T_total - 1, min(n_frames, T_total)).long().tolist()
    frames  = np.stack([all_frames[i] for i in indices], axis=0)          # (T, H, W, 3)
    frames  = torch.from_numpy(frames).float() / 255.0                    # (T, H, W, 3)
    frames  = rearrange(frames, 't h w c -> t c h w')                     # (T, 3, H, W)

    if target_h is not None and target_w is not None:
        import torch.nn.functional as _F
        frames = _F.interpolate(
            frames, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

    return frames.unsqueeze(1).unsqueeze(0)   # (1, T, 1, 3, H, W)


def save_dashcam_image(rgb, depth_pred, sem_pred=None, save_path=None, n_sample_frames=4):
    """
    Save a matplotlib image grid for real-world dashcam inference (no GT).

    Rows: n_sample_frames evenly-spaced time steps.
    Columns: RGB | Pred depth [| Pred semantic]

    Args:
        rgb:             (1, T, 1, 3, H, W)
        depth_pred:      (1, T, 1, 1, H, W)
        sem_pred:        (1, T, 1, NUM_CLS, H, W) or None
        save_path:       str or Path
        n_sample_frames: how many frames to show as rows
    """
    def _e6(t):
        return t.unsqueeze(1) if t is not None and t.dim() == 5 else t

    rgb        = _e6(rgb)
    depth_pred = _e6(depth_pred)
    if sem_pred is not None:
        sem_pred = _e6(sem_pred)

    T = rgb.shape[1]
    n_rows = min(n_sample_frames, T)
    t_indices = torch.linspace(0, T - 1, n_rows).long().tolist()

    has_sem  = sem_pred is not None
    n_cols   = 2 + (1 if has_sem else 0)
    col_titles = ["RGB", "Pred depth (log)"] + (["Pred semantic"] if has_sem else [])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    log_max = np.log1p(depth_pred.max().item())
    cmap_fn = plt.get_cmap("viridis")

    for row, t in enumerate(t_indices):
        # RGB
        img = rgb[0, t, 0].cpu().float()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[row, 0].imshow(rearrange(img, 'c h w -> h w c').numpy())
        axes[row, 0].axis("off")

        # Predicted depth
        pred = depth_pred[0, t, 0, 0].cpu().numpy()
        normed = np.clip(np.log1p(pred) / (log_max + 1e-8), 0, 1)
        axes[row, 1].imshow(cmap_fn(normed))
        axes[row, 1].axis("off")

        if has_sem:
            ps = sem_pred[0, t, 0].cpu()
            ps = ps.argmax(dim=0) if ps.shape[0] > 1 else ps[0]
            axes[row, 2].imshow(_colorize_semantic(ps.numpy()))
            axes[row, 2].axis("off")

        if row == 0:
            for c, title in enumerate(col_titles):
                axes[row, c].set_title(title)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def save_dashcam_video(rgb, depth_pred, sem_pred=None, save_path=None, fps=10, scale=0.5):
    """
    Save an mp4 for real-world dashcam inference (no GT).

    Rows per frame:
      Row 1: RGB
      Row 2: Predicted depth (viridis)
      Row 3: Predicted semantic  [only if sem_pred is not None]

    Args:
        rgb:        (1, T, 1, 3, H, W)
        depth_pred: (1, T, 1, 1, H, W)
        sem_pred:   (1, T, 1, NUM_CLS, H, W) or None
        save_path:  str or Path
        fps:        output video frame rate
        scale:      resize factor per tile
    """
    def _e6(t):
        return t.unsqueeze(1) if t is not None and t.dim() == 5 else t

    rgb        = _e6(rgb)
    depth_pred = _e6(depth_pred)
    if sem_pred is not None:
        sem_pred = _e6(sem_pred)

    num_cameras = rgb.shape[2]
    seq_len     = rgb.shape[1]
    H, W = rgb.shape[-2], rgb.shape[-1]
    th = max(2, int(H * scale) // 2 * 2)
    tw = max(2, int(W * scale) // 2 * 2)

    has_sem       = sem_pred is not None
    log_max_depth = np.log1p(depth_pred.max().item())
    cmap_fn       = plt.get_cmap("viridis")

    def resize(arr):
        from PIL import Image as _Image
        return np.array(_Image.fromarray(arr).resize((tw, th), _Image.BILINEAR))

    def depth_to_rgb(d_np):
        normed = np.clip(np.log1p(d_np) / (log_max_depth + 1e-8), 0, 1)
        return (cmap_fn(normed)[:, :, :3] * 255).astype(np.uint8)

    frames = []
    for t in range(seq_len):
        rows = []

        # RGB row
        rgb_row = []
        for j in range(num_cameras):
            img = rgb[0, t, j].cpu().float()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            rgb_row.append(resize((rearrange(img, 'c h w -> h w c').numpy() * 255).astype(np.uint8)))
        rows.append(np.concatenate(rgb_row, axis=1))

        # Depth row
        pred_row = []
        for j in range(num_cameras):
            pred_row.append(resize(depth_to_rgb(depth_pred[0, t, j, 0].cpu().numpy())))
        rows.append(np.concatenate(pred_row, axis=1))

        # Semantic row
        if has_sem:
            sem_row = []
            for j in range(num_cameras):
                ps = sem_pred[0, t, j].cpu()
                ps = ps.argmax(dim=0) if ps.shape[0] > 1 else ps[0]
                sem_row.append(resize(_colorize_semantic(ps.numpy())))
            rows.append(np.concatenate(sem_row, axis=1))

        frames.append(np.concatenate(rows, axis=0))

    video = torch.from_numpy(np.stack(frames, axis=0))
    torchvision.io.write_video(str(save_path), video, fps=fps)
