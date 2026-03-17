import os
import json
import gzip
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    
try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None
    np = None

try:
    import pytorch_lightning as pl
except ImportError:
    pl = None


CAMERA_NAMES = [
    "front",
    "front_left", 
    "front_right",
    "back",
    "back_left",
    "back_right",
]


class Bench2DriveDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        sequence_length: int = 1,
        cameras: List[str] = CAMERA_NAMES,
        transform=None,
        load_depth_as_label: bool = True,
        load_instance: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.sequence_length = sequence_length
        self.cameras = cameras
        self.transform = transform
        self.load_depth_as_label = load_depth_as_label
        self.load_instance = load_instance

        self.samples = self._collect_samples()
    
    def _collect_samples(self) -> List[Dict]:
        samples = []
        split_file = self.data_root / f"{self.split}_split.txt"

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Create train_split.txt and val_split.txt in {self.data_root} "
                f"before training. Each file should list one clip name per line."
            )
        with open(split_file, 'r') as f:
            clip_names = [line.strip() for line in f if line.strip()]

        for clip_name in clip_names:
            clip_path = self.data_root / clip_name
            if not clip_path.is_dir():
                continue

            camera_path = clip_path / "camera"
            if not camera_path.exists():
                continue

            frame_files = {}
            for cam in self.cameras:
                rgb_dir = camera_path / f"rgb_{cam}"
                if rgb_dir.exists():
                    frames = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
                    frame_files[cam] = [f.stem for f in frames]

            if not frame_files:
                continue

            all_frames = set()
            for frames in frame_files.values():
                all_frames.update(frames)

            common_frames = sorted(all_frames)
            S = self.sequence_length
            # Sliding window: each sample is a window of S consecutive frames.
            # sequence_length=1 produces one sample per frame (backward compatible).
            for start in range(len(common_frames) - S + 1):
                frame_ids = common_frames[start:start + S]
                sample = {
                    "clip_name": clip_name,
                    "frame_id": frame_ids[0],   # first frame (backward compat key)
                    "frame_ids": frame_ids,      # all S frames in the window
                    "clip_path": clip_path,
                }
                samples.append(sample)

        return samples
    
    def _load_image(self, path: Path) -> torch.Tensor:
        from einops import rearrange
        img = Image.open(path).convert("RGB")
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = rearrange(torch.from_numpy(img_array), 'h w c -> c h w')
        return img_tensor
    
    def _load_instance(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a CARLA instance segmentation PNG.

        Encoding (RGBA):
          R = semantic class ID (0-27, CARLA tags)
          G = instance ID low byte
          B = instance ID high byte
          A = ignored (always 255)

        Returns:
          semantic_class : (1, H, W) int32 — class ID per pixel
          instance_id    : (1, H, W) int32 — unique instance ID per pixel
                           (0 means background / no instance)
        """
        arr = np.array(Image.open(path))  # (H, W, 4) uint8
        semantic_class = torch.from_numpy(arr[:, :, 0].astype(np.int32)).unsqueeze(0)
        instance_id = torch.from_numpy(
            (arr[:, :, 1].astype(np.int32)) | (arr[:, :, 2].astype(np.int32) << 8)
        ).unsqueeze(0)
        return semantic_class, instance_id

    def _load_depth(self, path: Path) -> torch.Tensor:
        depth = Image.open(path)
        depth_array = np.array(depth, dtype=np.float32)
        if len(depth_array.shape) == 3:
            depth_array = depth_array[:, :, 0]
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)
        return depth_tensor
    
    def __len__(self) -> int:
        return len(self.samples)

    def _load_single_frame(self, clip_path: Path, frame_id: str):
        """Load all cameras for one frame.

        Returns (rgb_tensors, depth_tensors, sem_tensors, inst_tensors) —
        one tensor per camera in self.cameras order.
        """
        def load_camera(cam):
            cam_path = clip_path / "camera"
            rgb_path = cam_path / f"rgb_{cam}" / f"{frame_id}.jpg"
            if not rgb_path.exists():
                rgb_path = cam_path / f"rgb_{cam}" / f"{frame_id}.png"
            rgb = self._load_image(rgb_path) if rgb_path.exists() else torch.zeros(3, 900, 1600)

            depth = None
            if self.load_depth_as_label:
                depth_path = cam_path / f"depth_{cam}" / f"{frame_id}.png"
                depth = self._load_depth(depth_path) if depth_path.exists() else torch.zeros(1, 900, 1600)

            sem_cls = inst_id = None
            if self.load_instance:
                inst_path = cam_path / f"instance_{cam}" / f"{frame_id}.png"
                if inst_path.exists():
                    sem_cls, inst_id = self._load_instance(inst_path)
                else:
                    sem_cls = torch.zeros(1, 900, 1600, dtype=torch.int32)
                    inst_id = torch.zeros(1, 900, 1600, dtype=torch.int32)
            return rgb, depth, sem_cls, inst_id

        with ThreadPoolExecutor(max_workers=len(self.cameras)) as pool:
            results = list(pool.map(load_camera, self.cameras))

        rgb_tensors, depth_tensors, sem_tensors, inst_tensors = [], [], [], []
        for rgb, depth, sem_cls, inst_id in results:
            rgb_tensors.append(rgb)
            if depth is not None:
                depth_tensors.append(depth)
            if sem_cls is not None:
                sem_tensors.append(sem_cls)
                inst_tensors.append(inst_id)
        return rgb_tensors, depth_tensors, sem_tensors, inst_tensors

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        clip_path = sample["clip_path"]
        frame_ids = sample.get("frame_ids", [sample["frame_id"]])

        rgb_seq, depth_seq, sem_seq, inst_seq = [], [], [], []

        for frame_id in frame_ids:
            rgb_t, depth_t, sem_t, inst_t = self._load_single_frame(clip_path, frame_id)
            rgb_seq.append(torch.stack(rgb_t, dim=0))        # (C, 3, H, W)
            if depth_t:
                depth_seq.append(torch.stack(depth_t, dim=0))    # (C, 1, H, W)
            if sem_t:
                sem_seq.append(torch.stack(sem_t, dim=0))         # (C, 1, H, W)
                inst_seq.append(torch.stack(inst_t, dim=0))

        # (S, C, ch, H, W)
        rgb_video = torch.stack(rgb_seq, dim=0)

        result = {
            "rgb": rgb_video,
            "frame_id": frame_ids[0],
            "clip_name": sample["clip_name"],
        }

        if self.load_depth_as_label and depth_seq:
            result["depth"] = torch.stack(depth_seq, dim=0)      # (S, C, 1, H, W)

        if self.load_instance and sem_seq:
            result["instance_class"] = torch.stack(sem_seq,  dim=0)  # (S, C, 1, H, W)
            result["instance_id"]    = torch.stack(inst_seq, dim=0)

        if self.transform:
            result = self.transform(result)

        return result


class Bench2DriveDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 16,
        sequence_length: int = 1,
        cameras: List[str] = CAMERA_NAMES,
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.cameras = cameras
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = Bench2DriveDataset(
                self.data_root,
                split="train",
                sequence_length=self.sequence_length,
                cameras=self.cameras,
            )
            self.val_dataset = Bench2DriveDataset(
                self.data_root,
                split="val",
                sequence_length=self.sequence_length,
                cameras=self.cameras,
            )
        
        if stage == "test":
            self.test_dataset = Bench2DriveDataset(
                self.data_root,
                split="test",
                sequence_length=self.sequence_length,
                cameras=self.cameras,
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dataset = Bench2DriveDataset(sys.argv[1])
        print(f"Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"RGB shape: {sample['rgb'].shape}")
        print(f"Depth shape: {sample['depth'].shape}")
