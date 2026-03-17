# Experiments

Three experiments for joint or standalone depth estimation on the Bench2Drive
multi-camera driving dataset.  All share the same dataset pipeline
(`dataset.py`), visualization utilities (`visualization.py`), and path config
(`config.py`).

---

## Quick-start

```bash
# Baseline — single-frame depth + semantic segmentation
bash run_training.sh

# TinyViT + LSTM — multi-frame depth + semantic segmentation
bash run_video_training.sh

# VideoFormerDepth — streaming depth with depth tokens (pre-train then fine-tune)
bash run_former_depth.sh --single-frame   # step 1: single-frame pre-training
bash run_former_depth.sh                  # step 2: streaming fine-tuning

# Any script accepts --debug for a fast sanity check
bash run_former_depth.sh --debug
```

---

## Experiments

### 1. `baseline_seg_depth.py`

**Architecture**: from-scratch UNet encoder–decoder with two output heads.

```
Input (B, 1, C, 3, H, W)
  └─ ConvBlock encoder  ×4  (64 → 128 → 256 → 512 ch, maxpool between stages)
  └─ Bottleneck ConvBlock   (512 → 1024 ch)
  └─ ConvTranspose decoder  ×4 with skip connections
  └─ depth_head    → (B, 1, C, 1, H, W)
  └─ semantic_head → (B, 1, C, 23, H, W)
```

**Losses**: L1 (or MSE / SmoothL1) depth loss + cross-entropy semantic loss,
weighted by `--depth-weight` and `--sem-weight`.

**Launcher**: `run_training.sh`

**Key flags**:
| Flag | Default | Description |
|---|---|---|
| `--base-channels` | 64 | Base channel width of the UNet |
| `--depth-loss-fn` | `l1` | `l1`, `mse`, or `smooth_l1` |
| `--sequence-length` | 1 | Dataset window size (model always processes S=1) |
| `--img-h / --img-w` | 0 | Resize input; 0 = no resize |

---

### 2. `video_seg_depth.py`

**Architecture**: pretrained TinyViT-21M encoder (frozen) + LSTM bottleneck +
UNet decoder with two output heads.

```
Input (B, S, C, 3, H, W)
  └─ TinyViT-21M encoder (frozen)
       skip0  : (B·S·C,  96, H/4,  W/4)
       skip1  : (B·S·C, 192, H/8,  W/8)
       skip2  : (B·S·C, 384, H/16, W/16)
       bot    : (B·S·C, 576, H/32, W/32)
  └─ LSTM at bottleneck  (input=576, hidden=lstm_hidden)
       Reshapes (B·S·C, 576, h, w) → (B·C·h·w, S, 576) → LSTM
       → (B·S·C, lstm_hidden, h, w)
  └─ ConvTranspose decoder with TinyViT skip sizes
       up4/dec4: lstm_hidden→512, cat skip2(384)
       up3/dec3: 512→256,         cat skip1(192)
       up2/dec2: 256→128,         cat skip0 (96)
       up1/dec1: 128→64           (no skip)
       up0/dec0:  64→32           (no skip)
  └─ depth_head    → (B, S, C, 1, H, W)
  └─ semantic_head → (B, S, C, 23, H, W)
```

**Temporal mechanism**: LSTM treats each spatial location `(b, c, h, w)` as an
independent sequence across the S frames in a batch window.  No state is
carried between windows at training time.

**Trainable parameters**: LSTM + decoder only (~12.6 M of 33.2 M total).

**Launcher**: `run_video_training.sh`

**Key flags**:
| Flag | Default | Description |
|---|---|---|
| `--lstm-hidden` | 512 | LSTM hidden size (= bottleneck output channels) |
| `--sequence-length` | 4 | Frames per training window |
| `--img-h / --img-w` | 224 | Must be compatible with TinyViT window sizes |
| `--debug` | off | 1 % data, 1 % val, print tensor shapes |

---

### 3. `video_former_depth.py`

**Architecture**: pretrained TinyViT-21M encoder (frozen) + transformer decoder
with spatial **depth tokens** as a persistent depth memory.

```
Input (B, S, C, 3, H, W)  — processed frame by frame
  └─ TinyViT-21M encoder (frozen, same 4 scales as above)
  └─ Linear projections: each encoder level → token_dim
  └─ Depth tokens  (H/stride × W/stride grid)
       Initialisation: learnable token_init  (frame 0, or --single-frame mode)
       Carry-over:     enriched tokens from previous frame  (streaming mode)
  └─ Transformer decoder  ×num_decoder_layers
       Each layer (pre-norm):
         1. Self-attention among depth tokens
         2. DPT-style cross-attention (see below)
         3. FFN (Linear → GELU → Linear)
  └─ CNN depth head  (progressive ConvTranspose2d × log2(token_stride))
       → depth  (B, 1, H, W)
     Enriched tokens returned → init for next frame
```

**DPT-style cross-attention**: the depth tokens attend separately to each of
the 4 TinyViT encoder scales; the 4 cross-attention outputs are **summed**
before the residual add.  This allows the tokens to gather both coarse
(bottleneck, H/32) and fine (skip0, H/4) information without concatenating
into one expensive key-value sequence.

```
# Future directions noted in source:
#  - Learned per-scale weights instead of uniform sum
#  - Concatenate + project scale outputs
#  - Coarse-to-fine ordering
#  - FPN-style feature merge before a single cross-attention
```

**Streaming mechanism**:

```
t=0 : tokens ← token_init  (learned)
t=1 : tokens ← enriched tokens from t=0
t=2 : tokens ← enriched tokens from t=1
 ⋮
```

The enriched **feature** vector is carried forward (not projected scalar
depth), preserving full representational capacity across frames.

**Trainable parameters**: projections + decoder + head only (~6.4 M of 27 M
total).

**Launcher**: `run_former_depth.sh`

**Recommended training workflow**:
```bash
# 1. Single-frame pre-training: model learns depth from scratch each frame.
#    Trains token_init into a strong depth prior.
bash run_former_depth.sh --single-frame --max-epochs 50

# 2. Streaming fine-tuning: tokens carry temporal context between frames.
#    Load the pre-trained checkpoint and fine-tune end-to-end.
bash run_former_depth.sh --sequence-length 2 --max-epochs 50
```

**Key flags**:
| Flag | Default | Description |
|---|---|---|
| `--token-stride` | 8 | Token grid = H/stride × W/stride (must be power of 2) |
| `--token-dim` | 256 | Token feature dimension (must be divisible by 4) |
| `--num-decoder-layers` | 6 | Transformer decoder depth |
| `--num-heads` | 8 | Attention heads per layer |
| `--single-frame` | off | Pre-training mode: no token carry-over |
| `--sequence-length` | 2 | Frames per training window |
| `--img-h / --img-w` | 224 | Must be compatible with TinyViT window sizes |
| `--debug` | off | 1 % data, 1 % val, print tensor shapes |

---

## Dataset

`Bench2DriveDataset` (`dataset.py`) loads multi-camera clips from the
Bench2Drive dataset.  The split files `train_split.txt` and `val_split.txt`
must exist under `--data-root` and list one clip directory name per line.

**Temporal sequences**: set `--sequence-length N` to load sliding windows of N
consecutive frames.  `sequence_length=1` is backward-compatible with the
baseline (returns a single-frame batch).

**Camera layout** (6 cameras per frame):
`front`, `front_left`, `front_right`, `back`, `back_left`, `back_right`

**Expected directory structure per clip**:
```
<clip>/
  camera/
    rgb_front/        *.jpg or *.png
    depth_front/      *.png   (16-bit, single channel)
    instance_front/   *.png   (RGBA: R=class, G=inst_lo, B=inst_hi)
    rgb_front_left/   ...
    ...
```

---

## Shared utilities

| File | Purpose |
|---|---|
| `config.py` | `DATA_ROOT`, `LOG_ROOT`, `CHECKPOINT_ROOT` paths |
| `dataset.py` | `Bench2DriveDataset`, `Bench2DriveDataModule` |
| `visualization.py` | `save_depth_image`, `save_depth_video`, `save_joint_video`, viz mixins |

**Tensor convention** used throughout:

```
B  — batch size
S  — sequence length (frames per window)
C  — number of cameras
H, W — spatial dimensions
```

All models receive `(B, S, C, 3, H, W)` and return tensors shaped
`(B, S, C, *, H, W)` so that batch, time, and camera axes are always explicit.

---

## Dependencies

```bash
pip install timm yacs termcolor
git clone https://github.com/wkcn/TinyViT.git /workspace/TinyViT
```

TinyViT pretrained weights are downloaded automatically on first use from the
[TinyViT model zoo](https://github.com/wkcn/TinyViT).
