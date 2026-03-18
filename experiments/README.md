# Experiments

Six experiments for depth estimation and joint depth + semantic segmentation on the
Bench2Drive multi-camera driving dataset.  All share the same dataset pipeline
(`dataset.py`), visualization utilities (`visualization.py`), and path config
(`config.py`).

---

## Epoch definition and logging cadence

**One epoch = 500 gradient steps** (`--limit-train-batches 500`).  This keeps
epoch boundaries consistent across runs regardless of dataset size or batch
size, and makes TensorBoard curves directly comparable between experiments.

Validation runs every 5 epochs (`--val-check-interval 5`) on 10 % of the
validation set (`--limit-val-batches 0.1`), giving a fast signal without
dominating training time.  Training losses are averaged over the epoch
(`on_epoch=True`) so each TensorBoard data point corresponds to exactly 500
gradient steps.

All launcher scripts set these values by default; override with the
corresponding flags if needed.

---

## Quick-start

```bash
# Depth-only baselines
bash run_baseline_depth.sh                        # scratch UNet or ResNet encoder
bash run_baseline_depth.sh --backbone resnet18    # pretrained ResNet-18

# Joint depth + segmentation baselines
bash run_baseline_depth_seg.sh                    # scratch UNet, two heads

# TinyViT + LSTM — multi-frame, joint depth + seg
bash run_video_training.sh

# ResNet + LSTM — multi-frame, joint depth + seg (fine-tunable encoder)
bash run_video_seg_resnet.sh
bash run_video_seg_resnet.sh --backbone resnet34

# VideoFormerDepth — streaming depth-only with depth tokens
bash run_former_depth.sh --single-frame           # step 1: single-frame pre-training
bash run_former_depth.sh                          # step 2: streaming fine-tuning

# VideoFormerSegDepth — streaming depth + seg with parallel token stacks
bash run_former_seg_depth.sh --single-frame       # step 1: single-frame pre-training
bash run_former_seg_depth.sh                      # step 2: streaming fine-tuning

# Any script accepts --debug for a fast sanity check
bash run_former_seg_depth.sh --debug
```

---

## Experiments

### 1. `baseline_depth.py`

**Architecture**: depth-only UNet; scratch ConvNet by default, optional pretrained
ResNet encoder via `--backbone`.

```
Input (B, S, C, 3, H, W)
  └─ [backbone=none]  from-scratch ConvBlock encoder ×4
     OR
     [backbone=resnet18/34/50]  pretrained ResNet encoder (fine-tunable)
       stem   → H/2,   64ch
       layer1 → H/4,   64 / 256ch
       layer2 → H/8,  128 / 512ch
       layer3 → H/16, 256 /1024ch
       layer4 → H/32, 512 /2048ch
  └─ UNet decoder with skip connections (fixed widths: 256→128→64→32)
  └─ depth_head → (B, S, C, 1, H, W)
```

**Loss**: SILog (default), L1, or SmoothL1.

**Launcher**: `run_baseline_depth.sh`

**Key flags**:
| Flag | Default | Description |
|---|---|---|
| `--backbone` | `resnet18` | `none` (scratch UNet), `resnet18`, `resnet34`, `resnet50` |
| `--base-channels` | 64 | Channel width when using scratch UNet |
| `--depth-loss-fn` | `silog` | `l1`, `smooth_l1`, or `silog` |
| `--img-h / --img-w` | 0 | Resize input; 0 = no resize |

---

### 2. `baseline_seg_depth.py`

**Architecture**: depth + semantic segmentation, single from-scratch UNet with two output heads.

```
Input (B, S, C, 3, H, W)
  └─ ConvBlock encoder  ×4  (64 → 128 → 256 → 512 ch, maxpool between stages)
  └─ Bottleneck ConvBlock   (512 → 1024 ch)
  └─ ConvTranspose decoder  ×4 with skip connections
  └─ depth_head    → (B, S, C,  1, H, W)
  └─ semantic_head → (B, S, C, 23, H, W)
```

**Loss**: `depth_weight × depth_loss + sem_weight × (cross-entropy + 0.5 × Dice)`.

**Launcher**: `run_baseline_depth_seg.sh`

**Key flags**:
| Flag | Default | Description |
|---|---|---|
| `--base-channels` | 64 | Base channel width of the UNet |
| `--depth-loss-fn` | `silog` | `l1`, `smooth_l1`, or `silog` |
| `--depth-weight` | 1.0 | Scalar weight on the depth loss term |
| `--sem-weight` | 1.0 | Scalar weight on the semantic loss term |
| `--sequence-length` | 1 | Dataset window size (model always processes S=1) |
| `--img-h / --img-w` | 0 | Resize input; 0 = no resize |

---

### 3. `video_seg_depth.py`

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
  └─ depth_head    → (B, S, C,  1, H, W)
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
| `--depth-weight` | 1.0 | Scalar weight on the depth loss term |
| `--sem-weight` | 1.0 | Scalar weight on the semantic loss term |
| `--sequence-length` | 4 | Frames per training window |
| `--img-h / --img-w` | 224 | Must be compatible with TinyViT window sizes |
| `--debug` | off | 1 % data, 1 % val, print tensor shapes |

---

### 4. `video_seg_depth_resnet.py`

**Architecture**: pretrained ResNet encoder (fine-tunable) + LSTM bottleneck +
UNet decoder with two output heads.  Mirrors `video_seg_depth.py` but replaces
the frozen TinyViT with a fine-tunable ResNet and adds an extra decoder stage
using the stem skip.

```
Input (B, S, C, 3, H, W)
  └─ ResNet encoder (pretrained ImageNet, fine-tunable)
       stem   → H/2,   64ch
       layer1 → H/4,   64 / 256ch
       layer2 → H/8,  128 / 512ch
       layer3 → H/16, 256 /1024ch
       layer4 → H/32, 512 /2048ch
  └─ LSTM at layer4 bottleneck  (input=l4_ch, hidden=lstm_hidden)
  └─ UNet decoder with skip connections from all 5 ResNet stages
       Fixed decoder widths: 256→128→64→32 + final stem upsample
  └─ depth_head    → (B, S, C,  1, H, W)
  └─ semantic_head → (B, S, C, 23, H, W)
```

**vs `video_seg_depth.py`**:
- Encoder is fine-tunable (not frozen)
- Extra stem skip at H/2 gives one more decoder stage with real skip data
- No `img_size` constraint — works with any resolution
- Smaller receptive field than TinyViT attention layers

**Launcher**: `run_video_seg_resnet.sh`

**Key flags**:
| Flag | Default | Description |
|---|---|---|
| `--backbone` | `resnet18` | `resnet18`, `resnet34`, or `resnet50` |
| `--lstm-hidden` | 512 | LSTM hidden size |
| `--depth-weight` | 1.0 | Scalar weight on the depth loss term |
| `--sem-weight` | 1.0 | Scalar weight on the semantic loss term |
| `--sequence-length` | 4 | Frames per training window |
| `--debug` | off | 1 % data, 1 % val, print tensor shapes |

---

### 5. `video_former_depth.py`

**Architecture**: pretrained TinyViT-21M encoder (frozen) + transformer decoder
with spatial **depth tokens** as a persistent depth memory.

```
Input (B, S, C, 3, H, W)  — processed frame by frame
  └─ TinyViT-21M encoder (frozen, same 4 scales as above)
  └─ Linear projections: each encoder level → token_dim  (shared enc_proj_0..3)
  └─ Depth tokens  (H/stride × W/stride grid, dim=token_dim)
       Frame 0 / --single-frame: learnable depth_token_init
       Frame t (streaming):      enriched tokens from frame t-1
  └─ Transformer decoder  ×num_decoder_layers  (pre-norm)
       Each layer:
         1. Self-attention among depth tokens
         2. DPT-style multi-scale cross-attention (see below)
         3. FFN (Linear → GELU → Linear)
  └─ TokenCNNHead: progressive ConvTranspose2d × log2(token_stride)
       → depth  (B, 1, H, W)
     Enriched tokens returned → init for next frame
```

**DPT-style cross-attention**: depth tokens attend separately to each of the 4
TinyViT encoder scales; the 4 outputs are **summed** before the residual add.
This lets tokens gather coarse (bottleneck, H/32) and fine (skip0, H/4)
information without concatenating into one expensive key-value sequence.

**Streaming mechanism**:
```
t=0 : tokens ← depth_token_init  (learned)
t=1 : tokens ← enriched tokens from t=0
t=2 : tokens ← enriched tokens from t=1  …
```
The full token **feature vector** is carried forward (not a projected scalar),
preserving representational capacity across frames.

**Trainable parameters**: projections + decoder + head only (~6.4 M of 27 M total).

**Launcher**: `run_former_depth.sh`

**Recommended training workflow**:
```bash
bash run_former_depth.sh --single-frame --max-epochs 50   # 1. single-frame pre-train
bash run_former_depth.sh --sequence-length 2 --max-epochs 50  # 2. streaming fine-tune
```

**Key flags**:
| Flag | Default | Description |
|---|---|---|
| `--token-stride` | 8 | Token grid = H/stride × W/stride (power of 2) |
| `--token-dim` | 256 | Token feature dimension (divisible by 4) |
| `--num-decoder-layers` | 6 | Transformer decoder depth |
| `--num-heads` | 8 | Attention heads per layer |
| `--single-frame` | off | Pre-training mode: no token carry-over |
| `--sequence-length` | 2 | Frames per training window |
| `--img-h / --img-w` | 224 | Must be compatible with TinyViT window sizes |
| `--debug` | off | 1 % data, 1 % val, print tensor shapes |

---

### 6. `video_former_seg_depth.py`

**Architecture**: pretrained TinyViT-21M encoder (frozen) + **two parallel**
transformer decoder stacks — one for depth tokens, one for segmentation tokens —
sharing the same encoder features but maintaining independent temporal state.

```
Input (B, S, C, 3, H, W)  — processed frame by frame
  └─ TinyViT-21M encoder (frozen)
  └─ Shared linear projections enc_proj_0..3 → token_dim
  └─ Shared sinusoidal 2D position encodings

  ┌─ Depth branch ──────────────────────────────────────────┐
  │   depth_token_init  (learnable, frame 0 / single-frame) │
  │   depth_decoder_layers ×num_decoder_layers              │
  │     (DepthDecoderLayer: self-attn + DPT cross-attn + FFN)│
  │   TokenCNNHead(out_channels=1) → depth (B, 1, H, W)     │
  │   enriched depth_tokens → prev_depth_tokens next frame  │
  └─────────────────────────────────────────────────────────┘

  ┌─ Seg branch ────────────────────────────────────────────┐
  │   seg_token_init  (learnable, frame 0 / single-frame)   │
  │   seg_decoder_layers ×num_decoder_layers                │
  │     (same DepthDecoderLayer structure, separate weights) │
  │   TokenCNNHead(out_channels=23) → sem (B, 23, H, W)     │
  │   enriched seg_tokens → prev_seg_tokens next frame      │
  └─────────────────────────────────────────────────────────┘
```

**Key design choices**:
- No cross-task attention — depth and seg tokens process independently through
  their own decoder stacks and carry separate temporal state.
- Encoder projections and position encodings are **shared** between the two
  decoder stacks.
- Each token stack is a learned task-specific spatial working memory that
  accumulates temporal context independently.

**Streaming mechanism** (two independent token states):
```
t=0 : depth_tokens ← depth_token_init,  seg_tokens ← seg_token_init
t=1 : depth_tokens ← enriched from t=0, seg_tokens ← enriched from t=0
t=2 : …
```

**Loss**: `depth_weight × depth_loss + sem_weight × (cross-entropy + 0.5 × Dice)`.

**Trainable parameters**: both decoder stacks + projections + heads (~12 M of 27 M total).

**Launcher**: `run_former_seg_depth.sh`

**Recommended training workflow**:
```bash
bash run_former_seg_depth.sh --single-frame --max-epochs 50  # 1. single-frame pre-train
bash run_former_seg_depth.sh --sequence-length 2 --max-epochs 50  # 2. streaming fine-tune
```

**Key flags**:
| Flag | Default | Description |
|---|---|---|
| `--token-stride` | 8 | Token grid = H/stride × W/stride (power of 2) |
| `--token-dim` | 256 | Token feature dimension (divisible by 4) |
| `--num-decoder-layers` | 6 | Transformer decoder depth (per branch) |
| `--num-heads` | 8 | Attention heads per layer |
| `--num-classes` | 23 | Number of semantic classes |
| `--depth-weight` | 1.0 | Scalar weight on the depth loss term |
| `--sem-weight` | 1.0 | Scalar weight on the semantic loss term |
| `--single-frame` | off | Pre-training mode: no token carry-over |
| `--sequence-length` | 2 | Frames per training window |
| `--img-h / --img-w` | 224 | Must be compatible with TinyViT window sizes |
| `--debug` | off | 1 % data, 1 % val, print tensor shapes |

---

## Architecture comparison

| | encoder | temporal | depth | seg | img_size constraint |
|---|---|---|---|---|---|
| `baseline_depth` | scratch / ResNet (tunable) | none | ✓ | — | no |
| `baseline_seg_depth` | scratch UNet | none | ✓ | ✓ | no |
| `video_seg_depth` | TinyViT (frozen) | LSTM | ✓ | ✓ | yes (224+) |
| `video_seg_depth_resnet` | ResNet (tunable) | LSTM | ✓ | ✓ | no |
| `video_former_depth` | TinyViT (frozen) | depth tokens | ✓ | — | yes (224+) |
| `video_former_seg_depth` | TinyViT (frozen) | depth + seg tokens | ✓ | ✓ | yes (224+) |

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
| `losses.py` | `SILogLoss`, `DiceLoss`, `abs_rel` |
| `visualization.py` | `save_depth_image`, `save_depth_video`, `save_joint_video`, `DepthVizMixin`, `JointVizMixin` |

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
