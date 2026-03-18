#!/bin/bash
# VideoFormerDepth training launcher
#
# One "epoch" = 500 gradient steps (limit_train_batches=500).
# Validation runs every 5 epochs on 10% of the val set.
#
# Usage
# ─────
#   bash run_former_depth.sh                          # full streaming training
#   bash run_former_depth.sh --debug                  # 1% data/val + tensor shape table
#   bash run_former_depth.sh --single-frame           # single-frame pre-training mode
#   bash run_former_depth.sh --single-frame --debug   # pre-train with shape logging
#
# Common overrides (append after any of the above)
# ─────────────────────────────────────────────────
#   --max-epochs 1 --batch-size 1                     # quick smoke test
#   --sequence-length 4                               # longer temporal window
#   --token-stride 4                                  # finer token grid (H/4 × W/4)
#   --token-dim 128                                   # smaller/faster model
#   --num-decoder-layers 3                            # shallower decoder
#   --precision bf16-mixed                            # mixed precision
set -e

source /workspace/venv/bin/activate

echo "launching: $0 $*"

python experiments/video_former_depth.py \
  --data-root /workspace/bench2resize \
  --max-epochs 100 \
  --batch-size 2 \
  --num-workers 16 \
  --sequence-length 2 \
  --img-h 224 \
  --img-w 224 \
  --token-stride 8 \
  --token-dim 256 \
  --num-decoder-layers 6 \
  --num-heads 8 \
  --log-dir /workspace/logs/video_former_depth \
  --checkpoint-dir /workspace/checkpoints \
  --limit-train-batches 500 \
  --val-check-interval 5 \
  --limit-val-batches 0.1 \
  "$@"
