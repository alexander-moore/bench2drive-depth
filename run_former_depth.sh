#!/bin/bash
# VideoFormerDepth training launcher
#
# One "epoch" = 500 gradient steps (limit_train_batches=500).
# Validation runs every 5 epochs on 10% of the val set.
#
# Usage
# ─────
#   bash run_former_depth.sh                          # full streaming training
#   bash run_former_depth.sh --single-frame           # single-frame pre-training mode
#
# Common overrides
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

python train.py --model video_former_depth \
  --data-root /workspace/bench2resize \
  --max-epochs 100 \
  --batch-size 2 \
  --num-workers 16 \
  --sequence-length 2 \
  --token-stride 8 \
  --token-dim 256 \
  --num-decoder-layers 6 \
  --num-heads 8 \
  --log-dir /workspace/project/logs/video_former_depth \
  --checkpoint-dir /workspace/checkpoints \
  --limit-train-batches 500 \
  --val-check-interval 5 \
  --limit-val-batches 0.1 \
  "$@"
