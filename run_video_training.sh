#!/bin/bash
# One "epoch" = 500 gradient steps (limit_train_batches=500).
# Validation runs every 5 epochs on 10% of the val set.
#
# Usage:
#   bash run_video_training.sh               # normal training
#   bash run_video_training.sh --debug       # 1% data, 1% val, print tensor shapes
#   bash run_video_training.sh --max-epochs 1 --batch-size 1  # quick smoke test
set -e

source /workspace/venv/bin/activate

echo "launching: $0 $*"

python experiments/video_seg_depth.py \
  --data-root /workspace/bench2resize \
  --max-epochs 100 \
  --batch-size 2 \
  --num-workers 16 \
  --sequence-length 4 \
  --img-h 224 \
  --img-w 224 \
  --lstm-hidden 512 \
  --log-dir /workspace/logs/video_seg_depth \
  --checkpoint-dir /workspace/checkpoints \
  --limit-train-batches 500 \
  --val-check-interval 5 \
  --limit-val-batches 0.1 \
  "$@"
