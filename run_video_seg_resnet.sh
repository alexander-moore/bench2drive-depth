#!/bin/bash
# ResNet Video Seg+Depth training launcher
#
# One "epoch" = 500 gradient steps (limit_train_batches=500).
# Validation runs every 5 epochs on 10% of the val set.
#
# Usage:
#   bash run_video_seg_resnet.sh                   # normal training
#   bash run_video_seg_resnet.sh --backbone resnet34
set -e

source /workspace/venv/bin/activate

echo "launching: $0 $*"

python train.py --model video_seg_depth_resnet \
  --data-root /workspace/bench2resize \
  --max-epochs 100 \
  --batch-size 2 \
  --num-workers 16 \
  --sequence-length 4 \
  --backbone resnet18 \
  --lstm-hidden 512 \
  --log-dir /workspace/project/logs/video_seg_depth_resnet \
  --checkpoint-dir /workspace/checkpoints \
  --limit-train-batches 500 \
  --val-check-interval 5 \
  --limit-val-batches 0.1 \
  "$@"
