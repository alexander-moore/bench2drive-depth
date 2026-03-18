#!/bin/bash
# One "epoch" = 500 gradient steps (limit_train_batches=500).
# Validation runs every 5 epochs on 10% of the val set.
set -e

source /workspace/venv/bin/activate

echo "launching: $0 $*"

python train.py --model baseline_depth \
  --data-root /workspace/bench2resize \
  --max-epochs 100 \
  --batch-size 4 \
  --num-workers 16 \
  --base-channels 64 \
  --log-dir /workspace/project/logs/baseline_depth \
  --checkpoint-dir /workspace/checkpoints \
  --limit-train-batches 500 \
  --val-check-interval 5 \
  --limit-val-batches 0.05 \
  "$@"
