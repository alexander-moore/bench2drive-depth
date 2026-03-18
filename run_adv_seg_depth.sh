#!/bin/bash
# Adversarial baseline_seg_depth training launcher
#
# Adds a PatchGAN ImageDiscriminator on top of baseline_seg_depth.
# The discriminator sees (rgb, depth, semantic) and is trained with
# LSGAN loss + R1 gradient penalty.
#
# Schedule:
#   Epochs 0–9   : reconstruction only (CE + Dice + SILog), discriminator frozen
#   Epochs 10–99 : reconstruction + adversarial (adv_weight=0.1)
#
# One "epoch" = 500 gradient steps (limit_train_batches=500).
# Validation runs every 5 epochs on 10% of the val set.
#
# Usage
# ─────
#   bash run_adv_seg_depth.sh                        # default run
#   bash run_adv_seg_depth.sh --disc-mode depth      # depth discriminator only
#   bash run_adv_seg_depth.sh --disc-mode semantic   # semantic discriminator only
#   bash run_adv_seg_depth.sh --adv-weight 0.05      # lighter adversarial pressure
#   bash run_adv_seg_depth.sh --adv-warmup-epochs 20 # longer warmup
#   bash run_adv_seg_depth.sh --max-epochs 1 --batch-size 1  # quick smoke test
#   bash run_adv_seg_depth.sh --precision bf16-mixed # mixed precision
set -e

source /workspace/venv/bin/activate

echo "launching: $0 $*"

python train_adv.py --model baseline_seg_depth \
  --data-root /workspace/bench2resize \
  --max-epochs 100 \
  --batch-size 4 \
  --num-workers 16 \
  --base-channels 64 \
  --depth-weight 1.0 \
  --sem-weight 1.0 \
  --disc-mode both \
  --adv-weight 0.1 \
  --disc-lr 1e-4 \
  --disc-channels 64 \
  --adv-warmup-epochs 10 \
  --r1-weight 10.0 \
  --log-dir /workspace/project/logs/adv_seg_depth \
  --checkpoint-dir /workspace/checkpoints \
  --limit-train-batches 500 \
  --val-check-interval 5 \
  --limit-val-batches 0.1 \
  "$@"
