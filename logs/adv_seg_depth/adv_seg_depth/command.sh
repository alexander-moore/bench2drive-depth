================================================================
train_adv.py — resolved configuration
invoked : train_adv.py --model baseline_seg_depth --data-root /workspace/bench2resize --max-epochs 100 --batch-size 4 --num-workers 16 --base-channels 64 --depth-weight 1.0 --sem-weight 1.0 --disc-mode both --adv-weight 0.1 --disc-lr 1e-4 --disc-channels 64 --adv-warmup-epochs 10 --r1-weight 10.0 --log-dir /workspace/logs/adv_seg_depth --checkpoint-dir /workspace/checkpoints --limit-train-batches 500 --val-check-interval 5 --limit-val-batches 0.1 --trial-name adv_seg_depth
================================================================
python train_adv.py \
  --model baseline_seg_depth \
  --data-root /workspace/bench2resize \
  --max-epochs 100 \
  --batch-size 4 \
  --num-workers 16 \
  --prefetch-factor 2 \
  --learning-rate 0.0001 \
  --devices 1 \
  --accelerator auto \
  --gradient-clip-val 1.0 \
  --log-dir /workspace/logs/adv_seg_depth \
  --checkpoint-dir /workspace/checkpoints \
  --patience 10 \
  --trial-name adv_seg_depth \
  --sequence-length 1 \
  --precision 32 \
  --val-check-interval 5 \
  --limit-val-batches 0.1 \
  --limit-train-batches 500 \
  --depth-loss-fn silog \
  --depth-weight 1.0 \
  --sem-weight 1.0 \
  --img-h 0 \
  --img-w 0 \
  --disc-mode both \
  --adv-weight 0.1 \
  --disc-lr 0.0001 \
  --disc-channels 64 \
  --adv-warmup-epochs 10 \
  --r1-weight 10.0 \
  --base-channels 64
================================================================
all flags  (* = non-default):
  --model                 baseline_seg_depth  *
  --data-root             /workspace/bench2resize
  --max-epochs            100
  --batch-size            4
  --num-workers           16
  --prefetch-factor       2
  --learning-rate         0.0001
  --devices               1
  --accelerator           auto
  --gradient-clip-val     1.0
  --log-dir               /workspace/logs/adv_seg_depth  *
  --checkpoint-dir        /workspace/checkpoints
  --patience              10
  --trial-name            adv_seg_depth  *
  --sequence-length       1
  --precision             32
  --val-check-interval    5
  --limit-val-batches     0.1
  --limit-train-batches   500
  --depth-loss-fn         silog
  --depth-weight          1.0
  --sem-weight            1.0
  --img-h                 0
  --img-w                 0
  --single-frame          False
  --disc-mode             both
  --adv-weight            0.1
  --disc-lr               0.0001
  --disc-channels         64
  --adv-warmup-epochs     10
  --r1-weight             10.0
  --base-channels         64
================================================================
