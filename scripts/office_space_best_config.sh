#!/usr/bin/env bash
# Office_space dataset, using 360geer's current best validated config (exp42 on
# the bricks dataset: seeded pcd, resets effectively disabled, densify_grad_threshold
# 0.0003 - test SSIM 0.8475 there, beating every other lever/combo tried).
#
# This is the best TRANSFERRED setting, not something separately tuned for this
# scene - office_space is a genuinely different capture (28 train / 8 test images,
# vs bricks' 50/50; 640x320 native resolution vs 1920x960; 2585 initial SfM points
# vs 5958). Worth watching for: an indoor office scene likely has far better SfM
# coverage than an outdoor scene with open sky (COLMAP/OpenMVG can triangulate
# walls/furniture/ceiling far more reliably than featureless sky), so the sparse-
# view redundant-Gaussian-stacking problem this whole investigation has been
# fighting may simply be less severe here - or the much smaller train set (28
# images) could make it worse by cutting baseline diversity. No --resolution 1
# needed: native width (640) is already under the 1600px auto-downscale threshold.
#
# If results here don't match bricks' quality, the same systematic single-lever
# sweep methodology (scripts/exp20-41) can be re-run against this dataset rather
# than assuming the bricks-tuned settings transfer perfectly.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Own/Office_space/sfm/output}"
OUTPUT_DIR="${2:-./output/office_space_best_config}"
GPU="${3:-0}"
ITERS_NUM=30000

echo "== Seeding sparse-coverage directions =="
python scripts/seed_sparse_directions.py "$DATASET_DIR"

export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --eval --disable_viewer \
    --opacity_reset_interval 30000 \
    --size_prune_from_iter 30000 \
    --densify_grad_threshold 0.0003
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "office_space_best_config done: $OUTPUT_DIR"
