#!/usr/bin/env bash
# Experiment 114: 360Roam "office" scene, the full best-known recipe from the
# bricks/gallery_chair investigation, applied to a third scene for the first
# time - seed_sparse_directions.py + densify_grad_unscaled + lambda_aniso 0.0
# + lambda_smooth 0.01 + lambda_opacity_sparsity 0.012, resets off, no
# size-cull. Threshold left at 360geer's stock default (0.0002) as a starting
# point - office has no tuned threshold yet (bricks needed 0.00006,
# gallery_chair needed 0.0001, and this investigation has already shown
# threshold does NOT transfer between scenes) - this run's SSIM/PSNR/LPIPS
# and Gaussian count will tell us which direction (looser/stricter) to sweep
# next, same as the original bricks threshold search.
#
# Compare against exp113 (this scene's fully stock baseline, SSIM
# 0.665/PSNR 19.2/LPIPS 0.348, visibly scattered/shard artifacts on
# near-camera clutter).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/360Roam/360Roam_openmvg/office}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp114_office_full_recipe}"
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
    --densify_grad_unscaled \
    --densify_grad_threshold 0.0002 \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp114_office_full_recipe done: $OUTPUT_DIR"
