#!/usr/bin/env bash
# Experiment 112: gallery_chair, exp92's config (its own current best: seeded
# pcd, resets off, no size-cull, --densify_grad_unscaled, densify_grad_threshold
# 0.0001 - gallery_chair's own tuned value, NOT bricks' 0.00006, since
# threshold has already been shown not to transfer between scenes - lambda_aniso
# 0.0, SSIM 0.8954/PSNR 28.65/LPIPS 0.1673) + --lambda_smooth 0.01
# + --lambda_opacity_sparsity 0.012.
#
# Cross-dataset check of bricks' best-found recipe (exp108): the opacity-mass
# sparsity regularizer swept 0.0003->0.1 on bricks, peaking cleanly at 0.012
# (SSIM 0.8649->0.8681, turnover confirmed both sides). lambda_smooth 0.01
# was validated on bricks but never isolated on gallery_chair - stacking both
# here to test the full recipe transfer in one shot, on top of gallery_chair's
# own threshold (the one part of this investigation confirmed scene-specific).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/gallery_chair}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp112_gallery_chair_opacity_sparsity_012}"
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
    --densify_grad_threshold 0.0001 \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp112_gallery_chair_opacity_sparsity_012 done: $OUTPUT_DIR"
