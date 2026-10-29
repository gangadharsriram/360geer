#!/usr/bin/env bash
# Experiment 141: bricks, exp108's config (current best: densify_grad_unscaled,
# threshold 0.00006, lambda_smooth 0.01, lambda_aniso 0.0, lambda_opacity_sparsity
# 0.012 - SSIM 0.8681/PSNR 25.79/LPIPS 0.1653/1.12M Gaussians) + the new
# ray-aligned elongation penalty (lambda_ray_aniso 0.01, ray_aniso_max_ratio
# 3.0 default).
#
# This targets the needle-shape mechanism directly (elongation along each
# Gaussian's own camera ray - the axis narrow-baseline capture leaves
# unconstrained), verified on synthetic data to fire only on ray-aligned
# elongation and leave sideways-thin structures alone. 0.01 is a
# conservative starting weight (matches the original, now-superseded
# lambda_aniso default) - a single clean test before committing to a sweep,
# same pattern as every other new lever this session.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp141_bricks_ray_aniso_001}"
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
    --densify_grad_threshold 0.00006 \
    --lambda_smooth 0.01 \
    --resolution 1 \
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.012 \
    --lambda_ray_aniso 0.01 \
    --ray_aniso_max_ratio 3.0
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp141_bricks_ray_aniso_001 done: $OUTPUT_DIR - compare against exp108 (SSIM 0.8681/PSNR 25.79/LPIPS 0.1653)"
