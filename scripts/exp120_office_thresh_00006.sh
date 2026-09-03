#!/usr/bin/env bash
# Experiment 120: 360Roam "office" scene, exp114's recipe (seeding +
# densify_grad_unscaled + lambda_aniso 0.0 + lambda_smooth 0.01 +
# lambda_opacity_sparsity 0.012) with densify_grad_threshold 0.0006.
#
# Filling in the gap between the existing bracket points: 0.0002 (exp114,
# SSIM 0.668/PSNR 19.60/LPIPS 0.311/2.75M Gaussians), 0.0005 (exp115, SSIM
# 0.678/PSNR 19.77/LPIPS 0.333/664K), 0.001 (exp116, SSIM 0.684/PSNR 19.95/
# LPIPS 0.361/227K) - a real tradeoff (SSIM/PSNR up, LPIPS down as threshold
# tightens), not a clean peak. This finer sweep is to see whether there's a
# genuine sweet spot inside 0.0002-0.001 or whether it's a smooth monotonic
# tradeoff the whole way through.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/360Roam/360Roam_openmvg/office}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp120_office_thresh_00006}"
GPU="${3:-2}"
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
    --densify_grad_threshold 0.0006 \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp120_office_thresh_00006 done: $OUTPUT_DIR"
