#!/usr/bin/env bash
# Experiment 111: exp91's config (bricks' current best pre-sparsity) +
# --lambda_opacity_sparsity 0.1.
#
# Sweep kept climbing with no turnover through 0.012 (exp108: SSIM
# 0.8681/PSNR 25.79/LPIPS 0.1653, best so far). Jumping straight to 0.1 (a
# large step past exp109/110's 0.025/0.05) to sanity-check whether the trend
# holds at a much more aggressive weight or has clearly broken down by then -
# at some point this loss term should start suppressing real, needed opacity
# (everything becomes overly transparent/sparse) rather than just pruning
# redundant Gaussians.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp111_bricks_opacity_sparsity_01}"
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
    --densify_grad_threshold 0.00006 \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.1
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp111_bricks_opacity_sparsity_01 done: $OUTPUT_DIR"
