#!/usr/bin/env bash
# Experiment 145: exp108's exact recipe (bricks' current #1, SSIM 0.8681/
# PSNR 25.79/LPIPS 0.1653) but with lambda_smooth's edge-aware weighting
# switched from the raw GT-image gradient to the precomputed illumination-
# invariant structure prior (Structure Prior Extraction/, CIConv 'W',
# scale=0.8) via the new --smooth_prior_dir flag (train.py).
#
# Motivation: the raw-gradient weighting can't tell a real structural edge
# from a shading gradient - e.g. the brick sculpture's curved, uniformly-
# colored surface produces smooth RGB gradients from shading alone, which
# the old weighting reads as "no edge here" (same as flat background) even
# though it's genuinely hard geometry. The CIConv prior is specifically
# built to suppress illumination gradients while keeping real structural
# edges, so this should give the smoothness penalty better-targeted "where
# NOT to smooth" information without changing anything else in the recipe.
# Same --lambda_smooth 0.01 weight as exp108 kept unchanged for a clean,
# single-variable test of just the weighting source.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp145_bricks_smooth_prior}"
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
    --test_iterations 3000 7000 15000 20000 25000 30000 \
    --eval --disable_viewer \
    --opacity_reset_interval 30000 \
    --size_prune_from_iter 30000 \
    --densify_grad_unscaled \
    --densify_grad_threshold 0.00006 \
    --lambda_smooth 0.01 \
    --smooth_prior_dir "$DATASET_DIR/prior_W_0.8" \
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp145_bricks_smooth_prior done: $OUTPUT_DIR - compare against exp108 (SSIM 0.8681/PSNR 25.79/LPIPS 0.1653)"
