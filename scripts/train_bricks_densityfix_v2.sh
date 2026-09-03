#!/usr/bin/env bash
# Retrain bricks with the retuned (gentler) density/opacity-sparsity parameters:
#   densify_until_iter 15000->12000, densify_grad_threshold 0.0002->0.0005,
#   lambda_opacity_sparsity=0.0001, aniso_hard_ratio=1000 (backstop only)
# All defaults live in arguments/__init__.py - no special flags needed here.
#
# Usage: scripts/train_bricks_densityfix_v2.sh [DATASET_DIR] [OUTPUT_DIR]
set -e

if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first (has the EQUI-patched rasterizer build)." >&2
    echo "Currently active: '${CONDA_DEFAULT_ENV:-none}'. Run: conda activate 360geer" >&2
    exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/bricks_equi_30k_densityfix_v2}"

ITERS_NUM=30000

echo "== Training =="
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --eval \
    --disable_viewer

echo "== Rendering train + test sets =="
python render.py \
    -m "$OUTPUT_DIR" \
    -s "$DATASET_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --iteration "$ITERS_NUM"

echo "== Evaluating metrics (SSIM / PSNR / LPIPS) =="
python metrics.py \
    -m "$OUTPUT_DIR" \
    --block_mask \
    --iters "$ITERS_NUM"

echo "Done. Renders in $OUTPUT_DIR/{train,test}/ours_${ITERS_NUM}/, metrics in $OUTPUT_DIR/results.json"
