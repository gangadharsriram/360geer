#!/usr/bin/env bash
# Experiment 91: exp76's config (bricks' current best result - seeded pcd,
# resets off, no size-cull, --densify_grad_unscaled, densify_grad_threshold
# 0.00006, lambda_smooth 0.01, SSIM 0.8498/PSNR 25.07/LPIPS 0.1824) +
# --lambda_aniso 0.0.
#
# The soft scale-anisotropy hinge loss (OptimizationParams.lambda_aniso,
# default 0.01, penalizing scale_ratio above aniso_soft_ratio=10.0 in
# train.py) has been silently active through this entire investigation
# (exp42-exp90) without ever being revisited. It's a leftover from an
# earlier, pre-this-session fix attempt that was found to redistribute the
# redundant-Gaussian problem rather than fix it (made things measurably
# worse at the time: SSIM 0.667->0.616). This isolates whether it's still
# hurting on top of everything since found - the hard aniso cutoff in
# densify_and_prune (aniso_hard_ratio=50.0, aniso_prune_from_iter) is left
# untouched, only the soft penalty is disabled.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp91_bricks_lambda_aniso_off}"
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
    --lambda_aniso 0.0
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp91_bricks_lambda_aniso_off done: $OUTPUT_DIR"
