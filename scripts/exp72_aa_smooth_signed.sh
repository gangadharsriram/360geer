#!/usr/bin/env bash
# Experiment 72: exp64's base (seeded pcd, resets off, no size-cull,
# --densify_grad_unscaled --densify_grad_threshold 0.00008 - the confirmed peak
# of the threshold sweep, SSIM 0.8484/PSNR 25.08/LPIPS 0.1838) + antialiasing + lambda_smooth + densify_grad_signed, all stacked - the kitchen-sink combo. Given the repeated pattern in this investigation that stacking validated levers is sub-additive/overshoots, this is a check, not an expectation of the best result.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp72_aa_smooth_signed}"
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
    --densify_grad_threshold 0.00008 \
    --antialiasing \
    --lambda_smooth 0.01 \
    --densify_grad_signed
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp72_aa_smooth_signed done: $OUTPUT_DIR"
