#!/usr/bin/env bash
# Experiment 51: exp42's config (no antialiasing) + --lambda_smooth 0.01 alone -
# the isolated test of option 1 (edge-aware training-time smoothness regularizer)
# that exp50 never actually ran, since exp50 stacked it on top of antialiasing.
#
# exp49 (antialiasing alone) landed at SSIM 0.8489/@30000, 0.8607/@7000.
# exp50 (antialiasing + smoothness) landed at 0.8495/@30000, 0.8602/@7000 - only
# +0.0006 over exp49 at 30000, and *worse* than exp49 at 7000. With both levers
# stacked, it's impossible to tell whether that tiny 30000 delta is smoothness
# actually helping or just noise. This isolates lambda_smooth on its own against
# the exp42 baseline (0.8475/@30000, 0.8591/@7000) to get a real answer before
# concluding anything about whether the idea works.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp51_seeded_no_resets_smooth_only}"
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
    --densify_grad_threshold 0.0003 \
    --lambda_smooth 0.01
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp51_seeded_no_resets_smooth_only done: $OUTPUT_DIR"
