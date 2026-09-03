#!/usr/bin/env bash
# Experiment 56: exp43 (seeded pcd, resets off, no size-cull, densify_grad_threshold
# 0.0005 - the tightest train/test gap and most efficient Gaussian count of any
# strong result so far, 1,752,639 points) + --densify_stats_2d 1.
#
# Same rationale as exp55: exp43's threshold was tuned against the uncorrected,
# inflated gradient statistic. Worth re-checking specifically because exp43 was
# already the most efficient/best-generalizing config found pre-fix - if the fix
# compounds with it rather than requiring a re-tune, this could be the new best
# result outright.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp56_seeded_no_resets_grad0005_fixed}"
GPU="${3:-1}"
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
    --densify_grad_threshold 0.0005 \
    --densify_stats_2d 1
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp56_seeded_no_resets_grad0005_fixed done: $OUTPUT_DIR"
