#!/usr/bin/env bash
# Experiment 57: exp49 (exp42 + antialiasing, previously the best result at
# SSIM 0.8489) + --densify_stats_2d 1.
#
# Same rationale as exp55/56, applied to the antialiasing lever specifically -
# checking whether the gradient-statistic fix and antialiasing's benefit stack,
# or whether antialiasing's gain was partly masking/interacting with the same
# redundant-Gaussian population this fix targets.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp57_seeded_no_resets_antialiasing_fixed}"
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
    --densify_grad_threshold 0.0003 \
    --antialiasing \
    --densify_stats_2d 1
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp57_seeded_no_resets_antialiasing_fixed done: $OUTPUT_DIR"
