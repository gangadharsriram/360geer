#!/usr/bin/env bash
# Experiment 14: seeded point cloud + opacity_reset_interval 3000->10000 (fewer
# resets: ~2-3 over the full 30k run instead of 9-10). Training snapshots showed a
# SHARP quality discontinuity exactly across the first reset boundary (iter 2500
# crisp, iter 3500 already hazy) - this tests whether giving each region far more
# uninterrupted time to consolidate between resets preserves the seeded quality
# instead of repeatedly re-opening the same unresolved competition in weakly-
# constrained regions.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp14_seeded_fewer_resets}"
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
    --opacity_reset_interval 10000
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp14_seeded_fewer_resets done: $OUTPUT_DIR"
