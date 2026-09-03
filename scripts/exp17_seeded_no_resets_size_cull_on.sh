#!/usr/bin/env bash
# Experiment 17: seeded pcd (exp9) + opacity resets disabled, size-based Gaussian
# pruning left on its normal schedule. The complement to exp16 - completes the 2x2
# needed to cleanly separate "opacity reset" from "size-cull gate" as the cause of
# the post-iter-3000 sky/foliage degradation (see exp16 for the full root-cause
# writeup: the two were only coupled because train.py reused opacity_reset_interval
# to gate size_threshold, not because resets themselves are destructive).
#
# size_prune_from_iter is kept at its default (3000), so size-based culling still
# switches on at the normal point and keeps firing every 300 iters. If degradation
# still shows up here despite resets being off, that points back at the size-cull
# gate (matches exp16 being the fix, not this). If this one now stays clean instead,
# opacity reset itself was doing real damage independent of the size gate.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp17_seeded_no_resets_size_cull_on}"
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
    --resolution 1 \
    --opacity_reset_interval 999999
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp17_seeded_no_resets_size_cull_on done: $OUTPUT_DIR"
