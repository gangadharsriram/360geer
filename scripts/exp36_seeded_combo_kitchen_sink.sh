#!/usr/bin/env bash
# Experiment 36: COMBO (kitchen sink): prune_until_iter=30000 + min_opacity=0.02
# + densify_grad_threshold=0.0005 - all three levers stacked. Riskiest of the
# set (exp8 showed validated single levers don't always stack additively), but
# cheap to include alongside the isolated tests above.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp36_seeded_combo_kitchen_sink}"
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
    --resolution 1 \
    --size_prune_from_iter 8000 \
    --prune_until_iter 30000 \
    --min_opacity 0.02 \
    --densify_grad_threshold 0.0005
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp36_seeded_combo_kitchen_sink done: $OUTPUT_DIR"
