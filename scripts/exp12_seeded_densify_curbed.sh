#!/usr/bin/env bash
# Experiment 12: seeded point cloud + directly curbed densification window/threshold
# (densify_until_iter 15000->8000, densify_grad_threshold 0.0002->0.0005) - the
# combination not yet tried. exp9/10/11 all left densification volume unchanged,
# so it kept rebuilding the redundant-stacking problem on top of the good seeded
# start (point count still grew 93-175% during the densify window). This directly
# limits how much new material gets added instead of just tweaking shape/opacity.
#
# Note: densify_grad_threshold this high caused catastrophic under-fitting WITHOUT
# seeding (exp3, SSIM 0.22) - testing whether seeding's real initial coverage
# makes that safe to use this time, since the model has a real foundation instead
# of needing to invent everything from scratch.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp12_seeded_densify_curbed}"
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
    --densify_until_iter 8000 \
    --densify_grad_threshold 0.0005
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp12_seeded_densify_curbed done: $OUTPUT_DIR"
