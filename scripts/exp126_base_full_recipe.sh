#!/usr/bin/env bash
# Experiment 126: 360Roam "base" scene, the full best-known recipe -
# seed_sparse_directions.py + densify_grad_unscaled + lambda_aniso 0.0 +
# lambda_smooth 0.01 + lambda_opacity_sparsity 0.012, resets off, no
# size-cull. Threshold left at 360geer's stock default (0.0002) as a
# starting point, same approach as exp114 (office's first recipe attempt) -
# base has no tuned threshold yet.
#
# Compare against the paired stock baseline in this same batch.
#
# --resolution 1 forces true full resolution - base is 2048px wide, so the
# default -1 setting would auto-downscale to 1600px (camera_utils.py:48-55).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/360Roam/360Roam_openmvg/base}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp126_base_full_recipe}"
GPU="${3:-0}"
ITERS_NUM=30000

echo "== Seeding sparse-coverage directions =="
python scripts/seed_sparse_directions.py "$DATASET_DIR"

export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --resolution 1 \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --eval --disable_viewer \
    --opacity_reset_interval 30000 \
    --size_prune_from_iter 30000 \
    --densify_grad_unscaled \
    --densify_grad_threshold 0.0002 \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp126_base_full_recipe done: $OUTPUT_DIR"
