#!/usr/bin/env bash
# Experiment 115: 360Roam "office" scene, exp114's recipe (seeding +
# densify_grad_unscaled + lambda_aniso 0.0 + lambda_smooth 0.01 +
# lambda_opacity_sparsity 0.012) but with densify_grad_threshold 0.0005
# instead of exp114's stock 0.0002.
#
# exp114 at the stock threshold (0.0002) made Gaussian count EXPLODE (625K
# stock-baseline -> 2.75M) instead of shrinking like it did on bricks -
# opposite direction from bricks/gallery_chair. This scene's over-triggering
# problem is dominated by near-camera clutter (desks/cables/monitors), not
# distant background, so it likely needs a much stricter threshold to control
# it directly rather than relying on densify_grad_unscaled's depth-scaling
# removal (which mainly helps distant/background over-triggering).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/360Roam/360Roam_openmvg/office}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp115_office_thresh_00005}"
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
    --densify_grad_threshold 0.0005 \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp115_office_thresh_00005 done: $OUTPUT_DIR"
