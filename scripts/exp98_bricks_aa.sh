#!/usr/bin/env bash
# Experiment 98: exp91's config (bricks' current best - seeded pcd, resets
# off, no size-cull, --densify_grad_unscaled, densify_grad_threshold 0.00006,
# lambda_smooth 0.01, lambda_aniso 0.0 - SSIM 0.8614/PSNR 25.51/LPIPS 0.1708)
# + --antialiasing.
#
# Antialiasing was tested before (exp66/75/78) but always with the old
# lambda_aniso=0.01 penalty still silently active, which we now know (exp91)
# was hurting quality on its own. Never tested stacked on today's actual best
# config - this isolates whether it still helps now that baseline has moved.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp98_bricks_aa}"
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
    --densify_grad_threshold 0.00006 \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --antialiasing
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp98_bricks_aa done: $OUTPUT_DIR"
