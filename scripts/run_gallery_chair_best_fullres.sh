#!/usr/bin/env bash
# Full-resolution, best-known-config run for the gallery_chair dataset.
#
# --resolution 1 forces true full resolution - gallery_chair is 1920px wide,
# so the default -1 setting would auto-downscale to 1600px (camera_utils.py:48-55),
# same as every earlier gallery_chair experiment in this investigation.
#
# Config = exp112, the confirmed best result on gallery_chair (SSIM 0.8980/
# PSNR 28.82/LPIPS 0.1667 at auto-scaled resolution) - the bricks recipe
# transferred cleanly here, only densify_grad_threshold stayed scene-specific
# (0.0001, vs bricks' 0.00006 - confirmed not to transfer directly).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/gallery_chair}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/run_gallery_chair_best_fullres}"
GPU="${3:-1}"
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
    --densify_grad_threshold 0.0001 \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "run_gallery_chair_best_fullres done: $OUTPUT_DIR"
