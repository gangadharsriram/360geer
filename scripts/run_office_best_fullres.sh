#!/usr/bin/env bash
# Full-resolution, best-available-config run for the 360Roam "office" dataset.
#
# --resolution 1 forces true full resolution - office is 2048px wide, so the
# default -1 setting would auto-downscale to 1600px (camera_utils.py:48-55),
# same as every earlier office experiment in this investigation.
#
# NOTE - unlike bricks/gallery_chair, office does NOT have a clean single best
# config. The threshold sweep (exp114-117) found a genuine tradeoff: a
# stricter densify_grad_threshold raises SSIM/PSNR but worsens LPIPS (blurs
# out real near-camera clutter detail - desks/monitors/cables - instead of
# just suppressing narrow-baseline background artifacts). This uses exp116's
# threshold (0.001), the best SSIM/PSNR point found (0.684/19.95 at auto-scaled
# resolution, vs exp114's 0.0002 which had better LPIPS 0.311 but worse
# SSIM/PSNR and visible streaking). Treat this as the current best-available
# pick, not a confirmed clean win like bricks/gallery_chair - a finer sweep
# between 0.0002-0.001 was never completed.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/360Roam/360Roam_openmvg/office}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/run_office_best_fullres}"
GPU="${3:-2}"
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
    --densify_grad_threshold 0.001 \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "run_office_best_fullres done: $OUTPUT_DIR"
