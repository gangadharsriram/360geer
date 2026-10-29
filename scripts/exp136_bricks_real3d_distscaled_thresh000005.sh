#!/usr/bin/env bash
# Experiment 136: bricks, densify_stats_real3d_distscaled=1, threshold 0.00005,
# on top of exp91's otherwise-best config.
#
# Finer sweep around exp133's peak (7e-5, SSIM 0.857/PSNR 25.21/LPIPS 0.186,
# 587K Gaussians) - the 3-point bracket (2e-5/7e-5/2e-4) was non-monotonic
# with a peak near 7e-5 (looser overshoots into needle-scatter at 2e-5,
# stricter under-densifies at 2e-4), so this fills in both sides to find the
# actual peak and see whether it can close the remaining gap to exp108
# (SSIM 0.868/PSNR 25.79/LPIPS 0.165, 1.12M Gaussians).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp136_bricks_real3d_distscaled_thresh000005}"
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
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --densify_stats_real3d_distscaled 1 \
    --densify_grad_threshold 0.00005
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp136_bricks_real3d_distscaled_thresh000005 done: $OUTPUT_DIR"
