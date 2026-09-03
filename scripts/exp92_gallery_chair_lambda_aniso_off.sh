#!/usr/bin/env bash
# Experiment 92: gallery_chair, exp82's config (its own best pure-threshold
# result: seeded pcd, resets off, no size-cull, --densify_grad_unscaled,
# densify_grad_threshold 0.0001, SSIM 0.8812/PSNR 28.09/LPIPS 0.182,
# Gaussians 575,586) + --lambda_aniso 0.0.
#
# Cross-dataset confirmation of exp91's bricks finding: turning off the
# still-silently-active soft anisotropy hinge loss (lambda_aniso, default
# 0.01) improved SSIM/PSNR/LPIPS simultaneously AND cut the Gaussian count
# 32% on bricks (exp76->exp91) - the first result in this whole
# investigation to beat the old buggy-statistic baseline on all three
# metrics at once. This checks whether that holds on a second scene before
# treating lambda_aniso=0.0 as the new default. No lambda_smooth stacked in
# here - keeping this a single-variable test against exp82, the same way
# exp91 isolated it against exp76 on bricks.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/gallery_chair}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp92_gallery_chair_lambda_aniso_off}"
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
    --densify_grad_threshold 0.0001 \
    --lambda_aniso 0.0
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp92_gallery_chair_lambda_aniso_off done: $OUTPUT_DIR"
