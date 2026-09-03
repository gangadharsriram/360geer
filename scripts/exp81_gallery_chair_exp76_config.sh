#!/usr/bin/env bash
# Experiment 81: gallery_chair dataset, using exp76's config - the best result
# of the whole bricks-tuned investigation (seeded pcd, resets off, no size-cull,
# --densify_grad_unscaled, densify_grad_threshold 0.00006, lambda_smooth 0.01;
# bricks result: SSIM 0.8498/PSNR 25.07/LPIPS 0.1824).
#
# Cross-dataset check: everything from exp42 onward has been tuned on bricks
# alone. This transfers the winning config as-is (no re-tuning) to see whether
# the gains hold on a second scene before treating it as the new default.
# Structurally close to bricks (same 1920x960 native res, same 50/50 split,
# similar initial point count) - same reasoning gallery_chair_best_config.sh
# used when transferring exp42's config here.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/gallery_chair}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp81_gallery_chair_exp76_config}"
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
    --lambda_smooth 0.01
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp81_gallery_chair_exp76_config done: $OUTPUT_DIR"
