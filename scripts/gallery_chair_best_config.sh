#!/usr/bin/env bash
# gallery_chair dataset, using 360geer's current best validated config (exp42 on
# the bricks dataset: seeded pcd, resets effectively disabled, densify_grad_threshold
# 0.0003 - test SSIM 0.8475 there, beating every other lever/combo tried).
#
# Structurally much closer to bricks than office_space was: same 1920x960 native
# resolution (auto-downscaled to 1600x800, matching exp42's own setup - no
# --resolution 1 here, consistent with the reference config), same 50/50 train/
# test split, similar initial point-cloud density (6564 points vs bricks' 5958).
# Best transferred setting, not separately tuned for this scene yet - if results
# don't match bricks' quality, the same systematic single-lever sweep methodology
# (scripts/exp20-41) can be re-run against this dataset.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/gallery_chair}"
OUTPUT_DIR="${2:-./output/gallery_chair_best_config}"
GPU="${3:-1}"
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
    --densify_grad_threshold 0.0003
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "gallery_chair_best_config done: $OUTPUT_DIR"
