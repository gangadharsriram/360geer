#!/usr/bin/env bash
# Experiment 55: exp42 (seeded pcd, resets off, no size-cull, densify_grad_threshold
# 0.0003) + --densify_stats_2d 1.
#
# exp42 was 360geer's reference baseline for the whole exp42-51 "best config"
# lineage, run before the densification gradient-statistic bug was found (see
# exp52/54: at a matched threshold, densify_stats_2d=1 gave 20.8% fewer Gaussians
# and better SSIM/PSNR/LPIPS across the board). exp42 used 0.0003 specifically
# because it was tuned (via exp26/27/28) to partially compensate for the
# inflated, uncorrected gradient statistic - now that part of that inflation is
# fixed, 0.0003 may no longer be the right threshold, but this keeps every other
# variable matched to exp42 for a direct before/after comparison first.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp55_seeded_no_resets_grad0003_fixed}"
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
    --densify_grad_threshold 0.0003 \
    --densify_stats_2d 1
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp55_seeded_no_resets_grad0003_fixed done: $OUTPUT_DIR"
