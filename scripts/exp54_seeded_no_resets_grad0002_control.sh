#!/usr/bin/env bash
# Experiment 54: matched-threshold control for exp52. Identical to exp52
# (seeded pcd, resets off, no size-cull, densify_grad_threshold left at its
# default 0.0002) but WITHOUT --densify_stats_2d - i.e. the original 3-component
# gradient statistic (add_densification_stats, the one train.py has always used),
# at the exact same 0.0002 threshold exp52 used.
#
# exp52 (2-component fix, 0.0002) came back with MORE Gaussians than exp42
# (3-component, tuned 0.0003 threshold) - 4,284,920 vs 3,221,638 - and slightly
# worse SSIM (0.8414 vs 0.8475). That result is confounded: exp52 changed BOTH
# the statistic (3->2 components) AND the threshold (0.0003->0.0002, more
# permissive on its own) relative to exp42, so it couldn't tell us whether the
# fix itself helped, hurt, or did nothing.
#
# This run isolates that: same 0.0002 threshold as exp52, fix OFF. Comparing
# THIS run's Gaussian count/SSIM directly against exp52's (both at 0.0002, only
# --densify_stats_2d differs) gives a clean read on whether switching from 3 to
# 2 gradient components does anything on its own, without the threshold change
# muddying it.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp54_seeded_no_resets_grad0002_control}"
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
    --size_prune_from_iter 30000
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp54_seeded_no_resets_grad0002_control done: $OUTPUT_DIR"
