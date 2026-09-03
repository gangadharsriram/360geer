#!/usr/bin/env bash
# Experiment 46: exp42's config (seeded pcd, resets off, densify_grad_threshold
# 0.0003 - 360geer's current best, test SSIM 0.8475) + densification_interval
# 300->100, matching OmniGS's own config value for this exact dataset
# (cfg/lonlat/egonerf_lonlat_ricoh360.yaml: Optimization.densification_interval: 100).
#
# The other OmniGS-inspired lever (prune_by_extent split, exp44/exp45) barely
# moved the needle (0.8475 -> 0.8483, +0.0008) - nowhere near closing the gap to
# OmniGS's own 0.858 on this dataset. This tests the other untested config
# difference: OmniGS checks/densifies/prunes 3x more often (every 100 iterations
# instead of 300), which could matter independent of which criteria are checked -
# more frequent pruning means redundant/oversized Gaussians get caught sooner
# after forming instead of accumulating for up to 300 iterations first, and more
# frequent clone/split means gradient-driven growth responds faster to the
# current error signal instead of a stale one.
#
# Isolated against exp42 directly (not exp45) since exp45's own lever was shown
# to be near-noop - keeping this a clean one-variable test against the last
# well-established reference point.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp46_seeded_no_resets_densify_interval_100}"
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
    --densification_interval 100
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp46_seeded_no_resets_densify_interval_100 done: $OUTPUT_DIR"
