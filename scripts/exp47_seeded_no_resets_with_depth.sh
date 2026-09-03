#!/usr/bin/env bash
# Experiment 47: exp42's config (seeded pcd, resets off, densify_grad_threshold
# 0.0003 - 360geer's current best, test SSIM 0.8475) + real depth-map supervision
# via scripts/prepare_depth_maps.py.
#
# The dataset ships raw depth in depthmap/depth_npy/*.npy (depth/100m, confirmed
# by inspecting actual values: range [0.01, 1.0] matching the sibling
# depth_vis_*_100m folders). prepare_depth_maps.py converts these into the
# 16-bit-PNG inverse-depth format train.py's existing (previously unused for
# this dataset) depth-loss path expects, calibrated against this SfM
# reconstruction's own coordinate scale by reprojecting real SfM points into
# every camera and comparing against the depth maps at those pixels (median
# ratio ~1.26 SfM-units-per-metre for this dataset - not assumed, measured).
#
# This directly targets the sparse-view depth ambiguity this whole investigation
# has been fighting (redundant Gaussians stacked at wrong depths in
# weakly-constrained regions) with an actual depth signal instead of only
# indirect levers (pruning criteria, grad threshold, seeding). depth_l1_weight
# decays opt.depth_l1_weight_init->_final (0.1->0.01) over training, left at
# defaults here - a first, untuned test of whether real depth supervision helps
# at all before spending a round tuning its weight.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp47_seeded_no_resets_with_depth}"
GPU="${3:-2}"
ITERS_NUM=30000

echo "== Seeding sparse-coverage directions =="
python scripts/seed_sparse_directions.py "$DATASET_DIR"

echo "== Preparing depth maps (calibrating against SfM point cloud) =="
python scripts/prepare_depth_maps.py "$DATASET_DIR"

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
    --resolution 1 \
    --depths depth_png16
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp47_seeded_no_resets_with_depth done: $OUTPUT_DIR"
