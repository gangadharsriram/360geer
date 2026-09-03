#!/usr/bin/env bash
# Experiment 44: exp42's config (seeded pcd, resets off, densify_grad_threshold
# 0.0003 - 360geer's current best, test SSIM 0.8475) + OmniGS's actual pruning
# strategy instead of "no size-cull at all".
#
# OmniGS beats every 360geer experiment so far on this same dataset (test SSIM
# 0.858, see results/lonlat/EgoNeRF/Ricoh360_bricks/results.json in the OmniGS
# repo) despite using the *default* densify_grad_threshold (0.0002, untouched).
# Traced its actual densifyAndPrune call (gaussian_mapper.cpp:436-442) to two
# independent flags 360geer didn't have until now:
#   - prune_big_point_after_iter=0: screen-space size-cull (max_radii2D>20px)
#     active from essentially the start of training.
#   - prune_by_extent=false: world-space size-cull (scale>0.1*extent) is
#     permanently OFF - this is the exact criterion this investigation found
#     responsible for the one-shot bulk-deletion of ~39% of all Gaussians the
#     moment it first activated (checkpoint inspection, iter 3000 vs 7000).
# exp42 currently has size-cull off ENTIRELY (screen-space included) via
# size_prune_from_iter=30000 - this is new territory, not just a re-run: it
# adds continuous screen-space cleanup (which exp42 lacks) while keeping the
# destructive world-space criterion off, mirroring OmniGS's strategy exactly
# via the new size_prune_from_iter=0 + prune_by_extent=0 flags (gaussian_model.py,
# train.py, arguments/__init__.py).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp44_seeded_no_resets_omnigs_prune}"
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
    --densify_grad_threshold 0.0003 \
    --size_prune_from_iter 0 \
    --prune_by_extent 0
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp44_seeded_no_resets_omnigs_prune done: $OUTPUT_DIR"
