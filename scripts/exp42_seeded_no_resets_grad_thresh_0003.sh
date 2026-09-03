#!/usr/bin/env bash
# Experiment 42: exp15's config (seeded pcd, resets effectively disabled) +
# exp26's winning generalization lever (densify_grad_threshold 0.0002->0.0003).
#
# exp15 has the best raw test SSIM of any experiment so far (0.835) but a large
# train/test gap (0.955 vs 0.835, 0.120) - it's winning partly by overfitting to
# training views, visible as the "dotted" edge artifact on held-out test renders.
# exp26 (built on a different, resets-normal baseline) showed that raising
# densify_grad_threshold alone cuts that gap by a third (0.078) by creating fewer
# redundant edge Gaussians in the first place, for only a small hit to raw test
# SSIM (0.795). Nobody has combined these two specific levers yet - this tests
# whether exp15's headroom plus exp26's better-generalizing densification
# behavior beats both individually.
#
# IMPORTANT: exp15 originally ran before size_prune_from_iter was split out from
# opacity_reset_interval (they used to be the same parameter, so
# --opacity_reset_interval 30000 also accidentally disabled size-cull for the
# whole run). size_prune_from_iter now defaults independently to 3000, so
# without setting it explicitly here, size-cull would silently re-activate at
# iteration 3000 and this would NOT reproduce exp15's actual behavior - it'd be
# a different (worse) config. --size_prune_from_iter 30000 restores that.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp42_seeded_no_resets_grad_thresh_0003}"
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
    --densify_grad_threshold 0.0003
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp42_seeded_no_resets_grad_thresh_0003 done: $OUTPUT_DIR"
