#!/usr/bin/env bash
# Experiment 113: 360Roam "office" scene, fully STOCK 360geer/3DGEER defaults -
# no seed_sparse_directions.py, no densify_grad_unscaled, no lambda_aniso=0,
# no lambda_smooth, no lambda_opacity_sparsity. Every flag left at its
# argparse default (densify_grad_threshold=0.0002, opacity_reset_interval=3000,
# lambda_aniso=0.01, etc.) - the direct equivalent of running 3DGEER's own
# scripts/train.sh unmodified on its ScanNet++ scene, but on this new,
# narrower-baseline (2.4cm mean camera spacing, 1.22x0.03x0.72m extent -
# measured via data_extrinsics.json) Ricoh/360Roam-style dataset instead.
#
# Purpose: confirm/deny whether the narrow-baseline-driven scattered-dot
# artifact (established on bricks/gallery_chair, absent on 3DGEER's own
# room-scale scene) generalizes to a third, independent narrow-baseline
# scene, before assuming the same fixes (densify_grad_unscaled + lambda_aniso
# 0.0 + lambda_smooth 0.01 + lambda_opacity_sparsity 0.012) are needed here
# too. Deliberately does NOT apply any of this investigation's fixes yet -
# this is the "does the problem exist here at all" check, not a quality run.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/360Roam/360Roam_openmvg/office}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp113_office_baseline_stock}"
GPU="${3:-0}"
ITERS_NUM=30000

export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --eval --disable_viewer
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp113_office_baseline_stock done: $OUTPUT_DIR"
