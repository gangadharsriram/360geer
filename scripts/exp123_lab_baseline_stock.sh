#!/usr/bin/env bash
# Experiment 123: 360Roam "lab" scene, fully STOCK 360geer defaults -
# no seeding, no densify_grad_unscaled, no lambda_aniso=0, no lambda_smooth,
# no lambda_opacity_sparsity. Direct equivalent of exp113 (office) and
# exp122 (turtlebot) - a clean before-fix baseline.
#
# lab is one of office's narrowest 360Roam siblings (measured via the
# 43-scene camera-baseline survey): mean nearest-neighbor camera spacing
# well below the rest of the 360Roam family (which ranges up to 44.8cm at
# center1), though still wider than office itself (2.4cm, the narrowest
# scene in the whole survey).
#
# --resolution 1 forces true full resolution - lab is 2048px wide, so the
# default -1 setting would auto-downscale to 1600px (camera_utils.py:48-55).
# NOTE: unlike this, office's stock baseline (exp113) and turtlebot's
# (exp122) were run WITHOUT --resolution 1 (auto-downscaled) - so this run's
# numbers are not directly comparable to those two, only to exp124 (lab's
# own recipe run, same resolution).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/360Roam/360Roam_openmvg/lab}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp123_lab_baseline_stock}"
GPU="${3:-0}"
ITERS_NUM=30000

export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --resolution 1 \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --eval --disable_viewer
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp123_lab_baseline_stock done: $OUTPUT_DIR"
