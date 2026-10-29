#!/usr/bin/env bash
# Experiment 131: SHORT calibration run for the new
# densify_stats_real3d_distscaled statistic (real self._xyz.grad norm,
# scaled by each Gaussian's distance to the current camera - matches
# 3DGRT/gsplat's GSStrategy: ||grad_position|| * distance_to_camera / 2).
# Not a quality test - this scale is unknown ahead of time and unrelated to
# both the proxy statistic's tuned values (e.g. 0.00006) AND plain
# densify_stats_real3d's calibrated scale (mean=5.98e-5 on bricks, exp94) -
# the distance multiplier changes the units entirely.
#
# densify_grad_threshold set absurdly high (999999) so nothing actually
# triggers a split/clone during this short run - the calibration print
# fires regardless (computed before the threshold comparison), so this
# stays a safe, cheap, read-only probe. No render/metrics step - the print
# in the training log is the only thing this run is for.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp131_bricks_real3d_distscaled_calibration}"
GPU="${3:-0}"
ITERS_NUM=1000

echo "== Seeding sparse-coverage directions =="
python scripts/seed_sparse_directions.py "$DATASET_DIR"

export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations "$ITERS_NUM" \
    --save_iterations "$ITERS_NUM" \
    --test_iterations "$ITERS_NUM" \
    --eval --disable_viewer \
    --opacity_reset_interval 30000 \
    --size_prune_from_iter 30000 \
    --densify_grad_unscaled \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --densify_stats_real3d_distscaled 1 \
    --densify_grad_threshold 999999
echo "exp131_bricks_real3d_distscaled_calibration done: $OUTPUT_DIR - check the training log above for the [densify_stats_real3d_distscaled calibration] line"
