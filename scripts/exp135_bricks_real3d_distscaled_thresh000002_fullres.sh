#!/usr/bin/env bash
# Experiment 135: same as exp132 (bricks, densify_stats_real3d_distscaled=1,
# threshold 0.00002, on top of exp91's otherwise-best config), but at true
# full resolution (--resolution 1) - bricks is 1920px wide, so the default
# -1 setting would auto-downscale to 1600px (camera_utils.py:48-55), same
# catch as every other full-res run this session.
#
# Calibration (exp131, computed at auto-scaled resolution): mean=7.18e-5
# median=2.21e-5 p90=1.83e-4 max=1.26e-2 - not re-calibrated at full res,
# carried over as a reasonable starting point since the calibration probe's
# per-pixel gradient statistic is not expected to shift drastically with
# resolution alone.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp135_bricks_real3d_distscaled_thresh000002_fullres}"
GPU="${3:-0}"
ITERS_NUM=30000

echo "== Seeding sparse-coverage directions =="
python scripts/seed_sparse_directions.py "$DATASET_DIR"

export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --resolution 1 \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --eval --disable_viewer \
    --opacity_reset_interval 30000 \
    --size_prune_from_iter 30000 \
    --densify_grad_unscaled \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --densify_stats_real3d_distscaled 1 \
    --densify_grad_threshold 0.00002
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp135_bricks_real3d_distscaled_thresh000002_fullres done: $OUTPUT_DIR"
