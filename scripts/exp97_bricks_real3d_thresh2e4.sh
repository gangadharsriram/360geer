#!/usr/bin/env bash
# Experiment 97: exp91's config (bricks' current best - seeded pcd,
# resets off, no size-cull, --densify_grad_unscaled, lambda_smooth 0.01,
# lambda_aniso 0.0) + --densify_stats_real3d 1 (real self._xyz.grad-based
# statistic instead of the fabsf/depth-scaled proxy) + densify_grad_threshold
# 0.0002.
#
# Calibrated against exp94's iter-600 real-statistic readout: mean=5.98e-5,
# median=1.33e-5, p90=1.38e-4, max=2.7e-2 (heavily right-skewed). Above p90 - strict/selective; also happens to match the old proxy statistic's own original default value, a useful reference point.
# Note the calibration was measured early (iter 600, 6137 initial points) -
# real position gradients typically shrink as training converges, so a fixed
# threshold may behave more/less selectively later in training than this
# early snapshot suggests.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp97_bricks_real3d_thresh2e4}"
GPU="${3:-2}"
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
    --densify_grad_unscaled \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --densify_stats_real3d 1 \
    --densify_grad_threshold 0.0002
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp97_bricks_real3d_thresh2e4 done: $OUTPUT_DIR"
