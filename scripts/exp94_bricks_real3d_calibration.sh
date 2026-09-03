#!/usr/bin/env bash
# Experiment 94: SHORT calibration run, not a quality test. exp91's config
# (bricks' current best) + --densify_stats_real3d 1, capped at 1000
# iterations - just enough to hit the first densification check (iteration
# 800 = densify_from_iter 500 + densification_interval 300) and print the
# actual statistic's scale.
#
# The real dL_dmeans/xyz.grad-based statistic (see gaussian_model.py's new
# add_densification_stats_real3d, arguments/__init__.py's densify_stats_real3d)
# lives at a completely different, currently-unknown scale from the old
# fabsf/depth-scaled proxy - the tuned threshold 0.00006 does NOT carry over.
# densify_grad_threshold is set absurdly high here (999999) specifically so
# nothing actually triggers a split/clone during this short run - the
# calibration print fires regardless (it's computed before the threshold
# comparison), so this stays a safe, cheap, read-only probe. No render/metrics
# step - the print output in the training log is the only thing this run is
# for. Once the real scale is known, a real threshold can be picked and a
# proper 30000-iteration quality run built.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp94_bricks_real3d_calibration}"
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
    --densify_stats_real3d 1 \
    --densify_grad_threshold 999999
echo "exp94_bricks_real3d_calibration done: $OUTPUT_DIR - check the training log above for the [densify_stats_real3d calibration] line"
