#!/usr/bin/env bash
# Experiment 134: bricks, densify_stats_real3d_distscaled=1 (real gradient
# norm * distance-to-camera / 2, matching 3DGRT/gsplat's GSStrategy), on top
# of exp91's otherwise-best config, threshold 0.0002.
#
# Calibration (exp131): mean=7.18e-5 median=2.21e-5 p90=1.83e-4 max=1.26e-2.
# Bracketing that range to find the right operating point, same approach as
# the plain-real3d threshold sweep (exp95-97) - but this time the statistic
# has the distance-compensation term plain real3d was missing, so the
# expectation (not yet confirmed) is that it won't need to go as loose to
# get real signal in far/ambiguous regions.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp134_bricks_real3d_distscaled_thresh00002}"
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
    --densify_stats_real3d_distscaled 1 \
    --densify_grad_threshold 0.0002
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp134_bricks_real3d_distscaled_thresh00002 done: $OUTPUT_DIR"
