#!/usr/bin/env bash
# Experiment 99: exp91's config (bricks' current best) + --densify_stats_2d 1.
#
# densify_stats_2d strips only the "3 components instead of 2" part of the
# original inflated densification statistic (still fabsf + depth-scaled).
# Tested before (exp52/53/60) but only in very early runs - before seeded
# points, before threshold tuning, before lambda_aniso=0. Never tested
# stacked on today's actual best config.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp99_bricks_stats2d}"
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
    --size_prune_from_iter 30000 \
    --densify_grad_unscaled \
    --densify_grad_threshold 0.00006 \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --densify_stats_2d 1
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp99_bricks_stats2d done: $OUTPUT_DIR"
