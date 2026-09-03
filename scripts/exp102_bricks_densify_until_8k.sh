#!/usr/bin/env bash
# Experiment 102: exp91's config (bricks' current best) + --densify_until_iter 8000
# (was default 15000) - stop creating new Gaussians earlier, leave the rest
# of training to consolidate/prune instead of keep growing.
#
# Only tested very early (exp4=12000, exp12=8000, exp13=12000), stale
# relative to current baseline (predates seeded pcd/no-resets/threshold
# tuning/lambda_aniso=0). 8000 is exp12's more aggressive value - picked
# since the mild 12000 variant (exp4/13) showed weaker effect back then.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp102_bricks_densify_until_8k}"
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
    --densify_until_iter 8000
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp102_bricks_densify_until_8k done: $OUTPUT_DIR"
