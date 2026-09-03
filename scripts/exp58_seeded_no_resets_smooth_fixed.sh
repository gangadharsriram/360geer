#!/usr/bin/env bash
# Experiment 58: exp51 (exp42 + edge-aware smoothness regularizer alone,
# lambda_smooth=0.01) + --densify_stats_2d 1.
#
# Same rationale as exp55/56/57, applied to the smoothness lever - exp51 was the
# strongest individually-validated lever found before this fix (test SSIM 0.8493
# vs exp42's 0.8475). Checking whether it stacks with the gradient-statistic fix
# or whether they're targeting overlapping parts of the same redundant-Gaussian
# problem (in which case the combined gain would be sub-additive, matching the
# pattern seen repeatedly across this investigation's earlier lever-stacking
# attempts).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp58_seeded_no_resets_smooth_fixed}"
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
    --densify_grad_threshold 0.0003 \
    --lambda_smooth 0.01 \
    --densify_stats_2d 1
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp58_seeded_no_resets_smooth_fixed done: $OUTPUT_DIR"
