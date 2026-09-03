#!/usr/bin/env bash
# Experiment 41: same as exp40, but lambda_opacity_sparsity=0.005 instead of
# 0.001 - a stronger dose of the same never-isolated-before regularizer, to
# bracket it rather than guess a single value. If exp40 (0.001) helps but only
# a little, this checks whether more pressure helps further or starts trading
# away real detail (min_opacity=0.005 is the floor these Gaussians already get
# pruned against, so push too hard here and it's effectively the same failure
# mode as exp12/exp3's over-aggressive under-fitting, just from a different lever).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp41_seeded_opacity_sparsity_005}"
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
    --lambda_opacity_sparsity 0.005
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp41_seeded_opacity_sparsity_005 done: $OUTPUT_DIR"
