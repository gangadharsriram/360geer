#!/usr/bin/env bash
# Experiment 100: exp91's config (bricks' current best) + --lambda_opacity_sparsity 0.001.
#
# Small penalty on total opacity mass (gaussians.get_opacity.mean()), pushing
# the model toward explaining a spot with fewer, more confident Gaussians
# instead of many faint overlapping ones - existing min_opacity=0.005 prune
# then cleans up what this pushes down. 0.001 was the "middle" value tested
# in the old exp40 (vs exp7's gentle 0.00002, exp41's aggressive 0.005), but
# all three predate seeded pcd/no-resets/threshold-tuning/lambda_aniso=0 -
# never tested stacked on today's actual best config.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp100_bricks_opacity_sparsity}"
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
    --densify_grad_threshold 0.00006 \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.001
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp100_bricks_opacity_sparsity done: $OUTPUT_DIR"
