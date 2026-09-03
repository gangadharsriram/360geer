#!/usr/bin/env bash
# Experiment 89: gallery_chair, pure threshold sweep with
# --densify_grad_unscaled alone (no smooth/aa/signed), extending exp82-86's
# sweep upward (stricter) - that sweep showed the OPPOSITE trend from bricks:
# every step looser than 0.0001 made quality monotonically worse (exp82 at
# 0.0001/575K Gaussians was the best of that sweep, still climbing, not yet
# turned over). Matches the old-statistic baseline's own threshold (gallery_chair_best_config.sh) - direct apples-to-apples on the SAME threshold number, corrected statistic vs old buggy one.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/gallery_chair}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp89_gallery_chair_thresh0003}"
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
    --densify_grad_unscaled \
    --densify_grad_threshold 0.0003
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp89_gallery_chair_thresh0003 done: $OUTPUT_DIR"
