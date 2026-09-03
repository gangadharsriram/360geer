#!/usr/bin/env bash
# Resumes exp50 from its chkpnt15000.pth (cancelled partway through) instead of
# restarting from scratch. --start_checkpoint restores both the Gaussian model
# state AND the iteration counter (train.py:53-54: (model_params, first_iter) =
# torch.load(checkpoint)), so this continues exactly where exp50 left off, not
# a fresh run. Same exact flags as exp50_seeded_no_resets_edge_smooth.sh - no
# structural change happens after iteration 15000 anyway in this config
# (densify_until_iter/prune_until_iter both default to 15000), so the remaining
# 15000-30000 stretch is pure gradient-descent refinement either way.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp50_seeded_no_resets_edge_smooth}"
GPU="${3:-0}"
ITERS_NUM=30000

if [ ! -f "$OUTPUT_DIR/chkpnt15000.pth" ]; then
    echo "ERROR: $OUTPUT_DIR/chkpnt15000.pth not found - nothing to resume from." >&2
    exit 1
fi

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
    --antialiasing \
    --lambda_smooth 0.01 \
    --start_checkpoint "$OUTPUT_DIR/chkpnt15000.pth"
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp50 resumed and completed: $OUTPUT_DIR"
