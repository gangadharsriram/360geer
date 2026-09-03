#!/usr/bin/env bash
# Experiment 4: lower densify_until_iter 15000 -> 12000 (mild, isolated).
# Caps how long redundant stacking in weak-signal regions has to accumulate,
# while leaving 18k iterations of headroom to consolidate/recover (unlike the
# earlier failed attempt at 8000, which left no recovery room and hollowed out
# real geometry).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp4_densify_until_mild}"
GPU="${3:-0}"
ITERS_NUM=30000

export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --eval --disable_viewer \
    --densify_until_iter 12000
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp4_densify_until_mild done: $OUTPUT_DIR"
