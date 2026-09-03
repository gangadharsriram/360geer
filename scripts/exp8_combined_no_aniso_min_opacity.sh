#!/usr/bin/env bash
# Experiment 8: combine the two independently-validated winners -
# lambda_aniso=0 (exp1, SSIM 0.679) + min_opacity=0.02 (exp6, SSIM 0.658) -
# to see if the improvements stack, and whether the combination also helps
# the separate weak-wall region neither one fixed alone.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp8_combined_no_aniso_min_opacity}"
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
    --lambda_aniso 0.0 \
    --min_opacity 0.02
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp8_combined_no_aniso_min_opacity done: $OUTPUT_DIR"
