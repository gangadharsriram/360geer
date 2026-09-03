#!/usr/bin/env bash
# Experiment 38: same config as exp15 (seeded pcd + opacity resets effectively
# disabled via opacity_reset_interval=30000, which also disables the coupled
# size-cull gate for the whole run - exp15 is currently the best SSIM of any
# experiment so far, 0.835), but at the dataset's native 1920x960 resolution
# instead of the auto-downscaled 1600x800 (utils/camera_utils.py loadCam:
# images wider than 1600px get silently rescaled by orig_w/1600 unless
# --resolution is set explicitly). render.py/metrics.py reload resolution from
# the saved cfg_args automatically, so it only needs setting on the train.py call.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp38_seeded_no_resets_fullres}"
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
    --resolution 1 \
    --opacity_reset_interval 30000
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp38_seeded_no_resets_fullres done: $OUTPUT_DIR"
