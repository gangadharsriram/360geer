#!/usr/bin/env bash
# Experiment 18: same config as exp14 (seeded pcd + opacity_reset_interval 10000),
# but at the dataset's native 1920x960 resolution instead of the auto-downscaled
# 1600x800 (utils/camera_utils.py loadCam: any source image wider than 1600px gets
# silently rescaled by orig_w/1600 unless --resolution is set - every prior
# experiment in this run has been comparing renders at 1600x800, not native res).
# --resolution 1 forces global_down=1 (no downscale). render.py/metrics.py reload
# resolution from the saved cfg_args automatically, so it only needs setting here.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp18_seeded_fewer_resets_fullres}"
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
    --resolution 1 \
    --opacity_reset_interval 10000
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp18_seeded_fewer_resets_fullres done: $OUTPUT_DIR"
