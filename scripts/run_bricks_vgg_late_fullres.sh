#!/usr/bin/env bash
# Bricks full-res best recipe (threshold 0.00006, as the baseline) + VGG16 perceptual loss applied only LATE (from VGG_START, default 15000 = densification end).
# Compare against the lambda_vgg=0 baseline: SSIM 0.8509 / PSNR 25.12 / LPIPS 0.1841.
# Usage: [LAMBDA_VGG=0.05] [VGG_START=15000] [VGG_CROP=512] run_bricks_vgg_late_fullres.sh [DATASET_DIR] [OUTPUT_DIR] [GPU]
# Earlier runs (lambda 0.05 from iter 0): thr 0.00006 -> SSIM 0.8390/PSNR 24.73/LPIPS 0.1477 (4.4M G); thr 0.00015 -> 0.8327/24.68/0.1560 (1.22M G).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then echo "ERROR: activate '360geer' env." >&2; exit 1; fi
cd /data2/3dgs_pipeline/literature_code/360geer
DATASET_DIR="${1:-/data2/Dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
LAMBDA_VGG="${LAMBDA_VGG:-0.05}"; VGG_CROP="${VGG_CROP:-512}"; VGG_START="${VGG_START:-15000}"
OUTPUT_DIR="${2:-/data2/3dgs_pipeline/output/run_bricks_vgg${LAMBDA_VGG}_from${VGG_START}_fullres}"
GPU="${3:-0}"
ITERS_NUM=30000
python scripts/seed_sparse_directions.py "$DATASET_DIR"
export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI --resolution 1 \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 20000 25000 30000 \
    --eval --disable_viewer \
    --opacity_reset_interval 30000 --size_prune_from_iter 30000 \
    --densify_grad_unscaled --densify_grad_threshold 0.00006 \
    --lambda_smooth 0.01 --lambda_aniso 0.0 --lambda_opacity_sparsity 0.012 \
    --lambda_vgg "$LAMBDA_VGG" --vgg_crop "$VGG_CROP" --vgg_start_iter "$VGG_START"
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "done: $OUTPUT_DIR"
