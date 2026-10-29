#!/usr/bin/env bash
# exp108 recipe at AUTO resolution (1600 wide, like the old best exp108: SSIM 0.8681 / PSNR 25.79 / LPIPS 0.1653) + VGG16 perceptual loss.
# Goal: beat 26.2032 / 0.8634 / 0.1355 (PSNR / SSIM / LPIPS).
# Usage: [LAMBDA_VGG=0.05] [VGG_CROP=512] [VGG_START=0] run_bricks_vgg_autores.sh [DATASET_DIR] [OUTPUT_DIR] [GPU]
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then echo "ERROR: activate '360geer' env." >&2; exit 1; fi
cd /data2/3dgs_pipeline/literature_code/360geer
DATASET_DIR="${1:-/data2/Dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
LAMBDA_VGG="${LAMBDA_VGG:-0.02}"; VGG_CROP="${VGG_CROP:-512}"; VGG_START="${VGG_START:-0}"
OUTPUT_DIR="${2:-/data2/3dgs_pipeline/output/run_bricks_vgg${LAMBDA_VGG}_autores}"
GPU="${3:-0}"
ITERS_NUM=30000
python scripts/seed_sparse_directions.py "$DATASET_DIR"
export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
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
