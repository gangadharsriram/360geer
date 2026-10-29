#!/usr/bin/env bash
# Experiment 143: same as exp141/142 but lambda_ray_aniso 0.003.
#
# Filling out the weight sweep alongside exp142 (0.001): exp141 (0.01) was
# too strong (SSIM 0.849, worse than exp108 on all 3 metrics, no visible
# needle-scatter but a quiet quality loss - likely over-constraining
# legitimate elongated Gaussians). This point extends the log-scale
# coverage (0.01 / 0.003 / 0.001 / ...) so the trend is visible from one
# batch instead of guessing the next step after each single result.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp143_bricks_ray_aniso_0003}"
GPU="${3:-1}"
ITERS_NUM=30000

echo "== Seeding sparse-coverage directions =="
python scripts/seed_sparse_directions.py "$DATASET_DIR"

export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --resolution 1 \
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
    --lambda_opacity_sparsity 0.012 \
    --lambda_ray_aniso 0.003 \
    --ray_aniso_max_ratio 3.0
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp143_bricks_ray_aniso_0003 done: $OUTPUT_DIR - compare against exp108 (SSIM 0.8681/PSNR 25.79/LPIPS 0.1653) and exp141 (0.01: SSIM 0.849/PSNR 25.03/LPIPS 0.187)"
