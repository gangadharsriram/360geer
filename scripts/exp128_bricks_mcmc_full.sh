#!/usr/bin/env bash
# Experiment 128: bricks, full 30k MCMC densification run (--densify_mode
# mcmc), first real comparison against exp108 (this investigation's current
# best grad-threshold result: SSIM 0.8681/PSNR 25.79/LPIPS 0.1653/1,118,794
# Gaussians).
#
# cap_max = exp108's final Gaussian count, for a fair apples-to-apples
# budget comparison. densification_interval=100 (looser than 360geer's
# grad-threshold default of 300) per the reference implementation's own
# default - worth trying first for mcmc mode. densify_from_iter/
# densify_until_iter left at 360geer's existing defaults (500/15000) rather
# than the reference's (500/25000), to avoid changing two things at once
# relative to the established comparison baseline.
#
# lambda_smooth/lambda_aniso kept at their already-confirmed-best values
# (0.01 / 0.0) - both are unconditional image-space/regularization terms,
# independent of densify_mode. lambda_opacity_sparsity left at its default
# 0.0 (off) - that regularizer was tuned specifically for grad_threshold
# mode; mcmc mode has its own opacity regularizer (mcmc_opacity_reg,
# defaults to 0.01) instead, per the plan's design decision to keep them
# separate rather than double-regularize.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp128_bricks_mcmc_full}"
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
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --densify_mode mcmc \
    --cap_max 1118794 \
    --densification_interval 100
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp128_bricks_mcmc_full done: $OUTPUT_DIR - compare results.json against exp108 (SSIM 0.8681/PSNR 25.79/LPIPS 0.1653)"
