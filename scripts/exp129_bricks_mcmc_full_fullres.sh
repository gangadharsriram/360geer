#!/usr/bin/env bash
# Experiment 129: same as exp128 (bricks, full 30k MCMC densification run,
# --densify_mode mcmc), but at true full resolution (--resolution 1) -
# bricks is 1920px wide, so the default -1 setting would auto-downscale to
# 1600px (camera_utils.py:48-55), same catch as every other full-res run
# this session.
#
# Compare against run_bricks_best_fullres (grad-threshold mode's full-res
# best: SSIM 0.8509/PSNR 25.12/LPIPS 0.1841/1,118,794-ish Gaussians), not
# exp108's auto-scaled numbers - full res is a harder task at the same
# iteration budget, so the fair comparison point is the other full-res run,
# not the auto-scaled one.
#
# cap_max = exp108's final Gaussian count (same as exp128). All other flags
# identical to exp128 - see that script's comments for the full rationale
# (densification_interval=100, lambda_smooth/lambda_aniso kept at their
# confirmed-best values, lambda_opacity_sparsity left off in favor of
# mcmc_opacity_reg).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp129_bricks_mcmc_full_fullres}"
GPU="${3:-2}"
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
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --densify_mode mcmc \
    --cap_max 1118794 \
    --densification_interval 100
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp129_bricks_mcmc_full_fullres done: $OUTPUT_DIR - compare results.json against run_bricks_best_fullres (SSIM 0.8509/PSNR 25.12/LPIPS 0.1841)"
