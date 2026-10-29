#!/usr/bin/env bash
# Experiment 130: 360Roam "office" scene, full 30k MCMC densification run
# (--densify_mode mcmc) at true full resolution (--resolution 1) - office is
# 2048px wide, so the default -1 setting would auto-downscale to 1600px
# (camera_utils.py:48-55).
#
# office never had a clean grad-threshold "best" (the threshold sweep found
# a genuine SSIM/PSNR-vs-LPIPS tradeoff, not a single winner - see
# run_office_best_fullres, the best-available pick at threshold 0.001).
# cap_max = run_office_best_fullres's final Gaussian count (236,837), for a
# fair apples-to-apples budget comparison against that run.
#
# Same flags as exp128/129 (bricks' MCMC runs) otherwise -
# densification_interval=100 (reference's own default, worth trying first
# for mcmc mode), lambda_smooth/lambda_aniso at their confirmed-best values,
# lambda_opacity_sparsity left off in favor of mcmc_opacity_reg. Worth
# remembering going in: office has TWO overlapping problems (narrow-baseline
# ambiguity + genuine near-camera clutter detail) that a single
# grad-threshold knob couldn't fully separate - MCMC's opacity-driven
# relocation is a genuinely different mechanism, so this is a real,
# not-yet-answered question, not just another threshold guess.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/360Roam/360Roam_openmvg/office}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp130_office_mcmc_full_fullres}"
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
    --cap_max 236837 \
    --densification_interval 100
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp130_office_mcmc_full_fullres done: $OUTPUT_DIR - compare results.json against run_office_best_fullres (SSIM 0.6703/PSNR 19.76/LPIPS 0.3962)"
