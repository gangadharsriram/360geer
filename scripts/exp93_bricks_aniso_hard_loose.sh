#!/usr/bin/env bash
# Experiment 93: bricks, exp91's config (its own best result so far: seeded
# pcd, resets off, no size-cull, --densify_grad_unscaled, threshold 0.00006,
# lambda_smooth 0.01, lambda_aniso 0.0 - SSIM 0.8614/PSNR 25.51/LPIPS 0.1708,
# Gaussians 1,216,734) + aniso_hard_ratio raised from 50.0 to 1000.0.
#
# Now that the soft anisotropy penalty is confirmed off (exp91), check
# whether the remaining HARD cutoff in densify_and_prune (aniso_hard_ratio,
# still 50.0 by default) is also quietly constraining shape more than
# necessary. This is the exact change an earlier, since-abandoned fix
# attempt proposed (loosen the hard cutoff to a generous backstop rather
# than a tight constraint) but never actually tested, since that plan was
# superseded before this specific change was tried. Single variable vs
# exp91: aniso_hard_ratio only.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp93_bricks_aniso_hard_loose}"
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
    --opacity_reset_interval 30000 \
    --size_prune_from_iter 30000 \
    --densify_grad_unscaled \
    --densify_grad_threshold 0.00006 \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --aniso_hard_ratio 1000.0
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp93_bricks_aniso_hard_loose done: $OUTPUT_DIR"
