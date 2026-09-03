#!/usr/bin/env bash
# Experiment 59: exp54's control structure (seeded pcd, resets off, no size-cull,
# densify_grad_threshold at its default 0.0002) + --densify_grad_signed alone.
#
# UPDATED after this run's original result: --densify_grad_signed used to remove
# BOTH the fabsf'd accumulation AND the depth^2 scaling at once - that combined
# version (this script's original form) landed at 155,021 Gaussians, SSIM 0.744,
# badly overshooting past even OmniGS's own 944,467/0.858 into under-fitting.
# The flag has since been split in two: --densify_grad_signed now controls ONLY
# the fabsf-vs-signed accumulation (submodules/geer-rasterizer's backward.cu,
# renderCUDA's per-pixel dL_dmean2D accumulation); --densify_grad_unscaled
# (exp61) controls ONLY the depth^2 scaling, independently. A 1000-iteration
# smoke test isolating each found densify_grad_signed alone responsible for
# roughly 2x the reduction of densify_grad_unscaled alone (47.7% vs 23.9% fewer
# Gaussians at iter 1000) - this script (re-running it now tests the SIGNED-ONLY
# fix, not the old combined behavior) checks whether that milder, isolated
# effect lands in a better-calibrated range over a full 30000-iteration run.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp59_seeded_no_resets_grad_signed}"
GPU="${3:-0}"
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
    --densify_grad_signed
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp59_seeded_no_resets_grad_signed done: $OUTPUT_DIR"
