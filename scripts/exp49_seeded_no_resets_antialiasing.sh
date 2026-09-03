#!/usr/bin/env bash
# Experiment 49: exp42's config (seeded pcd, resets off, densify_grad_threshold
# 0.0003 - 360geer's current best, test SSIM 0.8475) + --antialiasing enabled.
#
# First of three "smooth out the scattered/dotted high-frequency artifact"
# experiments requested, tried cheapest-first: antialiasing (this one, one flag,
# no retraining logic changes) -> post-process blur/denoise on final renders ->
# a training-time smoothness/distortion regularizer (new code, biggest lift).
#
# antialiasing has been OFF (the default) in every experiment run so far -
# PipelineParams.antialiasing defaults False, --antialiasing flips it on. It's a
# real, wired mechanism (submodules/geer-rasterizer/cuda_rasterizer/forward.cu's
# omni_hvar, gated by this flag) that rescales each Gaussian's opacity based on
# its screen-space footprint relative to pixel size - the standard Mip-Splatting-
# style correction. Plausible mechanism for helping here: the "dots" are small,
# thin, redundant Gaussians that only line up correctly from their training
# camera; footprint-aware opacity correction could reduce how much they alias
# into visible specks from a different (test) viewing angle, without touching
# densification/pruning at all - a genuinely different, orthogonal lever from
# everything tried in exp20-48.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp49_seeded_no_resets_antialiasing}"
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
    --densify_grad_threshold 0.0003 \
    --antialiasing
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp49_seeded_no_resets_antialiasing done: $OUTPUT_DIR"
