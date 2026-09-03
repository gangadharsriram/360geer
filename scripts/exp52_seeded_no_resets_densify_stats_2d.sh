#!/usr/bin/env bash
# Experiment 52: exp42's core structure (seeded pcd, resets off, no size-cull) +
# --densify_stats_2d 1 (test #1 of the OmniGS investigation's central finding) +
# densify_grad_threshold left at its DEFAULT 0.0002, matching OmniGS's own value
# exactly - not exp42's tuned 0.0003, since that value was tuned specifically to
# compensate for the inflation this fix partially removes. Testing both at once
# would muddy the result.
#
# Root cause (verified directly against both CUDA backward.cu files this pass):
# OmniGS's add_densification_stats (gaussian_model.cpp) norms 2 SIGNED gradient
# components, written by OmniGS's backward.cu with no fabsf - pushes from
# different pixels on a Gaussian's footprint can cancel out. 360geer's version
# (the one actually used in train.py) norms 3 components, and 360geer's own
# backward.cu (submodules/geer-rasterizer/cuda_rasterizer/backward.cu:514-516)
# wraps every per-pixel contribution in fabsf() before accumulating - so nothing
# ever cancels, and there's an extra always-positive 3rd (ray-space) dimension on
# top. Same nominal threshold (0.0002 in both), structurally different statistic
# being compared against it - directly predicted to explain why 360geer grows to
# 520-560x its starting point count while OmniGS only grows 158x for equal/better
# quality (944,467 final Gaussians there vs 3.2-3.5M in 360geer's best runs).
#
# --densify_stats_2d 1 switches to gaussian_model.py's already-present
# add_densification_stats_direct (2-component). This does NOT fix the deeper
# fabsf/no-cancellation issue (that lives inside the CUDA kernel, untouched here)
# - only removes the 3rd-component contribution. Expect a partial, not full,
# reduction in point count/gap versus OmniGS - this isolates how much of the
# effect the easy, CUDA-free half of the fix accounts for.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp52_seeded_no_resets_densify_stats_2d}"
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
    --densify_stats_2d 1
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp52_seeded_no_resets_densify_stats_2d done: $OUTPUT_DIR"
