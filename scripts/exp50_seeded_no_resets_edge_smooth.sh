#!/usr/bin/env bash
# Experiment 50: exp49's config (exp42 + antialiasing, current best: test SSIM
# 0.8489) + a new edge-aware image-space smoothness regularizer (--lambda_smooth).
#
# Third of three "smooth out the scattered/dotted artifact" experiments, tried
# cheapest/lowest-risk first: antialiasing (exp49, small real win: SSIM 0.8475->
# 0.8489) -> post-process bilateral denoise on final renders (no retraining
# needed, tested directly: SSIM -0.0042, LPIPS +0.0327 - net negative, blurs
# real detail right along with the artifact, ruled out) -> this, a training-time
# regularizer (train.py loss section, arguments/__init__.py's lambda_smooth/
# smooth_edge_scale, new code this run).
#
# Unlike the post-process filter (which can't distinguish real detail from
# artifact scatter), this is weighted by the GROUND TRUTH's own local gradient -
# torch.exp(-gt_dx/dy * smooth_edge_scale) - so it only discourages high-frequency
# variation in the RENDER where the GT itself is flat (exactly where the dots
# have no business being), and backs off wherever GT has genuine detail (window
# grids, brick texture, leaves). Standard MonoDepth2-style edge-aware smoothness
# formula, adapted from depth to RGB. Starting weight (0.01) chosen small and
# cautious, matching how lambda_aniso/lambda_opacity_sparsity were introduced -
# this is a first, untuned test of whether the idea helps at all.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp50_seeded_no_resets_edge_smooth}"
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
    --antialiasing \
    --lambda_smooth 0.01
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp50_seeded_no_resets_edge_smooth done: $OUTPUT_DIR"
