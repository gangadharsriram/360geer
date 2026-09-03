#!/usr/bin/env bash
# Experiment 40: exp15's config (seeded pcd, resets off) + opacity-mass sparsity
# regularizer turned on: --lambda_opacity_sparsity 0.001.
#
# train.py:161-162 already has this term (loss += lambda_opacity_sparsity *
# mean_opacity) but it's been off (0.0) since early this investigation, when it
# was combined with a much more aggressive densify_grad_threshold hike and the
# COMBINATION caused a catastrophic point-count collapse (100k->24k, black-void
# render). That failure was never attributed cleanly to this term specifically -
# it was one of two untested changes stacked at once. This is the first isolated
# test of the sparsity term alone, on top of the current best-known-good baseline
# (exp15, SSIM 0.835) instead of a from-scratch config.
#
# Why this instead of just the exp20-37 sweep: those levers (prune_until_iter,
# min_opacity, densify_grad_threshold, aniso_hard_ratio) all clean up redundant
# Gaussians after densification creates them. This applies pressure *during*
# training itself - a real Gaussian with strong photometric support keeps its
# opacity high regardless (removing it visibly hurts the loss); a redundant one
# sitting in an already-explained-by-siblings region has much weaker individual
# pressure and gets pushed toward the existing min_opacity=0.005 prune floor
# faster than it would otherwise. Starting at the original (never-validated-
# alone) proposed value, 0.001 - small relative to the L1/D-SSIM terms it's
# added to, deliberately conservative given how badly the combined version
# failed last time.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp40_seeded_opacity_sparsity}"
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
    --lambda_opacity_sparsity 0.001
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp40_seeded_opacity_sparsity done: $OUTPUT_DIR"
