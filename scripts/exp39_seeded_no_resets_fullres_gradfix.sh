#!/usr/bin/env bash
# Experiment 39: same as exp38 (exp15's config at native 1920x960), plus
# densify_grad_threshold scaled down to compensate for resolution-driven
# gradient dilution.
#
# exp38 regressed hard vs exp15 (SSIM 0.648 vs 0.835) despite being the *same*
# config, just at native res instead of the 1600x800 auto-downscale. Root cause:
# utils/loss_utils.py's l1_loss/ssim use .mean() over all pixels, so the same
# absolute photometric error gets divided by more terms at 1920x960 (1.44x more
# pixels than 1600x800), shrinking the per-Gaussian gradient signal compared
# against the fixed absolute densify_grad_threshold=0.0002. Confirmed via point
# count (exp38: 1.45M vs exp15: 5.49M, and the same ~4-5x drop repeated for the
# exp14/exp18 pair) and a ray-cast diagnostic showing the starvation is uneven -
# well-constrained regions (tower) lose more than sparse-view-ambiguous ones
# (sky/tree, which never converge regardless of resolution), so the redundant
# depth-stacking artifact from the original investigation ends up as a *larger*
# share of a *smaller* total population - hence streaks, not just blur.
#
# This scales densify_grad_threshold by 1/1.44 (0.0002 -> ~0.00014), matching
# the direct pixel-count dilution ratio, to test whether that alone restores
# exp15-equivalent densification behavior at native resolution.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp39_seeded_no_resets_fullres_gradfix}"
GPU="${3:-2}"
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
    --resolution 1 \
    --opacity_reset_interval 30000 \
    --densify_grad_threshold 0.00014
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp39_seeded_no_resets_fullres_gradfix done: $OUTPUT_DIR"
