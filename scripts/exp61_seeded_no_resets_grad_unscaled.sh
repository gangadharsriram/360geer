#!/usr/bin/env bash
# Experiment 61: exp54's control structure (seeded pcd, resets off, no size-cull,
# densify_grad_threshold at its default 0.0002) + --densify_grad_unscaled alone -
# the companion isolated test to exp59, testing the OTHER half of the split fix.
#
# --densify_grad_unscaled removes ONLY the (depth^2+1)*0.3 scaling applied to the
# densification gradient statistic before comparison against densify_grad_threshold
# (submodules/geer-rasterizer's backward.cu, preprocessCUDA_mah). Leaves the
# fabsf'd (never-cancelling) accumulation untouched - that's exp59's job.
#
# A 1000-iteration smoke test found this flag alone gives a milder effect than
# densify_grad_signed alone (14,821 vs 10,178 Gaussians at iter 1000, vs 19,459
# default) - roughly half the reduction. Depth-scaling specifically inflates
# far/background content's statistic (sky, distant walls - the exact
# weakly-constrained regions this whole investigation traces the redundant-
# Gaussian-stacking artifact back to), so even though its isolated effect is
# smaller in raw Gaussian-count terms, it may be more targeted at the actual
# artifact than densify_grad_signed's broader effect.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp61_seeded_no_resets_grad_unscaled}"
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
    --densify_grad_unscaled
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp61_seeded_no_resets_grad_unscaled done: $OUTPUT_DIR"
