#!/usr/bin/env bash
# Experiment 45: same flags as exp44 (seeded pcd, resets off, densify_grad_threshold
# 0.0003, size_prune_from_iter=0, prune_by_extent=0), re-run against the fixed code -
# the clean, confound-free replication of OmniGS's actual pruning strategy.
#
# exp44 ran before aniso_hard_ratio's timing was split off from size_prune_from_iter's.
# With the old coupling, --size_prune_from_iter 0 didn't just replicate OmniGS's
# strategy (screen-space cull on early, world-space cull off) - it ALSO silently
# activated 360geer's own aniso hard-cutoff from essentially iteration 0, a
# criterion OmniGS has no equivalent of at all. That's a real extra variable exp44
# wasn't supposed to have. Now that aniso_prune_from_iter exists as its own
# parameter (defaulting to 3000, same as before the split), this run isolates
# OmniGS's strategy on its own - aniso stays on its normal schedule, untouched.
#
# Compare against: exp42 (0.8475, no size-cull at all), OmniGS's own result on
# this dataset (test SSIM 0.858, results/lonlat/EgoNeRF/Ricoh360_bricks/results.json
# in the OmniGS repo), and exp44 (confounded version of this same idea).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp45_seeded_no_resets_omnigs_prune_clean}"
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
    --densify_grad_threshold 0.0003 \
    --size_prune_from_iter 0 \
    --prune_by_extent 0
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp45_seeded_no_resets_omnigs_prune_clean done: $OUTPUT_DIR"
