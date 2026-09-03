#!/usr/bin/env bash
# Experiment 19: seeded pcd (exp9) + size-based Gaussian pruning re-enabled, but
# delayed to iteration 8000 instead of exp16's "never" (or the original bug's 3000).
#
# exp16 (size cull off for the whole run) fixed the early bulk-destruction problem
# (SSIM 0.826 vs ~0.6-0.67 before) but traded it for the opposite extreme: with
# max_screen_size never set, both big_points_vs/ws AND the aniso hard-cutoff prune
# (gaussian_model.py:465-474, all three share the same `if max_screen_size:` gate)
# stay off for all 30k iterations. Final model ballooned to 8,677,429 Gaussians
# (~8x exp9's ~1M). Ray-cast through a speckled window-edge test pixel found 697
# Gaussians within 1deg of the ray spread across 21.3 units of depth (half >50%
# opaque) vs just 11 for a clean sky pixel - the same redundant depth-stacking
# mechanism from the original artifact, now piling up unchecked at high-gradient
# edges (where densify_and_clone/split fire most) since nothing ever prunes it.
# That's the "sand scattered" look on background structure the renders show.
#
# This tests the middle ground: let seeded regions and edges consolidate past the
# first two reset boundaries (3000, 6000) with no size pressure, same as exp16,
# then turn size/aniso pruning on for the remaining 22k iterations so redundant
# edge-stacking actually gets cleaned up instead of accumulating for the full run.
# opacity_reset_interval stays default (3000, normal resets) - only the size-cull
# timing changes.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp19_seeded_delayed_size_cull}"
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
    --resolution 1 \
    --size_prune_from_iter 8000
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp19_seeded_delayed_size_cull done: $OUTPUT_DIR"
