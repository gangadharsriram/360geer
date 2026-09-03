#!/usr/bin/env bash
# Experiment 16: seeded pcd (exp9) + size-based Gaussian pruning disabled, opacity
# resets left at their normal 3000-iter cadence.
#
# Root-cause finding: train.py's size_threshold gate (`iteration > opacity_reset_interval`)
# was accidentally coupled to the reset schedule, not a real "give it time to settle"
# design choice for this dataset. Checkpoint inspection at iter 3000 showed 39.3% of
# all 439,884 Gaussians (172,889) already exceed the 0.1*extent world-space size cap -
# these are large, legitimate Gaussians covering weakly-constrained sky/foliage regions
# (no SfM triangulation there, so a few big blobs is how the optimizer cheaply explains
# them - this is why pre-iter-3000 snapshots look great). The instant the gate flips on
# (iteration > opacity_reset_interval, first checked at 3000+300=3300), ~39% of the
# model gets bulk-pruned in one shot, and every 300-iter check afterward re-prunes
# whatever regrew too large - producing the repeating degrade/recover sawtooth seen
# across every 500-iter snapshot in exp9/10/11, concentrated in exactly the regions
# that rely on large covering Gaussians (tower stays sharp throughout; sky doesn't).
#
# This experiment isolates that mechanism from the reset itself: opacity_reset_interval
# stays at its default 3000 (resets still fire normally), but size_prune_from_iter is
# pushed past the end of training so the size-based prune criterion never activates.
# If this alone recovers the sky/foliage quality lost after iter 3000, it confirms the
# size-cull gate (not the opacity reset) is the actual cause - see also exp17, its
# complement (resets disabled, size-cull left on normal schedule).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp16_seeded_no_size_cull}"
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
    --size_prune_from_iter 999999
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp16_seeded_no_size_cull done: $OUTPUT_DIR"
