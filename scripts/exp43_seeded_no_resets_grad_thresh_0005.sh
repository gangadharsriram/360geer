#!/usr/bin/env bash
# Experiment 43: same as exp42, but densify_grad_threshold=0.0005 instead of
# 0.0003 - one step further along the same axis that made exp42 the new best
# result (test SSIM 0.8475, gap 0.0931, beating exp15 on every metric).
#
# Chosen to match exp27's threshold exactly (0.0005, resets-normal baseline:
# test SSIM 0.765, gap 0.038) so this isolates the same "resets off" variable
# exp42 tested against exp26 - does resets-off + grad_threshold=0.0005 also
# beat resets-normal + grad_threshold=0.0005, the way exp42 beat exp26 at 0.0003?
# The exp26->27->28 progression (0.0003->0.0005->0.0008) showed the gap keeps
# shrinking as the threshold rises, but quality eventually collapses at 0.0008 -
# this checks whether 0.0005 is still on the safe side of that cliff when
# combined with resets off, or whether the sweet spot has already passed.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp43_seeded_no_resets_grad_thresh_0005}"
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
    --densify_grad_threshold 0.0005
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp43_seeded_no_resets_grad_thresh_0005 done: $OUTPUT_DIR"
