#!/usr/bin/env bash
# Experiment 60: exp54's control structure (seeded pcd, resets off, no size-cull,
# densify_grad_threshold at its default 0.0002, matching OmniGS's own value
# exactly) + BOTH validated fixes combined: --densify_stats_2d 1 (Python-level,
# drops the 3rd/ray-space gradient component - exp52 vs exp54: 20.8% fewer
# Gaussians, better SSIM) and --densify_grad_signed (CUDA-level, removes fabsf'd
# no-cancellation accumulation and depth^2 scaling - smoke-tested at 63% fewer
# Gaussians on its own). This is the most direct test yet of whether 360geer,
# with its densification gradient statistic corrected to actually mean the same
# thing as OmniGS's, can match or beat OmniGS's result on this dataset
# (944,467 Gaussians, SSIM 0.858) at the exact same nominal threshold.
#
# Flagging directly: this stacks two independently-validated levers without a
# fresh isolated test of the combination itself - this investigation has
# repeatedly found combined levers underperform the sum of their isolated
# effects (exp8, exp33-37, exp50 vs exp49+exp51). Worth comparing this result
# against exp59 (CUDA fix alone) to see whether they compound or not.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp60_seeded_no_resets_grad_fully_fixed}"
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
    --densify_stats_2d 1 \
    --densify_grad_signed
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp60_seeded_no_resets_grad_fully_fixed done: $OUTPUT_DIR"
