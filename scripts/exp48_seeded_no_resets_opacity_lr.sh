#!/usr/bin/env bash
# Experiment 48: exp42's config (seeded pcd, resets off, densify_grad_threshold
# 0.0003 - 360geer's current best, test SSIM 0.8475) + opacity_lr 0.025->0.05,
# matching OmniGS's own value for this dataset (cfg/lonlat/egonerf_lonlat_ricoh360.yaml:
# Optimization.opacity_lr: 0.05).
#
# Every other learning rate (position, feature, scaling, rotation) and
# percent_dense/lambda_dssim matched exactly between the two codebases when
# diffed directly - opacity_lr is the only difference found. The previous four
# OmniGS-inspired hypotheses (pruning-criterion split: exp44/45, near-noop;
# densification_interval: exp46, actively hurt; point cloud provenance: ruled
# out, identical clouds; metrics.py masking: ruled out, both full-image) were
# all negative, so this is the most promising untested lead so far - a 2x
# faster opacity learning rate could make the "prove your worth against
# min_opacity" dynamic more responsive, pushing redundant/low-value Gaussians
# toward the prune floor faster instead of lingering at a slowly-decaying
# opacity.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-./output/exp48_seeded_no_resets_opacity_lr}"
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
    --opacity_lr 0.05
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp48_seeded_no_resets_opacity_lr done: $OUTPUT_DIR"
