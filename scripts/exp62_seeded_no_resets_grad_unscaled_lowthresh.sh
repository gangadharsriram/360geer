#!/usr/bin/env bash
# Experiment 62: exp61's config (seeded pcd, resets off, no size-cull,
# --densify_grad_unscaled) + densify_grad_threshold LOWERED from the default
# 0.0002 to 0.0001.
#
# Why lower, not higher: exp61 already got its 92% Gaussian-count reduction
# (5.4M -> 422,899) purely from removing the depth^2 inflation - the threshold
# itself never changed from 0.0002. SSIM/PSNR held almost exactly steady vs
# the old buggy baseline, but LPIPS got measurably worse (0.2127 vs 0.2012),
# meaning some genuine fine-detail-capturing material was cut along with the
# redundancy. Raising the threshold further would only shrink the count more.
# Lowering it makes it easier again for genuinely-useful Gaussians to be
# created, on top of the now-uninflated (honest) statistic, aiming to recover
# the LPIPS gap while staying nowhere near the old 5.4M/158x-growth blowup.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp62_seeded_no_resets_grad_unscaled_lowthresh}"
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
    --densify_grad_unscaled \
    --densify_grad_threshold 0.0001
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp62_seeded_no_resets_grad_unscaled_lowthresh done: $OUTPUT_DIR"
