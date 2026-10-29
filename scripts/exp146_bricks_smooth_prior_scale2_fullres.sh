#!/usr/bin/env bash
# Experiment 146: run_bricks_best_fullres's exact recipe (bricks' full-res
# control: SSIM 0.8509/PSNR 25.12/LPIPS 0.1841) + the CIConv structure-prior
# edge weighting from exp145, with --smooth_edge_scale 2 instead of
# the untuned default (10.0).
#
# exp145 (auto-downscaled res) found the prior-weighted smooth loss a dead
# wash against exp108 at scale=10.0 - but that scale was carried over
# unchanged from the raw-gradient version, and the prior's value
# distribution is different (saturates toward 0/1, more binary, vs the raw
# gradient's small continuous values), so scale=10 may have been collapsing
# the smoothness weight to ~0 across too wide an area, masking any real
# effect. This sweeps the scale at the resolution that actually matters
# (full res, where the brick sculpture's curved/shaded surface - the
# original motivating case - renders at full detail instead of ~0.83x).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp146_bricks_smooth_prior_scale2_fullres}"
GPU="${3:-0}"
ITERS_NUM=30000

echo "== Seeding sparse-coverage directions =="
python scripts/seed_sparse_directions.py "$DATASET_DIR"

export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --resolution 1 \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 20000 25000 30000 \
    --eval --disable_viewer \
    --opacity_reset_interval 30000 \
    --size_prune_from_iter 30000 \
    --densify_grad_unscaled \
    --densify_grad_threshold 0.00006 \
    --lambda_smooth 0.01 \
    --smooth_prior_dir "$DATASET_DIR/prior_W_0.8" \
    --smooth_edge_scale 2 \
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp146_bricks_smooth_prior_scale2_fullres done: $OUTPUT_DIR - compare against run_bricks_best_fullres (SSIM 0.8509/PSNR 25.12/LPIPS 0.1841)"
