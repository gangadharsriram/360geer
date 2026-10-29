#!/usr/bin/env bash
# Full-resolution run on the OpenMVG reconstruction of bricks built from RoMa
# matches (RoMa/bricks_roma_openmvg), same recipe as run_bricks_best_fullres
# (exp108 + --resolution 1; original-bricks full-res test baseline: SSIM 0.8509 /
# PSNR 25.12 / LPIPS 0.1841).
#
# What this script does differently from the original bricks dataset:
#  - Trains on a rescaled copy (RoMa/bricks_roma_openmvg_scaled, built here, source left
#    untouched). This reconstruction is ~9x larger in world scale than the original
#    (similarity fit of camera centres: 0.111). Rendering is unaffected by a uniform
#    scale, but the densification statistic is not: on the sibling colmap_bricks_q8
#    set (~10x) the unscaled run stalled at ~95k Gaussians (36k at iter 3000 vs
#    209k for the original); dividing the world by 10 restored 258k / 1.6M at 3k / 7k.
#    Override with WORLD_SCALE=... if you want a different factor.
#  - pcd.ply here includes the 100 camera centres as green glyph points (15,677 =
#    15,577 structure + 100 cameras); the rescale step drops them.
#  - The source dir has no images/ folder; the copy links the views JSON's root_path.
# Caveat: on colmap_bricks_q8, rescaling fixed the Gaussian count but PSNR stayed
# well below the original bricks run (iter 7000 test 19.9 vs 25.0), so compare the
# final metrics against the baseline above before trusting this reconstruction.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/3dgs_pipeline/literature_code/360geer/RoMa/bricks_roma_openmvg}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/run_bricks_roma_openmvg_fullres}"
GPU="${3:-1}"
TRAIN_DIR="${4:-${DATASET_DIR%/}_scaled}"
WORLD_SCALE="${WORLD_SCALE:-0.111}"
ITERS_NUM="${ITERS_NUM:-30000}"

echo "== Building rescaled dataset copy (x$WORLD_SCALE) =="
python scripts/rescale_openmvg_dataset.py "$DATASET_DIR" "$TRAIN_DIR" "$WORLD_SCALE"

echo "== Seeding sparse-coverage directions =="
python scripts/seed_sparse_directions.py "$TRAIN_DIR"

export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$TRAIN_DIR" -m "$OUTPUT_DIR" \
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
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$TRAIN_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "run_bricks_roma_openmvg_fullres done: $OUTPUT_DIR - compare against run_bricks_best_fullres (SSIM 0.8509/PSNR 25.12/LPIPS 0.1841)"
