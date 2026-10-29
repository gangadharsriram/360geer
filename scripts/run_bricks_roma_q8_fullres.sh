#!/usr/bin/env bash
# Full-resolution run on the RoMa-matched COLMAP bricks reconstruction
# (RoMa/colmap_bricks_q8/openmvg), using the same recipe as run_bricks_best_fullres
# (exp108 + --resolution 1) so results are directly comparable to the original
# bricks full-res baseline (test SSIM 0.8509 / PSNR 25.12 / LPIPS 0.1841).
#
# Differences from the original bricks dataset that this script handles:
#  - No pcd.ply: openMVG/scene.ply is only the 100 camera centres. The real sparse
#    points are COLMAP's points3D (22,596 pts vs 5,958), in the same world frame as
#    data_extrinsics.json (camera centres/rotations verified identical), so they are
#    converted to pcd.ply first (only if pcd.ply isn't already there).
#  - World scale is ~10x the original bricks scene (COLMAP gauge): camera spacing
#    0.425 vs 0.043. Relative geometry is the same; cameras_extent and the position
#    LR scale with it automatically.
#  - No prior_W_0.8 (the CIConv smooth prior gave no gain on bricks), so lambda_smooth
#    uses the default raw-GT-gradient weighting, as in run_bricks_best_fullres.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/3dgs_pipeline/literature_code/360geer/RoMa/colmap_bricks_q8/openmvg}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/run_bricks_roma_q8_fullres}"
GPU="${3:-1}"
COLMAP_POINTS="${4:-$DATASET_DIR/../text/points3D.txt}"
ITERS_NUM=30000

if [ ! -f "$DATASET_DIR/pcd.ply" ]; then
    echo "== Converting COLMAP points3D -> pcd.ply =="
    python scripts/colmap_points_to_pcd.py "$DATASET_DIR" "$COLMAP_POINTS"
fi

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
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "run_bricks_roma_q8_fullres done: $OUTPUT_DIR - compare against run_bricks_best_fullres (SSIM 0.8509/PSNR 25.12/LPIPS 0.1841)"
