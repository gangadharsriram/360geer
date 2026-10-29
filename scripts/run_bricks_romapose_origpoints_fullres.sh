#!/usr/bin/env bash
# Full-res run: RoMa v2 poses (cell1 + angular filter) mapped into the original bricks frame, with the ORIGINAL sparse points.
# Same recipe as run_bricks_best_fullres. Compare: orig poses+orig pts 25.12 | orig poses+RoMa pts 25.03 | RoMa v2 poses+RoMa pts 23.16.
# Usage: run_bricks_romapose_origpoints_fullres.sh [POSES_DIR] [OUTPUT_DIR] [GPU] [TRAIN_DIR]
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then echo "ERROR: activate '360geer' env." >&2; exit 1; fi
ROOT=/data2/3dgs_pipeline/literature_code/360geer
ORIG=/data2/Dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks
POSES="${1:-$ROOT/RoMa/bricks_roma_v_q1_a}"
OUTPUT_DIR="${2:-/data2/3dgs_pipeline/output/run_bricks_romapose_origpoints_fullres}"
GPU="${3:-0}"
TRAIN_DIR="${4:-$ROOT/RoMa/bricks_romapose_origpoints}"
ITERS_NUM=30000
cd "$ROOT"
python scripts/poses_with_orig_points.py "$POSES" "$ORIG" "$TRAIN_DIR"
python scripts/seed_sparse_directions.py "$TRAIN_DIR"
export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$TRAIN_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI --resolution 1 \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 20000 25000 30000 \
    --eval --disable_viewer \
    --opacity_reset_interval 30000 --size_prune_from_iter 30000 \
    --densify_grad_unscaled --densify_grad_threshold 0.00006 \
    --lambda_smooth 0.01 --lambda_aniso 0.0 --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$TRAIN_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "done: $OUTPUT_DIR (baseline SSIM 0.8509/PSNR 25.12/LPIPS 0.1841)"
