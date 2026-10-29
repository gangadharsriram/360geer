#!/usr/bin/env bash
# Diagnostic: ORIGINAL bricks poses + RoMa sparse points (mapped into the original frame).
# Same recipe as run_bricks_best_fullres. If PSNR ~25 -> RoMa poses were the problem; if ~20 -> not the poses.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then echo "ERROR: activate '360geer' env." >&2; exit 1; fi
ORIG=/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks
ROMA=/4tb_hdd/sriram/3dgs_pipeline/literature_code/360geer/RoMa/bricks_roma_openmvg
TRAIN_DIR="${1:-/4tb_hdd/sriram/3dgs_pipeline/literature_code/360geer/RoMa/bricks_origposes_romapoints}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/run_bricks_origposes_romapoints_fullres}"
GPU="${3:-1}"
ITERS_NUM=30000
python scripts/roma_points_in_orig_frame.py "$ORIG" "$ROMA" "$TRAIN_DIR"
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
