#!/usr/bin/env bash
# Experiment 113: bricks, best confirmed config (exp108) at full native
# resolution instead of the auto-downscaled default.
#
# exp108's config: seeded pcd, resets off, no size-cull, --densify_grad_unscaled,
# densify_grad_threshold 0.00006, lambda_smooth 0.01, lambda_aniso 0.0,
# lambda_opacity_sparsity 0.012 (peak of the sweep, exp103-111) - best result
# of the investigation so far: SSIM 0.8681/PSNR 25.79/LPIPS 0.1653 at the
# default (downscaled) resolution.
#
# --resolution 1 forces global_down=1 (native res, images auto-scaled by
# orig_w/1600 otherwise) - render.py/metrics.py pick this up automatically
# from the saved cfg_args, no extra flag needed there.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp113_bricks_best_fullres}"
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
    --opacity_reset_interval 30000 \
    --size_prune_from_iter 30000 \
    --densify_grad_unscaled \
    --densify_grad_threshold 0.00006 \
    --lambda_smooth 0.01 \
    --lambda_aniso 0.0 \
    --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp113_bricks_best_fullres done: $OUTPUT_DIR"
