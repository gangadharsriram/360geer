#!/usr/bin/env bash
# Full-resolution run on OmniBlender/barbershop with the best-known 360geer recipe
# (exp108/bricks + gallery_chair recipe: seeded pcd, resets/size-prune off, unscaled densify grad,
# lambda_smooth 0.01, lambda_aniso 0, opacity sparsity 0.012).
#  - --resolution 1: barbershop is 2000px wide, the default would auto-downscale to 1600px.
#  - densify_grad_threshold is the only scene-specific knob (bricks 0.00006, gallery_chair 0.0001, office 0.001);
#    barbershop was never tuned, so 0.0001 is a starting guess. Override: THRESH=0.00006 scripts/run_barbershop_best_fullres.sh
#    Sanity check from the log: bricks reached ~209k / 852k Gaussians at iter 3000 / 7000 (very few => threshold too high).
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then echo "ERROR: activate '360geer' env." >&2; exit 1; fi
cd /data2/3dgs_pipeline/literature_code/360geer
DATASET_DIR="${1:-/data2/Dataset/Omnidirectional/OmniBlender/barbershop}"
OUTPUT_DIR="${2:-/data2/3dgs_pipeline/output/run_barbershop_best_fullres}"
GPU="${3:-0}"
THRESH="${THRESH:-0.0001}"
ITERS_NUM=30000

echo "== Seeding sparse-coverage directions =="
python scripts/seed_sparse_directions.py "$DATASET_DIR"

export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI --resolution 1 \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 20000 25000 30000 \
    --eval --disable_viewer \
    --opacity_reset_interval 30000 --size_prune_from_iter 30000 \
    --densify_grad_unscaled --densify_grad_threshold "$THRESH" \
    --lambda_smooth 0.01 --lambda_aniso 0.0 --lambda_opacity_sparsity 0.012
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "run_barbershop_best_fullres done: $OUTPUT_DIR"
