#!/usr/bin/env bash
# Bricks full-res baseline recipe with (a) an optional VGG loss weight and (b) a stretched training schedule.
# SCALE multiplies every iteration-based setting (iterations, densify/prune end, position-LR decay, reset/size-prune gates,
# checkpoint + test iterations). SCALE=1 is the original 30k schedule, SCALE=2 trains 60k with densification until 30k.
#   control (no VGG, this PC):  LAMBDA_VGG=0    SCALE=1 scripts/run_bricks_schedule_fullres.sh "" "" 0
#   long + denser, VGG 0.01:    LAMBDA_VGG=0.01 SCALE=2 scripts/run_bricks_schedule_fullres.sh "" "" 1
# Optional regularisers: OPAC=<opacity sparsity weight, default 0.012>, SCALE_REG=<scale prior weight, default 0>.
# Optional fixed per-row loss weight for the bottom rows: ROW_FRAC=0.12 ROW_W=0.5 (off by default).
# Target to beat (bricks full-res): SSIM 0.8634 / PSNR 26.2032 / LPIPS 0.1355. Reference: baseline 0.8509/25.12/0.1841.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then echo "ERROR: activate '360geer' env." >&2; exit 1; fi
cd /data2/3dgs_pipeline/literature_code/360geer
DATASET_DIR="${1:-/data2/Dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
LAMBDA_VGG="${LAMBDA_VGG:-0.01}"; SCALE="${SCALE:-1}"; VGG_CROP="${VGG_CROP:-512}"; ROW_FRAC="${ROW_FRAC:-0}"; ROW_W="${ROW_W:-1}"; OPAC="${OPAC:-0.012}"; SCALE_REG="${SCALE_REG:-0}"; BASE="${BASE_ITERS:-30000}"
OUTPUT_DIR="${2:-/data2/3dgs_pipeline/output/run_bricks_vgg${LAMBDA_VGG}_x${SCALE}_fullres}"
GPU="${3:-0}"
s() { echo $(( $1 * SCALE )); }
ITERS_NUM=$(s $BASE)
python scripts/seed_sparse_directions.py "$DATASET_DIR"
export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI --resolution 1 \
    --iterations "$ITERS_NUM" --position_lr_max_steps "$ITERS_NUM" \
    --densify_until_iter $(s $(( BASE / 2 ))) --prune_until_iter $(s $(( BASE / 2 ))) \
    --checkpoint_iterations $(s 3000) $(s 7000) $(s 15000) "$ITERS_NUM" \
    --save_iterations $(s 3000) $(s 7000) $(s 15000) "$ITERS_NUM" \
    --test_iterations $(s 3000) $(s 7000) $(s 15000) $(s 20000) $(s 25000) "$ITERS_NUM" \
    --eval --disable_viewer \
    --opacity_reset_interval "$ITERS_NUM" --size_prune_from_iter "$ITERS_NUM" \
    --densify_grad_unscaled --densify_grad_threshold 0.00006 \
    --lambda_smooth 0.01 --lambda_aniso 0.0 --lambda_opacity_sparsity "$OPAC" --lambda_scale_reg "$SCALE_REG" \
    --lambda_vgg "$LAMBDA_VGG" --vgg_crop "$VGG_CROP" \
    --row_weight_bottom_frac "$ROW_FRAC" --row_weight_value "$ROW_W"
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "done: $OUTPUT_DIR"
