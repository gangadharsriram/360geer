#!/usr/bin/env bash
# Train + render + evaluate an equirectangular (EQUI) scene, 30k iterations.
#
# Usage: scripts/train_equi.sh [DATASET_DIR] [OUTPUT_DIR]
# DATASET_DIR must be an OpenMVG-format scene: data_views.json, data_extrinsics.json,
# train.txt, test.txt, pcd.ply (or colorized.ply), images/ - see scene/dataset_readers.py
# readOpenMVGInfo. Defaults to the Sathyabama sample scene used to verify this pipeline.
set -e

# The EQUI render mode only exists in the geer-rasterizer build installed in the
# '360geer' conda env. The original '3dgeer' env still has the pre-EQUI binary,
# which silently falls through to a broken pinhole fallback (focal_x=focal_y=0,
# divide-by-zero ray directions) for render_model=EQUI - same inputs every time,
# so it deterministically "trains" to a fully black, collapsed result. Fail loud
# here instead of repeating that multi-hour debugging session.
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first (has the EQUI-patched rasterizer build)." >&2
    echo "Currently active: '${CONDA_DEFAULT_ENV:-none}'. Run: conda activate 360geer" >&2
    exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Own/sathyabama/sathyabama_equi/output/sathybama_sample_20/output}"
OUTPUT_DIR="${2:-./output/sathyabama_equi_30k}"

ITERS_NUM=30000

echo "== Training =="
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --eval \
    --disable_viewer

echo "== Rendering train + test sets =="
python render.py \
    -m "$OUTPUT_DIR" \
    -s "$DATASET_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --iteration "$ITERS_NUM"

echo "== Evaluating metrics (SSIM / PSNR / LPIPS) =="
python metrics.py \
    -m "$OUTPUT_DIR" \
    --block_mask \
    --iters "$ITERS_NUM"

echo "Done. Renders in $OUTPUT_DIR/{train,test}/ours_${ITERS_NUM}/, metrics in $OUTPUT_DIR/results.json"
