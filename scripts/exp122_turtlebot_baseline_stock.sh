#!/usr/bin/env bash
# Experiment 122: OmniScenes "turtlebot_pyebaekRoom_1_scene_1" scene, fully
# STOCK 360geer defaults - no seeding, no densify_grad_unscaled, no
# lambda_aniso=0, no lambda_smooth, no lambda_opacity_sparsity. Direct
# equivalent of exp113 (office's stock baseline) and 3DGEER's own scripts/
# train.sh run, but on a robot-captured (turtlebot) scene with a measured
# WIDE camera baseline: 6.07m x 0.38m x 5.78m extent, mean nearest-neighbor
# camera spacing 15.3cm (vs bricks 3.1cm, gallery_chair, office 2.4cm - all
# handheld-in-place captures under a meter across). This baseline is closer
# to 3DGEER's own room-scale scene (11.7cm) than to any of our narrow-baseline
# scenes.
#
# Purpose: direct test of the narrow-baseline hypothesis - if the artifact is
# really caused by weak parallax/narrow camera spacing (not something
# specific to the Ricoh360/360Roam capture rigs or EQUI rendering itself),
# this wide-baseline scene should come out much cleaner than bricks/
# gallery_chair/office even at fully stock settings, matching what we saw on
# 3DGEER's own scene.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/OmniScenes/turtlebot_pyebaekRoom_1_scene_1}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp122_turtlebot_baseline_stock}"
GPU="${3:-0}"
ITERS_NUM=30000

export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --eval --disable_viewer --resolution 1
python render.py -m "$OUTPUT_DIR" -s "$DATASET_DIR" --dataset EQUIRECT --render_model EQUI --iteration "$ITERS_NUM"
python metrics.py -m "$OUTPUT_DIR" --block_mask --iters "$ITERS_NUM"
echo "exp122_turtlebot_baseline_stock done: $OUTPUT_DIR"


