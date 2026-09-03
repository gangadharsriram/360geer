set -e
DATASET_DIR="/4tb_hdd/sriram/dataset/Omnidirectional/Own/sathyabama/sathyabama_fisheye/fisheye_splitter/sathyabama_315/input/SFM_output/altered_output/fisheye/"
OUTPUT_DIR="./output/scnt/sathyabama_315"

STEP=0.002
FOVMOD_TRAIN=0.7

ITERS_NUM=30000

BEAP_DIR_TRAIN="beap_fov_${FOVMOD_TRAIN}_step_${STEP}/"

TRAIN_MASK_FN="fov_${FOVMOD_TRAIN}_step_${STEP}_mask.png"

# Train: Generate beap
python data/scnt/scnt_kb2beap.py --path "$DATASET_DIR" --dst "$BEAP_DIR_TRAIN" --step "$STEP" --fov_mod "$FOVMOD_TRAIN" --mask_dst "$TRAIN_MASK_FN"

# Train: Run training script
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations 3000 7000 15000 30000 \
    --save_iterations 3000 7000 15000 30000 \
    --test_iterations 3000 7000 15000 30000 \
    --resolution 1 \
    --eval \
    --sample_step "$STEP" --fov_mod "$FOVMOD_TRAIN" \
    --mask_path "${DATASET_DIR}${BEAP_DIR_TRAIN}${TRAIN_MASK_FN}" \
    --sibr_mask_refcam "${DATASET_DIR}colmap/cameras_fish.txt"
