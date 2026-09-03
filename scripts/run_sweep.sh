#!/usr/bin/env bash
# Runs exp20-exp37 (the size-cull/min_opacity/densify_grad_threshold/aniso sweep)
# across all 3 GPUs, 6 experiments each in sequence, then aggregates every
# ./output/exp*/results.json into a CSV. No -e: one experiment failing shouldn't
# kill the rest of its GPU's queue.
set -uo pipefail
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_ROOT="${2:-./output}"
LOG_DIR="${OUTPUT_ROOT}/sweep_logs"
mkdir -p "$LOG_DIR"

SCRIPTS=(
  exp20_seeded_prune_until_20k
  exp21_seeded_prune_until_25k
  exp22_seeded_prune_until_30k
  exp23_seeded_min_opacity_01
  exp24_seeded_min_opacity_02
  exp25_seeded_min_opacity_05
  exp26_seeded_grad_thresh_0003
  exp27_seeded_grad_thresh_0005
  exp28_seeded_grad_thresh_0008
  exp29_seeded_size_cull_from_5k
  exp30_seeded_size_cull_from_12k
  exp31_seeded_aniso_hard_20
  exp32_seeded_aniso_hard_10
  exp33_seeded_combo_prune30k_minop02
  exp34_seeded_combo_prune30k_grad0005
  exp35_seeded_combo_minop02_grad0005
  exp36_seeded_combo_kitchen_sink
  exp37_seeded_combo_earlycull_longprune
)

GPUS=(0 1 2)
NUM_GPUS=${#GPUS[@]}

run_gpu_queue() {
    local gpu_idx=$1
    local gpu=${GPUS[$gpu_idx]}
    local i name
    for (( i=gpu_idx; i<${#SCRIPTS[@]}; i+=NUM_GPUS )); do
        name=${SCRIPTS[$i]}
        echo "[GPU $gpu] starting $name"
        if scripts/${name}.sh "$DATASET_DIR" "${OUTPUT_ROOT}/${name}" "$gpu" \
            > "${LOG_DIR}/${name}.log" 2>&1; then
            echo "[GPU $gpu] done $name"
        else
            echo "[GPU $gpu] FAILED $name (see ${LOG_DIR}/${name}.log)"
        fi
    done
}

for (( g=0; g<NUM_GPUS; g++ )); do
    run_gpu_queue "$g" &
done
wait

echo "== All queues finished, aggregating results =="
python scripts/summarize_experiments.py "${OUTPUT_ROOT}/exp*" --csv "${OUTPUT_ROOT}/sweep_results.csv"
echo "CSV written to ${OUTPUT_ROOT}/sweep_results.csv"
