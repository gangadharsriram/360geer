#!/usr/bin/env bash
# Runs all 8 isolated single-variable experiments (exp0-exp7), two at a time
# (GPU 0 + GPU 1 in parallel), waiting for each pair to finish before starting
# the next. Each script already hardcodes which GPU it uses (0 or 1) and its
# own output dir - see the individual exp*.sh files for what each one changes.
#
# Usage: scripts/run_all_experiments.sh [DATASET_DIR]
set -e
DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
cd "$(dirname "$0")/.."

pairs=(
  "exp0_baseline exp1_no_aniso"
  "exp2_densify_thresh_mild exp3_densify_thresh_strong"
  "exp4_densify_until_mild exp5_opacity_reset_freq"
  "exp6_min_opacity_strict exp7_opacity_sparsity_gentle"
)

for pair in "${pairs[@]}"; do
    read -r a b <<< "$pair"
    echo "=== Launching $a (GPU0) + $b (GPU1) ==="
    bash "scripts/${a}.sh" "$DATASET_DIR" &
    pid_a=$!
    bash "scripts/${b}.sh" "$DATASET_DIR" &
    pid_b=$!
    wait $pid_a $pid_b
    echo "=== Pair done: $a, $b ==="
done

echo "All 8 experiments complete."
echo "Summarize with: python scripts/summarize_experiments.py"
