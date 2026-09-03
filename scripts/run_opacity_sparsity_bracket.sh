#!/usr/bin/env bash
# Runs exp40 (lambda_opacity_sparsity=0.001) and exp41 (=0.005) in parallel on
# GPU 1 and 2. Always cd's to the repo root first so it works regardless of
# where it's invoked from (the exp scripts reference scripts/seed_sparse_directions.py
# and dataset paths relative to repo root).
set -uo pipefail
cd "$(dirname "$0")/.."

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"

scripts/exp40_seeded_opacity_sparsity.sh "$DATASET_DIR" ./output/exp40_seeded_opacity_sparsity 1 &
scripts/exp41_seeded_opacity_sparsity_005.sh "$DATASET_DIR" ./output/exp41_seeded_opacity_sparsity_005 2 &
wait
echo "Both exp40 and exp41 finished."
