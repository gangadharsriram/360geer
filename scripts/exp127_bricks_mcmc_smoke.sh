#!/usr/bin/env bash
# Experiment 127: SHORT smoke test for the new MCMC densification port
# (--densify_mode mcmc) - not a quality test. Checks: point count grows
# toward cap_max without exploding/NaN-ing, no crashes, opacity distribution
# looks sane. Same pattern as exp94's calibration probe - short run, no
# render/metrics step, just inspect the training log + final point count.
#
# cap_max set modest (300000) for a fast check; the real comparison run
# (exp128) uses exp108's actual final count (1,118,794) for a fair test.
set -e
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then
    echo "ERROR: activate the '360geer' conda env first." >&2; exit 1
fi

DATASET_DIR="${1:-/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks}"
OUTPUT_DIR="${2:-/4tb_hdd2/sriram/360geer/test_output/exp127_bricks_mcmc_smoke}"
GPU="${3:-0}"
ITERS_NUM=2000

echo "== Seeding sparse-coverage directions =="
python scripts/seed_sparse_directions.py "$DATASET_DIR"

export CUDA_VISIBLE_DEVICES=$GPU
python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    --dataset EQUIRECT --render_model EQUI \
    --iterations "$ITERS_NUM" \
    --checkpoint_iterations "$ITERS_NUM" \
    --save_iterations "$ITERS_NUM" \
    --test_iterations "$ITERS_NUM" \
    --eval --disable_viewer \
    --densify_mode mcmc \
    --cap_max 300000 \
    --densification_interval 100
echo "exp127_bricks_mcmc_smoke done: $OUTPUT_DIR - check the training log above for crashes/NaNs, and the final point count below"
python3 -c "
with open('$OUTPUT_DIR/point_cloud/iteration_$ITERS_NUM/point_cloud.ply','rb') as f:
    for line in f:
        if line.startswith(b'element vertex'):
            print('[smoke test] final point count:', line.decode().strip()); break
"
