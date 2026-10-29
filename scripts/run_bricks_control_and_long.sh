#!/usr/bin/env bash
# Bricks full-res experiments, two GPU queues running in parallel, then a comparison table.
#   GPU 0: (1) no-VGG control, 30k          (2) no-VGG, 2x schedule (60k)  -> isolates the schedule effect from VGG
#   GPU 1: (3) VGG 0.01, 2x schedule (60k)   (4) VGG 0.02, 2x schedule (60k)
# SCALE=2 -> 60k iterations, densify/prune until 30k (see run_bricks_schedule_fullres.sh).
# Logs: /data2/3dgs_pipeline/output/<run>.log   Usage: scripts/run_bricks_control_and_long.sh [GPU_A=0] [GPU_B=1]
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then echo "ERROR: activate '360geer' env." >&2; exit 1; fi
cd /data2/3dgs_pipeline/literature_code/360geer
OUT=/data2/3dgs_pipeline/output
GA="${1:-0}"; GB="${2:-1}"
run() {  # run LAMBDA SCALE GPU : one training run, logged
    LAMBDA_VGG=$1 SCALE=$2 scripts/run_bricks_schedule_fullres.sh "" "" "$3" > "$OUT/run_bricks_vgg$1_x$2_fullres.log" 2>&1 \
        && echo "finished vgg$1 x$2" || echo "FAILED vgg$1 x$2 (see $OUT/run_bricks_vgg$1_x$2_fullres.log)"
}
( run 0 1 "$GA"; run 0 2 "$GA" ) &
QA=$!
( run 0.01 2 "$GB"; run 0.02 2 "$GB" ) &
QB=$!
echo "started queues: GPU $GA (vgg0 x1 -> vgg0 x2) pid $QA, GPU $GB (vgg0.01 x2 -> vgg0.02 x2) pid $QB; logs in $OUT"
wait $QA; wait $QB
python - <<'PY'
import json
print("target                          SSIM 0.8634  PSNR 26.2032  LPIPS 0.1355")
print("old baseline (old PC)           SSIM 0.8509  PSNR 25.1214  LPIPS 0.1841")
for d in ("run_bricks_vgg0_x1_fullres", "run_bricks_vgg0_x2_fullres", "run_bricks_vgg0.01_x2_fullres", "run_bricks_vgg0.02_x2_fullres"):
    try:
        r = list(json.load(open(f"/data2/3dgs_pipeline/output/{d}/results.json")).values())[0]
        print(f"{d[11:-8]:<31} SSIM {r['SSIM']:.4f}  PSNR {r['PSNR']:.4f}  LPIPS {r['LPIPS']:.4f}")
    except Exception as e:
        print(d, "no results:", e)
PY
