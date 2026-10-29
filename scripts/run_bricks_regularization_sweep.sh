#!/usr/bin/env bash
# Stronger-regularisation sweep on the best recipe (dense PF-GS-360 init, original poses, VGG 0.01, 2x schedule = 60k).
# Base for comparison = bottom-12% row weight 0.2 (SSIM 0.8637 / PSNR 25.44 / LPIPS 0.1384; no row weight: 0.8596 / 25.24 / 0.1429).
# Two GPU queues, two runs each (~2.3 h per run):
#   GPU A: (1) rowmask    : bottom 12% weight 0.0 (loss fully ignores the strip)  (2) opac0.03 : base + opacity sparsity 0.03 (was 0.012)
#   GPU B: (3) row17w0.1  : bottom 17% weight 0.1                                 (4) scale0.02: base + scale prior 0.02 (new --lambda_scale_reg)
# Target: SSIM 0.8634 / PSNR 26.2032 / LPIPS 0.1355. Usage: scripts/run_bricks_regularization_sweep.sh [GPU_A=0] [GPU_B=1]
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then echo "ERROR: activate '360geer' env." >&2; exit 1; fi
cd /data2/3dgs_pipeline/literature_code/360geer
DS=/data2/3dgs_pipeline/literature_code/360geer/RoMa/bricks_pfgs_dense_init
OUT=/data2/3dgs_pipeline/output
[ -f "$DS/pcd.ply" ] || { echo "ERROR: $DS missing - run scripts/run_bricks_pfgs_init_fullres.sh first." >&2; exit 1; }
GA="${1:-0}"; GB="${2:-1}"
run() {  # run TAG GPU [env assignments...]   (e.g. run rowmask 0 ROW_FRAC=0.12 ROW_W=0)
    tag=$1; gpu=$2; shift 2; d="$OUT/run_bricks_pfgsinit_vgg0.01_x2_${tag}_fullres"
    env LAMBDA_VGG=0.01 SCALE=2 "$@" scripts/run_bricks_schedule_fullres.sh "$DS" "$d" "$gpu" > "$d.log" 2>&1 \
        && echo "finished $tag" || echo "FAILED $tag (see $d.log)"
}
( run rowmask "$GA" ROW_FRAC=0.12 ROW_W=0;      run opac0.03 "$GA" ROW_FRAC=0.12 ROW_W=0.2 OPAC=0.03 ) &
QA=$!
( run row17w0.1 "$GB" ROW_FRAC=0.17 ROW_W=0.1;  run scale0.02 "$GB" ROW_FRAC=0.12 ROW_W=0.2 SCALE_REG=0.02 ) &
QB=$!
echo "started queues: GPU $GA (rowmask -> opac0.03), GPU $GB (row17w0.1 -> scale0.02); logs: $OUT/run_bricks_pfgsinit_vgg0.01_x2_<tag>_fullres.log"
wait $QA; wait $QB
echo "target SSIM 0.8634 PSNR 26.2032 LPIPS 0.1355"
python - <<'PY'
import json
runs = {"no row weight (ref)": "", "row12 w0.2 (base)": "row0.12w0.2", "rowmask (12%, w0)": "rowmask", "row17 w0.1": "row17w0.1", "base + opacity 0.03": "opac0.03", "base + scale prior 0.02": "scale0.02"}
for name, tag in runs.items():
    d = "run_bricks_pfgsinit_vgg0.01_x2_" + (tag + "_" if tag else "") + "fullres"
    try:
        r = list(json.load(open(f"/data2/3dgs_pipeline/output/{d}/results.json")).values())[0]
        print(f"{name:<26} SSIM {r['SSIM']:.4f}  PSNR {r['PSNR']:.4f}  LPIPS {r['LPIPS']:.4f}")
    except Exception as e: print(name, "no results:", e)
PY
O=$OUT/run_bricks_pfgsinit_vgg0.01_x2
python scripts/strip_stats.py ${O}_fullres 60000 ${O}_row0.12w0.2_fullres 60000 ${O}_rowmask_fullres 60000 ${O}_row17w0.1_fullres 60000 ${O}_opac0.03_fullres 60000 ${O}_scale0.02_fullres 60000
