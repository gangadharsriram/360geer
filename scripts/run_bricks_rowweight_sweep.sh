#!/usr/bin/env bash
# Fixed per-row loss weight prior: down-weight the bottom 12% of rows (equirect nadir = rig / person), two weights in parallel.
# Everything else = best recipe so far: dense PF-GS-360 init, original poses, VGG 0.01, 2x schedule (60k).
#   GPU A: bottom 12% weight 0.5     GPU B: bottom 12% weight 0.2
# Reference (no row weight): SSIM 0.8596 / PSNR 25.24 / LPIPS 0.1429. Target: 0.8634 / 26.2032 / 0.1355.
# At the end prints PSNR plus the bottom-1/6 vs rest error split (train and test) for the reference and both runs.
# Usage: scripts/run_bricks_rowweight_sweep.sh [GPU_A=0] [GPU_B=1]
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then echo "ERROR: activate '360geer' env." >&2; exit 1; fi
cd /data2/3dgs_pipeline/literature_code/360geer
DS=/data2/3dgs_pipeline/literature_code/360geer/RoMa/bricks_pfgs_dense_init
OUT=/data2/3dgs_pipeline/output
[ -f "$DS/pcd.ply" ] || { echo "ERROR: $DS missing - run scripts/run_bricks_pfgs_init_fullres.sh first (it builds the dense-init dataset)." >&2; exit 1; }
GA="${1:-0}"; GB="${2:-1}"
run() {  # run WEIGHT GPU
    d="$OUT/run_bricks_pfgsinit_vgg0.01_x2_row0.12w$1_fullres"
    LAMBDA_VGG=0.01 SCALE=2 ROW_FRAC=0.12 ROW_W=$1 scripts/run_bricks_schedule_fullres.sh "$DS" "$d" "$2" > "$d.log" 2>&1 \
        && echo "finished weight $1" || echo "FAILED weight $1 (see $d.log)"
}
run 0.5 "$GA" &
P1=$!
run 0.2 "$GB" &
P2=$!
echo "started: weight 0.5 (GPU $GA), weight 0.2 (GPU $GB); logs: $OUT/run_bricks_pfgsinit_vgg0.01_x2_row0.12w*_fullres.log"
wait $P1; wait $P2
echo "target SSIM 0.8634 PSNR 26.2032 LPIPS 0.1355 | reference (no row weight) SSIM 0.8596 PSNR 25.24 LPIPS 0.1429"
python - <<'PY'
import json
for w in ("0.5", "0.2"):
    try:
        r = list(json.load(open(f"/data2/3dgs_pipeline/output/run_bricks_pfgsinit_vgg0.01_x2_row0.12w{w}_fullres/results.json")).values())[0]
        print(f"row weight {w}: SSIM {r['SSIM']:.4f}  PSNR {r['PSNR']:.4f}  LPIPS {r['LPIPS']:.4f}")
    except Exception as e: print("weight", w, "no results:", e)
PY
python scripts/strip_stats.py $OUT/run_bricks_pfgsinit_vgg0.01_x2_fullres 60000 $OUT/run_bricks_pfgsinit_vgg0.01_x2_row0.12w0.5_fullres 60000 $OUT/run_bricks_pfgsinit_vgg0.01_x2_row0.12w0.2_fullres 60000
