#!/usr/bin/env bash
# Tests dense initialisation: our pipeline started from the PF-GS-360 dense point cloud (2.27M pts, no sky, mapped into the
# original bricks frame) with the ORIGINAL poses. Two runs in parallel, one per GPU:
#   GPU A: dense init + VGG 0.01, 2x schedule (60k)  -> compare with sparse-init best: 0.8564 / 25.32 / 0.1495
#   GPU B: dense init, no VGG, 30k                   -> compare with sparse-init control: 0.8510 / 25.11 / 0.1833
# Target: SSIM 0.8634 / PSNR 26.2032 / LPIPS 0.1355.
# Usage: [MAX_POINTS=0] scripts/run_bricks_pfgs_init_fullres.sh [GPU_A=0] [GPU_B=1]     (MAX_POINTS>0 randomly subsamples the cloud)
if [ "${CONDA_DEFAULT_ENV:-}" != "360geer" ]; then echo "ERROR: activate '360geer' env." >&2; exit 1; fi
cd /data2/3dgs_pipeline/literature_code/360geer
PF=/data2/Resuts/360/pfgs360_outputs/Ricoh360/bricks/pfgs360
ORIG=/data2/Dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks
DS=/data2/3dgs_pipeline/literature_code/360geer/RoMa/bricks_pfgs_dense_init   # dataset dir (symlinks + converted cloud)
OUT=/data2/3dgs_pipeline/output
GA="${1:-0}"; GB="${2:-1}"
set -e
python scripts/pfgs_points_to_orig_frame.py "$PF" "$ORIG" "$DS" "${MAX_POINTS:-0}"
python scripts/seed_sparse_directions.py "$DS"
set +e
LAMBDA_VGG=0.01 SCALE=2 scripts/run_bricks_schedule_fullres.sh "$DS" "$OUT/run_bricks_pfgsinit_vgg0.01_x2_fullres" "$GA" > "$OUT/run_bricks_pfgsinit_vgg0.01_x2_fullres.log" 2>&1 &
P1=$!
LAMBDA_VGG=0 SCALE=1 scripts/run_bricks_schedule_fullres.sh "$DS" "$OUT/run_bricks_pfgsinit_vgg0_x1_fullres" "$GB" > "$OUT/run_bricks_pfgsinit_vgg0_x1_fullres.log" 2>&1 &
P2=$!
echo "started: dense+VGG0.01 x2 (GPU $GA, pid $P1), dense no-VGG x1 (GPU $GB, pid $P2); logs in $OUT"
wait $P1 && echo "dense+VGG0.01 x2 finished" || echo "dense+VGG0.01 x2 FAILED"
wait $P2 && echo "dense x1 finished" || echo "dense x1 FAILED"
python - <<'PY'
import json
print("target                      SSIM 0.8634  PSNR 26.2032  LPIPS 0.1355")
print("sparse init  vgg0   x1      SSIM 0.8510  PSNR 25.1097  LPIPS 0.1833")
print("sparse init  vgg0.01 x2     SSIM 0.8564  PSNR 25.3241  LPIPS 0.1495")
for d in ("run_bricks_pfgsinit_vgg0_x1_fullres", "run_bricks_pfgsinit_vgg0.01_x2_fullres"):
    try:
        r = list(json.load(open(f"/data2/3dgs_pipeline/output/{d}/results.json")).values())[0]
        print(f"dense init   {d[21:-8]:<14} SSIM {r['SSIM']:.4f}  PSNR {r['PSNR']:.4f}  LPIPS {r['LPIPS']:.4f}")
    except Exception as e:
        print(d, "no results:", e)
PY
