# 360geer progress summary (read this first when resuming)

Last updated 2026-10-27. Goal: bricks full-res test **SSIM 0.8634 / PSNR 26.2032 / LPIPS 0.1355** (PF-GS-360 output on the same 50 views; all three must improve). Details: `experiments/metrics.md`; commands: `experiments/commands.md`; daily logs: `progress/10-2026/`.

## Where we stand
**Best run: `run_bricks_pfgsinit_vgg0.01_x2_opac0.03_fullres` → SSIM 0.8667 / PSNR 25.71 / LPIPS 0.1339** (1.40M Gaussians, 2:15:42 training). SSIM and LPIPS already beat the target; **PSNR is 0.49 dB short.**
Recipe (script `scripts/run_bricks_schedule_fullres.sh`): dataset `RoMa/bricks_pfgs_dense_init` = original poses/images/split + PF-GS-360's dense cloud (`points3D_nosky.ply`, 2.27M pts, Sim3-mapped into the original frame), VGG16 loss λ=0.01 (512² crop, from iter 0), 2× schedule (60k its, densify/prune to 30k), bottom 12% rows loss weight 0.2, opacity sparsity 0.03, `--densify_grad_unscaled --densify_grad_threshold 0.00006`, `--lambda_smooth 0.01`, `--lambda_aniso 0`, `--resolution 1`.
Rerun: `LAMBDA_VGG=0.01 SCALE=2 ROW_FRAC=0.12 ROW_W=0.2 OPAC=0.03 scripts/run_bricks_schedule_fullres.sh /data2/3dgs_pipeline/literature_code/360geer/RoMa/bricks_pfgs_dense_init <out_dir> <gpu>`.

## Metric-protocol check (2026-10-28) - read before comparing with the target
Same code path (our `metrics.py` functions, all 50 test views): PF-GS-360 = SSIM 0.8658 / PSNR 26.2032 / LPIPS-vgg 0.1659 (LPIPS-alex 0.0989); best run (opac0.03) = 0.8667 / 25.71 / 0.1339 (alex 0.0823).
Like-for-like we are already ahead on LPIPS (−0.032 vgg, −0.017 alex; alex is independent of our VGG training loss) and marginally ahead on SSIM (+0.0009); **the only real gap is PSNR (−0.49 dB)**. The quoted 0.8634 / 0.1355 come from an unknown evaluator (my scipy SSIM on their PNGs also gives 0.8634) and are not reproducible with our LPIPS. On train views the best run is now almost level with PF-GS-360 (mean per-view PSNR 27.48 vs 27.68), so the train→test drop is 1.77 dB for ours vs 1.48 dB for PF-GS-360 (it was 3.8 vs 1.5 dB before the opacity/row-weight changes).
Heatmap reports for this run vs PF-GS-360: `/data2/3dgs_pipeline/output/heatmaps_vs_pfgs360_{test,train}/heatmap_report_{test,train}.pdf` (`scripts/heatmap_report.py`).

## How the numbers got here (test SSIM / PSNR / LPIPS)
| Step | Run | Result |
|---|---|---|
| baseline (old PC; reproduced on this PC) | sparse init, 30k, no VGG | 0.8510 / 25.11 / 0.1833 |
| + 60k schedule | | 0.8570 / 25.35 / 0.1763 |
| + VGG 0.01 | 60k | 0.8564 / 25.32 / 0.1495 |
| + dense PF-GS-360 init | | 0.8596 / 25.24 / 0.1429 |
| + bottom-12% row weight 0.2 | | 0.8637 / 25.44 / 0.1384 |
| + opacity sparsity 0.03 | **best** | **0.8667 / 25.71 / 0.1339** |

## What the diagnostics say (why PSNR is the hard one)
- Train PSNR ≈ 28-29 vs test ≈ 25-25.7: a generalisation gap (PF-GS-360: 27.6 → 26.1). More Gaussians/capacity raised train and lowered test PSNR in the old sweeps (exp132-138).
- Errors are concentrated: top 1% of pixels = 43% of squared error; edges 52%; the bottom 16.7% of rows (nadir/rig/person) = 40% and are memorised on train views. Test-view error maps may only be used for diagnosis.
- Pose agreement between COLMAP / OpenMVG / PF-GS-360 is good (~0.1-0.5 camera spacings, ≤ 0.4°); oracle test-time pose search only adds +0.23 dB → poses are not the main lever. RoMa-derived poses (v2 best: 1.38°/0.40 spacings) still give only 23.2 PSNR, so they are not usable.
- PF-GS-360 advantages we have not matched: depth supervision (UniK3D + GNCC), ~1M training steps, joint camera refinement, sky handling.

## Next steps (ranked)
1. Opacity sparsity beyond 0.03 (0.05, 0.1) - largest recent gain (+0.27 dB), trend untested; combine with row weight.
2. Depth supervision (dataset has depth folders; `--depths` path exists in train.py; origin of the depth maps unchecked) - most direct generalisation lever.
3. Other regularisation of the generalisation gap (stronger priors for the nadir strip beyond weighting, early-residual per-view masks).
4. Understand the rowmask NaN in `results.json`.
Not recommended: error-guided densification (evidence says more Gaussians hurt test PSNR), more iterations alone (+0.24 dB per 2×).

## Other scenes
Barbershop (OmniBlender): 0.9762 / 37.59 / 0.0736 at 30k (thr 0.0001, untuned). gallery_chair full-res best 0.8917 / 28.46 / 0.1836. Office is a tradeoff case (see `scripts/run_office_best_fullres.sh`).
