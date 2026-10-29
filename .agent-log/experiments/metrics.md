# Experiment metrics (results, timings, comparisons; commands are in commands.md under the same S-headings)

Backfilled 2026-10-27. Bricks, full resolution (1920×960), 50 test views, `metrics.py --block_mask` (= full-image metrics) unless stated. Format: SSIM / PSNR / LPIPS (G = final Gaussian count).
Training times are from each run's own tqdm line; run-to-run noise measured at ~0.02 dB PSNR (control on this PC vs old PC).
**Target: 0.8634 / 26.2032 / 0.1355** (PF-GS-360).

## Reference baselines (old PC, from `all_metrics.json`)
| Run | G | Test SSIM / PSNR / LPIPS | Train PSNR |
|---|---|---|---|
| exp108 bricks opacity-sparsity 0.012 (auto-res 1600 px, not comparable to full-res) | 1.12M | 0.8681 / 25.79 / 0.1653 | 28.26 |
| run_bricks_best_fullres (exp108 + `--resolution 1`) | 1.24M | 0.8509 / 25.12 / 0.1841 | 27.01 |
| exp128 MCMC (auto-res) | 1.12M | 0.8660 / 25.58 / 0.1707 | 26.49 |
| exp132 same recipe, 7.3M Gaussians (thresh 2e-6, auto-res) | 7.34M | 0.8234 / 24.05 / 0.2156 | 33.23 |
| exp138 / exp137 / exp136 (thresh 8e-6 / 6e-6 / 5e-6, auto-res) | 449k / 820k / 1.20M | 25.24 / 25.15 / 25.06 | 27.05 / 27.82 / 28.48 |
More Gaussians → higher train PSNR, lower test PSNR (exp132-138). gallery_chair best (exp112, auto-res): 0.8980 / 28.82 / 0.1667; full-res 0.8917 / 28.46 / 0.1836.

## S7 RoMa-based poses / datasets
| Run | Poses | Points | Test SSIM / PSNR / LPIPS | G | Train time | Date (log) |
|---|---|---|---|---|---|---|
| bricks_roma_openmvg (v1: cell 8, F-filter) | RoMa v1 | RoMa v1 | 0.6605 / 20.01 / 0.3411 | - | log: iter 3000 eval 21:21 → iter 30000 eval 22:02 | 2026-10-12 |
| origposes_romapoints (old PC) | original | RoMa v1 (mapped) | 0.8464 / 25.03 / 0.1886 | 1.27M | 1:04:11 | 2026-10-12/13 |
| roma_v_q1_a | RoMa v2 (cell 1 + angular) | RoMa v2 | 0.7732 / 23.16 / 0.2507 | 1.66M | 1:24:53 | 2026-10-13 |
| romapose_origpoints (new PC) | RoMa v2 (mapped into original frame) | original sparse (5,958) | 0.7708 / 23.17 / 0.2479 | 1.53M | - | 2026-10-14 |
Same RoMa v2 poses give 23.16-23.17 with either point source, original poses with RoMa points give 25.03 → the RoMa poses (not the points) limited those runs.
Pose agreement with the original poses (Sim3-fitted, 100 views; rot = centre-fit): v1 rot 4.5° / centre 0.84 spacings; `q1_f` 10.4° / 3.0; `q8_a` 3.8° / 2.1 (90 cams registered); **`q1_a` 1.38° / 0.40** (566k pts, 100/100 cams).
colmap_bricks_q8 (scale 0.0905 of original, rescaled ×0.1): 258k / 1.6M Gaussians at 3k / 7k vs 36k / 58k unscaled, but test PSNR 19.9 at 7k (original 25.0) → scale fixed the count, not quality.

## S8 OmniBlender barbershop
`run_barbershop_best_fullres` (30k, thr 0.0001): 0.9762 / 37.59 / 0.0736, G 239,938, completed 2026-10-15 15:04 (25 train / 25 test views).

## S1 VGG loss, sparse init, 30k (threshold 0.00006 unless stated)
| λ_vgg | Test SSIM / PSNR / LPIPS | G | Train time |
|---|---|---|---|
| 0 (control on this PC, S2) | 0.8510 / 25.11 / 0.1833 | 1.24M | 47:20 |
| 0.01 | 0.8495 / 25.08 / 0.1578 | 1.66M | 1:03:27 |
| 0.02 | 0.8472 / 24.98 / 0.1501 | 2.24M | 1:03:34 |
| 0.05 | 0.8390 / 24.73 / 0.1477 | 4.41M | 1:19:16 |
| 0.05, thr 0.00015 | 0.8327 / 24.68 / 0.1560 | 1.22M | 56:35 |
| 0.05 from iter 15000 | 0.8407 / 24.87 / 0.1513 | 1.24M | 52:41 |
VGG trades PSNR/SSIM for LPIPS smoothly with weight and inflates the Gaussian count (VGG gradient drives densification: raising the threshold to hold the count did not help).

## S2 Schedule (SCALE=2 → 60k) and control, sparse init
| Run | Test SSIM / PSNR / LPIPS | G | Train time |
|---|---|---|---|
| no VGG, 30k | 0.8510 / 25.11 / 0.1833 | 1.24M | 47:20 |
| no VGG, 60k | 0.8570 / 25.35 / 0.1763 | 1.31M | 1:34:10 |
| VGG 0.01, 60k | 0.8564 / 25.32 / 0.1495 | 1.80M | 1:58:03 |
| VGG 0.02, 60k | 0.8516 / 25.15 / 0.1472 | 2.57M | 2:10:00 |
The longer schedule alone gives ≈ +0.24 dB.

## S3 Dense initialisation (PF-GS-360 `points3D_nosky.ply`, 2.27M pts, original poses)
| Run | Test SSIM / PSNR / LPIPS | G | Train time |
|---|---|---|---|
| dense, no VGG, 30k | 0.8589 / 25.23 / 0.1644 | 2.23M | 1:01:07 |
| dense, VGG 0.01, 60k | 0.8596 / 25.24 / 0.1429 | 2.39M | 2:29:36 |
Dense init improves SSIM (+0.008) and LPIPS (−0.019 no-VGG) and converges much faster (test PSNR 25.0 already at iter 3000 vs 23.3), but test PSNR stays ≈ 25.2 while train PSNR reaches 28.8-29.1 (generalisation gap 3.8 dB; PF-GS-360: 27.58 train → 26.13 test = 1.5 dB).

## S4 Oracle test-time pose search (dense VGG 0.01 60k model, 50 test views, `scripts/test_time_pose_search.py`)
PSNR 25.241 → 25.468, SSIM 0.8602 → 0.8609, LPIPS 0.1443 → 0.1441. Pose moves: median ~0.035 camera spacings, rotation median 0.000° (max 0.087°, 0.149 spacings). Gain per view: median 0.18 dB, max 0.79, 5 views > 0.5 dB. → pose fitting explains ≈ 0.23 of the 0.96 dB gap.

## S5 Fixed per-row loss weight (dense init, VGG 0.01, 60k; bottom 12% of rows)
| Run | Test SSIM / PSNR / LPIPS | G | Train time |
|---|---|---|---|
| no row weight (ref) | 0.8596 / 25.24 / 0.1429 | 2.39M | 2:29:36 |
| weight 0.5 | 0.8622 / 25.35 / 0.1402 | 1.97M | 2:20:49 |
| weight 0.2 | 0.8637 / 25.44 / 0.1384 | 1.70M | 2:05:01 |

## S6 Stronger regularisation (base = row 12% w0.2, 60k)
| Run | Test SSIM / PSNR / LPIPS | G | Train time |
|---|---|---|---|
| row 12% w0.2 (base) | 0.8637 / 25.44 / 0.1384 | 1.70M | 2:05:01 |
| rowmask (12%, w 0) | results.json NaN; training-time eval at 60k: 0.8411 / 23.34 / 0.1490; pooled PSNR 23.30 | 1.55M | 2:10:59 |
| row 17% w0.1 | 0.8612 / 25.41 / 0.1391 | 1.57M | 2:01:35 |
| **base + opacity sparsity 0.03 (was 0.012)** | **0.8667 / 25.71 / 0.1339** | 1.40M | 2:15:42 |
| base + scale prior 0.02 | 0.8555 / 25.25 / 0.1463 | 1.44M | 1:47:37 |
`opac0.03` beats the target on SSIM (+0.0033) and LPIPS (−0.0016); PSNR is 0.49 dB short. Opacity sparsity 0.012 → 0.03 gave +0.27 dB / +0.003 SSIM / −0.0045 LPIPS vs its base - strongest lever of the last sweep; the trend beyond 0.03 is untested.
Bottom-sixth vs rest test MSE (×1e-3): ref 7.25 / 2.21; w0.2 6.62 / 2.17; opac0.03 6.45 / 2.00; row17w0.1 6.74 / 2.17; scale0.02 6.72 / 2.30; rowmask 17.27 / 2.16. PF-GS-360: 5.72 / 1.78.

## S9 PF-GS-360 benchmark analysis (`scripts/compare_with_pfgs360.py`, `scripts/error_heatmaps.py`, `scripts/strip_stats.py`)
- Same 50 test images (pixel-identical GT), same size. Its `results.json` reports 26.2166 / 0.8706 / 0.1367 (own evaluator); recomputed PSNR from its PNGs = 26.2032. 2.68M Gaussians, init 2.27M pts from UniK3D depth, depth supervision + GNCC, joint pose refinement (+ test-time pose opt), ~1M steps.
- vs our dense VGG 0.01 60k run: it wins 47/50 views, median +0.91 dB. MSE ratio ours/theirs by 8 latitude bands (top→bottom): 0.86 (we are better), 1.02, 1.19, 1.28, 1.27, 1.25, 1.09, 1.30. Edge detail: gradient energy GT 0.0450 / PF-GS 0.0354 / ours 0.0384 (we are not blurrier).
- Error concentration (test): top 1% of pixels carry 43% of squared error, top 5% 74%; edges (17.5% of pixels) 52%; bottom 16.7% of rows (nadir, rig/person) 39.6% (bottom 12%: 37.0%). Ours/PF-GS error by content: sky-like 1.21, edges 1.32, flat 1.18, bottom 12% 1.28.
- Train vs test (pooled MSE ×1e-3, bottom sixth / rest): ours train 1.16 / 1.30 → test 7.25 / 2.21 (PSNR 28.93 → 25.16); PF-GS train 2.40 / 1.62 → test 5.72 / 1.78 (27.58 → 26.13). Our nadir strip is memorised on train views and fails on test.

## S10 Pose-source agreement (no ground truth; `scripts/pose_comparison_report.py`, `scripts/pose_comparison_all_scenes.py`)
- Bricks, 50 common views: COLMAP vs OpenMVG ATE 0.19 spacings, rot (gauge-free) 0.14°; COLMAP vs PF-GS-360 0.30 / 0.23°; OpenMVG vs PF-GS-360 0.13 / 0.17° (closest). RTA AUC@10: 83 / 76 / 90.
- All 12 Ricoh360 scenes, 100 cams (PF-GS-360 poses decoded from checkpoints): 24 of 28 pairs rated Excellent/Good (my thresholds: Excellent < 0.2 spacings & < 0.3°), none Poor; worst gallery_park COLMAP-vs-PF-GS 0.63 spacings / 0.38°. gallery_pillar (ckpt step 1) and garden (step 0) PF-GS checkpoints are untrained → excluded; bridge has no COLMAP; gallery_chair has no PF-GS-360.
- PF-GS-360 train-camera poses differ from the dataset poses by ~0.17° per camera and 0.14 spacings (ATE 0.0027) - pose refinement is small.
Reports: `/data2/3dgs_pipeline/output/pose_comparison_report.pdf`, `pose_comparison_all_scenes.pdf`; heatmaps: `error_heatmaps_pfgsinit/` (test), `error_heatmaps_pfgsinit_train/` (train); `error_heatmaps_pfgs360_test/` (2026-10-19 10:32, created outside the chat's recorded commands - origin unknown, not analysed).

## S9b Metric-protocol check and heatmap reports (2026-10-28)
Same functions as `metrics.py` (`utils.loss_utils.ssim`, `lpipsPyTorch.lpips`), 50 test views, ours = `run_bricks_pfgsinit_vgg0.01_x2_opac0.03_fullres` @ 60000:
| Model | SSIM | PSNR | LPIPS vgg | LPIPS alex |
|---|---|---|---|---|
| PF-GS-360 (its PNGs) | 0.8658 | 26.2032 | 0.1659 | 0.0989 |
| ours (opac0.03) | 0.8667 | 25.71 | 0.1339 | 0.0823 |
The user-quoted target 0.8634 / 0.1355 and PF-GS-360's own results.json (0.8706 / 0.1367) come from other evaluators. My scipy-based SSIM (`heatmap_report.py`, reflect padding) gives PF-GS 0.8634 and ours 0.8644 (test); train: ours 0.8987 vs PF-GS 0.8820 SSIM, PSNR (mean of per-view) 27.48 vs 27.68.
Reports: `/data2/3dgs_pipeline/output/heatmaps_vs_pfgs360_test/heatmap_report_test.pdf` and `..._train/heatmap_report_train.pdf` (52 pages each: explanation, mean maps, 50 per-view pages; per-view JPEGs alongside).
