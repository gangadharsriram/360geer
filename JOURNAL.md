# JOURNAL (append-only; newest at the bottom). Detailed logs: `.agent-log/progress/MM-YYYY/DD-MM-YY.md`.

## 2026-10-14 12:00 - Claude Code - RoMa poses are the limit; moved to the new PC [ID: f726fa90-a7b3-4f2b-874f-392c26af976c]
**Did:** Found why RoMa-based reconstructions train badly (poses, not points/scale): cell-1 + angular-filter SfM cut pose error 4.5° → 1.38° but PSNR only 20.0 → 23.2 (orig poses 25.1). Moved repo to `/data2`; fixed the sm_89 nvrtc crash (`Tensor.prod` → `_prod3`); RoMa v2 poses + original points = 23.17.
**State:** RoMa poses unusable for the benchmark; original OpenMVG poses are the base for all later work.
**Next:** perceptual loss idea (VGG) to improve LPIPS.
**Watch out:** `nvrtc ... --gpu-architecture` = a JIT-compiled op on the RTX 4090; use plain tensor ops.

## 2026-10-15 23:50 - Claude Code - VGG perceptual loss added and swept [ID: f726fa90-a7b3-4f2b-874f-392c26af976c]
**Did:** `--lambda_vgg --vgg_crop --vgg_start_iter`; sweep over λ, threshold, start iteration (30k). Training now logs test/train SSIM+LPIPS. Barbershop script/run (0.9762 / 37.59 / 0.0736).
**State:** λ=0.01 ≈ keeps pixel metrics (0.8495 / 25.08) and LPIPS 0.1578; higher λ over-densifies (λ=0.05 → 4.4M Gaussians).
**Next:** no-VGG control and longer schedule on this PC.
**Watch out:** a mistyped `tee` target overwrote `scripts/run_bricks_vgg_fullres.sh` once - check paths after `tee`.

## 2026-10-16 13:00 - Claude Code - control, 60k schedule, benchmark analysis, dense init [ID: f726fa90-a7b3-4f2b-874f-392c26af976c]
**Did:** Control reproduces old baseline (25.11 vs 25.12). 60k schedule +0.24 dB. Identified the target as PF-GS-360's output; compared per view/region; started our pipeline from its dense cloud (0.8596 / 25.24 / 0.1429); oracle test-time pose search +0.23 dB; pose agreement reports (bricks + all scenes).
**State:** SSIM/LPIPS close to target, PSNR stuck ≈ 25.2 due to the train→test gap.
**Next:** attack the nadir strip / generalisation.
**Watch out:** PF-GS-360 checkpoints for gallery_pillar and garden are untrained (steps 1, 0); its ckpts need `numpy._core` aliasing to load.

## 2026-10-19 02:50 - Claude Code - heatmaps, row weight, regularisation sweep [ID: f726fa90-a7b3-4f2b-874f-392c26af976c]
**Did:** Error heatmaps (test+train): nadir strip = 40% of test error, memorised on train. Row-weight prior (`--row_weight_*`) +0.2 dB; regularisation sweep: opacity sparsity 0.03 → **0.8667 / 25.71 / 0.1339 (best; SSIM+LPIPS beat target, PSNR −0.49 dB)**, scale prior and full masking worse.
**State:** best run `run_bricks_pfgsinit_vgg0.01_x2_opac0.03_fullres` (see `.agent-log/progress_summary.md` for the recipe).
**Next:** opacity > 0.03, depth supervision.
**Watch out:** rowmask's `results.json` is NaN (per-view finite, in-training eval 23.34 PSNR).

## 2026-10-27 18:10 - Claude Code - journal setup and backfill [ID: f726fa90-a7b3-4f2b-874f-392c26af976c]
**Did:** Adopted the journal convention from `/data2/3dgs_pipeline/agents` for 360geer: AGENTS.md/CLAUDE.md/JOURNAL.md/.agent-log created; 10-12 → 10-19 history backfilled from run logs, result files and timestamps (including three results never discussed before: the reg sweep, barbershop, romapose_origpoints).
**State:** all runs and results are recorded in `.agent-log/experiments/metrics.md`; commands in `commands.md`; read `.agent-log/progress_summary.md` first.
**Next:** see progress_summary.md "Next steps".
**Watch out:** commands.md was reconstructed (no live logging before today); other repos under literature_code (SeedVR etc.) were changed by other sessions and are not covered here.

## 2026-10-28 00:10 - Claude Code - per-view heatmap reports vs PF-GS-360 and a metric-protocol finding [ID: f726fa90-a7b3-4f2b-874f-392c26af976c]
**Did:** `scripts/heatmap_report.py` builds, for every test and train view, "our render vs GT" next to "PF-GS-360 render vs GT" (error heatmap, structure-loss = 1 − local SSIM heatmap, ours-minus-theirs maps) with an explanation page; outputs in `/data2/3dgs_pipeline/output/heatmaps_vs_pfgs360_{test,train}/`. Re-ran PF-GS-360's PNGs through our metrics code.
**State:** only PSNR of the target reproduces (26.2032). Same code path: PF-GS-360 SSIM 0.8658 / LPIPS-vgg 0.1659 / alex 0.0989; our best run 0.8667 / 0.1339 / 0.0823 → we lead on LPIPS and marginally SSIM; the only real gap is PSNR (−0.49 dB).
**Next:** opacity sparsity > 0.03, depth supervision; confirm with the user which evaluator produced 0.8634 / 0.1355.
**Watch out:** do not claim "beats the target on LPIPS/SSIM" without saying which evaluator; use `.agent-log/experiments/metrics.md` S9b.
