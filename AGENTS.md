# Project Agent Guidelines: 360geer (equirectangular 3D Gaussian Splatting)

Canonical context for any AI coding agent working in this repo. Adapted from `/data2/3dgs_pipeline/agents/AGENTS.md` (that file is for the
Relighting project - only the journaling protocol and the Unicode-math rule are reused here).
Resume a session by reading, in order: this file → `JOURNAL.md` (last entries) → `.agent-log/progress_summary.md` →
latest `.agent-log/progress/MM-YYYY/DD-MM-YY.md`. Raw numbers: `.agent-log/experiments/metrics.md`; exact commands: `.agent-log/experiments/commands.md`.

## Goal (set 2026-10-16)
Beat, on Ricoh360 **bricks** at full resolution (`--resolution 1`, 50 test views, 1920×960, full-image metrics = `metrics.py --block_mask`):
`SSIM 0.8634 / PSNR 26.2032 / LPIPS 0.1355` - all three must improve. These come from PF-GS-360 (`/data2/Resuts/360/pfgs360_outputs/Ricoh360/bricks/pfgs360/`,
`eval_renders` pred vs gt reproduces PSNR 26.2032 exactly). Current best run and the remaining gap: see `.agent-log/progress_summary.md`.
**Metric-protocol caveat (checked 2026-10-28):** only PSNR of the target reproduces exactly from PF-GS-360's PNGs. Pushed through our `metrics.py` code path the same PF-GS-360 renders score SSIM 0.8658 / LPIPS(vgg) 0.1659
(LPIPS alex 0.0989); the 0.8634 / 0.1355 figures come from some other evaluator (its own results.json says 0.8706 / 0.1367). Compare SSIM/LPIPS only against numbers computed with the same code, and always report both.

## Environment
- Conda env `360geer` (python 3.7, torch 1.12.1+cu116). Always `conda activate 360geer`; scripts refuse to run otherwise.
  Non-interactive: `PATH=/home/sriram/miniconda3/envs/360geer/bin:$PATH CONDA_DEFAULT_ENV=360geer`.
- Machine (since 2026-10-14): 2× RTX 4090 24 GB (sm_89), paths under `/data2/...`. The old PC's `/4tb_hdd`, `/4tb_hdd2` no longer exist; some old JSON/log files still mention them.
  torch 1.12/cu116's nvrtc rejects sm_89 JIT kernels → `Tensor.prod()` was replaced by `_prod3` in `scene/gaussian_model.py`. Avoid other ops that JIT-compile (reductions via jiterator); if
  `nvrtc: error: invalid value for --gpu-architecture` appears, find the op.
- Datasets: `/data2/Dataset/Omnidirectional/{Ricoh360,OmniBlender,360Roam,...}` (OpenMVG format `Data_openmvg/<scene>`: `data_views.json`, `data_extrinsics.json`, `images/`, `train.txt`, `test.txt`, `pcd.ply`).
- Outputs: `/data2/3dgs_pipeline/output/` (runs + logs + heatmaps + reports). Working repo: `/data2/3dgs_pipeline/literature_code/360geer`. GPU may be shared with another user (check `nvidia-smi`).
- Memory notes for Claude Code live in `~/.claude/projects/-4tb-hdd-sriram-3dgs-pipeline-literature-code-360geer/memory/` (old-path folder name; still the active one).

## Pipeline facts that cost time to rediscover
- Training: `train.py --dataset EQUIRECT --render_model EQUI` (CUDA equirect rasterizer in `submodules/geer-rasterizer`). Reader `readOpenMVGInfo` prefers `pcd_seeded.ply`; `scripts/seed_sparse_directions.py DATASET_DIR`
  writes it (adds synthetic points in uncovered directions; handles zero seeds since 2026-10-16).
- Camera convention: OpenMVG `rotation` = world→cam, x right / y down / z forward. COLMAP images.txt (qvec,t) maps to the same. PF-GS-360 stores poses as `[t(3), rotvec(3)]` → `R_w2c = diag(1,-1,-1) · exp(rotvec)ᵀ`, centre = t (verified exact).
- Equirect rendering only depends on directions → a uniform world rescale (SfM gauge) leaves images unchanged, BUT the densification statistic (`--densify_grad_unscaled`, threshold 0.00006 tuned at the original bricks scale) is not scale invariant.
  Scenes ~10× larger under-densify; `scripts/rescale_openmvg_dataset.py` fixes the count (not quality). Per-scene threshold: bricks 0.00006, gallery_chair 0.0001, office 0.001.
- Test-time eval prints PSNR/SSIM/LPIPS for train (5 views) and test (all) at `--test_iterations` (python output is flushed). `render/` dir holds fixed-subset test renders per checkpoint.
- Loss extras added to `train.py` (all default OFF): `--lambda_vgg/--vgg_crop/--vgg_start_iter` (VGG16 perceptual loss, `utils/vgg_loss.py`), `--row_weight_bottom_frac/--row_weight_value` (fixed per-row loss weight),
  `--lambda_scale_reg` (L1 on Gaussian scales). Existing: `--lambda_smooth`, `--lambda_aniso`, `--lambda_opacity_sparsity`, `--smooth_prior_dir`.
- Standard bricks full-res recipe: see `scripts/run_bricks_schedule_fullres.sh` (env `LAMBDA_VGG SCALE ROW_FRAC ROW_W OPAC SCALE_REG`; `SCALE=2` = 60k iterations, densify/prune to 30k, reset/size-prune gates moved past the end).
- Baselines must be compared like-for-like: same PC, same resolution. A no-VGG control on this PC reproduced the old PC's baseline (25.11 vs 25.12 PSNR) → run-to-run noise is tiny (~0.02 dB).

## Working rules (learned from the user)
- The user usually runs training himself: give a ready CLI (`tee`/`nohup`, both GPUs when 2 jobs fit) and then check results when told it is done. Do not launch long jobs without being asked; short smoke tests (≤ ~100 iterations) on a free GPU are fine.
- Never build a prior or tune from TEST-view errors (leakage). Test heatmaps are for diagnosis only.
- Report outcomes faithfully (wrong hypotheses included), and report the gap to the goal above, not only the delta vs our own baseline.
- Before deleting outputs, look at them; the user once asked for "keep only top 5 per scene" explicitly. Do not delete otherwise.
- Check `tee`/redirect targets in pasted commands: a mistyped `tee scripts/x.sh` once overwrote a script.
- Chat output: no raw LaTeX; use Unicode symbols (Δ θ λ ≈ ≤ ≥ ×). Keep answers concise, direct answer first.

## Journaling & Progress Logging Protocol (from `/data2/3dgs_pipeline/agents/skills/journal/SKILL.md`)
- **Start of session**: read this file, the last 3-5 `JOURNAL.md` entries and the latest daily log; summarise in 2-4 sentences where things stand before working.
- **During the session**: the moment a training / SfM / data-generation command runs (not `tail`, `nvidia-smi`, `ps`), append it to `.agent-log/experiments/commands.md` (exact command, env/GPU, output path - reproduction only, **no metrics/timings**).
  Results and timings go to `.agent-log/experiments/metrics.md` under a matching heading. Timings come from the run's own log/file timestamps, never estimated.
- **End of session**: (1) append a `JOURNAL.md` entry (**Did / State / Next / Watch out**, header `## YYYY-MM-DD HH:MM - <agent> - <topic> [ID: <conversation-id>]`);
  (2) create/update `.agent-log/progress/MM-YYYY/DD-MM-YY.md` (headline sentences first, tables for numbers/durations, one `##` per thread, separate **Bugs hit and fixed**, **Open items** last);
  (3) update `.agent-log/progress_summary.md` when the standings change. Append-only, concrete (paths, flags, numbers), no secrets.
