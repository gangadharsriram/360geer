# Experiment commands (reproduction only - no metrics, results or timings; results are in metrics.md under the same heading)

Backfilled 2026-10-27 from the conversation `f726fa90-a7b3-4f2b-874f-392c26af976c` and the scripts/logs on disk. Dates = the day the run's own log / output files were written.
All commands: `cd /data2/3dgs_pipeline/literature_code/360geer && conda activate 360geer` (since 2026-10-14; before that the repo lived under `/4tb_hdd/sriram/3dgs_pipeline/literature_code/360geer`, outputs under `/4tb_hdd2/sriram/360geer/test_output/`).
Common recipe flags inside `scripts/run_bricks_*fullres.sh`: `--dataset EQUIRECT --render_model EQUI --resolution 1 --eval --disable_viewer --opacity_reset_interval <ITERS> --size_prune_from_iter <ITERS> --densify_grad_unscaled --densify_grad_threshold 0.00006 --lambda_smooth 0.01 --lambda_aniso 0.0 --lambda_opacity_sparsity 0.012`.

## S7 RoMa-based poses / datasets (old PC, 2026-10-12 → 13)
- RoMa matching + OpenMVG SfM (user's script, old PC): `demo/run_roma_openmvg.sh <imgs> <dir with train.txt/test.txt> <out_dir> [window=5] [cell_px=8] [sfm_mode=STELLAR]` → dataset `RoMa/bricks_roma_openmvg` (v1: cell 8, fundamental-matrix filter). Other RoMa datasets used: `RoMa/colmap_bricks_q8/openmvg`, `RoMa/openmvg_all_q8`.
- Variant SfM (env `colmap-cuda`, `RoMa/demo/roma_to_openmvg_v2.py IMG NPZ OUT SPLIT CELL_PX MODE CERT GEOM`, 2026-10-13 00:17-00:25), all STELLAR, certainty 0.5, reusing `bricks_roma_openmvg/roma_npz`:
  - `q1_f`: cell 1, geom `f` → `RoMa/bricks_roma_v_q1_f`;  `q1_a`: cell 1, geom `a` (angular essential matrix; output file named `.e.bin`) → `RoMa/bricks_roma_v_q1_a`;  `q8_a`: cell 8, geom `a` → `RoMa/bricks_roma_v_q8_a`.
- Train on RoMa v1 poses/points (GPU 1, old PC): `scripts/run_bricks_roma_openmvg_fullres.sh` (rescales ×0.111 into `RoMa/bricks_roma_openmvg_scaled`, seeds, trains) → `/4tb_hdd2/sriram/360geer/test_output/run_bricks_roma_openmvg_fullres`. Log date 12/10 21:15-22:02.
- Train on RoMa v2 (`q1_a`) poses/points (old PC): `WORLD_SCALE=0.0343 scripts/run_bricks_roma_openmvg_fullres.sh <RoMa/bricks_roma_v_q1_a> /4tb_hdd2/sriram/360geer/test_output/run_bricks_roma_v_q1_a_fullres 1`. Log date 13/10 00:35-02:00 (metrics step OOM'd on a shared GPU; re-run: `CUDA_VISIBLE_DEVICES=2 python metrics.py -m <out> --block_mask --iters 30000`).
- Original poses + RoMa points (old PC): `scripts/run_bricks_origposes_romapoints_fullres.sh` (builds `RoMa/bricks_origposes_romapoints` with `scripts/roma_points_in_orig_frame.py`) → `/4tb_hdd2/sriram/360geer/test_output/run_bricks_origposes_romapoints_fullres`, GPU 1, log 12/10 22:46 → 13/10.
- RoMa v2 poses + ORIGINAL sparse points (new PC, GPU 0 default): `scripts/run_bricks_romapose_origpoints_fullres.sh` (builds `RoMa/bricks_romapose_origpoints` with `scripts/poses_with_orig_points.py`) → `/data2/3dgs_pipeline/output/run_bricks_romapose_origpoints_fullres`. Written 2026-10-14; the first attempt hit the nvrtc `prod` error, rerun after the `_prod3` fix; finished 2026-10-14 11:11.

## S8 Other scene
- OmniBlender barbershop (GPU 0 default, 2026-10-15): `scripts/run_barbershop_best_fullres.sh 2>&1 | tee ...` (THRESH=0.0001 default, `--resolution 1`, 30k) → `/data2/3dgs_pipeline/output/run_barbershop_best_fullres`. Dataset `/data2/Dataset/Omnidirectional/OmniBlender/barbershop` (25 train / 25 test views, 2000×1000 RGBA).

## S1 VGG perceptual loss on bricks (original sparse init, original poses; output root `/data2/3dgs_pipeline/output/`; 2026-10-15 → 16)
- `LAMBDA_VGG=0.05 scripts/run_bricks_vgg_fullres.sh` → `run_bricks_vgg0.05_fullres` (GPU 0).
- `scripts/run_bricks_vgg_thresh00015_fullres.sh` (LAMBDA_VGG 0.05, THRESH 0.00015) → `run_bricks_vgg0.05_thresh0.00015_fullres`.
- `scripts/run_bricks_vgg_late_fullres.sh` (LAMBDA_VGG 0.05, VGG_START 15000) → `run_bricks_vgg0.05_from15000_fullres`.
- `LAMBDA_VGG=0.01 scripts/run_bricks_vgg_fullres.sh "" "" 0` → `run_bricks_vgg0.01_fullres`;  `LAMBDA_VGG=0.02 scripts/run_bricks_vgg_fullres.sh "" "" 1` → `run_bricks_vgg0.02_fullres` (ran in parallel).

## S2 Schedule / control (2026-10-16, `scripts/run_bricks_control_and_long.sh`, two GPU queues; SCALE=2 → 60k iterations, densify/prune until 30k)
- `nohup scripts/run_bricks_control_and_long.sh > /data2/3dgs_pipeline/output/control_and_long.log 2>&1 &` runs, via `scripts/run_bricks_schedule_fullres.sh "" "" <GPU>`:
  GPU 0: `LAMBDA_VGG=0 SCALE=1` → `run_bricks_vgg0_x1_fullres`, then `LAMBDA_VGG=0 SCALE=2` → `run_bricks_vgg0_x2_fullres`.
  GPU 1: `LAMBDA_VGG=0.01 SCALE=2` → `run_bricks_vgg0.01_x2_fullres`, then `LAMBDA_VGG=0.02 SCALE=2` → `run_bricks_vgg0.02_x2_fullres`.

## S3 Dense initialisation from PF-GS-360 (2026-10-16)
- `nohup scripts/run_bricks_pfgs_init_fullres.sh > /data2/3dgs_pipeline/output/pfgs_init.log 2>&1 &` (GPU A=0, B=1). Step 1 builds the dataset `RoMa/bricks_pfgs_dense_init`:
  `python scripts/pfgs_points_to_orig_frame.py /data2/Resuts/360/pfgs360_outputs/Ricoh360/bricks/pfgs360 /data2/Dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks RoMa/bricks_pfgs_dense_init 0` (points3D_nosky.ply, 2,272,699 pts, Sim3 scale 1.085, original poses/images/split symlinked), then `scripts/seed_sparse_directions.py` (0 seeds). Runs:
  GPU 0: `LAMBDA_VGG=0.01 SCALE=2` → `run_bricks_pfgsinit_vgg0.01_x2_fullres`;  GPU 1: `LAMBDA_VGG=0 SCALE=1` → `run_bricks_pfgsinit_vgg0_x1_fullres`.

## S5 Fixed per-row loss weight (2026-10-18, dense-init dataset, VGG 0.01, SCALE=2)
- `nohup scripts/run_bricks_rowweight_sweep.sh > /data2/3dgs_pipeline/output/rowweight_sweep.log 2>&1 &`: GPU 0 `ROW_FRAC=0.12 ROW_W=0.5` → `run_bricks_pfgsinit_vgg0.01_x2_row0.12w0.5_fullres`; GPU 1 `ROW_FRAC=0.12 ROW_W=0.2` → `..._row0.12w0.2_fullres`.

## S6 Stronger-regularisation sweep (2026-10-18 22:14 → 10-19 02:50, same base + `ROW_FRAC=0.12 ROW_W=0.2` unless stated)
- `nohup scripts/run_bricks_regularization_sweep.sh > /data2/3dgs_pipeline/output/regularization_sweep.log 2>&1 &`
  GPU 0: `rowmask` (`ROW_FRAC=0.12 ROW_W=0`) then `opac0.03` (`OPAC=0.03`);  GPU 1: `row17w0.1` (`ROW_FRAC=0.17 ROW_W=0.1`) then `scale0.02` (`SCALE_REG=0.02`). Output `..._pfgsinit_vgg0.01_x2_<tag>_fullres`.
- Best run so far = `opac0.03`; to rerun it alone: `LAMBDA_VGG=0.01 SCALE=2 ROW_FRAC=0.12 ROW_W=0.2 OPAC=0.03 scripts/run_bricks_schedule_fullres.sh /data2/3dgs_pipeline/literature_code/360geer/RoMa/bricks_pfgs_dense_init /data2/3dgs_pipeline/output/<name> 0`.
