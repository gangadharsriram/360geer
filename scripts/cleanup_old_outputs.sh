#!/usr/bin/env bash
# Cleanup of old/superseded experiment outputs, agreed with the user 2026-09-07.
#
# Two locations:
# 1. The repo's own output/ dir (/4tb_hdd, which was at 100% full) - entirely
#    pre-investigation runs (Aug 3-12), all superseded by later results
#    already recorded in memory and in test_output/. Keeps only the two
#    small summary CSVs.
# 2. /4tb_hdd2/.../test_output/ - keeps only the final/best-designated
#    results per scene, the exp91/92 lambda_aniso pivot (biggest single
#    finding of the investigation), and the tiny calibration probe. Deletes
#    every intermediate single-variable sweep step, whose numeric result is
#    already permanently recorded in
#    ~/.claude/projects/-4tb-hdd-sriram/memory/360geer_densification_investigation.md
set -e

echo "== Step 1: repo output/ dir (178G, on the full disk) =="
OLD_OUT="/4tb_hdd/sriram/3dgs_pipeline/literature_code/360geer/output"
du -sh "$OLD_OUT" 2>/dev/null || true
find "$OLD_OUT" -mindepth 1 -maxdepth 1 \
    ! -name "sweep_results.csv" \
    ! -name "point_count_comparison.csv" \
    -exec rm -rf {} +
echo "-- after --"
ls -la "$OLD_OUT"
du -sh "$OLD_OUT"
df -h /4tb_hdd

echo
echo "== Step 2: test_output/ (244G) - keep only final/pivotal results =="
TEST_OUT="/4tb_hdd2/sriram/360geer/test_output"
du -sh "$TEST_OUT" 2>/dev/null || true
find "$TEST_OUT" -mindepth 1 -maxdepth 1 -type d \
    ! -name "exp108_bricks_opacity_sparsity_0012" \
    ! -name "run_bricks_best_fullres" \
    ! -name "exp112_gallery_chair_opacity_sparsity_012" \
    ! -name "run_gallery_chair_best_fullres" \
    ! -name "run_office_best_fullres" \
    ! -name "exp113_office_baseline_stock" \
    ! -name "exp114_office_full_recipe" \
    ! -name "exp122_turtlebot_baseline_stock" \
    ! -name "exp123_lab_baseline_stock" \
    ! -name "exp124_lab_full_recipe" \
    ! -name "exp125_base_baseline_stock" \
    ! -name "exp126_base_full_recipe" \
    ! -name "exp91_bricks_lambda_aniso_off" \
    ! -name "exp92_gallery_chair_lambda_aniso_off" \
    ! -name "exp94_bricks_real3d_calibration" \
    -exec rm -rf {} +
echo "-- after --"
ls "$TEST_OUT"
du -sh "$TEST_OUT"
df -h /4tb_hdd2

echo
echo "=== cleanup done ==="
