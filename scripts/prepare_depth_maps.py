"""Convert this dataset's raw depth_npy maps into the 16-bit-PNG inverse-depth
format train.py's depth-supervision loss expects (utils/camera_utils.py's
loadCam, cv2.imread(path,-1).astype(float32)/2**16), with a depth_params.json
scale/offset so the values land in the SfM reconstruction's own coordinate
scale - not blindly assumed to already match it.

Two things had to be checked against the actual data rather than assumed:
1. depthmap/depth_npy/*.npy values are in [0.01, 1.0], matching the sibling
   depth_vis_*_100m folder naming - these are depth/100m, not raw metric depth
   and not inverse depth. Metric depth = npy_value * 100.
2. This SfM reconstruction's own coordinate scale is NOT guaranteed to be real
   metres (cameras_extent for this dataset is ~0.45, an OpenMVG/SfM gauge that
   may or may not be metric) - so metric depth can't be used directly as the
   supervision target without checking. This script calibrates a single global
   scale factor by reprojecting every real SfM point (pcd.ply / pcd_seeded.ply)
   into every camera, sampling the depth map at that pixel, and comparing
   against the point's own known distance from that camera (which IS in the
   SfM reconstruction's true coordinate scale, no ambiguity there) - the median
   ratio across all of these correspondences is the calibration factor.

Usage: python scripts/prepare_depth_maps.py DATASET_DIR
       [--depth_npy_dir depthmap/depth_npy] [--out_subdir depth_png16]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.dataset_readers import fetchPly

EQUI_PI = np.pi
METRIC_SCALE = 100.0  # npy values are depth_meters / 100, per the _100m folder naming


def world_to_pixel(dir_world, R_wc, W, H):
    """Same convention as seed_sparse_directions.py / the CUDA rasterizer's auxiliary.h."""
    dir_cam = R_wc @ dir_world
    lon = np.arctan2(dir_cam[0], dir_cam[2])
    lat = np.arcsin(np.clip(dir_cam[1] / np.linalg.norm(dir_cam), -1.0, 1.0))
    px = (lon / EQUI_PI + 1.0) * 0.5 * W
    py = (lat * (2.0 / EQUI_PI) + 1.0) * 0.5 * H
    return px, py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--depth_npy_dir", default="depthmap/depth_npy")
    ap.add_argument("--out_subdir", default="depth_png16")
    ap.add_argument("--calib_samples_per_cam", type=int, default=2000,
                     help="Cap on how many SfM points to reproject per camera for calibration.")
    args = ap.parse_args()
    path = args.dataset_dir
    npy_dir = os.path.join(path, args.depth_npy_dir)
    out_dir = os.path.join(path, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(path, "data_extrinsics.json")) as f:
        extrinsics = json.load(f)["extrinsics"]
    with open(os.path.join(path, "data_views.json")) as f:
        cam_dict = {v["key"]: v["value"]["ptr_wrapper"]["data"]["filename"] for v in json.load(f)["views"]}

    seeded_path = os.path.join(path, "pcd_seeded.ply")
    pcd_path = seeded_path if os.path.exists(seeded_path) else os.path.join(path, "pcd.ply")
    pcd = fetchPly(pcd_path)
    points = np.asarray(pcd.points, dtype=np.float64)
    print(f"Calibrating against {len(points)} SfM points from {os.path.basename(pcd_path)}, "
          f"{len(extrinsics)} cameras.")

    # --- Step 1: calibrate a single global (true SfM distance) / (claimed metric depth) ratio ---
    ratios = []
    rng = np.random.default_rng(0)
    for e in extrinsics:
        cam_name = cam_dict[e["key"]]
        stem = os.path.splitext(cam_name)[0]
        npy_path = os.path.join(npy_dir, f"{stem}.npy")
        if not os.path.exists(npy_path):
            continue
        depth_npy = np.load(npy_path)
        H, W = depth_npy.shape[:2]

        center = np.array(e["value"]["center"])
        R_wc = np.array(e["value"]["rotation"])

        rel = points - center
        dist = np.linalg.norm(rel, axis=1)
        valid = dist > 1e-6
        idx = np.where(valid)[0]
        if len(idx) > args.calib_samples_per_cam:
            idx = rng.choice(idx, size=args.calib_samples_per_cam, replace=False)

        for i in idx:
            direction = rel[i] / dist[i]
            px, py = world_to_pixel(direction, R_wc, W, H)
            ix, iy = int(round(px)), int(round(py))
            if not (0 <= ix < W and 0 <= iy < H):
                continue
            claimed_metric_depth = float(depth_npy[iy, ix]) * METRIC_SCALE
            if claimed_metric_depth <= 1e-3:
                continue
            ratios.append(dist[i] / claimed_metric_depth)

    if len(ratios) < 100:
        raise RuntimeError(f"Only found {len(ratios)} usable calibration correspondences - "
                            f"too few to trust a global scale factor.")
    ratios = np.array(ratios)
    calib_ratio = float(np.median(ratios))
    print(f"Calibration: {len(ratios)} correspondences, median ratio (SfM units per claimed metre) "
          f"= {calib_ratio:.6f}  (IQR {np.percentile(ratios,25):.6f}-{np.percentile(ratios,75):.6f})")

    # --- Step 2: convert every depth map to inverse depth in SfM-coordinate scale ---
    npy_files = sorted(f for f in os.listdir(npy_dir) if f.endswith(".npy"))
    inv_depths = {}
    for fname in npy_files:
        depth_npy = np.load(os.path.join(npy_dir, fname))
        depth_metric = depth_npy.astype(np.float64) * METRIC_SCALE
        depth_sfm = depth_metric * calib_ratio
        inv_depths[fname] = 1.0 / np.clip(depth_sfm, 1e-6, None)

    all_inv = np.concatenate([d.ravel() for d in inv_depths.values()])
    p999 = float(np.percentile(all_inv, 99.9))
    encode_scale = 60000.0 / max(p999, 1e-6)
    depth_params_scale = 65536.0 / encode_scale
    print(f"Inverse-depth range (SfM scale): p50={np.median(all_inv):.4f} p99.9={p999:.4f} "
          f"max={all_inv.max():.4f}  -> encode_scale={encode_scale:.4f}")

    # --- Step 3: write 16-bit PNGs + depth_params.json ---
    depth_params = {}
    for fname in npy_files:
        stem = os.path.splitext(fname)[0]
        inv = inv_depths[fname]
        png16 = np.clip(np.round(inv * encode_scale), 0, 65535).astype(np.uint16)
        Image.fromarray(png16).save(os.path.join(out_dir, f"{stem}.png"))
        depth_params[stem] = {"scale": depth_params_scale, "offset": 0.0, "med_scale": depth_params_scale}

    with open(os.path.join(path, "depth_params.json"), "w") as f:
        json.dump(depth_params, f, indent=2)

    print(f"Wrote {len(npy_files)} depth maps to {out_dir}/ and {path}/depth_params.json")
    print(f"Train with: --depths {args.out_subdir}")


if __name__ == "__main__":
    main()
