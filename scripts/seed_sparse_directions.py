"""Augment a sparse OpenMVG/equirect scene's initial point cloud with synthetic seed
points in angular directions the real SfM cloud barely covers (sky, featureless/
grazing-angle walls COLMAP/OpenMVG can't triangulate).

Densification during training can still *reach* those directions, but only by
repeatedly cloning/splitting outward from whatever's nearby - with near-zero initial
signal and a narrow camera baseline, that process has no correct depth to converge
to, so it scatters redundant Gaussians across many plausible-but-wrong depths
instead (the actual cause of the streak/shard artifacts - see plan docs from this
investigation). Seeding a coarse point at a *reasonable* depth guess in those
directions gives training something sane to converge from instead of nothing.

Writes pcd_seeded.ply next to the real pcd.ply. scene/dataset_readers.py's
readOpenMVGInfo prefers pcd_seeded.ply automatically if present - no other config
needed to use it, just point --source_path at the same dataset dir as always.

Usage: python scripts/seed_sparse_directions.py DATASET_DIR
       [--grid_step_deg 10] [--coverage_threshold_deg 5] [--min_depth_samples 3]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.dataset_readers import fetchPly, storePly

EQUI_PI = np.pi


def world_to_pixel(dir_world, R_wc, W, H):
    """Project a world-space unit direction into an EQUI camera's pixel space,
    using the exact convention baked into the CUDA rasterizer (auxiliary.h)."""
    dir_cam = R_wc @ dir_world
    lon = np.arctan2(dir_cam[0], dir_cam[2])
    lat = np.arcsin(np.clip(dir_cam[1] / np.linalg.norm(dir_cam), -1.0, 1.0))
    px = (lon / EQUI_PI + 1.0) * 0.5 * W
    py = (lat * (2.0 / EQUI_PI) + 1.0) * 0.5 * H
    return px, py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--grid_step_deg", type=float, default=10.0)
    ap.add_argument("--coverage_threshold_deg", type=float, default=5.0,
                     help="Directions with no real SfM point within this angle of the "
                          "camera-centroid ray are considered uncovered and get seeded.")
    ap.add_argument("--min_depth_samples", type=int, default=3,
                     help="Widen the search cone until at least this many real points "
                          "are found, to estimate a depth guess for the seed point.")
    args = ap.parse_args()
    path = args.dataset_dir

    with open(os.path.join(path, "data_extrinsics.json")) as f:
        extrinsics = json.load(f)["extrinsics"]
    with open(os.path.join(path, "data_views.json")) as f:
        cam_dict = {v["key"]: v["value"]["ptr_wrapper"]["data"]["filename"] for v in json.load(f)["views"]}

    centers = np.array([e["value"]["center"] for e in extrinsics])
    centroid = centers.mean(axis=0)

    # Pick one camera (the first) purely for sampling a plausible color per direction.
    ref = extrinsics[0]
    R_wc = np.array(ref["value"]["rotation"])  # world->cam, OpenMVG convention directly
    ref_image_path = os.path.join(path, "images", cam_dict[ref["key"]])
    ref_image = np.array(Image.open(ref_image_path).convert("RGB"))
    ref_h, ref_w = ref_image.shape[:2]

    pcd = fetchPly(os.path.join(path, "pcd.ply"))
    xyz = np.asarray(pcd.points, dtype=np.float64)
    rgb = np.asarray(pcd.colors, dtype=np.float64) * 255.0  # fetchPly normalizes to [0,1]

    rel = xyz - centroid
    dist = np.linalg.norm(rel, axis=1)
    rel_dir = rel / np.maximum(dist[:, None], 1e-9)

    # Coarse full-sphere grid of candidate directions (world-space, arbitrary pole
    # choice - just needs to be a consistent full-sphere parameterization for
    # enumerating candidates, unrelated to any camera's own convention).
    lons = np.arange(-180.0, 180.0, args.grid_step_deg)
    lats = np.arange(-80.0, 80.0 + 1e-6, args.grid_step_deg)  # skip the exact poles

    seed_xyz, seed_rgb = [], []
    n_checked, n_seeded = 0, 0
    for lat_deg in lats:
        lat = np.radians(lat_deg)
        cl = np.cos(lat)
        for lon_deg in lons:
            n_checked += 1
            lon = np.radians(lon_deg)
            direction = np.array([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)])

            cosang = rel_dir @ direction
            ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            if ang.min() <= args.coverage_threshold_deg:
                continue  # already covered by a real SfM point

            # Widen the search cone until we have enough points for a depth estimate.
            depth = None
            for cone_deg in (20, 40, 80, 180):
                near = ang <= cone_deg
                if near.sum() >= args.min_depth_samples:
                    depth = np.median(dist[near])
                    break
            if depth is None:
                depth = np.median(dist)  # global fallback (should be rare)

            seed_pos = centroid + depth * direction

            px, py = world_to_pixel(direction, R_wc, ref_w, ref_h)
            ix = int(np.clip(round(px), 0, ref_w - 1))
            iy = int(np.clip(round(py), 0, ref_h - 1))
            color = ref_image[iy, ix].astype(np.float64)

            seed_xyz.append(seed_pos)
            seed_rgb.append(color)
            n_seeded += 1

    seed_xyz = np.array(seed_xyz)
    seed_rgb = np.array(seed_rgb)

    out_xyz = np.concatenate([xyz, seed_xyz], axis=0)
    out_rgb = np.concatenate([rgb, seed_rgb], axis=0)

    out_path = os.path.join(path, "pcd_seeded.ply")
    storePly(out_path, out_xyz, out_rgb)
    print(f"Checked {n_checked} candidate directions ({args.grid_step_deg} deg grid), "
          f"seeded {n_seeded} uncovered ones (threshold {args.coverage_threshold_deg} deg).")
    print(f"Wrote {out_path}: {len(xyz)} real + {n_seeded} synthetic = {len(out_xyz)} points total.")


if __name__ == "__main__":
    main()
