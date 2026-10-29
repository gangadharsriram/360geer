"""Make a uniformly rescaled copy of an OpenMVG/equirect dataset dir.

Equirect rendering only uses directions from each camera centre, so scaling camera
centres and points by the same factor leaves every image unchanged - it is the free
gauge of SfM. But the densification statistic (backward.cu: dL_dmean2D mapped through
the per-Gaussian world-to-object transform) is not invariant to that scale, so
--densify_grad_threshold tuned on one scene scale under-fires on a scene ~10x larger
(RoMa/colmap_bricks_q8: Gaussians stuck at ~95k instead of ~1.1M).

Writes DST with scaled data_extrinsics.json + scaled pcd.ply, and symlinks to the
files the reader needs (views json, train/test split, images). SRC is never modified.
Points sitting exactly on a camera centre (OpenMVG scene.ply draws cameras as green
glyphs) are dropped - they are not structure.

Usage: python scripts/rescale_openmvg_dataset.py SRC DST SCALE
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.dataset_readers import fetchPly, storePly

src, dst, scale = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), float(sys.argv[3])
os.makedirs(dst, exist_ok=True)


def link(target, name):
    path = os.path.join(dst, name)
    if os.path.lexists(path):
        os.remove(path)
    os.symlink(target, path)


for name in ("data_views.json", "train.txt", "test.txt"):
    link(os.path.join(src, name), name)

images = os.path.join(src, "images")
if not os.path.exists(images):
    with open(os.path.join(src, "data_views.json")) as f:
        images = json.load(f)["root_path"]
link(images, "images")

with open(os.path.join(src, "data_extrinsics.json")) as f:
    ext = json.load(f)
centers = np.array([e["value"]["center"] for e in ext["extrinsics"]])
for e in ext["extrinsics"]:
    e["value"]["center"] = [c * scale for c in e["value"]["center"]]
with open(os.path.join(dst, "data_extrinsics.json"), "w") as f:
    json.dump(ext, f, indent=1)

pcd = fetchPly(os.path.join(src, "pcd.ply"))
xyz = np.asarray(pcd.points, dtype=np.float64)
rgb = np.asarray(pcd.colors, dtype=np.float64) * 255.0
on_camera = np.array([np.linalg.norm(centers - p, axis=1).min() < 1e-4 for p in xyz])
storePly(os.path.join(dst, "pcd.ply"), xyz[~on_camera] * scale, rgb[~on_camera])
print(f"Wrote {dst}: {len(centers)} cameras, {(~on_camera).sum()} points (dropped {on_camera.sum()} on camera centres), scaled by {scale}")
