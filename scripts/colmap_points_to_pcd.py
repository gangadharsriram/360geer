"""Write DATASET_DIR/pcd.ply from a COLMAP text-format points3D.txt.

The OpenMVG/equirect reader (scene/dataset_readers.py) needs pcd.ply (or
pcd_seeded.ply / colorized.ply) next to data_extrinsics.json. Datasets built from
a COLMAP reconstruction (e.g. RoMa/colmap_bricks_q8) only have openMVG/scene.ply,
which is just the camera centres - the real sparse points live in COLMAP's
points3D, in the same world frame as the openMVG extrinsics.

Usage: python scripts/colmap_points_to_pcd.py DATASET_DIR POINTS3D_TXT
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.dataset_readers import storePly

dataset_dir, points_txt = sys.argv[1], sys.argv[2]
rows = [l.split()[1:7] for l in open(points_txt) if not l.startswith("#") and l.strip()]
pts = np.array(rows, dtype=np.float64)
out = os.path.join(dataset_dir, "pcd.ply")
storePly(out, pts[:, :3], pts[:, 3:6])
print(f"Wrote {out}: {len(pts)} points")
