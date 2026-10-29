"""Build DST = original bricks poses/images/split + the PF-GS-360 dense point cloud mapped into the original frame.
The raw->original similarity is read off pfgs_extrinsics_raw.json vs pfgs_extrinsics_aligned.json (camera centres, exact fit).
Usage: python scripts/pfgs_points_to_orig_frame.py PFGS_DIR ORIG_DIR DST [MAX_POINTS=0 (all)] [points file=points3D_nosky.ply]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.dataset_readers import fetchPly, storePly
pf, orig, dst = [os.path.abspath(a) for a in sys.argv[1:4]]
maxn = int(sys.argv[4]) if len(sys.argv) > 4 else 0
ply = sys.argv[5] if len(sys.argv) > 5 else 'points3D_nosky.ply'
os.makedirs(dst, exist_ok=True)
ld = lambda f: {e['key']: np.array(e['value']['center']) for e in json.load(open(f))['extrinsics']}
raw, al = ld(pf + '/pfgs_extrinsics_raw.json'), ld(pf + '/pfgs_extrinsics_aligned.json')
k = sorted(set(raw) & set(al)); X, Y = np.array([raw[i] for i in k]), np.array([al[i] for i in k]); mx, my = X.mean(0), Y.mean(0)
U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx)); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); Rg = U @ D @ Vt
s = (S * np.diag(D)).sum() / ((X - mx) ** 2).sum()
res = np.linalg.norm((s * Rg @ (X - mx).T).T + my - Y, axis=1).max()
print(f'{len(k)} cameras, scale {s:.4f}, max fit residual {res:.2e}')
assert res < 1e-3, 'raw->aligned is not a clean similarity'
p = fetchPly(os.path.join(pf, ply)); xyz = np.asarray(p.points, np.float64); rgb = np.asarray(p.colors, np.float64) * 255
if maxn and len(xyz) > maxn:
    idx = np.random.RandomState(0).choice(len(xyz), maxn, replace=False); xyz, rgb = xyz[idx], rgb[idx]
storePly(dst + '/pcd.ply', (s * Rg @ (xyz - mx).T).T + my, rgb)
print(f'wrote {len(xyz)} points -> {dst}/pcd.ply')
for f in ('data_views.json', 'data_extrinsics.json', 'train.txt', 'test.txt', 'images'):
    l = os.path.join(dst, f)
    if os.path.lexists(l): os.remove(l)
    os.symlink(os.path.join(orig, f), l)
