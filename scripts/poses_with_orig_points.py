"""Build DST = poses from POSES_DIR (OpenMVG dir) mapped into the ORIGINAL bricks frame + the original sparse points.
Similarity (Umeyama on camera centres, matched by filename) so world scale/frame = original; images/split from ORIG.
Usage: python scripts/poses_with_orig_points.py POSES_DIR ORIG_DIR DST"""
import json, os, sys
import numpy as np
poses, orig, dst = [os.path.abspath(a) for a in sys.argv[1:4]]
os.makedirs(dst, exist_ok=True)
names = lambda d: {x['key']: x['value']['ptr_wrapper']['data']['filename'] for x in json.load(open(d + '/data_views.json'))['views']}
ext = json.load(open(poses + '/data_extrinsics.json'))
na, nb = names(poses), names(orig)
ca = {na[e['key']]: np.array(e['value']['center']) for e in ext['extrinsics']}
cb = {nb[e['key']]: np.array(e['value']['center']) for e in json.load(open(orig + '/data_extrinsics.json'))['extrinsics']}
n = sorted(set(ca) & set(cb))
X, Y = np.array([ca[k] for k in n]), np.array([cb[k] for k in n]); mx, my = X.mean(0), Y.mean(0)
U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx)); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); Rg = U @ D @ Vt
s = (S * np.diag(D)).sum() / ((X - mx) ** 2).sum()
print(f'{len(n)} matched cameras, scale {s:.4f}')
for e in ext['extrinsics']:   # x' = s Rg (x - mx) + my  =>  R' = R Rg^T
    v = e['value']
    v['rotation'] = (np.array(v['rotation']) @ Rg.T).tolist()
    v['center'] = (s * Rg @ (np.array(v['center']) - mx) + my).tolist()
json.dump(ext, open(dst + '/data_extrinsics.json', 'w'), indent=1)
def link(target, name):
    p = os.path.join(dst, name)
    if os.path.lexists(p): os.remove(p)
    os.symlink(target, p)
for f in ('data_views.json', 'train.txt', 'test.txt'): link(os.path.join(poses, f), f)
link(os.path.join(orig, 'pcd.ply'), 'pcd.ply')       # original sparse points, already in the original frame
link(os.path.join(orig, 'images'), 'images')
