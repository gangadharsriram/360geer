"""Build DST = original bricks poses/images/split + RoMa sparse points mapped into the original frame
(Umeyama similarity on camera centres matched by filename). Tests poses-vs-points as the cause of low PSNR.
Usage: python scripts/roma_points_in_orig_frame.py ORIG_DIR ROMA_DIR DST"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.dataset_readers import fetchPly, storePly
orig, roma, dst = [os.path.abspath(a) for a in sys.argv[1:4]]
os.makedirs(dst, exist_ok=True)
def cams(d):
    names = {x['key']: x['value']['ptr_wrapper']['data']['filename'] for x in json.load(open(d + '/data_views.json'))['views']}
    return {names[x['key']]: np.array(x['value']['center']) for x in json.load(open(d + '/data_extrinsics.json'))['extrinsics']}
ca, cb = cams(roma), cams(orig)
n = sorted(set(ca) & set(cb))
X, Y = np.array([ca[k] for k in n]), np.array([cb[k] for k in n])
mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
U, S, Vt = np.linalg.svd(Yc.T @ Xc); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt
s = (S * np.diag(D)).sum() / (Xc ** 2).sum()
print(f"{len(n)} matched cameras, scale {s:.4f}, mean residual {np.linalg.norm(s*(R@Xc.T).T-Yc,axis=1).mean():.4f}")
pcd = fetchPly(roma + '/pcd.ply')
p = np.asarray(pcd.points, np.float64); rgb = np.asarray(pcd.colors, np.float64) * 255
cen = np.array(list(ca.values()))
keep = np.array([np.linalg.norm(cen - q, axis=1).min() >= 1e-4 for q in p])
q = (s * (R @ (p[keep] - mx).T).T) + my
storePly(dst + '/pcd.ply', q, rgb[keep])
print(f"wrote {keep.sum()} points (dropped {(~keep).sum()} camera glyphs) -> {dst}/pcd.ply")
for f in ('data_views.json', 'data_extrinsics.json', 'train.txt', 'test.txt'):
    l = os.path.join(dst, f)
    if os.path.lexists(l): os.remove(l)
    os.symlink(os.path.join(orig, f), l)
img = os.path.join(orig, 'images')
if not os.path.exists(img): img = json.load(open(orig + '/data_views.json'))['root_path']
l = dst + '/images'
if os.path.lexists(l): os.remove(l)
os.symlink(img, l)
