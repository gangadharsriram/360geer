"""Compare camera pose sets vs a reference (original bricks OpenMVG) after a global similarity fit.
Usage: python scripts/compare_pose_sets.py   (sets hard-coded below)"""
import json, numpy as np
from scipy.spatial.transform import Rotation as Rot
D = '/4tb_hdd/sriram/3dgs_pipeline/literature_code/'
def openmvg(d):
    nm = {x['key']: x['value']['ptr_wrapper']['data']['filename'] for x in json.load(open(d + '/data_views.json'))['views']}
    return {nm[x['key']]: (np.array(x['value']['rotation']), np.array(x['value']['center'])) for x in json.load(open(d + '/data_extrinsics.json'))['extrinsics']}
def colmap(f):
    out = {}
    for l in open(f):
        t = l.split()
        if len(t) == 10 and t[-1].endswith(('.png', '.jpg')):
            q = np.array(t[1:5], float); R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            out[t[-1]] = (R, -R.T @ np.array(t[5:8], float))
    return out
S = {
 'Original (Data_openmvg)': openmvg('/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks'),
 'Dataset COLMAP (output_dir/colmap)': colmap('/4tb_hdd/sriram/dataset/Omnidirectional/Ricoh360/Data_colmap/bricks/output_dir/colmap/images.txt'),
 'PanoVGGT': colmap(D + 'PanoVGGT/panovggt_session_20261008_182706_499216/colmap_format/images.txt'),
 'RoMa v1 (cell8, F-filter)': openmvg(D + '360geer/RoMa/bricks_roma_openmvg'),
 'RoMa v2 (cell1, angular)': openmvg(D + '360geer/RoMa/bricks_roma_v_q1_a'),
}
ref = S['Original (Data_openmvg)']
print('| Pose set | cams | matched | scale vs orig | centre err (mean / max, camera spacings) | rot err, centre-fit (mean / max deg) | rot err, rotation-fit (mean / max deg) |\n|---|---|---|---|---|---|---|')
for name, A in S.items():
    n = sorted(set(A) & set(ref))
    X, Y = np.array([A[k][1] for k in n]), np.array([ref[k][1] for k in n]); mx, my = X.mean(0), Y.mean(0)
    U, Sv, Vt = np.linalg.svd((Y - my).T @ (X - mx)); Dm = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); Rg = U @ Dm @ Vt
    s = (Sv * np.diag(Dm)).sum() / ((X - mx) ** 2).sum()
    cd = np.linalg.norm((s * Rg @ (X - mx).T).T + my - Y, axis=1) / 0.043   # orig camera spacing ~0.043
    ang = lambda R_: np.degrees([np.arccos(np.clip((np.trace(ref[k][0] @ R_ @ A[k][0].T) - 1) / 2, -1, 1)) for k in n])
    a1 = ang(Rg)
    Rm = Rot.from_matrix(np.array([ref[k][0].T @ A[k][0] for k in n])).mean().as_matrix(); a2 = ang(Rm)
    print(f'| {name} | {len(A)} | {len(n)} | {s:.4f} | {cd.mean():.2f} / {cd.max():.2f} | {a1.mean():.2f} / {a1.max():.2f} | {a2.mean():.2f} / {a2.max():.2f} |')
