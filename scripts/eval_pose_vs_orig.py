"""Pose error of an OpenMVG dataset dir vs the original bricks poses after a global similarity fit.
Usage: python scripts/eval_pose_vs_orig.py DIR [ORIG_DIR]"""
import json, sys
import numpy as np
O = '/data2/Dataset/Omnidirectional/Ricoh360/Data_openmvg/bricks'
def L(d):
    nm = {x['key']: x['value']['ptr_wrapper']['data']['filename'] for x in json.load(open(d + '/data_views.json'))['views']}
    return {nm[x['key']]: (np.array(x['value']['rotation']), np.array(x['value']['center'])) for x in json.load(open(d + '/data_extrinsics.json'))['extrinsics']}
A, B = L(sys.argv[1]), L(sys.argv[2] if len(sys.argv) > 2 else O)
n = sorted(set(A) & set(B)); print(f'{len(A)} cams registered, {len(n)} matched')
X, Y = np.array([A[k][1] for k in n]), np.array([B[k][1] for k in n])
mx, my = X.mean(0), Y.mean(0)
U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx)); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); Rg = U @ D @ Vt
s = (S * np.diag(D)).sum() / ((X - mx) ** 2).sum()
ang = np.degrees([np.arccos(np.clip((np.trace(B[k][0] @ Rg @ A[k][0].T) - 1) / 2, -1, 1)) for k in n])
cd = np.array([np.linalg.norm(s * Rg @ (A[k][1] - mx) + my - B[k][1]) for k in n])
print(f'scale {s:.4f} | rot err deg mean {ang.mean():.2f} med {np.median(ang):.2f} max {ang.max():.2f} | centre err/spacing(0.043) mean {cd.mean()/0.043:.2f} max {cd.max()/0.043:.2f}')
