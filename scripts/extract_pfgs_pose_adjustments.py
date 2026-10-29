"""Pull the per-camera pose parameters (train + eval/test cameras) out of PF-GS-360 nerfstudio checkpoints.
The ckpts were pickled with numpy 2.x, so numpy._core is aliased to numpy.core. Usage: python ... OUT_DIR  (writes <scene>.npz)"""
import glob, os, sys
import numpy, numpy.core, numpy.core.multiarray, numpy.core.numeric
sys.modules['numpy._core'] = numpy.core
for n in ('multiarray', 'numeric', '_multiarray_umath', 'umath'): sys.modules['numpy._core.' + n] = getattr(numpy.core, n)
import torch
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for ck in sorted(glob.glob('/data2/Resuts/360/pfgs360_outputs/Ricoh360/*/pfgs360/nerfstudio_models/*.ckpt')):
    scene = ck.split('/')[-4]; c = torch.load(ck, map_location='cpu'); p = c['pipeline']
    numpy.savez(f'{out}/{scene}.npz', step=c['step'], train=p['_model.camera_optimizer.pose_adjustment'].numpy(),
                eval=p['_model.eval_camera_optimizer.pose_adjustment'].numpy())
    print(scene, 'step', c['step'], p['_model.camera_optimizer.pose_adjustment'].shape, p['_model.eval_camera_optimizer.pose_adjustment'].shape, flush=True)
    del c, p
