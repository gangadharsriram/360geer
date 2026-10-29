"""Oracle test-time pose search: freeze a trained model and, for every test view, coordinate-descent over the 6 camera DoF
(rotation about the camera axes, translation along the camera axes) to maximise PSNR against that test image - mimicking the
test-time camera optimisation PF-GS-360 does at evaluation. Reports PSNR/SSIM/LPIPS before/after and how far the poses moved.
NOT a normal evaluation (the test image is used to fit its own pose): it measures how much of the PSNR gap is pose alignment.
Usage: python scripts/test_time_pose_search.py MODEL_DIR --iteration N [--max_views K] [--spacing 0.043]"""
import argparse, copy, json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams
from utils.loss_utils import ssim
from lpipsPyTorch import lpips

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("--iteration", type=int, required=True)
ap.add_argument("--max_views", type=int, default=0); ap.add_argument("--spacing", type=float, default=0.043)
ap.add_argument("--rot_steps_deg", default="0.4,0.2,0.1,0.05"); ap.add_argument("--trans_steps", default="0.004,0.002,0.001,0.0005")
ap.add_argument("--passes", type=int, default=2)
args = ap.parse_args()

parser = argparse.ArgumentParser(); model = ModelParams(parser, sentinel=True)
from argparse import Namespace
cfg = eval(open(os.path.join(args.model_dir, "cfg_args")).read()); cfg.model_path = args.model_dir
dataset = model.extract(cfg)
pipe = argparse.Namespace(debug=False, antialiasing=False, densify_grad_signed=False, densify_grad_unscaled=False,
                          convert_SHs_python=False, compute_cov3D_python=False)
dataset.fov_mod = None; dataset.sample_step = None; dataset.render_model = "EQUI"
dataset.focal_scaling = 1.0; dataset.distortion_scaling = 1.0; dataset.mirror_shift = 0.0; dataset.raymap = None
gaussians = GaussianModel(dataset.sh_degree)
bg = torch.zeros(3, device="cuda")

def rodrigues(w):
    th = torch.linalg.norm(w)
    if th < 1e-12: return torch.eye(3, device=w.device)
    k = w / th; K = torch.zeros(3, 3, device=w.device); K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=w.device) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)

def posed(base, R0, C0, x):  # x = [rot(3) rad, trans(3) in camera axes]
    R = rodrigues(x[:3]) @ R0; C = C0 + R0.T @ x[3:]
    M = torch.eye(4, device="cuda"); M[:3, :3] = R; M[:3, 3] = -(R @ C)
    cam = copy.copy(base); cam.world_view_transform = M.T; cam.camera_center = C
    return cam

with torch.no_grad():
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cams = scene.getTestCameras(); cams = cams[:args.max_views] if args.max_views else cams
    print(f"{gaussians.get_xyz.shape[0]} Gaussians, {len(cams)} test views", flush=True)
    rs, ts = [float(v) for v in args.rot_steps_deg.split(",")], [float(v) for v in args.trans_steps.split(",")]
    rows = []; t0 = time.time()
    for n, base in enumerate(cams):
        gt = base.original_image[0:3].cuda(); M = base.world_view_transform.T; R0, C0 = M[:3, :3].clone(), base.camera_center.clone()
        def score(x):
            img = render(posed(base, R0, C0, x), gaussians, pipe, bg)["render"].clamp(0, 1)
            return (10 * torch.log10(1 / ((img - gt) ** 2).mean())).item(), img
        x = torch.zeros(6, device="cuda"); best, img0 = score(x); p0 = best
        for rd, tstep in zip(rs, ts):
            for _ in range(args.passes):
                for d in range(6):
                    for sgn in (1, -1):
                        y = x.clone(); y[d] += sgn * (math.radians(rd) if d < 3 else tstep)
                        s, _ = score(y)
                        if s > best + 1e-6: best, x = s, y; break
        _, img1 = score(x)
        rows.append(dict(view=base.image_name, psnr0=p0, psnr1=best,
                         ssim0=ssim(img0[None], gt[None]).item(), ssim1=ssim(img1[None], gt[None]).item(),
                         lpips0=lpips(img0[None], gt[None], net_type='vgg').item(), lpips1=lpips(img1[None], gt[None], net_type='vgg').item(),
                         rot_deg=math.degrees(x[:3].norm().item()), trans=x[3:].norm().item()))
        print(f"{n+1}/{len(cams)} {base.image_name}: {p0:.2f} -> {best:.2f}  rot {rows[-1]['rot_deg']:.3f} deg  trans {rows[-1]['trans']/args.spacing:.3f} spacings  ({time.time()-t0:.0f}s)", flush=True)
m = lambda k: float(np.mean([r[k] for r in rows]))
print(f"MEAN  PSNR {m('psnr0'):.3f} -> {m('psnr1'):.3f} | SSIM {m('ssim0'):.4f} -> {m('ssim1'):.4f} | LPIPS {m('lpips0'):.4f} -> {m('lpips1'):.4f}")
print(f"pose moved: rot mean {m('rot_deg'):.3f} deg, trans mean {m('trans')/args.spacing:.3f} camera spacings (PF-GS-360 test poses: RPE_r 0.053 deg, ATE 0.0027 = 0.06 spacings)")
json.dump(rows, open(os.path.join(args.model_dir, f"test_time_pose_search_{args.iteration}.json"), "w"), indent=1)
