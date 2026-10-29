"""Render one training-camera pose at increasing lateral offsets (0/1/2/5/10cm),
using an already-trained model, to directly visualize how quickly the
scattered/needle-artifact illusion breaks down as the camera moves away from
where it was trained - the mechanism this whole investigation has been about
(redundant/stretched Gaussians that only line up correctly from angles near
the training cameras).

Offset is applied along the camera's own local right axis (a lateral dolly
move), keeping rotation and all other camera parameters fixed - the most
direct way to ask "how much real camera movement does it take before this
falls apart," which is exactly the baseline-sensitivity question this whole
session has been measuring.

Usage: python scripts/render_pose_offsets.py MODEL_DIR [--iteration 30000]
    [--cam_index N] [--offsets_cm 0,1,2,5,10]
"""
import argparse
import copy
import os
import sys
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--cam_index", type=int, default=None,
                     help="Training-camera index to use (default: middle of the list)")
    ap.add_argument("--offsets_cm", type=str, default="0,1,2,5,10")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    # cfg_args only ever saves ModelParams fields (confirmed by reading a
    # real one - train.py's prepare_output_and_logger writes just the
    # `dataset` object, not the full combined args), so reconstruct the
    # dataset from it directly. Pipeline flags (antialiasing, densify_grad_
    # signed/unscaled) only affect the backward/densification pass, never
    # the forward render - confirmed earlier this session that
    # omni_hvar's antialiasing arg is hardcoded true in forward.cu
    # regardless of pipe.antialiasing - so plain defaults are exact here,
    # not an approximation.
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    cfg_path = os.path.join(args.model_dir, "cfg_args")
    with open(cfg_path) as f:
        cfg = eval(f.read())
    cfg.model_path = args.model_dir
    dataset = model.extract(cfg)
    pipe = argparse.Namespace(debug=False, antialiasing=False,
                               densify_grad_signed=False, densify_grad_unscaled=False,
                               convert_SHs_python=False, compute_cov3D_python=False)
    dataset.fov_mod = None
    dataset.sample_step = None
    dataset.render_model = "EQUI"
    dataset.focal_scaling = 1.0
    dataset.distortion_scaling = 1.0
    dataset.mirror_shift = 0.0
    dataset.raymap = None

    gaussians = GaussianModel(dataset.sh_degree)
    with torch.no_grad():
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians from iteration {scene.loaded_iter}")

        train_cams = scene.getTrainCameras()
        cam_index = args.cam_index if args.cam_index is not None else len(train_cams) // 2
        base_cam = train_cams[cam_index]
        print(f"Using training camera index {cam_index} ({base_cam.image_name})")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # World-to-camera rotation, recovered from the stored (transposed)
        # world_view_transform - M[:3,:3] is the world2cam rotation R.
        M = base_cam.world_view_transform.T
        R = M[:3, :3]
        C = base_cam.camera_center.clone()

        # Camera's own local right axis (local +x), expressed in world space:
        # for a rotation matrix, the world-space direction of camera-local
        # axis i is row i of R (since R maps world vectors into camera space).
        right_world = R[0, :] / R[0, :].norm()

        out_dir = args.out_dir or os.path.join(args.model_dir, "pose_offset_renders")
        os.makedirs(out_dir, exist_ok=True)

        offsets_cm = [float(x) for x in args.offsets_cm.split(",")]
        for off_cm in offsets_cm:
            off_m = off_cm / 100.0
            cam = copy.copy(base_cam)  # shallow copy - reuse everything except pose

            if off_cm == 0.0:
                new_center = C
                new_wvt = base_cam.world_view_transform
            else:
                new_center = C + right_world * off_m
                t_new = -(R @ new_center)
                M_new = torch.eye(4, device=M.device, dtype=M.dtype)
                M_new[:3, :3] = R
                M_new[:3, 3] = t_new
                new_wvt = M_new.T

                # Sanity check: recovering the camera center from the new
                # world_view_transform the same way Camera.__init__ does
                # should match what we intended, to catch any convention
                # mistake before trusting the render.
                recovered = new_wvt.inverse()[3, :3]
                assert torch.allclose(recovered, new_center, atol=1e-4), \
                    f"camera_center convention mismatch: {recovered} vs {new_center}"

            cam.world_view_transform = new_wvt
            cam.camera_center = new_center

            pkg = render(cam, gaussians, pipe, background)
            img = pkg["render"].clamp(0.0, 1.0)
            out_path = os.path.join(out_dir, f"offset_{off_cm:05.1f}cm.png")
            torchvision.utils.save_image(img, out_path)
            print(f"wrote {out_path}")

    print(f"done - {len(offsets_cm)} renders in {out_dir}")


if __name__ == "__main__":
    main()
