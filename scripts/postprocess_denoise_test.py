"""Quick, no-retraining test of option 2 from the "smooth out the scattered/
dotted artifact" plan: apply a post-process denoise filter to already-rendered
test images and see whether it actually improves SSIM/PSNR/LPIPS against GT, or
just looks smoother while quietly hurting the metrics (a real risk - a naive
filter can't distinguish real fine detail like window grids/bricks/leaves from
the artifact scatter, and would blur both).

Uses an edge-preserving bilateral filter, not a plain Gaussian blur - the dots
are salt-and-pepper/speckle-like (individual scattered pixels/small clusters),
which is exactly what bilateral filtering targets (smooths flat/noisy regions
while preserving strong edges), unlike a uniform blur which would soften real
edges just as much as the artifact.

Usage: python scripts/postprocess_denoise_test.py MODEL_DIR
       [--iteration 30000] [--d 5] [--sigma_color 40] [--sigma_space 40]
       [--save_dir <path to also save denoised renders, optional>]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.image_utils import psnr as compute_psnr
from utils.loss_utils import ssim as compute_ssim
from lpipsPyTorch import lpips as compute_lpips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--d", type=int, default=5, help="Bilateral filter pixel neighborhood diameter.")
    ap.add_argument("--sigma_color", type=float, default=40.0)
    ap.add_argument("--sigma_space", type=float, default=40.0)
    ap.add_argument("--save_dir", default=None, help="Optional: save denoised renders here for visual inspection.")
    args = ap.parse_args()

    method = f"ours_{args.iteration}"
    render_dir = os.path.join(args.model_dir, "test", method, "renders")
    gt_dir = os.path.join(args.model_dir, "test", method, "gt")
    names = sorted(f for f in os.listdir(render_dir) if f.endswith(".png") and "mask" not in f)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    orig_ssim, orig_psnr, orig_lpips = [], [], []
    denoised_ssim, denoised_psnr, denoised_lpips = [], [], []

    with torch.no_grad():
        for n in names:
            render_bgr = cv2.imread(os.path.join(render_dir, n))
            denoised_bgr = cv2.bilateralFilter(render_bgr, args.d, args.sigma_color, args.sigma_space)
            if args.save_dir:
                cv2.imwrite(os.path.join(args.save_dir, n), denoised_bgr)

            render = tf.to_tensor(Image.open(os.path.join(render_dir, n))).unsqueeze(0)[:, :3].cuda()
            denoised = tf.to_tensor(Image.fromarray(cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0)[:, :3].cuda()
            gt = tf.to_tensor(Image.open(os.path.join(gt_dir, n))).unsqueeze(0)[:, :3].cuda()

            orig_ssim.append(compute_ssim(render, gt).item())
            orig_psnr.append(compute_psnr(render, gt).mean().item())
            orig_lpips.append(compute_lpips(render, gt, net_type="vgg").item())

            denoised_ssim.append(compute_ssim(denoised, gt).item())
            denoised_psnr.append(compute_psnr(denoised, gt).mean().item())
            denoised_lpips.append(compute_lpips(denoised, gt, net_type="vgg").item())

    def avg(x):
        return sum(x) / len(x)

    print(f"{'':12} {'SSIM':>8} {'PSNR':>8} {'LPIPS':>8}")
    print(f"{'original':12} {avg(orig_ssim):>8.4f} {avg(orig_psnr):>8.2f} {avg(orig_lpips):>8.4f}")
    print(f"{'denoised':12} {avg(denoised_ssim):>8.4f} {avg(denoised_psnr):>8.2f} {avg(denoised_lpips):>8.4f}")
    print(f"{'delta':12} {avg(denoised_ssim)-avg(orig_ssim):>+8.4f} {avg(denoised_psnr)-avg(orig_psnr):>+8.2f} "
          f"{avg(denoised_lpips)-avg(orig_lpips):>+8.4f}")


if __name__ == "__main__":
    main()
