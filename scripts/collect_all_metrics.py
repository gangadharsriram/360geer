"""Collect SSIM/PSNR/LPIPS for both train and test splits across every
experiment output dir, plus Gaussian count, into one CSV.

Test metrics are read straight from each dir's existing results.json
(metrics.py already computed them with --block_mask --iters 30000).
Train metrics don't exist anywhere yet (metrics.py hardcodes test_dir), so
they're computed here with the same readImages/ssim/psnr/lpips helpers
metrics.py uses, same --block_mask/--iters 30000 convention, just pointed at
train/ instead of test/.
"""
import glob
import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from plyfile import PlyData

from metrics import readImages
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips

OUTPUT_ROOT = "/4tb_hdd2/sriram/360geer/test_output"
ITERS = 30000


def eval_split(model_dir, split, iters=ITERS):
    split_dir = Path(model_dir) / split / f"ours_{iters}"
    renders_dir = split_dir / "renders"
    gt_dir = split_dir / "gt"
    if not renders_dir.is_dir():
        return None
    renders_list = sorted(glob.glob(str(renders_dir / "*.png")))
    renders_list = [f for f in renders_list if "mask" not in f]
    if not renders_list:
        return None
    N = 20
    ssims, psnrs, lpipss = [], [], []
    for i in range(math.ceil(len(renders_list) / N)):
        renders, gts, _ = readImages(renders_dir, gt_dir, renders_list, i * N, min(len(renders_list), (i + 1) * N))
        for r, g in zip(renders, gts):
            ssims.append(ssim(r, g, mask=None))
            psnrs.append(psnr(r, g, mask=None))
            lpipss.append(lpips(r, g, net_type='vgg'))
    return {
        "SSIM": torch.tensor(ssims).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "LPIPS": torch.tensor(lpipss).mean().item(),
    }


def gaussian_count(model_dir, iters=ITERS):
    ply_path = Path(model_dir) / "point_cloud" / f"iteration_{iters}" / "point_cloud.ply"
    if not ply_path.is_file():
        return None
    return PlyData.read(str(ply_path))["vertex"].count


def main():
    rows = []
    for d in sorted(os.listdir(OUTPUT_ROOT)):
        model_dir = os.path.join(OUTPUT_ROOT, d)
        if not os.path.isdir(model_dir):
            continue
        results_path = os.path.join(model_dir, "results.json")
        if not os.path.isfile(results_path):
            print(f"skip {d}: no results.json (incomplete run)")
            continue
        with open(results_path) as f:
            test_metrics = json.load(f).get(f"ours_{ITERS}")
        if test_metrics is None:
            print(f"skip {d}: results.json has no ours_{ITERS} entry")
            continue

        cfg_path = os.path.join(model_dir, "cfg_args")
        scene = ""
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = eval(f.read())
            scene = os.path.basename(cfg.source_path)

        print(f"{d}: computing train metrics...")
        with torch.no_grad():
            train_metrics = eval_split(model_dir, "train")
        if train_metrics is None:
            print(f"  no train renders for {d}")

        row = {
            "folder": d,
            "scene": scene,
            "num_gaussians": gaussian_count(model_dir),
            "train_SSIM": train_metrics["SSIM"] if train_metrics else None,
            "train_PSNR": train_metrics["PSNR"] if train_metrics else None,
            "train_LPIPS": train_metrics["LPIPS"] if train_metrics else None,
            "test_SSIM": test_metrics["SSIM"],
            "test_PSNR": test_metrics["PSNR"],
            "test_LPIPS": test_metrics["LPIPS"],
        }
        rows.append(row)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "all_metrics.json")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
