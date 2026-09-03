"""Summarize results.json + final point count across all exp* output dirs, plus
train-set SSIM/PSNR (computed directly from train/ours_N/{renders,gt}, since
metrics.py only ever evaluates the test split) and the train-test SSIM gap - a
direct measure of the redundant-Gaussian/train-overfitting artifact this project
keeps running into. Run after scripts/run_all_experiments.sh (or any subset of
the exp*.sh scripts) finishes. Usage:
    python scripts/summarize_experiments.py [output_dir_glob] [--csv path] [--skip-train]
"""
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from plyfile import PlyData

from utils.image_utils import psnr as compute_psnr
from utils.loss_utils import ssim as compute_ssim

args = sys.argv[1:]
csv_path = None
if "--csv" in args:
    i = args.index("--csv")
    csv_path = args[i + 1]
    del args[i:i + 2]
skip_train = "--skip-train" in args
if skip_train:
    args.remove("--skip-train")
# Experiments now live in two places (older ones under ./output/, newer ones
# under /4tb_hdd2/sriram/360geer/test_output/) - search both by default, plus
# the non-"exp*"-prefixed baseline configs used for cross-dataset comparison.
patterns = [args[0]] if args else [
    "./output/exp*", "./output/*_best_config",
    "/4tb_hdd2/sriram/360geer/test_output/exp*",
    "/4tb_hdd2/sriram/360geer/test_output/*_best_config",
]


def exp_number(name):
    """Leading exp<N> number, e.g. 'exp42_seeded_...' -> 42. None for dirs that
    don't follow that convention (e.g. office_space_best_config) - plain
    alphabetical sort ('exp1', 'exp10', 'exp2', ...) puts exp37 right before
    exp38, exp39, exp3, exp40... which reads as if the list stopped at exp37."""
    m = re.match(r"exp(\d+)", name)
    return int(m.group(1)) if m else None


seen = set()
dirs = []
for pattern in patterns:
    for d in sorted(glob.glob(pattern)):
        if os.path.isdir(d) and os.path.basename(d) not in seen:
            seen.add(os.path.basename(d))
            dirs.append(d)
dirs.sort(key=lambda d: (exp_number(os.path.basename(d)) is None,
                          exp_number(os.path.basename(d)) or 0,
                          os.path.basename(d)))

fieldnames = ["experiment", "exp_num", "dataset", "SSIM", "PSNR", "LPIPS", "train_SSIM", "train_PSNR", "SSIM_gap", "num_gaussians"]
rows = []


def dataset_name(d):
    """Dataset this experiment trained on, read from its cfg_args' source_path."""
    cfg_path = os.path.join(d, "cfg_args")
    if not os.path.exists(cfg_path):
        return ""
    m = re.search(r"source_path='([^']*)'", open(cfg_path).read())
    return os.path.basename(m.group(1).rstrip("/")) if m else ""


def best_method_key(results):
    """Pick the ours_N entry with the highest N - results.json only ever holds
    whichever iteration(s) metrics.py was last run against, not always 30000."""
    keys = [k for k in results if k.startswith("ours_")]
    if not keys:
        return None
    return max(keys, key=lambda k: int(k.split("_")[1]))


def eval_split(renders_dir, gt_dir):
    if not (os.path.exists(renders_dir) and os.path.exists(gt_dir)):
        return None, None
    names = sorted(f for f in os.listdir(renders_dir) if f.endswith(".png") and "mask" not in f)
    if not names:
        return None, None
    ssims, psnrs = [], []
    with torch.no_grad():
        for n in names:
            r = tf.to_tensor(Image.open(os.path.join(renders_dir, n))).unsqueeze(0)[:, :3].cuda()
            g = tf.to_tensor(Image.open(os.path.join(gt_dir, n))).unsqueeze(0)[:, :3].cuda()
            ssims.append(compute_ssim(r, g).item())
            psnrs.append(compute_psnr(r, g).mean().item())
    return sum(ssims) / len(ssims), sum(psnrs) / len(psnrs)


header = f"{'exp#':>4} {'experiment':<40} {'dataset':<14} {'SSIM':>8} {'PSNR':>8} {'LPIPS':>8} {'train_SSIM':>10} {'gap':>7} {'#gaussians':>12}"
print(header)
for d in dirs:
    name = os.path.basename(d)
    num = exp_number(name)
    num_str = str(num) if num is not None else "-"
    ds = dataset_name(d)
    results_path = os.path.join(d, "results.json")
    ply_path = os.path.join(d, "point_cloud", "iteration_30000", "point_cloud.ply")
    n_gaussians = PlyData.read(ply_path)["vertex"].count if os.path.exists(ply_path) else ""

    row = {"experiment": name, "exp_num": num if num is not None else "", "dataset": ds, "SSIM": "", "PSNR": "", "LPIPS": "",
           "train_SSIM": "", "train_PSNR": "", "SSIM_gap": "", "num_gaussians": n_gaussians}

    if not os.path.exists(results_path):
        print(f"{num_str:>4} {name:<40} {ds:<14} (no results.json yet)")
        rows.append(row)
        continue

    results = json.load(open(results_path))
    method = best_method_key(results)
    m = results.get(method, {}) if method else {}
    ssim_v, psnr_v, lpips_v = m.get("SSIM"), m.get("PSNR"), m.get("LPIPS")

    if ssim_v is None:
        print(f"{num_str:>4} {name:<40} {ds:<14} (incomplete results.json)")
        rows.append(row)
        continue

    row.update({"SSIM": ssim_v, "PSNR": psnr_v, "LPIPS": lpips_v})

    train_ssim = train_psnr = gap = ""
    if not skip_train and method is not None:
        iters = method.split("_")[1]
        train_ssim, train_psnr = eval_split(
            os.path.join(d, "train", method, "renders"),
            os.path.join(d, "train", method, "gt"),
        )
        if train_ssim is not None:
            gap = train_ssim - ssim_v
            row.update({"train_SSIM": train_ssim, "train_PSNR": train_psnr, "SSIM_gap": gap})

    gap_str = f"{gap:>7.4f}" if gap != "" else f"{'-':>7}"
    train_ssim_str = f"{train_ssim:>10.4f}" if train_ssim not in ("", None) else f"{'-':>10}"
    print(f"{num_str:>4} {name:<40} {ds:<14} {ssim_v:>8.4f} {psnr_v:>8.2f} {lpips_v:>8.4f} {train_ssim_str} {gap_str} {n_gaussians:>12}")
    rows.append(row)

if csv_path:
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")
