"""Pooled PSNR and error split (bottom 1/6 of rows vs rest) for train and test renders of a run.
Usage: python scripts/strip_stats.py RUN_DIR ITERATION [RUN_DIR2 ITERATION2 ...]"""
import sys
import numpy as np
from PIL import Image
L = lambda p: np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
args = sys.argv[1:]
print(f"{'run / split':<62}{'PSNR':>7}{'bottom-1/6 MSE':>16}{'rest MSE':>10}{'bottom share':>14}")
for run, it in zip(args[::2], args[1::2]):
    for split in ("train", "test"):
        t = b = n = 0
        for j in range(50):
            e = ((L(f"{run}/{split}/ours_{it}/renders/{j:05d}.png") - L(f"{run}/{split}/ours_{it}/gt/{j:05d}.png")) ** 2).mean(-1)
            H = e.shape[0]; c = int(H * 5 / 6); t += e.sum(); b += e[c:].sum(); n += e.size
        print(f"{run.rstrip('/').split('/')[-1][:44] + ' ' + split:<62}{10 * np.log10(1 / (t / n)):7.2f}{b / (n / 6) * 1e3:13.2f}e-3{(t - b) / (n * 5 / 6) * 1e3:7.2f}e-3{b / t * 100:12.1f}%")
