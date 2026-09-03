"""Generate a nadir-region mask for equirectangular 360 captures.

Handheld/pole-mounted 360 cameras always show the operator or rig mount near
the bottom of the panorama, at essentially the same on-screen position in
every frame (it's rigidly attached to the camera, so it never gets real
multi-view parallax). Training on that region pushes the optimizer to grow
degenerate, opaque, needle-shaped Gaussians right next to the camera trying
to explain unexplainable content. Masking it out of the loss (via the
existing --mask_path mechanism in train.py) removes that pressure entirely.

Usage: python scripts/make_nadir_mask.py OUT.png [--fraction 0.20] [--width 1024]
"""
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("out_path")
parser.add_argument("--fraction", type=float, default=0.20,
                     help="Bottom fraction of image rows to mask out (0-1). "
                          "Inspect a source frame's bottom rows first and adjust "
                          "so it covers the operator/rig without cutting real content.")
parser.add_argument("--width", type=int, default=1024, help="Mask width (auto-resized to match render resolution at train time).")
args = parser.parse_args()

height = args.width // 2  # standard equirect 2:1 aspect ratio
mask = np.full((height, args.width), 255, dtype=np.uint8)
cutoff = int(height * (1.0 - args.fraction))
mask[cutoff:, :] = 0

Image.fromarray(mask, mode="L").save(args.out_path)
print(f"Wrote {args.out_path}: {args.width}x{height}, bottom {args.fraction*100:.0f}% masked (rows {cutoff}-{height})")
