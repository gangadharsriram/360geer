"""Per-view heatmap report: "our render vs GT" next to "PF-GS-360 render vs GT", for every test or train view of bricks.
One page per view (3x3): col 0 = ground truth + two difference maps (ours minus PF-GS-360), col 1 = our render + its error / structure-loss
heatmaps, col 2 = PF-GS-360 render + its heatmaps. Pages are also written as JPEGs next to a single PDF (explanation page, summary page, then all views).
Usage: python scripts/heatmap_report.py OUR_SPLIT_DIR(.../test/ours_N | .../train/ours_N) OUT_DIR [--label test|train] [--run_name NAME]
PF-GS-360 reference: test -> eval_renders/NNNNNN-{gt,pred}.png ; train -> train/view_NNNN_rgb.png (left half GT, right half render)."""
import argparse, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

ap = argparse.ArgumentParser(); ap.add_argument("our_dir"); ap.add_argument("out_dir")
ap.add_argument("--label", default="test"); ap.add_argument("--run_name", default="our run")
ap.add_argument("--pf", default="/data2/Resuts/360/pfgs360_outputs/Ricoh360/bricks/pfgs360")
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True); O = a.our_dir.rstrip("/") + "/"
L = lambda p: np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
n = len(os.listdir(O + "renders"))

def pf_pair(i):
    if a.label == "train":
        v = L(f"{a.pf}/train/view_{i:04d}_rgb.png"); return v[:, :v.shape[1] // 2], v[:, v.shape[1] // 2:]
    return L(f"{a.pf}/eval_renders/{i:06d}-gt.png"), L(f"{a.pf}/eval_renders/{i:06d}-pred.png")

def ssim_map(x, y, C1=0.01 ** 2, C2=0.03 ** 2, s=1.5):   # standard SSIM: 11x11 Gaussian window (sigma 1.5), mean over R,G,B
    g = lambda z: np.stack([gaussian_filter(z[..., c], s, truncate=3.5) for c in range(3)], -1)
    mx, my = g(x), g(y); sxx, syy, sxy = g(x * x) - mx ** 2, g(y * y) - my ** 2, g(x * y) - mx * my
    return (((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))).mean(-1)

D = lambda m: uniform_filter(m, 2)[::2, ::2]          # 2x2 average down to 960x480 for display
def rgb2(x): return np.stack([D(x[..., c]) for c in range(3)], -1)
pf = [pf_pair(i) for i in range(50)]
gts = [L(O + f"gt/{j:05d}.png") for j in range(n)]
match = [int(np.argmin([np.abs(pf[i][0] - g).mean() for i in range(50)])) for g in gts]       # PF-GS view index for each of our views
assert all(np.abs(pf[match[j]][0] - gts[j]).max() < 1e-3 for j in range(n)), "PF-GS-360 ground truth does not match ours"

pages, rows = [], []
fig_to_img = lambda fig: Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
for j in range(n):
    gt, r = gts[j], L(O + f"renders/{j:05d}.png"); p = pf[match[j]][1]
    e_o, e_p = ((r - gt) ** 2).mean(-1), ((p - gt) ** 2).mean(-1); s_o, s_p = 1 - ssim_map(r, gt), 1 - ssim_map(p, gt)
    d = dict(j=j, psnr_o=10 * np.log10(1 / e_o.mean()), psnr_p=10 * np.log10(1 / e_p.mean()), ssim_o=1 - s_o.mean(), ssim_p=1 - s_p.mean(),
             e_o=D(np.sqrt(e_o)), e_p=D(np.sqrt(e_p)), s_o=D(s_o), s_p=D(s_p)); rows.append(d)
    fig, ax = plt.subplots(3, 3, figsize=(18, 10.3)); fig.subplots_adjust(left=0.01, right=0.93, top=0.93, bottom=0.01, wspace=0.03, hspace=0.12)
    ax[0, 0].imshow(rgb2(gt)); ax[0, 0].set_title("ground truth", fontsize=11)
    ax[0, 1].imshow(rgb2(r)); ax[0, 1].set_title(f"{a.run_name}  (PSNR {d['psnr_o']:.2f} dB, SSIM {d['ssim_o']:.3f})", fontsize=11)
    ax[0, 2].imshow(rgb2(p)); ax[0, 2].set_title(f"PF-GS-360 render  (PSNR {d['psnr_p']:.2f} dB, SSIM {d['ssim_p']:.3f})", fontsize=11)
    im1 = ax[1, 1].imshow(d["e_o"], cmap="inferno", vmin=0, vmax=0.2); ax[1, 1].set_title("OURS vs GT: error heatmap", fontsize=11)
    ax[1, 2].imshow(d["e_p"], cmap="inferno", vmin=0, vmax=0.2); ax[1, 2].set_title("PF-GS-360 vs GT: error heatmap", fontsize=11)
    im2 = ax[2, 1].imshow(d["s_o"], cmap="magma", vmin=0, vmax=0.8); ax[2, 1].set_title("OURS vs GT: structure-loss heatmap", fontsize=11)
    ax[2, 2].imshow(d["s_p"], cmap="magma", vmin=0, vmax=0.8); ax[2, 2].set_title("PF-GS-360 vs GT: structure-loss heatmap", fontsize=11)
    im3 = ax[1, 0].imshow(d["e_o"] - d["e_p"], cmap="coolwarm", vmin=-0.08, vmax=0.08); ax[1, 0].set_title("error: ours minus PF-GS-360\n(red = we are worse, blue = we are better)", fontsize=10)
    ax[2, 0].imshow(d["s_o"] - d["s_p"], cmap="coolwarm", vmin=-0.3, vmax=0.3); ax[2, 0].set_title("structure loss: ours minus PF-GS-360\n(red = we are worse, blue = we are better)", fontsize=10)
    for x in ax.ravel(): x.axis("off")
    cb = fig.colorbar(im1, ax=ax[1, :], fraction=0.012, pad=0.01); cb.set_label("RMS colour error (0-1)", fontsize=8); cb.ax.tick_params(labelsize=7)
    cb = fig.colorbar(im2, ax=ax[2, :], fraction=0.012, pad=0.01); cb.set_label("1 - local SSIM", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"{a.label.upper()} view {j:02d}   |   {a.run_name} vs PF-GS-360, each compared with the ground truth", fontsize=13, weight="bold", y=0.985)
    fig.canvas.draw(); img = fig_to_img(fig); plt.close(fig)
    img.save(f"{a.out_dir}/{a.label}_view{j:02d}_ours{d['psnr_o']:.2f}_pfgs{d['psnr_p']:.2f}.jpg", quality=85); pages.append(img)
    print(j, f"ours {d['psnr_o']:.2f} / pfgs {d['psnr_p']:.2f}", flush=True)

def text_page(title, paras, size=12):
    fig = plt.figure(figsize=(18, 10.3)); fig.text(0.04, 0.95, title, fontsize=20, weight="bold", va="top"); y = 0.88
    for p in paras:
        for line in (textwrap.wrap(p, 150) or [""]): fig.text(0.04, y, line, fontsize=size, va="top"); y -= 0.034
        y -= 0.016
    fig.canvas.draw(); img = fig_to_img(fig); plt.close(fig); return img

m = lambda k: float(np.mean([d[k] for d in rows]))
mean_maps = {k: np.mean([d[k] for d in rows], 0) for k in ("e_o", "e_p", "s_o", "s_p")}
fig, ax = plt.subplots(2, 2, figsize=(18, 10.3)); fig.subplots_adjust(left=0.02, right=0.92, top=0.9, bottom=0.02, wspace=0.05, hspace=0.15)
for x, k, t, cm, vm in ((ax[0, 0], "e_o", f"{a.run_name}: mean error heatmap", "inferno", 0.12), (ax[0, 1], "e_p", "PF-GS-360: mean error heatmap", "inferno", 0.12),
                        (ax[1, 0], "s_o", f"{a.run_name}: mean structure-loss heatmap", "magma", 0.5), (ax[1, 1], "s_p", "PF-GS-360: mean structure-loss heatmap", "magma", 0.5)):
    im = x.imshow(mean_maps[k], cmap=cm, vmin=0, vmax=vm); x.set_title(t, fontsize=12); x.axis("off")
    if k in ("e_p", "s_p"): fig.colorbar(im, ax=ax[0 if k == "e_p" else 1, :], fraction=0.012, pad=0.01)
fig.suptitle(f"{a.label.upper()} views - averaged over all {n} views.  Mean PSNR: ours {m('psnr_o'):.2f} dB, PF-GS-360 {m('psnr_p'):.2f} dB   |   mean SSIM: ours {m('ssim_o'):.4f}, PF-GS-360 {m('ssim_p'):.4f}",
             fontsize=13, weight="bold"); fig.canvas.draw(); summary = fig_to_img(fig); plt.close(fig)

expl = text_page(f"How to read these heatmaps ({a.label} views)", [
    "Every page compares one view of the bricks scene twice: OUR render against the ground-truth photo (GT), and the PF-GS-360 render against the same GT. All maps use one fixed colour scale for every view, so pages can be compared.",
    "ERROR HEATMAP (inferno: black → red → yellow). For each pixel: how different the rendered colour is from the GT colour (root-mean-square over R, G, B, on a 0-1 scale; bright = large, the scale is clipped at 0.2). "
    "PSNR is computed from the average of these squared errors, so every bright area costs PSNR. Typical bright places: sharp edges (a small shift or blur makes a thin bright line), foliage, repeated textures such as the brick tower, and the strip at the very bottom (camera rig / person).",
    "STRUCTURE-LOSS HEATMAP (magma: black → purple → white). For each pixel: 1 − local SSIM, with SSIM computed in a small Gaussian window (11×11 pixels). SSIM ignores small uniform colour shifts and instead checks whether the local contrast, edges and texture pattern match the GT. "
    "0 (black) = the local pattern matches; 1 (white) = unrelated structure. Bright areas mean blur, wrong or missing texture, misplaced edges or noise. The SSIM metric of a view is 1 minus the average of this map.",
    "DIFFERENCE MAPS (left column, blue-white-red). Our error minus the PF-GS-360 error for the same pixel: red = our render is worse there, blue = ours is better, white = about equal. Use these to see where the gap to PF-GS-360 comes from.",
    f"Colour scales: error 0-0.2, structure loss 0-0.8, differences ±0.08 (error) and ±0.3 (structure loss). Maps are shown at half resolution (2×2 average). Our model: {a.run_name} (iteration 60000). PF-GS-360 test renders come from its eval_renders; "
    "its train renders from the right half of its train/view_NNNN_rgb.png images. SSIM here is my own implementation of the standard SSIM; it can differ slightly from the metrics.py number.",
    f"Next pages: one overview of the mean maps over all {n} {a.label} views, then {n} per-view pages."], size=12.5)
out = [expl, summary] + pages
out[0].save(f"{a.out_dir}/heatmap_report_{a.label}.pdf", save_all=True, append_images=out[1:], resolution=100)
print("mean PSNR ours %.3f pfgs %.3f | mean SSIM ours %.4f pfgs %.4f" % (m("psnr_o"), m("psnr_p"), m("ssim_o"), m("ssim_p")))
print("wrote", f"{a.out_dir}/heatmap_report_{a.label}.pdf", "and", n, "JPEG pages")
