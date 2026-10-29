"""Error heatmaps: where do the test renders differ from ground truth (and from the PF-GS-360 reference renders)?
Per view PNG: GT | our render | |error| heatmap | (1-SSIM) heatmap | [reference |error| | our-minus-reference error].
Summary PDF: mean error / SSIM-loss maps over all test views, tile shares of the total squared error, error by latitude / edge-vs-flat /
sky-like regions / hardest pixels, and the same regions ours-vs-reference (where we lose to them). Metrics are over the full image
(metrics.py --block_mask means no mask). Usage:
  python scripts/error_heatmaps.py OUR_TEST_DIR(.../test/ours_N) OUT_DIR [--ref /path/to/pfgs360/eval_renders|none] [--worst 8]"""
import argparse, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter, sobel

ap = argparse.ArgumentParser(); ap.add_argument("our_dir"); ap.add_argument("out_dir")
ap.add_argument("--ref", default="/data2/Resuts/360/pfgs360_outputs/Ricoh360/bricks/pfgs360/eval_renders"); ap.add_argument("--worst", type=int, default=8); ap.add_argument("--label", default="test")
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True); O = a.our_dir.rstrip("/") + "/"
L = lambda p: np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
n = len(glob.glob(O + "renders/*.png")); gts = [L(O + f"gt/{i:05d}.png") for i in range(n)]
ref = None if a.ref.lower() == "none" else a.ref.rstrip("/") + "/"
TRAIN_REF = bool(ref) and os.path.exists(ref + "view_0000_rgb.png")   # pfgs360/train/view_NNNN_rgb.png = [ground truth | render] side by side
def ref_pair(i):  # (gt, pred) of reference view i
    if TRAIN_REF:
        v = L(ref + f"view_{i:04d}_rgb.png"); return v[:, :v.shape[1] // 2], v[:, v.shape[1] // 2:]
    return L(ref + f"{i:06d}-gt.png"), L(ref + f"{i:06d}-pred.png")
if ref:  # match reference views to ours by GT pixels (their order differs)
    rp_all = [ref_pair(i) for i in range(50)]; rg = [x[0] for x in rp_all]
    rmap = {j: int(np.argmin([np.abs(rg[i] - gts[j]).mean() for i in range(50)])) for j in range(n)}
    assert all(np.abs(rg[rmap[j]] - gts[j]).max() < 1e-3 for j in range(n)), "reference GT does not match ours"

def ssim_map(x, y, C1=0.01 ** 2, C2=0.03 ** 2, s=1.5):
    g = lambda z: np.stack([gaussian_filter(z[..., c], s, truncate=3.5) for c in range(3)], -1)
    mx, my = g(x), g(y); sxx, syy, sxy = g(x * x) - mx ** 2, g(y * y) - my ** 2, g(x * y) - mx * my
    return (((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))).mean(-1)

def regions(gt):  # approximate content classes from the GT image alone
    gray = gt.mean(-1); gx, gy = sobel(gaussian_filter(gray, 1.0), 1), sobel(gaussian_filter(gray, 1.0), 0); grad = np.hypot(gx, gy)
    H = gt.shape[0]; top = np.arange(H)[:, None] < H * 0.55
    sky = top & (gt[..., 2] > gt[..., 0] + 0.04) & (gt[..., 2] >= gt[..., 1]) & (gt.mean(-1) > 0.45) | (top & (gt.min(-1) > 0.8))
    return grad, sky

H, W = gts[0].shape[:2]; bands = 8
acc = {k: np.zeros((H, W)) for k in ("mse", "ssimloss", "mse_ref")}; rows = []
G = lambda e: uniform_filter(e, 2)[::2, ::2]
for j in range(n):
    gt = gts[j]; r = L(O + f"renders/{j:05d}.png"); e = ((r - gt) ** 2).mean(-1); sm = 1 - ssim_map(r, gt)
    psnr = 10 * np.log10(1 / e.mean()); d = dict(j=j, gt=gt, r=r, e=e, sm=sm, psnr=psnr, ssim=1 - sm.mean())
    if ref:
        rp = rp_all[rmap[j]][1]; er = ((rp - gt) ** 2).mean(-1); d.update(rp=rp, er=er, psnr_ref=10 * np.log10(1 / er.mean())); acc["mse_ref"] += er
    acc["mse"] += e; acc["ssimloss"] += sm; rows.append(d)
for k in acc: acc[k] /= n
print(f"mean PSNR over views {np.mean([d['psnr'] for d in rows]):.3f}" + (f" | reference {np.mean([d['psnr_ref'] for d in rows]):.3f}" if ref else ""), flush=True)

# ---- per-view heatmaps
def view_fig(d, path):
    nr = 3 if ref else 2; fig, axs = plt.subplots(nr, 2, figsize=(14, 3.7 * nr + 0.4)); ax = axs.ravel()
    ax[0].imshow(d["gt"]); ax[0].set_title("ground truth"); ax[1].imshow(d["r"]); ax[1].set_title(f"our render  (PSNR {d['psnr']:.2f}, SSIM {d['ssim']:.3f})")
    im = ax[2].imshow(G(np.sqrt(d["e"])), cmap="inferno", vmin=0, vmax=0.2); ax[2].set_title("error heatmap  (RMS colour error 0-0.2; bright = large)")
    ax[3].imshow(G(d["sm"]), cmap="magma", vmin=0, vmax=0.8); ax[3].set_title("structure-loss heatmap  (1 - local SSIM; bright = structure wrong)")
    if ref:
        ax[4].imshow(G(np.sqrt(d["er"])), cmap="inferno", vmin=0, vmax=0.2); ax[4].set_title(f"PF-GS-360 error heatmap  (PSNR {d['psnr_ref']:.2f})")
        diff = G(np.sqrt(d["e"]) - np.sqrt(d["er"])); ax[5].imshow(diff, cmap="coolwarm", vmin=-0.08, vmax=0.08); ax[5].set_title("ours minus PF-GS-360 error  (red = we are worse)")
    for x in ax: x.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)
for d in rows: view_fig(d, f"{a.out_dir}/view_{d['j']:02d}_psnr{d['psnr']:.2f}.png")

# ---- aggregate statistics
cat = {k: np.zeros(2) for k in ("sky-like", "edges (sharpest 20% of pixels)", "flat / textured non-edge", "bottom 12% (rig / ground)")}  # [sum sq err ours, pixel count]
cat_ref = {k: 0.0 for k in cat}; share_top = {1: 0.0, 5: 0.0, 20: 0.0}
lat = np.zeros(bands); lat_ref = np.zeros(bands)
for d in rows:
    grad, sky = regions(d["gt"]); edge = grad >= np.percentile(grad, 80); bottom = np.arange(H)[:, None] >= H * 0.88; bottom = np.broadcast_to(bottom, (H, W))
    masks = {"sky-like": sky, "edges (sharpest 20% of pixels)": edge & ~sky, "flat / textured non-edge": ~edge & ~sky, "bottom 12% (rig / ground)": bottom}
    for k, m in masks.items():
        cat[k] += [d["e"][m].sum(), m.sum()]
        if ref: cat_ref[k] += d["er"][m].sum()
    s = np.sort(d["e"].ravel())[::-1]; cs = s.cumsum() / s.sum()
    for q in share_top: share_top[q] += cs[int(len(s) * q / 100) - 1] / n
    lat += np.array([d["e"][b * H // bands:(b + 1) * H // bands].mean() for b in range(bands)]) / n
    if ref: lat_ref += np.array([d["er"][b * H // bands:(b + 1) * H // bands].mean() for b in range(bands)]) / n
tot = sum(v[0] for v in cat.values()) - cat["bottom 12% (rig / ground)"][0]   # categories sky/edge/flat partition the image; bottom overlaps them
part = {k: cat[k] for k in list(cat)[:3]}; tot = sum(v[0] for v in part.values())

with PdfPages(f"{a.out_dir}/error_heatmap_report.pdf") as pdf:
    # page 1: aggregate maps
    fig, axs = plt.subplots(3 if ref else 2, 2, figsize=(11.7, 11 if ref else 7.7)); ax = axs.ravel()
    mm = G(np.sqrt(acc["mse"])); im = ax[0].imshow(mm, cmap="inferno", vmin=0, vmax=0.12); ax[0].set_title("Mean error over all " + a.label + " views (RMS colour error)\nbright = where our renders are consistently wrong"); plt.colorbar(im, ax=ax[0], fraction=.03)
    im = ax[1].imshow(G(acc["ssimloss"]), cmap="magma", vmin=0, vmax=0.5); ax[1].set_title("Mean structure loss (1 - SSIM)"); plt.colorbar(im, ax=ax[1], fraction=.03)
    th, tw = 6, 12; tiles = acc["mse"][:H // th * th, :W // tw * tw].reshape(th, H // th, tw, W // tw).sum((1, 3)); share = tiles / tiles.sum() * 100
    ax[2].imshow(G(np.sqrt(acc["mse"])), cmap="inferno", vmin=0, vmax=0.12, extent=[0, tw, th, 0], aspect="auto")
    for yy in range(th):
        for xx in range(tw): ax[2].text(xx + .5, yy + .5, f"{share[yy, xx]:.1f}", ha="center", va="center", color="white", fontsize=7)
    ax[2].set_title("Share of the TOTAL squared error per tile (%)\n(this is what drives PSNR)"); ax[2].set_xticks([]); ax[2].set_yticks([])
    ax[3].barh(np.arange(bands), lat * 1e3, color="#d62728" if not ref else "#1f77b4", label="ours"); ax[3].invert_yaxis()
    if ref: ax[3].barh(np.arange(bands) + 0.0, lat_ref * 1e3, height=0.35, color="#2ca02c", alpha=.8, label="PF-GS-360")
    ax[3].set_yticks(range(bands)); ax[3].set_yticklabels([f"{int(90 - 180 * b / bands)}° to {int(90 - 180 * (b + 1) / bands)}°" for b in range(bands)], fontsize=7)
    ax[3].set_xlabel("mean squared error (x1e-3)"); ax[3].set_title("Error by latitude band (top = sky/zenith)"); ax[3].legend(fontsize=7); ax[3].grid(axis="x", alpha=.3)
    if ref:
        ratio = G(acc["mse"]) / np.maximum(G(acc["mse_ref"]), 1e-9); im = ax[4].imshow(np.log2(ratio), cmap="coolwarm", vmin=-1, vmax=1); ax[4].set_title("Where we lose to PF-GS-360: log2(our MSE / their MSE)\nred = we are worse, blue = we are better"); plt.colorbar(im, ax=ax[4], fraction=.03)
        ax[5].axis("off")
    for x in [ax[0], ax[1]] + ([ax[4]] if ref else []): x.axis("off")
    fig.suptitle(f"Where do the renders lose PSNR / SSIM?  (all {a.label} views)", fontsize=13, weight="bold"); fig.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)

    # page 2: region shares + per-view
    fig = plt.figure(figsize=(11.7, 8.3)); a1 = fig.add_axes([0.07, 0.56, 0.4, 0.34]); a2 = fig.add_axes([0.57, 0.56, 0.38, 0.34]); a3 = fig.add_axes([0.07, 0.08, 0.88, 0.36])
    ks = list(part); err_share = [part[k][0] / tot * 100 for k in ks]; pix_share = [part[k][1] / sum(v[1] for v in part.values()) * 100 for k in ks]
    xx = np.arange(len(ks)); a1.bar(xx - .2, pix_share, .4, label="% of pixels", color="#bbbbbb"); a1.bar(xx + .2, err_share, .4, label="% of squared error", color="#d62728")
    a1.set_xticks(xx); a1.set_xticklabels([k.replace(" (", "\n(") for k in ks], fontsize=7); a1.set_title("Error by image content (classes estimated from the GT)", fontsize=9); a1.legend(fontsize=7); a1.grid(axis="y", alpha=.3)
    if ref:
        a2.bar(xx, [part[k][0] / cat_ref[k] for k in ks], color=["#d62728" if part[k][0] / cat_ref[k] > 1.15 else "#999999" for k in ks]); a2.axhline(1, color="k", lw=.8)
        a2.set_xticks(xx); a2.set_xticklabels([k.replace(" (", "\n(") for k in ks], fontsize=7); a2.set_title("Our squared error / PF-GS-360 squared error, by content\n(>1 = we are worse there)", fontsize=9); a2.grid(axis="y", alpha=.3)
    else: a2.axis("off")
    ps = [d["psnr"] for d in rows]; a3.bar(np.arange(n) - (.2 if ref else 0), ps, .4 if ref else .8, color="#d62728", label="ours")
    if ref: a3.bar(np.arange(n) + .2, [d["psnr_ref"] for d in rows], .4, color="#2ca02c", label="PF-GS-360"); a3.legend(fontsize=7)
    a3.set_ylim(min(ps) - 1.5, max(ps) + 1.5); a3.set_xlabel(a.label + " view index (our render order)"); a3.set_ylabel("PSNR (dB)"); a3.grid(axis="y", alpha=.3); a3.set_title("PSNR per " + a.label + " view", fontsize=9)
    fig.suptitle(f"Hardest pixels: top 1% of pixels carry {share_top[1]*100:.0f}% of the squared error, top 5% carry {share_top[5]*100:.0f}%, top 20% carry {share_top[20]*100:.0f}%", fontsize=11, weight="bold")
    pdf.savefig(fig); plt.close(fig)

    # worst views
    for d in sorted(rows, key=lambda d: d["psnr"])[:a.worst]:
        view_fig(d, f"{a.out_dir}/_tmp.png"); im = plt.imread(f"{a.out_dir}/_tmp.png"); fig = plt.figure(figsize=(11.7, 11.7 * im.shape[0] / im.shape[1] + 0.3))
        ax = fig.add_axes([0, 0, 1, 0.97]); ax.imshow(im); ax.axis("off"); fig.suptitle(f"{a.label} view {d['j']} (one of the {a.worst} lowest-PSNR views)", fontsize=10, y=0.995); pdf.savefig(fig); plt.close(fig)
    os.remove(f"{a.out_dir}/_tmp.png")
print("regions (share of squared error): ", {k: round(part[k][0] / tot * 100, 1) for k in ks}, "| pixel share:", {k: round(v, 1) for k, v in zip(ks, pix_share)})
if ref: print("ours/ref squared error by region:", {k: round(part[k][0] / cat_ref[k], 2) for k in ks}, "| bottom12%:", round(cat["bottom 12% (rig / ground)"][0] / cat_ref["bottom 12% (rig / ground)"], 2))
print("top-pixel share of error:", {q: round(v * 100, 1) for q, v in share_top.items()}, "\nwrote", a.out_dir)
