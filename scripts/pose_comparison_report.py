"""Pose comparison report for the Ricoh360 'bricks' scene: COLMAP vs OpenMVG vs PF-GS-360 camera poses -> PDF.
No ground truth exists, so every number is AGREEMENT between two pose sets. Relative metrics (RRA/RTA/AUC) are gauge-free;
absolute ones (ATE, rotation error) use a Sim3 fit of the camera centres. Usage: python scripts/pose_comparison_report.py [OUT.pdf]"""
import glob, itertools, json, os, sys, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.transform import Rotation as Rot

R5 = "/data2/Dataset/Omnidirectional/Ricoh360"
OPENMVG = f"{R5}/Data_openmvg/bricks"
COLMAP = f"{R5}/Data_colmap/bricks/output_dir/colmap/images.txt"
PFGS = "/data2/Resuts/360/pfgs360_outputs/Ricoh360"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/data2/3dgs_pipeline/output/pose_comparison_report.pdf"

def load_openmvg(d):
    nm = {v["key"]: v["value"]["ptr_wrapper"]["data"]["filename"] for v in json.load(open(d + "/data_views.json"))["views"]}
    ex = json.load(open(d + "/data_extrinsics.json"))["extrinsics"]
    return {nm[e["key"]]: (np.array(e["value"]["rotation"]), np.array(e["value"]["center"])) for e in ex}, nm

def load_colmap(f):
    out = {}
    for l in open(f):
        t = l.split()
        if len(t) == 10 and t[-1].endswith((".png", ".jpg")):
            q = np.array(t[1:5], float); R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            out[t[-1]] = (R, -R.T @ np.array(t[5:8], float))
    return out

def load_pfgs(f, nm):
    return {nm[e["key"]]: (np.array(e["value"]["rotation"]), np.array(e["value"]["center"])) for e in json.load(open(f))["extrinsics"]}

omv, nm = load_openmvg(OPENMVG)
SETS = {"COLMAP": load_colmap(COLMAP), "OpenMVG": omv,
        "PF-GS-360": load_pfgs(f"{PFGS}/bricks/pfgs360/pfgs_extrinsics_aligned.json", nm)}
split = {s: set(open(f"{OPENMVG}/{s}.txt").read().split()) for s in ("train", "test")}
stem = lambda n: n.rsplit(".", 1)[0]
common = sorted(set.intersection(*[set(v) for v in SETS.values()]), key=lambda n: int(stem(n)))
common100 = sorted(set(SETS["COLMAP"]) & set(SETS["OpenMVG"]), key=lambda n: int(stem(n)))
print("cameras per set:", {k: len(v) for k, v in SETS.items()}, "| common to all 3:", len(common), "| COLMAP&OpenMVG:", len(common100))

def stack(S, names): return np.array([S[n][0] for n in names]), np.array([S[n][1] for n in names])

def sim3(X, Y):  # X -> Y
    mx, my = X.mean(0), Y.mean(0); U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt; s = (S * np.diag(D)).sum() / ((X - mx) ** 2).sum()
    return s, R, mx, my

def ang(M): return np.degrees(np.arccos(np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))

def auc(err, tau):  # mean accuracy over thresholds 0.05..tau deg (fine grid, so sub-degree differences are not lost)
    e = np.sort(err); t = np.arange(1, int(tau / 0.05) + 1) * 0.05; return (np.searchsorted(e, t, side='right') / len(e)).mean()

def compare(A, B, names):
    RA, CA = stack(A, names); RB, CB = stack(B, names); n = len(names)
    s, Rg, mx, my = sim3(CA, CB)
    cen = np.linalg.norm((s * Rg @ (CA - mx).T).T + my - CB, axis=1)
    d = np.linalg.norm(CB[:, None] - CB[None], axis=2); np.fill_diagonal(d, np.inf); spacing = np.median(d.min(1))
    rot = ang(np.einsum("nij,njk->nik", RB, np.einsum("nij,kj->nik", RA, Rg).transpose(0, 2, 1)))  # RB * (RA Rg^T)^T
    G = np.einsum("nji,njk->nik", RB, RA)   # per-camera estimate of the global rotation: RB^T RA
    rot_gf = ang(np.einsum("nij,kj->nik", G, Rot.from_matrix(G).mean().as_matrix()))
    i, j = np.triu_indices(n, 1)
    rel = lambda R, C: (R[j] @ R[i].transpose(0, 2, 1), np.einsum("pab,pb->pa", R[i], C[j] - C[i]))
    (rA, tA), (rB, tB) = rel(RA, CA), rel(RB, CB)
    rra = ang(np.einsum("pji,pjk->pik", rA, rB))
    u = tA / np.linalg.norm(tA, axis=1, keepdims=True); v = tB / np.linalg.norm(tB, axis=1, keepdims=True)
    rta = np.degrees(np.arccos(np.clip((u * v).sum(1), -1, 1)))
    rpe_t = np.linalg.norm(s * tA - tB, axis=1) / np.linalg.norm(tB, axis=1)  # scale-aligned relative-translation error
    both = np.maximum(rra, rta)
    return dict(n=n, pairs=len(i), scale=s, ate_rmse=float(np.sqrt((cen ** 2).mean())), ate_spacings=float(np.sqrt((cen ** 2).mean()) / spacing),
                rot_mean=float(rot.mean()), rot_max=float(rot.max()), rot_gf_mean=float(rot_gf.mean()), rot_gf_max=float(rot_gf.max()), rot_gf=rot_gf, rpe_t_med=float(np.median(rpe_t) * 100),
                rra=rra, rta=rta, both=both, rra_med=float(np.median(rra)), rta_med=float(np.median(rta)), cen=cen / spacing, rot=rot, spacing=spacing,
                aligned=(s * Rg @ (CA - mx).T).T + my, ref=CB,
                **{f"{k}_auc{t}": auc(e, t) * 100 for k, e in (("rra", rra), ("rta", rta), ("both", both)) for t in (5, 10, 30)})

PAIRS = [("COLMAP", "OpenMVG"), ("COLMAP", "PF-GS-360"), ("OpenMVG", "PF-GS-360")]
RES = {p: compare(SETS[p[0]], SETS[p[1]], common) for p in PAIRS}
RES100 = compare(SETS["COLMAP"], SETS["OpenMVG"], common100)
for p, r in RES.items():
    print(p, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})

# ---------------- PDF ----------------
COL = {"COLMAP": "#1f77b4", "OpenMVG": "#2ca02c", "PF-GS-360": "#d62728"}
PCOL = {PAIRS[0]: "#1f77b4", PAIRS[1]: "#ff7f0e", PAIRS[2]: "#9467bd"}
lab = lambda p: f"{p[0]} vs {p[1]}"

def text_page(pdf, title, body, size=9.5):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.07, 0.95, title, fontsize=15, weight="bold", va="top")
    y = 0.91
    for para in body:
        for line in (textwrap.wrap(para, 105) or [""]):
            fig.text(0.07, y, line, fontsize=size, va="top", family="DejaVu Sans"); y -= 0.0185
        y -= 0.008
    pdf.savefig(fig); plt.close(fig)

with PdfPages(OUT) as pdf:
    best = min(PAIRS, key=lambda p: RES[p]["ate_spacings"])
    text_page(pdf, "Camera pose comparison - Ricoh360 / bricks", [
        "Three pose sources for the same capture: COLMAP (Data_colmap/bricks/output_dir/colmap), OpenMVG (Data_openmvg/bricks, "
        "the poses all our 360geer training uses) and PF-GS-360 (pfgs360_outputs/Ricoh360/bricks, joint pose refinement; its poses cover the 50 "
        f"TRAIN views only). Only {len(common)} cameras are common to all three, so the three-way comparison uses those {len(common)} views "
        f"({len(common100)} for COLMAP vs OpenMVG).",
        "IMPORTANT: there is no ground-truth pose in this dataset. Every number below measures AGREEMENT between two pose sets, not accuracy. "
        "A pair that agrees well is not necessarily right, and the OpenMVG poses are not a reference by definition.",
        "Metrics. RRA = angular error of the relative rotation between a camera pair; RTA = angular error of the relative translation DIRECTION "
        "between a pair (both over all camera pairs, gauge-free, no alignment needed). AUC@T = area under the accuracy-vs-threshold curve up to "
        "T degrees (percent, higher is better), computed on max(RRA, RTA) as in VGGT/CO3D. ATE = RMSE of camera centres after a Sim3 fit, "
        "given in units of the median nearest-neighbour camera spacing (so 1.0 = one camera spacing). Rotation error = per-camera angle, reported both after "
        "the centre Sim3 and gauge-free (global rotation fitted from the rotations). Relative-translation error = median of |s*t_A - t_B| / |t_B| over pairs (scale-aligned), in percent.",
        "Summary (common views):"] + [
        f"  {lab(p):<24} ATE {r['ate_spacings']:.2f} spacings | rot err {r['rot_mean']:.2f} deg | RTA AUC@10 {r['rta_auc10']:.1f} | "
        f"RRA AUC@10 {r['rra_auc10']:.1f} | pose AUC@30 {r['both_auc30']:.1f}" for p, r in RES.items()] + [
        f"Rotation note: the centre-fit rotation error is dominated by a near-constant offset (the camera ring is almost planar, so a Sim3 fit of the centres leaves a roll-like ambiguity). "
        "The gauge-free column removes it by fitting the global rotation from the camera rotations themselves - it is the better number for rotation agreement.",
        f"Closest pair by ATE: {lab(best)}.",
        "Caveat on PF-GS-360: its saved poses are already Sim3-aligned to the OpenMVG frame by its own pipeline, but all metrics here either are "
        "invariant to that alignment (relative ones) or re-fit their own Sim3, so that does not bias them toward OpenMVG."], size=9.2)

    # table page
    fig = plt.figure(figsize=(8.27, 11.69)); ax = fig.add_axes([0.02, 0.5, 0.96, 0.43]); ax.axis("off")
    fig.text(0.07, 0.95, "Metrics table", fontsize=15, weight="bold", va="top")
    hdr = ["Pair", "scale", "ATE\n(spacings)", "rot err, centre-fit\nmean/max (deg)", "rot err, gauge-free\nmean/max (deg)", "rel-trans\nerr (%)", "RRA med (deg)\nAUC@5/10/30", "RTA med (deg)\nAUC@5/10/30", "pose AUC\n@5/10/30"]
    rows = []
    for p, r in list(RES.items()) + [(("COLMAP", "OpenMVG [100 views]"), RES100)]:
        rows.append([lab(p).replace("[100 views]", "(100 views)"), f"{r['scale']:.3f}", f"{r['ate_spacings']:.2f}", f"{r['rot_mean']:.2f} / {r['rot_max']:.2f}", f"{r['rot_gf_mean']:.2f} / {r['rot_gf_max']:.2f}", f"{r['rpe_t_med']:.1f}",
                     f"{r['rra_med']:.2f}  " + "/".join(f"{r[f'rra_auc{t}']:.1f}" for t in (5, 10, 30)), f"{r['rta_med']:.2f}  " + "/".join(f"{r[f'rta_auc{t}']:.0f}" for t in (5, 10, 30)),
                     "/".join(f"{r[f'both_auc{t}']:.0f}" for t in (5, 10, 30))])
    tb = ax.table(cellText=rows, colLabels=hdr, loc="upper center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(6); tb.auto_set_column_width(list(range(len(hdr)))); tb.scale(1, 2.8)
    fig.text(0.07, 0.50, "Scale = Sim3 scale taking the first set into the second one's units (COLMAP and PF-GS-360 use their own arbitrary units).\n"
             "spacing = median nearest-neighbour camera distance of the second set. AUC values in percent.", fontsize=8, va="top")
    pdf.savefig(fig); plt.close(fig)

    # curves
    fig, axs = plt.subplots(2, 2, figsize=(8.27, 8.5)); ths = np.arange(0, 31)
    for ax, (key, name) in zip(axs.flat, (("rra", "Relative rotation accuracy (RRA), 0-5 deg"), ("rta", "Relative translation direction accuracy (RTA)"),
                                          ("both", "Relative pose accuracy, max(RRA,RTA)"), ("rra_zoom", "Zoom: relative rotation accuracy up to 2 deg"))):
        for p in PAIRS:
            z = key == "rra_zoom"; e = RES[p]["rra" if z else key]; T = np.linspace(0, 2, 81) if z else (np.linspace(0, 5, 101) if key == "rra" else ths)
            acc = np.array([(e <= t).mean() for t in T]) * 100
            ax.plot(T, acc, color=PCOL[p], lw=2, label=f"{lab(p)}  " + (f"median {RES[p]['rra_med']:.2f} deg" if z else f"AUC@30={RES[p][key + '_auc30']:.1f}"))
        ax.set_title(name, fontsize=9); ax.set_xlabel("threshold (deg)"); ax.set_ylabel("% of camera pairs"); ax.grid(alpha=.3); ax.legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Accuracy vs threshold (area under each curve = AUC)", fontsize=12, weight="bold"); fig.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)

    # bar chart of AUCs
    fig, axs = plt.subplots(1, 3, figsize=(8.27, 4.2))
    for ax, key, name in zip(axs, ("rra", "rta", "both"), ("RRA", "RTA", "max(RRA,RTA)")):
        w = 0.26
        for k, t in enumerate((5, 10, 30)):
            ax.bar(np.arange(3) + (k - 1) * w, [RES[p][f"{key}_auc{t}"] for p in PAIRS], w, label=f"AUC@{t}")
        ax.set_xticks(range(3)); ax.set_xticklabels(["COL-\nOMVG", "COL-\nPFGS", "OMVG-\nPFGS"], fontsize=7); ax.set_title(name, fontsize=9); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=.3)
    axs[0].set_ylabel("AUC (%)"); axs[0].legend(fontsize=7); fig.suptitle("AUC by pose-set pair", fontsize=12, weight="bold"); fig.tight_layout(rect=[0, 0, 1, 0.94]); pdf.savefig(fig); plt.close(fig)

    # trajectories: everything Sim3-aligned into the OpenMVG frame
    fig, axs = plt.subplots(1, 2, figsize=(8.27, 4.6)); ref = stack(SETS["OpenMVG"], common)[1]
    for ax, (a, b), nm_ in zip(axs, ((0, 2), (0, 1)), ("top-down (x-z)", "side (x-y)")):
        ax.scatter(ref[:, a], ref[:, b], s=10, c=COL["OpenMVG"], label="OpenMVG", zorder=3)
        for name in ("COLMAP", "PF-GS-360"):
            C = stack(SETS[name], common)[1]; s, Rg, mx, my = sim3(C, ref); Q = (s * Rg @ (C - mx).T).T + my
            ax.scatter(Q[:, a], Q[:, b], s=10, c=COL[name], label=name, zorder=2)
            for q, r_ in zip(Q, ref): ax.plot([q[a], r_[a]], [q[b], r_[b]], color=COL[name], lw=.5, alpha=.5)
        ax.set_title(nm_, fontsize=9); ax.axis("equal"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.suptitle(f"Camera centres ({len(common)} common views), each set Sim3-aligned to OpenMVG; lines = per-camera displacement", fontsize=9.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); pdf.savefig(fig); plt.close(fig)

    # per-camera errors
    fig, axs = plt.subplots(2, 1, figsize=(8.27, 7)); idx = np.arange(len(common))
    for p in PAIRS:
        axs[0].plot(idx, RES[p]["cen"], color=PCOL[p], marker=".", lw=1, label=lab(p)); axs[1].plot(idx, RES[p]["rot_gf"], color=PCOL[p], marker=".", lw=1, label=lab(p))
    axs[0].set_ylabel("centre error (spacings)"); axs[1].set_ylabel("rotation error, gauge-free (deg)")
    for ax in axs: ax.set_xlabel("camera (sorted by image number)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.suptitle("Per-camera error (centres: Sim3 fit; rotations: global rotation fitted from the rotations)", fontsize=12, weight="bold"); fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)

    # appendix: PF-GS-360 self-reported pose metrics, all scenes
    rows = []
    for d in sorted(glob.glob(f"{PFGS}/*/pfgs360/pose.json")):
        j = json.load(open(d)); rows.append([d.split("/")[-3], f"{j['Points']:,}", f"{j['RPE_t']:.3f}", f"{j['RPE_r']:.3f}", f"{j['ATE']:.4f}"])
    fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.07, 0.95, "Appendix - PF-GS-360 self-reported pose metrics (all scenes)", fontsize=13, weight="bold", va="top")
    ax = fig.add_axes([0.07, 0.6, 0.86, 0.3]); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=["scene", "points", "RPE_t", "RPE_r", "ATE"], loc="upper center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.5)
    fig.text(0.07, 0.56, textwrap.fill("Copied from each scene's pose.json (pfgs360 output) - NOT recomputed here; the definition and units of RPE_t / RPE_r / ATE are those of the "
             "PF-GS-360 code, measured by it against the dataset poses. Only the bricks scene has saved per-camera extrinsics, so only bricks could be compared "
             "pose-by-pose above. COLMAP poses exist for all scenes except bridge; extending the three-way comparison to other scenes needs PF-GS-360's "
             "extrinsics exported for them.", 110), fontsize=8, va="top")
    pdf.savefig(fig); plt.close(fig)
print("wrote", OUT)
