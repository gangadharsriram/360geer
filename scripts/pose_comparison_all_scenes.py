"""Pose-agreement report for every Ricoh360 scene: COLMAP vs OpenMVG vs PF-GS-360 (poses read out of the PF-GS-360 checkpoints,
train + test cameras). Output: a plain-language PDF. No ground truth exists - every number is agreement between two pose sets.
Usage: python scripts/pose_comparison_all_scenes.py ADJ_DIR [OUT.pdf]    (ADJ_DIR from scripts/extract_pfgs_pose_adjustments.py)"""
import json, os, sys, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.transform import Rotation as Rot

ADJ = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "/data2/3dgs_pipeline/output/pose_comparison_all_scenes.pdf"
RIC = "/data2/Dataset/Omnidirectional/Ricoh360"; PFGS = "/data2/Resuts/360/pfgs360_outputs/Ricoh360"
PX_PER_DEG = 1920 / 360
F = np.diag([1, -1, -1.])

def load_openmvg(d):
    nm = {v["key"]: v["value"]["ptr_wrapper"]["data"]["filename"] for v in json.load(open(d + "/data_views.json"))["views"]}
    return {nm[e["key"]]: (np.array(e["value"]["rotation"]), np.array(e["value"]["center"])) for e in json.load(open(d + "/data_extrinsics.json"))["extrinsics"]}

def load_colmap(f):
    out = {}
    for l in open(f):
        t = l.split()
        if len(t) == 10 and t[-1].endswith((".png", ".jpg")):
            q = np.array(t[1:5], float); R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            out[t[-1]] = (R, -R.T @ np.array(t[5:8], float))
    return out

def load_pfgs(scene, d):  # row order = train.txt (train cams) / test.txt (eval cams); decoded and verified against the exported bricks file
    z = np.load(f"{ADJ}/{scene}.npz"); ext = ".png"
    out = {}
    for rows, lst in ((z["train"], "train"), (z["eval"], "test")):
        names = open(f"{d}/{lst}.txt").read().split()
        R = F @ Rot.from_rotvec(rows[:, 3:]).as_matrix().transpose(0, 2, 1)
        for n, r, c in zip(names, R, rows[:, :3]): out[n + ext] = (r, c.astype(float))
    return out, int(z["step"])

def sim3(X, Y):
    mx, my = X.mean(0), Y.mean(0); U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt
    return (S * np.diag(D)).sum() / ((X - mx) ** 2).sum(), R, mx, my

def ang(M): return np.degrees(np.arccos(np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
def auc(err, tau):
    e = np.sort(err); t = np.arange(1, int(tau / 0.05) + 1) * 0.05; return (np.searchsorted(e, t, side="right") / len(e)).mean() * 100

def compare(A, B, names):
    RA, CA = np.array([A[n][0] for n in names]), np.array([A[n][1] for n in names]); RB, CB = np.array([B[n][0] for n in names]), np.array([B[n][1] for n in names])
    s, Rg, mx, my = sim3(CA, CB); cen = np.linalg.norm((s * Rg @ (CA - mx).T).T + my - CB, axis=1)
    d = np.linalg.norm(CB[:, None] - CB[None], axis=2); np.fill_diagonal(d, np.inf); spacing = np.median(d.min(1))
    G = np.einsum("nji,njk->nik", RB, RA); rot_gf = ang(np.einsum("nij,kj->nik", G, Rot.from_matrix(G).mean().as_matrix()))
    i, j = np.triu_indices(len(names), 1)
    rel = lambda R, C: (R[j] @ R[i].transpose(0, 2, 1), np.einsum("pab,pb->pa", R[i], C[j] - C[i]))
    (rA, tA), (rB, tB) = rel(RA, CA), rel(RB, CB)
    rra = ang(np.einsum("pji,pjk->pik", rA, rB)); u = tA / np.linalg.norm(tA, axis=1, keepdims=True); v = tB / np.linalg.norm(tB, axis=1, keepdims=True)
    rta = np.degrees(np.arccos(np.clip((u * v).sum(1), -1, 1))); both = np.maximum(rra, rta)
    ate = float(np.sqrt((cen ** 2).mean()) / spacing)
    return dict(n=len(names), ate=ate, cen=cen / spacing, rot_gf=rot_gf, rot=float(rot_gf.mean()), rra=rra, rta=rta, rta_med=float(np.median(rta)), rra_med=float(np.median(rra)),
                auc10=auc(both, 10), auc30=auc(both, 30), rta_auc10=auc(rta, 10), aligned=(s * Rg @ (CA - mx).T).T + my, ref=CB, scale=s)

def grade(r):
    a, d = r["ate"], r["rot"]
    return ("Excellent", "#2e7d32") if a < .2 and d < .3 else ("Good", "#7cb342") if a < .5 and d < 1 else ("Fair", "#f9a825") if a < 1 and d < 2 else ("Poor", "#c62828")

PAIRS = [("COLMAP", "OpenMVG"), ("COLMAP", "PF-GS-360"), ("OpenMVG", "PF-GS-360")]
PC = {PAIRS[0]: "#1f77b4", PAIRS[1]: "#ff7f0e", PAIRS[2]: "#9467bd"}; SC = {"COLMAP": "#1f77b4", "OpenMVG": "#2ca02c", "PF-GS-360": "#d62728"}
lab = lambda p: f"{p[0]} vs {p[1]}"
scenes = sorted(os.listdir(f"{RIC}/Data_openmvg")); ALL = {}
for sc in scenes:
    S = {"OpenMVG": load_openmvg(f"{RIC}/Data_openmvg/{sc}")}; note = []
    f = f"{RIC}/Data_colmap/{sc}/output_dir/colmap/images.txt"
    if os.path.exists(f): S["COLMAP"] = load_colmap(f)
    else: note.append("no COLMAP reconstruction available")
    if os.path.exists(f"{ADJ}/{sc}.npz"):
        pf, step = load_pfgs(sc, f"{RIC}/Data_openmvg/{sc}")
        if step < 1000: note.append(f"the PF-GS-360 checkpoint for this scene is at training step {step} (never trained - its poses are garbage), so PF-GS-360 is left out")
        else: S["PF-GS-360"] = pf
    else: note.append("no PF-GS-360 output")
    names = sorted(set.intersection(*[set(v) for v in S.values()]), key=lambda n: int(n.rsplit(".", 1)[0]))
    res = {p: compare(S[p[0]], S[p[1]], names) for p in PAIRS if p[0] in S and p[1] in S}
    ALL[sc] = dict(S=S, names=names, res=res, note=note)
    print(sc, len(names), {lab(p): (round(r["ate"], 2), round(r["rot"], 2)) for p, r in res.items()}, note)

# ------------------------------------------------------------------ PDF
def page_text(pdf, title, paras, size=10):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.07, 0.95, title, fontsize=16, weight="bold", va="top"); y = 0.905
    for p in paras:
        for line in (textwrap.wrap(p, 98) or [""]): fig.text(0.07, y, line, fontsize=size, va="top"); y -= 0.0195
        y -= 0.01
    pdf.savefig(fig); plt.close(fig)

def verdict(sc):
    r = ALL[sc]["res"]
    if len(r) < 3: return "only one pair available (" + "; ".join(ALL[sc]["note"]) + ")" if ALL[sc]["note"] else "only one pair available"
    mean = {s: np.mean([r[p]["ate"] for p in r if s in p]) for s in SC}; odd = max(mean, key=mean.get); lo = min(mean.values())
    return "no clear odd one out" if mean[odd] < 1.25 * lo + 0.03 else f"{odd} is the odd one out"

with PdfPages(OUT) as pdf:
    ngood = sum(1 for sc in ALL for p, r in ALL[sc]["res"].items() if grade(r)[0] in ("Excellent", "Good"))
    page_text(pdf, "Do the camera poses agree? - Ricoh360, all scenes", [
        "What this compares. For each of the 12 Ricoh360 scenes there are up to three independent estimates of where every camera was and which way it pointed: "
        "COLMAP (Data_colmap), OpenMVG (Data_openmvg - the poses our 360geer training uses) and PF-GS-360 (poses the PF-GS-360 method refined while training; read straight "
        "out of its saved checkpoints, covering both its train and test cameras). The report asks how closely each pair of estimates agree.",
        "Important: nobody has ground truth here. Agreement does not prove accuracy - two methods can share the same mistake - and a disagreement does not say which one is wrong. "
        "Treat it as a consistency check: if three independent methods disagree, at least some of the poses are unreliable.",
        "How to read the numbers (each one is a difference between two pose estimates):",
        "- Position gap: how far the camera positions differ, as a fraction of the distance between neighbouring cameras (1.0 = one full camera step). 0.1 means a 10% gap. "
        "(Overall scale and placement are removed first, because each method uses its own units.)",
        f"- Rotation gap: how many degrees the viewing direction differs. For reference, 1 degree is about {PX_PER_DEG:.1f} pixels on a 1920-pixel-wide panorama, so 0.2 degrees "
        "is roughly one pixel of image shift.",
        "- Direction-between-cameras accuracy (AUC, 0-100): for every pair of cameras, does each method agree on which direction the other camera lies? "
        "100 = perfect agreement on every pair, lower = more pairs where the methods disagree by many degrees. 'AUC@10' counts disagreements up to 10 degrees.",
        "- Rating: Excellent = position gap < 0.2 and rotation gap < 0.3 deg; Good = < 0.5 and < 1 deg; Fair = < 1 and < 2 deg; Poor = worse. "
        "(Thresholds are my own, chosen so differences visible as sub-pixel to few-pixel image shifts get the top grades.)",
        f"Headline: {ngood} of {sum(len(ALL[s]['res']) for s in ALL)} pose-set pairs across the scenes rate Excellent or Good. The next page lists every scene; later pages show one scene each."])

    # scoreboard
    fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.05, 0.95, "Scoreboard - every scene, every pair", fontsize=14, weight="bold", va="top")
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.86]); ax.axis("off")
    hdr = ["Scene", "cams", "Pair", "Position gap\n(camera steps)", "Rotation gap\n(deg / ~px)", "Direction AUC@10\n(0-100)", "Rating"]
    rows, colors = [], []
    for sc in ALL:
        first = True
        for p, r in ALL[sc]["res"].items():
            g, c = grade(r)
            rows.append([sc if first else "", r["n"] if first else "", lab(p).replace("PF-GS-360", "PFGS"), f"{r['ate']:.2f}", f"{r['rot']:.2f} / {r['rot']*PX_PER_DEG:.1f}", f"{r['auc10']:.0f}", g]); colors.append(c); first = False
        if not ALL[sc]["res"]: rows.append([sc, "", "no pair available", "", "", "", ""]); colors.append("#bdbdbd")
    tb = ax.table(cellText=rows, colLabels=hdr, loc="upper center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(6.8); tb.auto_set_column_width(list(range(len(hdr)))); tb.scale(1, 1.55)
    for i, c in enumerate(colors): tb[(i + 1, 6)].set_facecolor(c); tb[(i + 1, 6)].get_text().set_color("white")
    pdf.savefig(fig); plt.close(fig)

    # bar summary
    fig, axs = plt.subplots(2, 1, figsize=(8.27, 8.8)); x = np.arange(len(scenes)); w = 0.27
    for k, p in enumerate(PAIRS):
        for ax, key in zip(axs, ("ate", "rot")):
            ax.bar(x + (k - 1) * w, [ALL[s]["res"][p][key] if p in ALL[s]["res"] else np.nan for s in scenes], w, color=PC[p], label=lab(p))
    axs[0].set_ylabel("position gap (camera steps)\nlower = better"); axs[1].set_ylabel("rotation gap (degrees)\nlower = better")
    for ax in axs: ax.set_xticks(x); ax.set_xticklabels(scenes, rotation=35, ha="right", fontsize=8); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7)
    fig.suptitle("How far apart are the pose estimates, scene by scene", fontsize=12, weight="bold"); fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)

    # one page per scene
    for sc in scenes:
        d = ALL[sc]; res = d["res"]; fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.965, f"Scene: {sc}", fontsize=16, weight="bold", va="top")
        lines = [f"{len(d['names'])} cameras compared. Verdict: {verdict(sc)}."] + [f"Note: {n}." for n in d["note"]]
        for p, r in res.items():
            g, _ = grade(r)
            lines.append(f"{lab(p)}: {g}. Positions differ by {r['ate']*100:.0f}% of a camera step on average; viewing directions by {r['rot']:.2f} deg (~{r['rot']*PX_PER_DEG:.1f} px). "
                         f"Direction-between-cameras AUC@10 = {r['auc10']:.0f}/100.")
        y = 0.925
        for l in lines:
            for t in textwrap.wrap(l, 105): fig.text(0.06, y, t, fontsize=8.6, va="top"); y -= 0.016
            y -= 0.004
        top = y - 0.015; h = (top - 0.06) / 3 - 0.045
        if res:
            a1 = fig.add_axes([0.08, top - h, 0.38, h]); a2 = fig.add_axes([0.58, top - h, 0.38, h])
            for p, r in res.items():
                T = np.linspace(0, 30, 121); a1.plot(T, [(max(0, 1) * (np.maximum(r['rra'], r['rta']) <= t).mean()) * 100 for t in T], color=PC[p], lw=2, label=f"{lab(p)} (AUC@30 {r['auc30']:.0f})")
            a1.set_xlabel("allowed disagreement (deg)", fontsize=7); a1.set_ylabel("% of camera pairs in agreement", fontsize=7); a1.set_title("Pairwise direction agreement", fontsize=8); a1.grid(alpha=.3); a1.legend(fontsize=5.5, loc="lower right"); a1.tick_params(labelsize=7)
            for p, r in res.items(): a2.plot(np.arange(r["n"]), r["cen"], color=PC[p], lw=1, label=lab(p))
            a2.set_xlabel("camera (image order)", fontsize=7); a2.set_ylabel("position gap (camera steps)", fontsize=7); a2.set_title("Per-camera position gap", fontsize=8); a2.grid(alpha=.3); a2.tick_params(labelsize=7)
            a3 = fig.add_axes([0.08, top - 2 * h - 0.075, 0.38, h]); a4 = fig.add_axes([0.58, top - 2 * h - 0.075, 0.38, h]); S = d["S"]; ref = np.array([S["OpenMVG"][n][1] for n in d["names"]])
            for ax, (a, b), t in ((a3, (0, 2), "Camera positions, top-down"), (a4, (0, 1), "Camera positions, side view")):
                ax.scatter(ref[:, a], ref[:, b], s=7, c=SC["OpenMVG"], label="OpenMVG", zorder=3)
                for nm_ in ("COLMAP", "PF-GS-360"):
                    if nm_ not in S: continue
                    C = np.array([S[nm_][n][1] for n in d["names"]]); s, Rg, mx, my = sim3(C, ref); Q = (s * Rg @ (C - mx).T).T + my
                    ax.scatter(Q[:, a], Q[:, b], s=7, c=SC[nm_], label=nm_, zorder=2)
                    for q, r_ in zip(Q, ref): ax.plot([q[a], r_[a]], [q[b], r_[b]], color=SC[nm_], lw=.4, alpha=.5)
                ax.set_title(t + " (aligned to OpenMVG)", fontsize=8); ax.axis("equal"); ax.grid(alpha=.3); ax.legend(fontsize=6); ax.tick_params(labelsize=7)
            a5 = fig.add_axes([0.08, 0.06, 0.88, h * 0.9])
            for p, r in res.items(): a5.hist(r["rot_gf"], bins=20, alpha=.5, color=PC[p], label=lab(p))
            a5.set_xlabel("per-camera rotation gap (degrees)", fontsize=7); a5.set_ylabel("# cameras", fontsize=7); a5.set_title("How big are the rotation gaps?", fontsize=8); a5.legend(fontsize=6); a5.grid(alpha=.3); a5.tick_params(labelsize=7)
        pdf.savefig(fig); plt.close(fig)

    page_text(pdf, "Method notes and limits", [
        "PF-GS-360 poses: read from each scene's checkpoint (nerfstudio_models/*.ckpt). Each camera's pose is stored as a position plus a rotation vector; I decoded them and verified "
        "the decoding on bricks against the extrinsics file you exported earlier (exact match, error 7e-8) and against the dataset poses of the held-out test cameras. Train cameras follow train.txt order and test cameras "
        "(the checkpoint's test-time camera set) follow test.txt order. Using them avoids relying on an export that only exists for bricks.",
        "Alignment. COLMAP, OpenMVG and PF-GS-360 each live in their own coordinate frame and scale. 'Position gap' is measured after fitting one overall scale+rotation+shift between the two sets "
        "(Umeyama); 'rotation gap' removes the best single global rotation fitted from the rotations themselves. 'Direction AUC' compares, for every pair of cameras, the angle of the baseline direction "
        "and the relative rotation, and needs no alignment. AUC@T is the mean accuracy over disagreement thresholds from 0 to T degrees.",
        "Camera spacing unit: the median distance from a camera to its nearest neighbour in the second set of each pair.",
        "Limits. (1) No ground truth - this is agreement, not accuracy. (2) The 'Poor' scenes may be poor because one set is wrong, or because the trajectory is hard (long straight paths, little parallax). "
        "(3) The rating thresholds are mine. (4) PF-GS-360 test-camera poses come from a test-time optimisation that looked at the test images, so those poses are fit to the pictures and not an independent estimate. "
        "(5) gallery_pillar and garden: their saved PF-GS-360 checkpoints are at step 1 and step 0 (untrained, poses wildly off), although pose.json metrics exist for them - PF-GS-360 is left out there. gallery_chair has no PF-GS-360 output and bridge has no COLMAP reconstruction in Data_colmap.",
        "Reproduce: python scripts/extract_pfgs_pose_adjustments.py <adj_dir>; python scripts/pose_comparison_all_scenes.py <adj_dir>."], size=9.5)
print("wrote", OUT)
