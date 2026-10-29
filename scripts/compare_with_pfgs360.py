"""Per-view comparison of our test renders vs the PF-GS-360 eval_renders (same 50 test images, matched by GT pixels).
Writes composite PNGs (GT | theirs | ours | |err| theirs | |err| ours) for the views with the largest PSNR gap.
Usage: python scripts/compare_with_pfgs360.py OUR_TEST_DIR(.../test/ours_N) [N_VIEWS=4]"""
import sys, numpy as np
from PIL import Image
T = '/data2/Resuts/360/pfgs360_outputs/Ricoh360/bricks/pfgs360/eval_renders/'
O = sys.argv[1].rstrip('/') + '/'; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
L = lambda p: np.asarray(Image.open(p).convert('RGB'), np.float32) / 255
tg = [L(T + f'{i:06d}-gt.png') for i in range(50)]; og = [L(O + f'gt/{j:05d}.png') for j in range(50)]
ps = lambda a, b: 10 * np.log10(1 / ((a - b) ** 2).mean())
rows = []
for i in range(50):
    j = int(np.argmin([np.abs(tg[i] - g).mean() for g in og]))
    tp, op = L(T + f'{i:06d}-pred.png'), L(O + f'renders/{j:05d}.png')
    rows.append((ps(tp, tg[i]) - ps(op, og[j]), i, j, ps(tp, tg[i]), ps(op, og[j]), tg[i], tp, op))
for gap, i, j, pt, po, g, tp, op in sorted(rows, key=lambda r: -r[0])[:n]:
    e = lambda x: np.clip(np.abs(x - g).mean(2, keepdims=True).repeat(3, 2) * 4, 0, 1)
    img = np.concatenate([g, tp, op, e(tp), e(op)], 0)
    out = f'/data2/3dgs_pipeline/output/compare_vs_pfgs360/view{i:02d}_theirs{pt:.2f}_ours{po:.2f}.png'
    Image.fromarray((img * 255).astype(np.uint8)).resize((960, 2400)).save(out); print(out)
