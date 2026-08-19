"""Ve box GT (YOLO-OBB da chuyen doi) len ca hai modality de kiem tra bang mat.

Day la sanity check bat buoc truoc khi train: neu phep chuyen doi toa do sai thi
moi ket qua phia sau deu vo nghia ma khong co dau hieu bao loi.
"""
import argparse, glob, os, random

import cv2
import numpy as np

PALETTE = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (0, 255, 255),
           (255, 128, 0), (128, 0, 255), (0, 0, 255), (255, 255, 0)]


def draw(img, label_path, names):
    vis = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = vis.copy()
    h, w = vis.shape[:2]
    if not os.path.exists(label_path):
        return vis, 0
    n = 0
    for line in open(label_path):
        p = line.split()
        if len(p) != 9:
            continue
        c = int(p[0])
        pts = np.array([float(v) for v in p[1:]], dtype=np.float64).reshape(4, 2)
        pts *= np.array([w, h])
        cv2.polylines(vis, [pts.astype(np.int32)], True, PALETTE[c % len(PALETTE)], 1, cv2.LINE_AA)
        cv2.circle(vis, tuple(pts[0].astype(int)), 2, (255, 255, 255), -1)  # goc dau tien
        n += 1
    cv2.putText(vis, f"{os.path.basename(label_path)}  n={n}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dronevehicle", "vedai"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/gt_check")
    ap.add_argument("--ids", nargs="*", default=None)
    a = ap.parse_args()

    if a.dataset == "dronevehicle":
        ir_dir, rgb_dir = f"data/dronevehicle/imagesr/{a.split}", f"data/dronevehicle/images/{a.split}"
        lab_dir, ext = f"data/dronevehicle/labels/{a.split}", ".jpg"
        tag = f"dronevehicle_{a.split}"
    else:
        ir_dir, rgb_dir = "data/vedai/imagesr", "data/vedai/images"
        lab_dir, ext = "data/vedai/labels", ".png"
        tag = "vedai"

    ids = a.ids or sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{rgb_dir}/*{ext}"))
    if not a.ids:
        rng = random.Random(a.seed)
        ids = rng.sample(ids, min(a.n, len(ids)))
    os.makedirs(a.out, exist_ok=True)

    cells, total = [], 0
    for sid in sorted(ids):
        rgb = cv2.imread(f"{rgb_dir}/{sid}{ext}", cv2.IMREAD_COLOR)
        ir = cv2.imread(f"{ir_dir}/{sid}{ext}", cv2.IMREAD_GRAYSCALE)
        if rgb is None or ir is None:
            continue
        a1, n = draw(rgb, f"{lab_dir}/{sid}.txt", None)
        a2, _ = draw(ir, f"{lab_dir}/{sid}.txt", None)
        total += n
        cell = np.vstack([a1, a2])
        cells.append(cv2.copyMakeBorder(cell, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(60, 60, 60)))
    while len(cells) % a.cols:
        cells.append(np.zeros_like(cells[0]))
    m = np.vstack([np.hstack(cells[i:i + a.cols]) for i in range(0, len(cells), a.cols)])
    scale = min(1.0, 1900 / m.shape[1])
    if scale < 1.0:
        m = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    p = f"{a.out}/{tag}.jpg"
    cv2.imwrite(p, m, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {p}  ({len(ids)} anh, {total} box; hang tren RGB, hang duoi IR)")


if __name__ == "__main__":
    main()
