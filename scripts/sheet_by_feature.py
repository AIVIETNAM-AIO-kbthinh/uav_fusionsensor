"""Dung contact sheet cac anh co gia tri dac trung gan mot moc cho truoc.

Dung de do nguong bang mat: `--feature p10 --at 20 30 40 50` se sinh moi moc
mot tam anh gom cac mau co p10 gan moc do nhat.
"""
import argparse, csv, os

import cv2
import numpy as np


def montage(root, split, rows, tile=(240, 192), cols=6):
    cells = []
    for sid, val in rows:
        rgb = cv2.imread(f"{root}/images/{split}/{sid}.jpg")
        ir = cv2.imread(f"{root}/imagesr/{split}/{sid}.jpg")
        if rgb is None or ir is None:
            continue
        cell = np.vstack([cv2.resize(rgb, tile), cv2.resize(ir, tile)])
        cv2.putText(cell, f"{sid} {val:.1f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cells.append(cv2.copyMakeBorder(cell, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40)))
    if not cells:
        return None
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    return np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dronevehicle")
    ap.add_argument("--split", default="test")
    ap.add_argument("--feature", default="p10")
    ap.add_argument("--at", nargs="+", type=float, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="results/illumination")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(f"{a.root}/meta/{a.split}_feat.csv")))
    vals = [(r["id"], float(r[a.feature])) for r in rows]
    os.makedirs(a.out, exist_ok=True)
    for t in a.at:
        near = sorted(vals, key=lambda kv: abs(kv[1] - t))[:a.n]
        m = montage(a.root, a.split, sorted(near, key=lambda x: x[1]))
        if m is not None:
            p = f"{a.out}/feat_{a.feature}_{t:g}.jpg"
            cv2.imwrite(p, m, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print("wrote", p)


if __name__ == "__main__":
    main()
