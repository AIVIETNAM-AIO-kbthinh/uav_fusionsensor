"""Tinh dac trung chieu sang tot hon do sang trung binh.

Do sang trung binh (mean) KHONG tach duoc "dem co den duong manh" khoi "ban ngay":
den nhan tao keo mean len toi ~105-130 trong khi phan lon canh van toi.

Dac trung tot hon xuat phat tu vat ly anh sang:
  - ban ngay : toan canh duoc chieu sang -> phan vi thap (p10) van cao
  - ban dem  : chi cac vung gan nguon sang moi sang -> p10 rat thap, do tuong phan cao
Ngoai ra den natri/LED tao am mau manh -> do lech kenh mau cao.

Xuat data/dronevehicle/meta/{split}_feat.csv
"""
import argparse, csv, glob, os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np


def feats(task):
    root, split, sid = task
    im = cv2.imread(f"{root}/images/{split}/{sid}.jpg", cv2.IMREAD_COLOR)
    if im is None:
        return None
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    p = np.percentile(g, [5, 10, 25, 50, 75, 90, 95])
    b, gr, r = im[:, :, 0].mean(), im[:, :, 1].mean(), im[:, :, 2].mean()
    return dict(
        id=sid,
        mean=round(float(g.mean()), 3),
        p05=round(float(p[0]), 2), p10=round(float(p[1]), 2), p25=round(float(p[2]), 2),
        median=round(float(p[3]), 2), p75=round(float(p[4]), 2),
        p90=round(float(p[5]), 2), p95=round(float(p[6]), 2),
        frac_dark40=round(float((g < 40).mean()), 4),
        frac_bright200=round(float((g > 200).mean()), 4),
        sat=round(float(hsv[:, :, 1].mean()), 2),
        # am mau: lech giua kenh am nhat va sang nhat, chuan hoa theo do sang
        cast=round(float((max(b, gr, r) - min(b, gr, r)) / (g.mean() + 1e-6)), 4),
    )


COLS = ["id", "mean", "p05", "p10", "p25", "median", "p75", "p90", "p95",
        "frac_dark40", "frac_bright200", "sat", "cast"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dronevehicle")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()

    for split in a.splits:
        ids = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(f"{a.root}/images/{split}/*.jpg"))
        print(f"[{split}] {len(ids)}", flush=True)
        rows = []
        with ProcessPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(feats, [(a.root, split, s) for s in ids], chunksize=64)):
                if r:
                    rows.append(r)
                if i % 4000 == 0:
                    print(f"   {i}/{len(ids)}", flush=True)
        with open(f"{a.root}/meta/{split}_feat.csv", "w", newline="") as f:
            w = csv.DictWriter(f, COLS)
            w.writeheader()
            w.writerows(rows)
        print(f"[{split}] wrote {len(rows)}", flush=True)
    print("FEATURES DONE", flush=True)


if __name__ == "__main__":
    main()
