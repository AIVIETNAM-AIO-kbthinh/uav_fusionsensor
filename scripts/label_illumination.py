"""Phan tang chieu sang cho DroneVehicle (phuc vu RQ3).

BOI CANH — mot phat hien cua qua trinh chuan bi du lieu:
DroneVehicle KHONG cung cap nhan chieu sang. XML chi co folder/filename/size/object
(da kiem chung bang scripts/scan_labels.py tren ca 57k file).

Da thu suy dien nhan day/night tu anh RGB va KHONG dat do chinh xac chap nhan duoc:
  - do sang trung binh (mean): den duong keo mean len ~105-130 tren canh dem
    -> gop nham "dem sang den" vao "ngay" (xem results/illumination/sheet_boundary_day.jpg)
  - luat 2 dac trung (p10 >= 55 va cast <= 0.15): tot hon nhung van sai ~25% o bien,
    vi canh dem duoc chieu sang manh nhin tu tren cao that su giong ngay am u
    (xem results/illumination/rule_day_boundary.jpg)

QUYET DINH: khong tuyen bo nhan "day/night". Phan tang theo DO CHIEU SANG DO DUOC
cua canh — mot dai luong lien tuc, khach quan, khong can ground truth:

    illum_proxy = p10 cua do xam anh RGB (phan vi 10%)

Ly do p10: ban ngay toan canh duoc chieu sang nen phan vi thap van cao; ban dem
chi vung quanh nguon sang moi sang nen p10 rat thap. p10 mien nhiem voi viec bi
mot vai den pha keo lech, khac voi mean.

Bao cao chinh nen la ĐƯỜNG CONG delta mAP theo illum_proxy (hoac theo decile),
bang tom tat dung 3 bin duoi day. Cot `day_heuristic` duoc giu lai nhu mot nhan
phu, PHAI ghi ro trong bao cao la heuristic ~75% chinh xac o bien, khong dung de
ket luan.
"""
import argparse, csv, os, random

import cv2
import numpy as np

# nguong co dinh (dung chung cho ca 3 split de so sanh duoc), chon tu phan bo p10
BINS = [("lowlight", 0.0, 10.0), ("midlight", 10.0, 55.0), ("bright", 55.0, 1e9)]
DAY_P10, DAY_CAST = 55.0, 0.15


def classify(v, bins=BINS):
    return next(n for n, lo, hi in bins if lo <= v < hi)


def montage(root, split, rows, tile=(240, 192), cols=6):
    cells = []
    for sid, val in rows:
        rgb = cv2.imread(f"{root}/images/{split}/{sid}.jpg")
        ir = cv2.imread(f"{root}/imagesr/{split}/{sid}.jpg")
        if rgb is None or ir is None:
            continue
        cell = np.vstack([cv2.resize(rgb, tile), cv2.resize(ir, tile)])
        cv2.putText(cell, f"{sid} p10={val:.0f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
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
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--sheet-split", default="test")
    ap.add_argument("--per-bin", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    outdir = "results/illumination"
    os.makedirs(outdir, exist_ok=True)
    sheets_src = None

    for split in a.splits:
        feat = {r["id"]: r for r in csv.DictReader(open(f"{a.root}/meta/{split}_feat.csv"))}
        base = {r["id"]: r for r in csv.DictReader(open(f"{a.root}/meta/{split}.csv"))}
        counts = {n: 0 for n, _, _ in BINS}
        boxes = {n: 0 for n, _, _ in BINS}
        n_day = 0
        with open(f"{a.root}/meta/{split}_illum.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "illum_proxy", "illum_bin", "day_heuristic", "mean_lum", "n_obj_ir"])
            for sid, r in feat.items():
                p10, cast = float(r["p10"]), float(r["cast"])
                b = classify(p10)
                day = int(p10 >= DAY_P10 and cast <= DAY_CAST)
                n_obj = int(base[sid]["n_obj_ir"]) if sid in base else 0
                counts[b] += 1
                boxes[b] += n_obj
                n_day += day
                w.writerow([sid, p10, b, day, r["mean"], n_obj])
        n = len(feat)
        print(f"[{split}] n={n}")
        for c, _, _ in BINS:
            print(f"    {c:9s} {counts[c]:6d} anh ({100*counts[c]/n:4.1f}%)  "
                  f"{boxes[c]:7d} box IR ({100*boxes[c]/max(sum(boxes.values()),1):4.1f}%)")
        print(f"    day_heuristic={n_day} ({100*n_day/n:.1f}%)  [nhan phu, khong dung ket luan]")
        if split == a.sheet_split:
            sheets_src = feat

    if sheets_src:
        vals = {sid: float(r["p10"]) for sid, r in sheets_src.items()}
        for c, lo, hi in BINS:
            ids = [(i, v) for i, v in vals.items() if lo <= v < hi]
            pick = rng.sample(ids, min(a.per_bin, len(ids)))
            m = montage(a.root, a.sheet_split, sorted(pick, key=lambda x: x[1]))
            if m is not None:
                cv2.imwrite(f"{outdir}/bin_{c}.jpg", m, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"\n-> contact sheet: {outdir}/bin_*.jpg (hang tren RGB, hang duoi IR)")


if __name__ == "__main__":
    main()
