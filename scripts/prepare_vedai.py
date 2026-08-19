"""Tien xu ly VEDAI (subset 512) sang cung dinh dang voi DroneVehicle.

Dinh dang annotation goc (annotation512.txt, 1 dong = 1 doi tuong), theo DevKit
watch_image.m: cot 1=id, 2=xc, 3=yc, 4=orientation, 5-8=x1..x4, 9-12=y1..y4,
13=class, 14=fully_contained, 15=occluded.

Khac biet quan trong so voi DroneVehicle: VEDAI chi co MOT bo ground truth dung
chung cho ca hai modality (khong co GT rieng RGB/IR). Vi vay khong ton tai van de
"chon GT nao" -> VEDAI la phep kiem chung sach hon cho H1, doi lai khong co kich
ban ban dem de kiem chung H2.

Output: data/vedai/
          images/ID.png    RGB 512x512
          imagesr/ID.png   NIR 512x512 grayscale
          labels/ID.txt    YOLO-OBB
          folds/fold{NN}_{train,test}.txt
          meta/summary.json
"""
import argparse, collections, glob, json, os

import cv2
import numpy as np

# Ma lop VEDAI -> ten. Cac lop cuc hiem (7, 8, 31: 3-48 mau) gop vao 'other'
# theo thong le cua cac cong trinh multimodal gan day tren VEDAI.
RAW2NAME = {1: "car", 2: "truck", 4: "tractor", 5: "camping", 9: "van",
            10: "other", 11: "pickup", 23: "boat",
            7: "other", 8: "other", 31: "other"}
CLASSES = ["car", "pickup", "camping", "truck", "other", "tractor", "boat", "van"]
CID = {c: i for i, c in enumerate(CLASSES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/vedai_raw")
    ap.add_argument("--out", default="data/vedai")
    ap.add_argument("--res", type=int, default=512)
    a = ap.parse_args()
    S, R = a.src, a.res
    for sub in ("images", "imagesr", "labels", "folds", "meta"):
        os.makedirs(f"{a.out}/{sub}", exist_ok=True)

    # 1. doc annotation
    objs = collections.defaultdict(list)
    n_lines = 0
    cls_count = collections.Counter()
    for line in open(f"{S}/Annotations{R}/annotation{R}.txt"):
        p = line.split()
        if len(p) < 13:
            continue
        n_lines += 1
        sid = p[0].zfill(8)
        xs = [float(v) for v in p[4:8]]
        ys = [float(v) for v in p[8:12]]
        raw = int(float(p[12]))
        name = RAW2NAME.get(raw)
        if name is None:
            cls_count[f"UNMAPPED:{raw}"] += 1
            continue
        cls_count[name] += 1
        objs[sid].append((name, np.array(list(zip(xs, ys)), dtype=np.float64)))

    # 2. anh + nhan
    ids = sorted({os.path.basename(p)[:8] for p in glob.glob(f"{S}/Vehicules{R}/*_co.png")})
    stats = dict(n_images=0, n_boxes=0, clipped=0, dropped=0, ir_3ch_identical=0, ir_3ch_diff=0)
    empty = []
    for sid in ids:
        co = cv2.imread(f"{S}/Vehicules{R}/{sid}_co.png", cv2.IMREAD_COLOR)
        ir = cv2.imread(f"{S}/Vehicules{R}/{sid}_ir.png", cv2.IMREAD_UNCHANGED)
        if co is None or ir is None:
            continue
        if ir.ndim == 3:
            if int(np.abs(ir[:, :, 0].astype(int) - ir[:, :, 2].astype(int)).max()) == 0:
                stats["ir_3ch_identical"] += 1
            else:
                stats["ir_3ch_diff"] += 1
            ir = ir[:, :, 0]
        h, w = co.shape[:2]
        cv2.imwrite(f"{a.out}/images/{sid}.png", co)
        cv2.imwrite(f"{a.out}/imagesr/{sid}.png", ir)

        lines = []
        for name, pts in objs.get(sid, []):
            if not (0 <= pts[:, 0].mean() < w and 0 <= pts[:, 1].mean() < h):
                stats["dropped"] += 1
                continue
            if pts[:, 0].min() < 0 or pts[:, 0].max() > w or pts[:, 1].min() < 0 or pts[:, 1].max() > h:
                stats["clipped"] += 1
            q = pts.copy()
            q[:, 0] = np.clip(q[:, 0], 0, w)
            q[:, 1] = np.clip(q[:, 1], 0, h)
            q /= np.array([w, h])
            lines.append(f"{CID[name]} " + " ".join(f"{v:.6f}" for v in q.reshape(-1)))
        with open(f"{a.out}/labels/{sid}.txt", "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        if not lines:
            empty.append(sid)
        stats["n_images"] += 1
        stats["n_boxes"] += len(lines)

    # 3. fold
    have = set(ids)
    folds = {}
    for f in sorted(glob.glob(f"{S}/Annotations{R}/fold[0-9]*.txt")):
        base = os.path.basename(f)[:-4]
        kind = "test" if base.endswith("test") else "train"
        num = base.replace("fold", "").replace("test", "")
        got = [l.strip().zfill(8) for l in open(f) if l.strip()]
        keep = [i for i in got if i in have]
        folds.setdefault(num, {})[kind] = len(keep)
        with open(f"{a.out}/folds/fold{num}_{kind}.txt", "w") as g:
            g.write("\n".join(keep) + "\n")

    summary = dict(res=R, classes=CLASSES, class_counts=dict(cls_count.most_common()),
                   n_annotation_lines=n_lines, empty_images=len(empty), folds=folds, **stats)
    json.dump(summary, open(f"{a.out}/meta/summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=1))
    print("folds:", {k: v for k, v in list(folds.items())[:3]}, "...", len(folds), "fold")
    print("VEDAI PREPARE DONE")


if __name__ == "__main__":
    main()
