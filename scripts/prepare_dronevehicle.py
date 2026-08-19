"""Tien xu ly DroneVehicle: cat vien trang 100px, chuan hoa nhan, xuat YOLO-OBB.

Input : data/raw/{split}/{split}img|imgr|label|labelr
Output: data/dronevehicle/
          images/{split}/ID.jpg     RGB 640x512
          imagesr/{split}/ID.jpg    IR  640x512 grayscale
          labels/{split}/ID.txt     YOLO-OBB tu annotation IR  (GT thong nhat - dung de train/eval)
          labels_rgb/{split}/ID.txt YOLO-OBB tu annotation RGB (chi dung cho phan tich phu)
          meta/{split}.csv          id, luminance, n_obj, thong ke lam sach
          meta/summary.json         tong hop + danh sach cap bi loai

Nhan chieu sang KHONG chot o day: cot `luminance` duoc luu tho de
scripts/label_illumination.py suy dien lai voi nguong tuy chinh ma khong phai
chay lai toan bo tien xu ly.
"""
import argparse, csv, glob, json, os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

BORDER = 100
W, H = 640, 512
CLASSES = ["car", "truck", "bus", "van", "freight_car"]
CID = {c: i for i, c in enumerate(CLASSES)}

# Da quet toan bo XML (scripts/scan_labels.py) -> day la tap ten lop DAY DU.
# Dataset goc co loi chinh ta: "feright" thay vi "freight", "truvk" thay vi "truck".
CLASS_ALIAS = {
    "car": "car", "truck": "truck", "bus": "bus", "van": "van",
    "feright_car": "freight_car", "feright car": "freight_car", "feright": "freight_car",
    "truvk": "truck",
    "*": None,            # nhan rac, bo
    "": None,
}

RAW, OUT = "data/raw", "data/dronevehicle"


def parse_objects(xml_path):
    """-> (list[(cls, 4x2 corners)] | None, n_no_geom, n_bndbox, n_bad_class)."""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None, 0, 0, 0
    out, no_geom, nbb, bad_cls = [], 0, 0, 0
    for obj in root.findall("object"):
        raw = (obj.findtext("name") or "").strip()
        if raw not in CLASS_ALIAS:
            bad_cls += 1
            continue
        name = CLASS_ALIAS[raw]
        poly, bb = obj.find("polygon"), obj.find("bndbox")
        try:
            if poly is not None:
                pts = np.array([[float(poly.findtext(f"x{i}")), float(poly.findtext(f"y{i}"))]
                                for i in (1, 2, 3, 4)], dtype=np.float64)
            elif bb is not None:
                x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
                x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
                nbb += 1
            else:
                no_geom += 1          # 75 object trong toan dataset khong co hinh hoc
                continue
        except (TypeError, ValueError):
            no_geom += 1
            continue
        if name is None:
            bad_cls += 1
            continue
        out.append((name, pts))
    return out, no_geom, nbb, bad_cls


def to_yolo_obb(objs):
    """Doi sang he toa do anh da cat, bo box co tam ngoai khung, clip, chuan hoa."""
    lines, dropped, clipped = [], 0, 0
    for name, pts in objs:
        p = pts - BORDER
        if not (0 <= p[:, 0].mean() < W and 0 <= p[:, 1].mean() < H):
            dropped += 1              # tam box nam trong vung vien -> khong con trong anh
            continue
        if p[:, 0].min() < 0 or p[:, 0].max() > W or p[:, 1].min() < 0 or p[:, 1].max() > H:
            clipped += 1              # box vuot vien (dataset co chu y cho phep dieu nay)
        p[:, 0] = np.clip(p[:, 0], 0, W)
        p[:, 1] = np.clip(p[:, 1], 0, H)
        n = p / np.array([W, H])
        lines.append(f"{CID[name]} " + " ".join(f"{v:.6f}" for v in n.reshape(-1)))
    return lines, dropped, clipped


def process_one(task):
    split, sid = task
    rgb_p = f"{RAW}/{split}/{split}img/{sid}.jpg"
    ir_p = f"{RAW}/{split}/{split}imgr/{sid}.jpg"
    lab_p = f"{RAW}/{split}/{split}label/{sid}.xml"
    labr_p = f"{RAW}/{split}/{split}labelr/{sid}.xml"
    rec = {"id": sid, "status": "ok"}

    missing = [p for p in (rgb_p, ir_p, lab_p, labr_p) if not os.path.exists(p)]
    if missing:
        rec["status"] = "missing:" + ",".join(os.path.basename(os.path.dirname(m)) for m in missing)
        return rec
    rgb = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
    ir = cv2.imread(ir_p, cv2.IMREAD_UNCHANGED)
    if rgb is None or ir is None:
        rec["status"] = "unreadable"
        return rec
    if rgb.shape[:2] != (712, 840) or ir.shape[:2] != (712, 840):
        rec["status"] = f"bad_shape:{rgb.shape[:2]}/{ir.shape[:2]}"
        return rec

    rgb_c = rgb[BORDER:-BORDER, BORDER:-BORDER]
    ir_c = ir[BORDER:-BORDER, BORDER:-BORDER]
    if ir_c.ndim == 3:
        ir_c = ir_c[:, :, 0]          # 3 kenh la ban sao giong het nhau (da kiem chung)

    objs_ir, ng_ir, bb_ir, bc_ir = parse_objects(labr_p)
    objs_rgb, ng_rgb, bb_rgb, bc_rgb = parse_objects(lab_p)
    if objs_ir is None or objs_rgb is None:
        rec["status"] = "xml_parse_error"
        return rec
    lines_ir, d_ir, c_ir = to_yolo_obb(objs_ir)
    lines_rgb, d_rgb, c_rgb = to_yolo_obb(objs_rgb)

    for sub, arr in (("images", rgb_c), ("imagesr", ir_c)):
        cv2.imwrite(f"{OUT}/{sub}/{split}/{sid}.jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    for sub, lines in (("labels", lines_ir), ("labels_rgb", lines_rgb)):
        with open(f"{OUT}/{sub}/{split}/{sid}.txt", "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    rec.update(
        luminance=round(float(cv2.cvtColor(rgb_c, cv2.COLOR_BGR2GRAY).mean()), 3),
        ir_mean=round(float(ir_c.mean()), 3),
        n_obj_ir=len(lines_ir), n_obj_rgb=len(lines_rgb),
        no_geom_ir=ng_ir, no_geom_rgb=ng_rgb, bndbox_ir=bb_ir, bndbox_rgb=bb_rgb,
        bad_cls_ir=bc_ir, bad_cls_rgb=bc_rgb,
        dropped_ir=d_ir, dropped_rgb=d_rgb, clipped_ir=c_ir, clipped_rgb=c_rgb,
    )
    return rec


COLS = ["id", "luminance", "ir_mean", "n_obj_ir", "n_obj_rgb", "no_geom_ir", "no_geom_rgb",
        "bndbox_ir", "bndbox_rgb", "bad_cls_ir", "bad_cls_rgb",
        "dropped_ir", "dropped_rgb", "clipped_ir", "clipped_rgb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    os.makedirs(f"{OUT}/meta", exist_ok=True)
    # gop voi ket qua cu thay vi ghi de, de chay tung split roi le van giu du summary
    sum_path = f"{OUT}/meta/summary.json"
    summary = json.load(open(sum_path)) if os.path.exists(sum_path) else {}
    for split in a.splits:
        for sub in ("images", "imagesr", "labels", "labels_rgb"):
            os.makedirs(f"{OUT}/{sub}/{split}", exist_ok=True)
        # duyet theo file NHAN (day du 100%) chu khong theo anh, de phat hien cap thieu anh
        ids = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(f"{RAW}/{split}/{split}labelr/*.xml"))
        print(f"[{split}] {len(ids)} ids", flush=True)
        recs = []
        with ProcessPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(process_one, [(split, s) for s in ids], chunksize=32)):
                recs.append(r)
                if i % 2000 == 0:
                    print(f"   {i}/{len(ids)}", flush=True)
        ok = [r for r in recs if r["status"] == "ok"]
        bad = [r for r in recs if r["status"] != "ok"]
        with open(f"{OUT}/meta/{split}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, COLS)
            w.writeheader()
            for r in ok:
                w.writerow({k: r.get(k) for k in COLS})
        summary[split] = {
            "n_ids": len(ids), "n_ok": len(ok), "n_excluded": len(bad),
            "excluded": [{"id": r["id"], "reason": r["status"]} for r in bad],
            **{k: sum(r[k] for r in ok) for k in COLS if k not in ("id", "luminance", "ir_mean")},
        }
        print(f"[{split}] ok={len(ok)} excluded={len(bad)} "
              f"boxes_ir={summary[split]['n_obj_ir']} boxes_rgb={summary[split]['n_obj_rgb']}", flush=True)
    json.dump(summary, open(sum_path, "w"), indent=1)
    print("PREPARE DONE", flush=True)


if __name__ == "__main__":
    main()
