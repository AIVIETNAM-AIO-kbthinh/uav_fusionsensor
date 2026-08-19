"""Phân tích định tính: tìm những ảnh mà fusion thắng/thua rõ nhất so với đối chứng.

Plan mục 8 (tuần 10) yêu cầu chọn ~20 trường hợp fusion thắng và ~20 trường hợp
fusion thua rồi tìm quy luật. Script này chọn tự động theo tiêu chí định lượng —
hiệu số AP50 **trên từng ảnh** giữa hai cấu hình — rồi kết xuất ảnh so sánh để
xem bằng mắt.

    python scripts/qualitative.py --a runs/dronevehicle/F2a_seed0 \\
        --b runs/dronevehicle/C2_seed0 --dataset dronevehicle --split test --n 20

Mỗi ô: hàng trên RGB, hàng dưới IR. Xanh = GT, vàng = dự đoán của A, đỏ = của B.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML  # noqa: E402

from src.eval.harness import load_dump, match_all  # noqa: E402
from src.eval.metrics import evaluate  # noqa: E402

GT_COLOR, A_COLOR, B_COLOR = (0, 255, 0), (0, 255, 255), (0, 0, 255)


def image_ap(rec, nc, conf_thr=0.25):
    """AP50 của một ảnh (dùng để xếp hạng, không phải để báo cáo)."""
    keep = rec["conf"] >= conf_thr
    return evaluate(rec["tp"][keep], rec["conf"][keep], rec["pred_cls"][keep],
                    rec["gt_cls"], nc=nc)["map50"]


def draw(img, polys, color, conf=None, thr=0.25):
    for k, p in enumerate(polys):
        if conf is not None and conf[k] < thr:
            continue
        cv2.polylines(img, [np.asarray(p, np.float32).reshape(4, 2).astype(np.int32)],
                      True, color, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run dir cua cau hinh A (vd F2a)")
    ap.add_argument("--b", required=True, help="run dir cua cau hinh B (vd C2, doi chung)")
    ap.add_argument("--dataset", choices=["dronevehicle", "vedai"], default="dronevehicle")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="results/qualitative")
    a = ap.parse_args()

    A, B = Path(a.a), Path(a.b)
    nc = len(YAML.load(str(A / "data.yaml"))["names"])
    pa = match_all(load_dump(A / f"preds_{a.split}.npz"))
    pb = match_all(load_dump(B / f"preds_{a.split}.npz"))
    common = sorted(set(pa) & set(pb))

    deltas = []
    for sid in common:
        if len(pa[sid]["gt_cls"]) == 0:
            continue
        d = image_ap(pa[sid], nc, a.conf) - image_ap(pb[sid], nc, a.conf)
        deltas.append((d, sid))
    deltas.sort()

    if a.dataset == "dronevehicle":
        rgb_dir = ROOT / f"data/dronevehicle/images/{a.split}"
        ir_dir = ROOT / f"data/dronevehicle/imagesr/{a.split}"
        ext = ".jpg"
    else:
        rgb_dir, ir_dir, ext = ROOT / "data/vedai/images", ROOT / "data/vedai/imagesr", ".png"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    picks = {"fusion_thang": deltas[-a.n:][::-1], "fusion_thua": deltas[:a.n]}
    report = {}

    for name, sel in picks.items():
        cells = []
        report[name] = [{"id": s, "delta_ap50": round(d, 4)} for d, s in sel]
        for d, sid in sel:
            rgb = cv2.imread(str(rgb_dir / f"{sid}{ext}"), cv2.IMREAD_COLOR)
            ir = cv2.imread(str(ir_dir / f"{sid}{ext}"), cv2.IMREAD_GRAYSCALE)
            if rgb is None or ir is None:
                continue
            ir = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
            for canvas in (rgb, ir):
                draw(canvas, pa[sid]["gt_poly"], GT_COLOR)
                draw(canvas, pa[sid]["pred_poly"], A_COLOR, pa[sid]["conf"], a.conf)
                draw(canvas, pb[sid]["pred_poly"], B_COLOR, pb[sid]["conf"], a.conf)
            cell = np.vstack([rgb, ir])
            cv2.putText(cell, f"{sid} dAP={d:+.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(cv2.copyMakeBorder(cell, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(60, 60, 60)))
        if not cells:
            continue
        cols = 4
        while len(cells) % cols:
            cells.append(np.zeros_like(cells[0]))
        m = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
        scale = min(1.0, 1900 / m.shape[1])
        if scale < 1.0:
            m = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        p = out / f"{A.name}_vs_{B.name}_{name}.jpg"
        cv2.imwrite(str(p), m, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"-> {p}  ({len(sel)} anh)")

    d = np.array([x[0] for x in deltas])
    report["summary"] = {
        "n_images": len(d),
        "A_better": int((d > 0.01).sum()), "B_better": int((d < -0.01).sum()),
        "tie": int((np.abs(d) <= 0.01).sum()),
        "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
    }
    (out / f"{A.name}_vs_{B.name}.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    print("\nXanh = GT, vang = " + A.name + ", do = " + B.name)


if __name__ == "__main__":
    main()
