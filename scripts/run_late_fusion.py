"""F3 — late fusion: hợp nhất dự đoán của S1 và S2 bằng WBF, không train gì thêm.

    python scripts/run_late_fusion.py --s1 runs/dronevehicle/S1_seed0 \
        --s2 runs/dronevehicle/S2_seed0 --out runs/dronevehicle/F3_seed0 --split test

Chi phí train bằng 0 (dùng lại weight của S1 và S2), chi phí suy luận gấp đôi.
Đây là cấu hình bền nhất với sai lệch đồng đăng ký nên là điểm chính của RQ4.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML  # noqa: E402

from src.eval.harness import load_dump, run_inference, save_dump  # noqa: E402
from src.fusion.wbf_obb import wbf_obb  # noqa: E402


def ensure_preds(run_dir: Path, split: str, imgsz: int, batch: int, device: str) -> Path:
    p = run_dir / f"preds_{split}.npz"
    if not p.exists():
        p = run_inference(run_dir, split=split, imgsz=imgsz, batch=batch, device=device,
                          out_name=f"preds_{split}.npz")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True, help="run dir cua S1 (RGB)")
    ap.add_argument("--s2", required=True, help="run dir cua S2 (IR)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--iou-thr", type=float, default=0.55)
    ap.add_argument("--weights", nargs=2, type=float, default=[1.0, 1.0])
    a = ap.parse_args()

    s1, s2, out = Path(a.s1), Path(a.s2), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    d1 = load_dump(ensure_preds(s1, a.split, a.imgsz, a.batch, a.device))
    d2 = load_dump(ensure_preds(s2, a.split, a.imgsz, a.batch, a.device))
    common = sorted(set(d1) & set(d2))
    if len(common) != len(d1) or len(common) != len(d2):
        print(f"canh bao: S1 co {len(d1)} anh, S2 co {len(d2)}, chung {len(common)}")

    fused = {}
    n_in, n_out = 0, 0
    for sid in common:
        r1, r2 = d1[sid], d2[sid]
        p, s, l = wbf_obb(
            [r1["pred_poly"], r2["pred_poly"]],
            [r1["conf"], r2["conf"]],
            [r1["pred_cls"], r2["pred_cls"]],
            weights=a.weights, iou_thr=a.iou_thr,
        )
        n_in += len(r1["conf"]) + len(r2["conf"])
        n_out += len(s)
        fused[sid] = {
            "pred_poly": p.astype(np.float32), "conf": s.astype(np.float32),
            "pred_cls": l.astype(np.int32),
            # GT giống nhau ở cả hai run (cùng nhãn IR thống nhất)
            "gt_poly": r2["gt_poly"], "gt_cls": r2["gt_cls"],
        }

    save_dump(fused, out / f"preds_{a.split}.npz")
    # sao chép data.yaml/provenance để scripts/evaluate_run.py dùng lại được
    (out / "data.yaml").write_text((s2 / "data.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    meta = {"id": "F3", "source_s1": str(s1), "source_s2": str(s2), "split": a.split,
            "iou_thr": a.iou_thr, "weights": a.weights,
            "n_pred_in": n_in, "n_pred_out": n_out, "n_images": len(common)}
    (out / "result.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"WBF: {n_in} box vao -> {n_out} box ra tren {len(common)} anh")
    print(f"-> {out / f'preds_{a.split}.npz'}")
    print(f"Danh gia: python scripts/evaluate_run.py --run {out} --split {a.split} --skip-inference")


if __name__ == "__main__":
    main()
