"""Chẩn đoán chênh lệch giữa harness và validator của Ultralytics.

Tách hai nguyên nhân có thể:
  1. IoU: đa giác chính xác (harness) vs ProbIoU xấp xỉ (Ultralytics)
  2. Thứ tự khớp: theo confidence (chuẩn COCO) vs theo IoU giảm dần (Ultralytics)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.utils.metrics import batch_probiou  # noqa: E402
from ultralytics.utils.ops import xyxyxyxy2xywhr  # noqa: E402

from src.eval import metrics as M  # noqa: E402
from src.eval.harness import evaluate_strata, load_dump  # noqa: E402


def probiou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ra = xyxyxyxy2xywhr(torch.from_numpy(a.reshape(-1, 4, 2)).float())
    rb = xyxyxyxy2xywhr(torch.from_numpy(b.reshape(-1, 4, 2)).float())
    return batch_probiou(ra, rb).numpy()


def match_ultra(iou, n_pred, iou_thrs):
    """Khớp kiểu Ultralytics: duyệt cặp theo IoU giảm dần."""
    tp = np.zeros((n_pred, len(iou_thrs)), dtype=bool)
    for t, thr in enumerate(iou_thrs):
        pairs = np.argwhere(iou >= thr)
        if not len(pairs):
            continue
        order = iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]
        pairs = pairs[order]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
        tp[pairs[:, 0].astype(int), t] = True
    return tp


def build(dump, iou_fn, matcher):
    per = {}
    diffs = []
    for sid, r in dump.items():
        order = np.argsort(-r["conf"]) if len(r["conf"]) else np.zeros(0, int)
        pp, pc, cf = r["pred_poly"][order], r["pred_cls"][order], r["conf"][order]
        gp, gc = r["gt_poly"], r["gt_cls"]
        if len(pp) and len(gp):
            iou = iou_fn(pp, gp)
            same = pc[:, None] == gc[None, :]
            iou = np.where(same, iou, 0.0)
            exact = M.rotated_iou_matrix(pp, gp)
            exact = np.where(same, exact, 0.0)
            m = exact > 0.1
            if m.any():
                diffs.append((iou[m] - exact[m]))
            tp = matcher(iou, len(pp), M.IOU_THRS)
        else:
            tp = np.zeros((len(pp), len(M.IOU_THRS)), bool)
        per[sid] = {"tp": tp, "conf": cf, "pred_cls": pc, "gt_cls": gc}
    return per, (np.concatenate(diffs) if diffs else np.zeros(0))


def conf_matcher(iou, n_pred, iou_thrs):
    tp = np.zeros((n_pred, len(iou_thrs)), dtype=bool)
    for t, thr in enumerate(iou_thrs):
        taken = np.zeros(iou.shape[1], dtype=bool)
        for i in range(n_pred):
            row = iou[i]
            for k in np.argsort(-row):
                if row[k] < thr:
                    break
                if not taken[k]:
                    taken[k] = True
                    tp[i, t] = True
                    break
    return tp


def main():
    dump = load_dump(sys.argv[1] if len(sys.argv) > 1 else "runs/smoke/F2a_smoke/preds_val.npz")
    nc = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    combos = [
        ("exact IoU + khop theo confidence (COCO)", M.rotated_iou_matrix, conf_matcher),
        ("exact IoU + khop kieu Ultralytics", M.rotated_iou_matrix, match_ultra),
        ("ProbIoU  + khop theo confidence", probiou_matrix, conf_matcher),
        ("ProbIoU  + khop kieu Ultralytics", probiou_matrix, match_ultra),
    ]
    print(f"{'cau hinh':45s} {'mAP50':>8s} {'mAP50-95':>9s}")
    for name, iou_fn, matcher in combos:
        per, diff = build(dump, iou_fn, matcher)
        res = evaluate_strata(per, {"all": {"all": list(per)}}, nc=nc)["all"]["all"]
        print(f"{name:45s} {res['map50']:8.4f} {res['map5095']:9.4f}")
        if "ProbIoU" in name and len(diff):
            print(f"{'':45s}  ProbIoU - exact: trung binh {diff.mean():+.4f}, "
                  f"p95 {np.percentile(diff, 95):+.4f}, max {diff.max():+.4f}")


if __name__ == "__main__":
    main()
