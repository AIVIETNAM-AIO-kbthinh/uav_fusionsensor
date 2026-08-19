"""mAP cho oriented bounding box, tính lại từ đầu.

Vì sao không dùng thẳng validator của Ultralytics: đề tài cần (a) phân tầng theo
điều kiện chiếu sáng / kích thước / lớp, và (b) khoảng tin cậy bootstrap theo ảnh.
Cả hai đều đòi hỏi tính AP trên một TẬP CON ẢNH TUỲ Ý và tính lại hàng nghìn lần.
Validator của Ultralytics không cho điểm móc đó.

Bù lại, hàm ở đây được đối chiếu với validator của Ultralytics trên cùng dữ liệu
(tests/test_metrics.py) để chắc chắn không lệch chuẩn.

IoU: dùng giao đa giác CHÍNH XÁC qua `cv2.rotatedRectangleIntersection`, không
dùng ProbIoU xấp xỉ. Để chạy nhanh, lọc trước bằng IoU của hộp bao trục (AABB)
rồi mới tính chính xác cho các cặp còn lại.

⚠️ ĐIỀU NÀY LÀM SỐ LIỆU KHÁC VALIDATOR CỦA ULTRALYTICS — có chủ đích
-------------------------------------------------------------------
Đã đo bằng scripts/diag_metrics.py trên cùng một bộ dự đoán:

    exact IoU + khớp theo confidence (COCO)   mAP50 0,1585   mAP50-95 0,0576
    exact IoU + khớp kiểu Ultralytics         mAP50 0,1552   mAP50-95 0,0571
    ProbIoU   + khớp theo confidence          mAP50 0,1761   mAP50-95 0,1034
    ProbIoU   + khớp kiểu Ultralytics         mAP50 0,1698   mAP50-95 0,1014   <- Ultralytics báo 0,176 / 0,103

Kết luận:
* Thứ tự khớp gần như không ảnh hưởng (0,1585 vs 0,1552).
* **ProbIoU cao hơn IoU thật trung bình +0,13, tối đa +0,26.** Đó là toàn bộ
  chênh lệch. Dùng ProbIoU thì harness tái tạo đúng số của Ultralytics, nên
  harness được xác nhận là đúng.
* ProbIoU là xấp xỉ khả vi dùng cho HÀM MẤT MÁT; dùng nó để ĐÁNH GIÁ sẽ cho số
  liệu lạc quan có hệ thống, và lệch mạnh nhất ở các ngưỡng IoU cao (mAP50-95).

Vì các baseline công bố trên DroneVehicle (mmrotate / DOTA devkit) đều dùng IoU
đa giác chính xác, đề tài **báo cáo theo IoU chính xác**. Hệ quả: số của
`model.val()` Ultralytics KHÔNG so trực tiếp được với số trong báo cáo — phải
ghi rõ điều này. Nó cũng có nghĩa là sanity check "đối chiếu S2 với ~83% của
literature" chỉ hợp lệ khi dùng harness này, không dùng `model.val()`.
"""

from __future__ import annotations

import cv2
import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)


def poly_to_rrect(poly: np.ndarray) -> tuple:
    """(8,) hoặc (4,2) -> ((cx,cy),(w,h),angle_deg) cho cv2."""
    pts = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    return cv2.minAreaRect(pts)


def poly_aabb(polys: np.ndarray) -> np.ndarray:
    """(N,8) -> (N,4) hộp bao trục [x1,y1,x2,y2]."""
    p = polys.reshape(-1, 4, 2)
    return np.concatenate([p.min(1), p.max(1)], axis=1)


def _aabb_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def rotated_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU chính xác giữa hai tập OBB dạng (N,8) và (M,8)."""
    n, m = len(a), len(b)
    out = np.zeros((n, m), dtype=np.float32)
    if n == 0 or m == 0:
        return out
    cand = _aabb_iou(poly_aabb(a), poly_aabb(b)) > 0        # lọc thô, rất rẻ
    ra = [poly_to_rrect(x) for x in a]
    rb = [poly_to_rrect(x) for x in b]
    area_a = np.array([r[1][0] * r[1][1] for r in ra], dtype=np.float32)
    area_b = np.array([r[1][0] * r[1][1] for r in rb], dtype=np.float32)
    for i, j in zip(*np.nonzero(cand)):
        ret, region = cv2.rotatedRectangleIntersection(ra[i], rb[j])
        if ret == cv2.INTERSECT_NONE or region is None:
            continue
        inter = cv2.contourArea(cv2.convexHull(region))
        union = area_a[i] + area_b[j] - inter
        if union > 1e-9:
            out[i, j] = inter / union
    return out


def match_image(pred_poly, pred_cls, gt_poly, gt_cls, iou_thrs=IOU_THRS) -> np.ndarray:
    """Khớp dự đoán với GT trong MỘT ảnh.

    Dự đoán phải được sắp theo confidence giảm dần.

    Returns:
        (n_pred, n_thr) bool — TP tại từng ngưỡng IoU.
    """
    n_thr = len(iou_thrs)
    tp = np.zeros((len(pred_poly), n_thr), dtype=bool)
    if len(pred_poly) == 0 or len(gt_poly) == 0:
        return tp
    iou = rotated_iou_matrix(pred_poly, gt_poly)
    same = pred_cls[:, None] == gt_cls[None, :]
    iou = np.where(same, iou, 0.0)
    for t, thr in enumerate(iou_thrs):
        taken = np.zeros(len(gt_poly), dtype=bool)
        for i in range(len(pred_poly)):
            j = -1
            best = thr
            row = iou[i]
            for k in np.argsort(-row):
                if row[k] < best:
                    break
                if not taken[k]:
                    j = k
                    break
            if j >= 0:
                taken[j] = True
                tp[i, t] = True
    return tp


REC_THRS = np.linspace(0, 1, 101)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP nội suy 101 điểm theo đúng công thức của pycocotools.

    Lưu ý: KHÔNG dùng `trapz` như `ultralytics.utils.metrics.compute_ap`. Với
    trapz, một dự đoán hoàn hảo duy nhất cho AP = 0,995 chứ không phải 1,0 (đoạn
    cuối chỉ được tính nửa trọng số). pycocotools lấy TRUNG BÌNH của 101 giá trị
    precision đã nội suy, nên AP(GT, GT) = 1,0 chính xác — đúng như sanity check
    mà plan mục 7.3 yêu cầu, và khớp với cách DOTA/COCO tính.
    """
    if len(recall) == 0:
        return 0.0
    pr = np.asarray(precision, dtype=np.float64).copy()
    rc = np.asarray(recall, dtype=np.float64)
    for i in range(len(pr) - 1, 0, -1):          # ép precision đơn điệu giảm
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    inds = np.searchsorted(rc, REC_THRS, side="left")
    q = np.zeros(len(REC_THRS), dtype=np.float64)
    valid = inds < len(pr)
    q[valid] = pr[inds[valid]]
    return float(q.mean())


def evaluate(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray,
             gt_cls: np.ndarray, nc: int, iou_thrs=IOU_THRS) -> dict:
    """Gộp TP/FP của nhiều ảnh thành AP.

    Args:
        tp: (n_pred, n_thr) bool đã gộp mọi ảnh.
        conf, pred_cls: (n_pred,)
        gt_cls: (n_gt,) nhãn của toàn bộ GT trong tập đang xét.
        nc: số lớp.

    Returns:
        {"map50", "map5095", "ap50_per_class", "ap5095_per_class", "n_gt_per_class"}
    """
    n_thr = len(iou_thrs)
    ap = np.zeros((nc, n_thr), dtype=np.float64)
    n_gt = np.bincount(gt_cls.astype(int), minlength=nc) if len(gt_cls) else np.zeros(nc, int)
    present = n_gt > 0

    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]

    for c in range(nc):
        if not present[c]:
            continue
        m = pred_cls == c
        if not m.any():
            continue
        tpc = tp[m].cumsum(0)
        fpc = (~tp[m]).cumsum(0)
        recall = tpc / n_gt[c]
        precision = tpc / np.clip(tpc + fpc, 1e-9, None)
        for t in range(n_thr):
            ap[c, t] = compute_ap(recall[:, t], precision[:, t])

    idx = np.where(present)[0]
    return {
        "map50": float(ap[idx, 0].mean()) if len(idx) else 0.0,
        "map5095": float(ap[idx].mean()) if len(idx) else 0.0,
        "ap50_per_class": {int(c): float(ap[c, 0]) for c in range(nc)},
        "ap5095_per_class": {int(c): float(ap[c].mean()) for c in range(nc)},
        "n_gt_per_class": {int(c): int(n_gt[c]) for c in range(nc)},
        "n_classes_present": int(present.sum()),
    }
