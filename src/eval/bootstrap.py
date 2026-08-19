"""Khoảng tin cậy bootstrap cho HIỆU SỐ mAP giữa hai cấu hình.

Hai điểm thiết kế
-----------------

**1. Bootstrap ghép cặp theo ảnh.** Cùng một mẫu ảnh được lấy lại cho cả hai cấu
hình, nên phương sai do độ khó của ảnh bị triệt tiêu và khoảng tin cậy của hiệu số
hẹp hơn nhiều so với bootstrap độc lập. Đây là phép đo mà plan mục 6.3 dùng để kết
luận RQ1.

**2. Công thức có trọng số thay vì nhân bản vật lý.** Cách ngây thơ — mỗi vòng lặp
gom lại danh sách phát hiện của các ảnh được chọn rồi tính AP — tốn ~150 ms/vòng
trên tập test 8.980 ảnh, tức ~25 phút cho MỘT phép so sánh. Với vài phép so sánh
nhân vài tầng thì không chạy nổi.

Nhận xét: nếu ảnh i được chọn c_i lần thì mọi phát hiện của nó xuất hiện c_i lần
với cùng confidence. Do đó đường cong P-R chỉ phụ thuộc vào

    TP(t) = Σ_{conf_d ≥ t} c_{img(d)} · tp_d       n_gt[c] = Σ_i c_i · n_gt_i[c]

nên chỉ cần sắp xếp toàn cục theo confidence MỘT LẦN, rồi mỗi vòng bootstrap là
một phép `cumsum` có trọng số. Kết quả **giống hệt** cách nhân bản, nhanh hơn hai
bậc độ lớn.

Mặc định bootstrap ở ngưỡng IoU 0,5 (mAP50 — chỉ số chính theo plan mục 6.1 đã
cập nhật). Bootstrap toàn bộ 10 ngưỡng của mAP50-95 tốn gấp 10 lần và không cần
thiết cho kết luận.
"""

from __future__ import annotations

import numpy as np

from src.eval.metrics import compute_ap


class BootstrapData:
    """Chuẩn bị sẵn các mảng đã sắp xếp để bootstrap chạy nhanh.

    per_image: {image_id: {"tp": (n,T) bool, "conf": (n,), "pred_cls": (n,),
                           "gt_cls": (m,)}}
    """

    def __init__(self, per_image: dict, ids: list[str], nc: int, iou_idx: int = 0):
        self.nc = nc
        self.ids = list(ids)
        pos = {i: k for k, i in enumerate(self.ids)}

        tp, conf, cls, img = [], [], [], []
        gt_counts = np.zeros((len(self.ids), nc), dtype=np.float64)
        for i in self.ids:
            rec = per_image.get(i)
            if rec is None:
                continue
            k = pos[i]
            if len(rec["gt_cls"]):
                gt_counts[k] += np.bincount(np.asarray(rec["gt_cls"], dtype=int), minlength=nc)
            n = len(rec["conf"])
            if n:
                tp.append(np.asarray(rec["tp"])[:, iou_idx])
                conf.append(np.asarray(rec["conf"]))
                cls.append(np.asarray(rec["pred_cls"], dtype=int))
                img.append(np.full(n, k, dtype=np.int64))

        self.tp = np.concatenate(tp) if tp else np.zeros(0, dtype=bool)
        self.conf = np.concatenate(conf) if conf else np.zeros(0)
        self.cls = np.concatenate(cls) if cls else np.zeros(0, dtype=int)
        self.img = np.concatenate(img) if img else np.zeros(0, dtype=np.int64)
        self.gt_counts = gt_counts

        order = np.argsort(-self.conf, kind="stable")
        self.tp, self.cls, self.img = self.tp[order], self.cls[order], self.img[order]
        # tách sẵn theo lớp để vòng bootstrap không phải lọc lại
        self.by_class = [np.where(self.cls == c)[0] for c in range(nc)]

    def map50(self, counts: np.ndarray) -> float:
        """mAP tại ngưỡng IoU đã chọn, với trọng số ảnh `counts`."""
        n_gt = counts @ self.gt_counts                      # (nc,)
        aps, present = [], n_gt > 0
        w_all = counts[self.img]
        for c in range(self.nc):
            if not present[c]:
                continue
            idx = self.by_class[c]
            if len(idx) == 0:
                aps.append(0.0)
                continue
            w = w_all[idx]
            t = self.tp[idx]
            tpc = np.cumsum(w * t)
            fpc = np.cumsum(w * ~t)
            recall = tpc / n_gt[c]
            precision = tpc / np.clip(tpc + fpc, 1e-9, None)
            aps.append(compute_ap(recall, precision))
        return float(np.mean(aps)) if aps else 0.0


def bootstrap_delta(per_image_a: dict, per_image_b: dict, ids, nc: int,
                    n_boot: int = 10000, seed: int = 0, iou_idx: int = 0) -> dict:
    """Khoảng tin cậy 95% cho `mAP(A) − mAP(B)` tại một ngưỡng IoU.

    Returns:
        dict gồm giá trị điểm, CI 95%, p-value hai phía, tỉ lệ lần A thắng B.
    """
    ids = list(ids)
    n = len(ids)
    A = BootstrapData(per_image_a, ids, nc, iou_idx)
    B = BootstrapData(per_image_b, ids, nc, iou_idx)

    ones = np.ones(n, dtype=np.float64)
    a0, b0 = A.map50(ones), B.map50(ones)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
        deltas[k] = A.map50(counts) - B.map50(counts)

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "a": float(a0), "b": float(b0), "delta": float(a0 - b0),
        "ci95_low": float(lo), "ci95_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "frac_a_better": float((deltas > 0).mean()),
        "significant": bool(lo > 0 or hi < 0),
        "n_images": n, "n_boot": n_boot, "iou_idx": iou_idx,
    }
