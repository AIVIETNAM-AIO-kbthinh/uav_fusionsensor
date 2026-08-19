"""Late fusion (F3): Weighted Box Fusion cho oriented bounding box.

Khác NMS ở chỗ WBF không VỨT BỎ box trùng mà HỢP NHẤT chúng: toạ độ là trung bình
có trọng số theo confidence của cả cụm. Nhờ vậy hai detector nhìn thấy cùng một
xe sẽ cho một box chính xác hơn cả hai, thay vì giữ nguyên box của cái tự tin hơn.

Chi tiết cho box xoay
---------------------
Không trung bình 4 góc trực tiếp: thứ tự góc giữa hai box có thể lệch nhau nên
phép trung bình sẽ tạo ra hình méo. Thay vào đó chuyển sang (cx, cy, w, h, θ),
trung bình có trọng số từng thành phần, và xử lý riêng θ:

* θ của box xoay có chu kỳ π (xoay 180° là chính nó), và (w,h,θ) ≡ (h,w,θ±π/2).
* Ta đưa mọi box trong cụm về cùng "quy ước" với box tự tin nhất: nếu hoán đổi
  w↔h rồi xoay θ đi π/2 làm θ gần θ_ref hơn thì hoán đổi.
* Sau đó trung bình θ theo vector đơn vị góc kép (2θ) để không bị lỗi quấn vòng.

Điểm số của cụm theo Solovyev et al. (2021): trung bình có trọng số, rồi nhân với
`min(n_models_gop_mat, T) / T` để phạt các cụm chỉ một mô hình nhìn thấy.
"""

from __future__ import annotations

import numpy as np

from src.eval.metrics import rotated_iou_matrix


def poly_to_xywhr(polys: np.ndarray) -> np.ndarray:
    """(N,8) -> (N,5) dạng cx, cy, w, h, theta(rad)."""
    import cv2

    out = np.zeros((len(polys), 5), dtype=np.float64)
    for i, p in enumerate(polys):
        (cx, cy), (w, h), ang = cv2.minAreaRect(np.asarray(p, np.float32).reshape(4, 2))
        out[i] = (cx, cy, w, h, np.deg2rad(ang))
    return out


def xywhr_to_poly(b: np.ndarray) -> np.ndarray:
    """(N,5) -> (N,8)."""
    cx, cy, w, h, r = b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4]
    cos, sin = np.cos(r), np.sin(r)
    dx = np.stack([-w / 2, w / 2, w / 2, -w / 2], axis=1)
    dy = np.stack([-h / 2, -h / 2, h / 2, h / 2], axis=1)
    x = cx[:, None] + dx * cos[:, None] - dy * sin[:, None]
    y = cy[:, None] + dx * sin[:, None] + dy * cos[:, None]
    return np.stack([x, y], axis=2).reshape(-1, 8)


def _align(box: np.ndarray, ref_theta: float) -> np.ndarray:
    """Đưa (w,h,θ) về quy ước gần `ref_theta` nhất."""
    b = box.copy()
    for _ in range(2):
        d0 = abs(np.angle(np.exp(1j * (b[4] - ref_theta))))
        alt = b.copy()
        alt[2], alt[3] = b[3], b[2]
        alt[4] = b[4] + np.pi / 2
        d1 = abs(np.angle(np.exp(1j * (alt[4] - ref_theta))))
        if d1 < d0:
            b = alt
        b[4] = (b[4] + np.pi / 2) % np.pi - np.pi / 2
    return b


def wbf_obb(polys_list, scores_list, labels_list, weights=None, iou_thr: float = 0.55,
            skip_thr: float = 0.0, conf_type: str = "avg"):
    """Hợp nhất dự đoán của nhiều mô hình.

    Args:
        polys_list: list theo mô hình, mỗi phần tử (N_m, 8).
        scores_list, labels_list: tương ứng.
        weights: trọng số từng mô hình (mặc định bằng nhau).
        iou_thr: ngưỡng IoU để gộp vào một cụm.
        skip_thr: bỏ box có score thấp hơn ngưỡng này trước khi gộp.

    Returns:
        (polys (K,8), scores (K,), labels (K,))
    """
    T = len(polys_list)
    if weights is None:
        weights = np.ones(T, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights * T / weights.sum()

    polys, scores, labels, model_id = [], [], [], []
    for m in range(T):
        p = np.asarray(polys_list[m], dtype=np.float64).reshape(-1, 8)
        s = np.asarray(scores_list[m], dtype=np.float64).ravel() * weights[m]
        l = np.asarray(labels_list[m]).ravel().astype(int)
        keep = s > skip_thr
        polys.append(p[keep])
        scores.append(s[keep])
        labels.append(l[keep])
        model_id.append(np.full(keep.sum(), m, dtype=int))
    if not len(polys) or sum(len(p) for p in polys) == 0:
        return np.zeros((0, 8)), np.zeros(0), np.zeros(0, dtype=int)

    polys = np.concatenate(polys)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    model_id = np.concatenate(model_id)

    out_p, out_s, out_l = [], [], []
    for c in np.unique(labels):
        m = labels == c
        p_c, s_c, mid_c = polys[m], scores[m], model_id[m]
        order = np.argsort(-s_c)
        p_c, s_c, mid_c = p_c[order], s_c[order], mid_c[order]
        r_c = poly_to_xywhr(p_c)

        used = np.zeros(len(p_c), dtype=bool)
        for i in range(len(p_c)):
            if used[i]:
                continue
            rest = np.where(~used)[0]
            rest = rest[rest != i]
            members = [i]
            if len(rest):
                iou = rotated_iou_matrix(p_c[i:i + 1], p_c[rest])[0]
                members += list(rest[iou >= iou_thr])
            used[members] = True

            w = s_c[members]
            ref = r_c[members[0], 4]
            aligned = np.stack([_align(r_c[j], ref) for j in members])
            fused = np.empty(5)
            for k in range(4):
                fused[k] = (aligned[:, k] * w).sum() / w.sum()
            z = (np.exp(2j * aligned[:, 4]) * w).sum() / w.sum()
            fused[4] = np.angle(z) / 2

            n_models = len(np.unique(mid_c[members]))
            if conf_type == "avg":
                sc = w.mean()
            elif conf_type == "max":
                sc = w.max()
            else:
                raise ValueError(conf_type)
            sc = sc * min(n_models, T) / T

            out_p.append(fused)
            out_s.append(sc)
            out_l.append(c)

    if not out_p:
        return np.zeros((0, 8)), np.zeros(0), np.zeros(0, dtype=int)
    fused_polys = xywhr_to_poly(np.stack(out_p))
    return fused_polys, np.asarray(out_s), np.asarray(out_l, dtype=int)
