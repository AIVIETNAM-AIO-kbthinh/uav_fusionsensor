"""Test chỉ số đánh giá: IoU xoay, mAP, bootstrap, WBF."""

import numpy as np
import pytest

from src.eval import metrics as M
from src.eval.bootstrap import BootstrapData, bootstrap_delta
from src.fusion.wbf_obb import poly_to_xywhr, wbf_obb, xywhr_to_poly


def rect(cx, cy, w, h, deg=0.0):
    r = np.deg2rad(deg)
    dx = np.array([-w / 2, w / 2, w / 2, -w / 2])
    dy = np.array([-h / 2, -h / 2, h / 2, h / 2])
    x = cx + dx * np.cos(r) - dy * np.sin(r)
    y = cy + dx * np.sin(r) + dy * np.cos(r)
    return np.stack([x, y], 1).reshape(8)


# --------------------------------------------------------------------- IoU

def test_iou_self_is_one():
    """IoU của một box với chính nó phải bằng 1,0 CHÍNH XÁC.

    Đây là lý do không dùng ProbIoU: ProbIoU cho ~0,9995, đủ ở ngưỡng 0,5 nhưng
    sai ở các ngưỡng cao của mAP50-95.
    """
    a = np.stack([rect(50, 50, 30, 10, 25.0), rect(10, 80, 20, 20, 0.0)])
    iou = M.rotated_iou_matrix(a, a)
    assert np.allclose(np.diag(iou), 1.0, atol=1e-4), np.diag(iou)


def test_iou_disjoint_is_zero():
    a = np.stack([rect(10, 10, 10, 10)])
    b = np.stack([rect(500, 500, 10, 10)])
    assert M.rotated_iou_matrix(a, b)[0, 0] == 0.0


def test_iou_known_value():
    """Hai hình vuông 10x10 lệch nhau 5px theo x -> giao 50, hợp 150, IoU = 1/3."""
    a = np.stack([rect(0, 0, 10, 10)])
    b = np.stack([rect(5, 0, 10, 10)])
    assert abs(M.rotated_iou_matrix(a, b)[0, 0] - 1 / 3) < 1e-3


def test_iou_rotation_90_equivalent():
    """Xoay hình vuông 90° là chính nó."""
    a = np.stack([rect(0, 0, 10, 10, 0)])
    b = np.stack([rect(0, 0, 10, 10, 90)])
    assert M.rotated_iou_matrix(a, b)[0, 0] > 0.99


# --------------------------------------------------------------------- mAP

def test_map_perfect_prediction():
    """Dự đoán = GT thì mAP phải bằng 1,0 ở MỌI ngưỡng IoU (sanity check plan 7.3)."""
    gt = np.stack([rect(20, 20, 10, 6, 15.0), rect(60, 40, 12, 8, -30.0)])
    gt_cls = np.array([0, 1])
    tp = M.match_image(gt, gt_cls, gt, gt_cls)
    assert tp.all()
    res = M.evaluate(tp, np.array([0.9, 0.8]), gt_cls, gt_cls, nc=2)
    assert abs(res["map50"] - 1.0) < 1e-6, res
    assert abs(res["map5095"] - 1.0) < 1e-6, res


def test_map_empty_prediction():
    gt_cls = np.array([0, 1])
    res = M.evaluate(np.zeros((0, 10), bool), np.zeros(0), np.zeros(0, int), gt_cls, nc=2)
    assert res["map50"] == 0.0 and res["map5095"] == 0.0


def test_map_all_false_positives():
    gt = np.stack([rect(20, 20, 10, 6)])
    pred = np.stack([rect(300, 300, 10, 6)])
    tp = M.match_image(pred, np.array([0]), gt, np.array([0]))
    assert not tp.any()
    res = M.evaluate(tp, np.array([0.9]), np.array([0]), np.array([0]), nc=1)
    assert res["map50"] == 0.0


def test_match_one_gt_one_prediction_only():
    """Hai dự đoán trùng lên một GT: chỉ cái tự tin hơn là TP, cái kia là FP."""
    gt = np.stack([rect(20, 20, 10, 10)])
    pred = np.stack([rect(20, 20, 10, 10), rect(21, 20, 10, 10)])
    tp = M.match_image(pred, np.array([0, 0]), gt, np.array([0]))
    assert tp[0, 0] and not tp[1, 0]


def test_wrong_class_not_matched():
    gt = np.stack([rect(20, 20, 10, 10)])
    pred = np.stack([rect(20, 20, 10, 10)])
    tp = M.match_image(pred, np.array([1]), gt, np.array([0]))
    assert not tp.any()


# --------------------------------------------------------------------- bootstrap

def _fake_per_image(n_img=40, seed=0, quality=0.8):
    rng = np.random.default_rng(seed)
    per = {}
    for i in range(n_img):
        k = rng.integers(1, 6)
        tp = (rng.random((k, 10)) < quality)
        tp = np.sort(tp, axis=1)[:, ::-1]        # TP ở ngưỡng cao thì cũng TP ở ngưỡng thấp
        per[f"img{i}"] = {
            "tp": tp, "conf": np.sort(rng.random(k))[::-1],
            "pred_cls": rng.integers(0, 2, k), "gt_cls": rng.integers(0, 2, k),
        }
    return per


def test_bootstrap_weighted_matches_naive():
    """Công thức có trọng số phải cho ĐÚNG kết quả như cách nhân bản vật lý."""
    per = _fake_per_image(20)
    ids = list(per)
    bd = BootstrapData(per, ids, nc=2, iou_idx=0)

    rng = np.random.default_rng(1)
    for _ in range(3):
        pick_idx = rng.integers(0, len(ids), len(ids))
        counts = np.bincount(pick_idx, minlength=len(ids)).astype(float)
        w = bd.map50(counts)

        # cách ngây thơ: nhân bản vật lý rồi tính lại
        dup = {}
        for n, c in enumerate(counts.astype(int)):
            for j in range(c):
                dup[f"{ids[n]}__{j}"] = per[ids[n]]
        bd2 = BootstrapData(dup, list(dup), nc=2, iou_idx=0)
        naive = bd2.map50(np.ones(len(dup)))
        assert abs(w - naive) < 1e-9, (w, naive)


def test_bootstrap_identical_configs_ci_contains_zero():
    per = _fake_per_image(30)
    r = bootstrap_delta(per, per, list(per), nc=2, n_boot=200, seed=0)
    assert r["delta"] == 0.0
    assert r["ci95_low"] <= 0 <= r["ci95_high"]
    assert not r["significant"]


def test_bootstrap_detects_clear_difference():
    good = _fake_per_image(60, seed=2, quality=0.95)
    bad = {k: {**v, "tp": v["tp"] & (np.random.default_rng(3).random(v["tp"].shape) < 0.3)}
           for k, v in good.items()}
    r = bootstrap_delta(good, bad, list(good), nc=2, n_boot=300, seed=0)
    assert r["delta"] > 0
    assert r["significant"], r


# --------------------------------------------------------------------- WBF

def test_wbf_identity():
    """WBF hai bản sao giống hệt nhau phải trả về chính nó (plan 7.3)."""
    polys = np.stack([rect(50, 50, 20, 10, 20.0), rect(120, 80, 16, 8, -40.0)])
    scores = np.array([0.9, 0.7])
    labels = np.array([0, 1])
    p, s, l = wbf_obb([polys, polys.copy()], [scores, scores.copy()], [labels, labels.copy()],
                      iou_thr=0.55)
    assert len(p) == 2, len(p)
    order = np.argsort(-s)
    p, s, l = p[order], s[order], l[order]
    iou = M.rotated_iou_matrix(p, polys)
    assert iou.max(axis=1).min() > 0.97, iou
    assert np.allclose(np.sort(s)[::-1], [0.9, 0.7], atol=1e-6)


def test_wbf_single_model_penalised():
    """Box chỉ một mô hình nhìn thấy bị phạt điểm theo tỉ lệ mô hình đồng ý."""
    a = np.stack([rect(50, 50, 20, 10)])
    b = np.zeros((0, 8))
    p, s, l = wbf_obb([a, b], [np.array([0.8])], [np.array([0])], iou_thr=0.55) \
        if False else wbf_obb([a, b], [np.array([0.8]), np.zeros(0)],
                              [np.array([0]), np.zeros(0, int)], iou_thr=0.55)
    assert len(p) == 1
    assert abs(s[0] - 0.8 * 0.5) < 1e-6, s


def test_wbf_merges_overlapping():
    a = np.stack([rect(50, 50, 20, 10)])
    b = np.stack([rect(52, 50, 20, 10)])
    p, s, l = wbf_obb([a, b], [np.array([0.9]), np.array([0.9])],
                      [np.array([0]), np.array([0])], iou_thr=0.5)
    assert len(p) == 1
    cx = poly_to_xywhr(p)[0, 0]
    assert 50 <= cx <= 52, cx


def test_xywhr_poly_roundtrip():
    b = np.array([[50.0, 40.0, 20.0, 10.0, 0.3]])
    p = xywhr_to_poly(b)
    b2 = poly_to_xywhr(p)
    assert abs(b2[0, 0] - 50) < 1e-3 and abs(b2[0, 1] - 40) < 1e-3
    assert abs(sorted(b2[0, 2:4])[1] - 20) < 1e-3
