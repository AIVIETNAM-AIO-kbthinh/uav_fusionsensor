"""Phân tầng tập test theo điều kiện vận hành (RQ3).

Ba chiều phân tầng, tất cả đều dựa trên đại lượng ĐO ĐƯỢC, không dựa trên nhãn
thủ công nào:

* `illum`  — độ chiếu sáng cảnh, từ data/dronevehicle/meta/{split}_illum.csv.
             Xem DATA_REPORT.md mục 2.4 về vì sao không dùng nhãn day/night.
* `size`   — diện tích OBB theo chuẩn COCO: small <32², medium 32²–96², large >96².
             Phân tầng theo TỪNG BOX chứ không theo ảnh.
* `class`  — 5 lớp của DroneVehicle (car chiếm 85,8%).

Ngoài ra hàm `illum_deciles` chia theo decile để vẽ đường cong Δ mAP theo độ
chiếu sáng — đây mới là báo cáo chính của RQ3, ba bin chỉ để tóm tắt.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

COCO_SIZE_BINS = [("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, np.inf)]


def load_illum(meta_csv: str | Path) -> dict[str, dict]:
    """id -> {"illum_proxy": float, "illum_bin": str}."""
    out = {}
    for r in csv.DictReader(open(meta_csv, encoding="utf-8")):
        out[r["id"]] = {"illum_proxy": float(r["illum_proxy"]), "illum_bin": r["illum_bin"]}
    return out


def image_groups_by_illum(ids, illum: dict) -> dict[str, list[str]]:
    """Nhóm ảnh theo bin chiếu sáng."""
    g: dict[str, list[str]] = {}
    for i in ids:
        b = illum.get(i, {}).get("illum_bin", "unknown")
        g.setdefault(b, []).append(i)
    return g


def illum_deciles(ids, illum: dict, n: int = 10) -> dict[str, list[str]]:
    """Chia ảnh thành n nhóm đều theo `illum_proxy` (dùng vẽ đường cong RQ3)."""
    vals = np.array([illum.get(i, {}).get("illum_proxy", np.nan) for i in ids], dtype=float)
    ok = ~np.isnan(vals)
    ids_ok = np.array(list(ids))[ok]
    vals_ok = vals[ok]
    edges = np.quantile(vals_ok, np.linspace(0, 1, n + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    g = {}
    for k in range(n):
        m = (vals_ok >= edges[k]) & (vals_ok < edges[k + 1])
        g[f"d{k + 1:02d}"] = list(ids_ok[m])
    return g


def poly_area(polys: np.ndarray) -> np.ndarray:
    """Diện tích đa giác (công thức shoelace) cho (N,8)."""
    p = np.asarray(polys, dtype=np.float64).reshape(-1, 4, 2)
    x, y = p[..., 0], p[..., 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))


def size_bin(areas: np.ndarray) -> np.ndarray:
    """Gán nhãn bin kích thước COCO cho từng box."""
    out = np.empty(len(areas), dtype=object)
    for name, lo, hi in COCO_SIZE_BINS:
        out[(areas >= lo) & (areas < hi)] = name
    return out


def strata_spec(split_ids, illum: dict, n_deciles: int = 10) -> dict[str, dict[str, list[str]]]:
    """Toàn bộ nhóm ảnh cần đánh giá.

    Returns:
        {"illum": {bin: [ids]}, "illum_decile": {d01: [ids], ...}, "all": {"all": [ids]}}
    """
    ids = list(split_ids)
    return {
        "all": {"all": ids},
        "illum": image_groups_by_illum(ids, illum),
        "illum_decile": illum_deciles(ids, illum, n_deciles),
    }
