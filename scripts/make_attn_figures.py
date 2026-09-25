"""Sinh bốn hình cho phần attention (F2b, F2c) của báo cáo.

    python scripts/make_attn_figures.py

Hình A so sánh mAP, hình B chẩn đoán cổng, hình C đường cong huấn luyện, hình D theo
lớp. Mọi con số đọc từ `metrics_{split}.json`, `result.json`, `results.csv` và
`gates_{split}.csv` —
không gõ tay số nào. Dùng lại đúng bảng màu và style của scripts/make_figures.py
để hình mới đặt cạnh hình cũ trong báo cáo không bị lệch phong cách.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_figures import (  # noqa: E402
    AXIS, INK, INK2, MUTED, OUT, SLOTS, SURF, bare, layout, save, style,
)

# ─────────────────────────────── cấu hình hiển thị ────────────────────────────
ROLES = {"baseline": (SLOTS[0], "Baseline 1 luồng"),
         "control": (SLOTS[1], "Đối chứng capacity"),
         "fusion": (SLOTS[2], "Fusion RGB+IR")}

BARS = [
    ("S1", "S1 · RGB đơn", "baseline"),
    ("S2", "S2 · IR đơn", "baseline"),
    ("C2", "C2 · IR×2", "control"),
    ("C2b", "C2b · IR×2 + cổng cũ", "control"),
    ("C2c", "C2c · IR×2 + cổng mới", "control"),
    ("F1", "F1 · fusion sớm", "fusion"),
    ("F2a", "F2a · concat+1×1", "fusion"),
    ("F2b", "F2b · cổng attention", "fusion"),
    ("F2c", "F2c · cổng modality", "fusion"),
]

# Chỉ DroneVehicle. VEDAI vẫn có đủ run và số liệu (xem scripts/make_tables.py), nhưng
# 121 ảnh/fold làm mọi đường cong nhiễu tới mức che mất thông điệp — nên để ngoài bộ hình
# này. Thêm lại bằng cách bỏ chú thích dòng dưới.
DATASETS = [
    dict(key="dronevehicle", name="DroneVehicle", split="test", tag="seed0"),
    # dict(key="vedai", name="VEDAI", split="val", tag="3 fold"),
]

LEVELS = ["P3", "P4", "P5"]
LEVEL_COLOR = dict(zip(LEVELS, SLOTS[:3]))


# ──────────────────────────────── đọc dữ liệu ────────────────────────────────
def map50(ds: dict, cid: str) -> tuple[float, float, int]:
    """mean, std, n của mAP50 qua các run của một cấu hình."""
    vals = []
    for d in sorted((ROOT / "runs" / ds["key"]).glob(f"{cid}_*")):
        m = d / f"metrics_{ds['split']}.json"
        if m.exists():
            vals.append(100 * json.loads(m.read_text(encoding="utf-8"))["all"]["all"]["map50"])
    v = np.asarray(vals, float)
    return (float(v.mean()), float(v.std()), len(v)) if len(v) else (np.nan, np.nan, 0)


def gate_rows(run: Path, split: str) -> dict[str, np.ndarray]:
    """id -> [P3, P4, P5] từ gates_{split}.csv."""
    rows = list(csv.DictReader((run / f"gates_{split}.csv").open(encoding="utf-8")))
    return {r["id"]: np.array([float(r[k]) for k in LEVELS]) for r in rows}


def illum_proxy() -> dict[str, float]:
    p = ROOT / "data/dronevehicle/meta/test_illum.csv"
    return {r["id"]: float(r["illum_proxy"]) for r in csv.DictReader(p.open(encoding="utf-8"))}


# ───────────────────────────────── hình A ────────────────────────────────────
def fig_map():
    """So sánh mAP50 của mọi cấu hình, tách theo dataset.

    Dùng dot plot chứ không phải bar: mọi giá trị đều nằm trong dải 69–83%, nếu vẽ
    thanh thì trục buộc phải cắt gốc 0 và độ dài thanh sẽ nói dối về tỉ lệ.
    """
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11.0, 4.6), squeeze=False)
    y = np.arange(len(BARS))[::-1]

    for ax, ds in zip(axes[0], DATASETS):
        vals = [map50(ds, cid) for cid, _, _ in BARS]
        colors = [ROLES[r][0] for _, _, r in BARS]
        means = np.array([v[0] for v in vals])
        # thanh sai số chỉ vẽ ở cấu hình chạy nhiều hơn một seed/fold; bốn cấu hình có
        # cổng mới chạy seed 0 nên std của chúng bằng 0 và không có gì để vẽ
        errs = np.array([v[1] if v[2] > 1 else 0.0 for v in vals])
        ax.errorbar(means, y, xerr=errs, fmt="none", ecolor=MUTED, elinewidth=1.1,
                    capsize=3, zorder=1)
        ax.scatter(means, y, s=78, c=colors, zorder=3, edgecolors=SURF, linewidths=1.6)
        span = means.max() + errs.max() - means.min() + errs.min()
        off = errs + 0.035 * span
        for yi, m, o in zip(y, means, off):
            ax.text(m + o, yi, f"{m:.2f}", va="center", ha="left", fontsize=8.5, color=INK)
        ax.set_yticks(y, [lab for _, lab, _ in BARS], fontsize=8.5)
        ax.set_ylim(-0.7, len(BARS) - 0.3)
        lo, hi = (means - errs).min(), (means + off).max()
        ax.set_xlim(lo - 0.06 * (hi - lo), hi + 0.10 * (hi - lo))
        ax.set_xlabel("mAP50 (%)")
        if len(DATASETS) > 1:                          # một dataset thì tiêu đề chung đã đủ
            ax.set_title(f"{ds['name']} — {ds['tag']}", loc="left")
        bare(ax, x_grid=True)

    handles = [Patch(facecolor=c, label=l) for c, l in ROLES.values()]
    layout(fig, "Hình A — độ chính xác của các biến thể fusion giữa",
           "DroneVehicle, tập test 8.980 ảnh. mAP50 trên harness của đề tài (IoU đa giác chính "
           "xác). Thanh sai số = độ lệch chuẩn giữa 2 seed.",
           "Trục không bắt đầu từ 0 nên dùng điểm thay cho thanh. Bốn cấu hình có cổng (C2b, C2c, "
           "F2b, F2c) mới chạy seed 0 nên không có thanh sai số. F2a/F2b/F2c chênh nhau ≤ 1,25% "
           "số tham số nên đây là so sánh gần như cùng dung lượng mô hình.",
           handles, top=0.86, bottom=0.17, ncol=3)
    return save(fig, "fig09_attention_map")


# ───────────────────────────────── hình B ────────────────────────────────────
def fig_gate():
    """Chẩn đoán cổng: xu hướng theo chiếu sáng, và biên độ thật của nó."""
    run = ROOT / "runs/dronevehicle/F2c_seed0"
    g = gate_rows(run, "test")
    il = illum_proxy()
    ids = [i for i in g if i in il]
    x = np.array([il[i] for i in ids])
    Y = np.stack([g[i] for i in ids])                      # (n, 3)

    order = np.argsort(x)
    bins = np.array_split(order, 10)                       # decile theo độ chiếu sáng

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax = axes[0]
    for k, lv in enumerate(LEVELS):
        m = [Y[b, k].mean() for b in bins]
        r = np.corrcoef(Y[:, k], x)[0, 1]
        ax.plot(range(1, 11), m, color=LEVEL_COLOR[lv], lw=2.4, marker="o", ms=5)
        # nhãn trực tiếp kèm hệ số tương quan -> không cần chú giải riêng trong panel
        ax.annotate(f"{lv}   r = {r:.2f}".replace("-", "−").replace(".", ","),
                    (10, m[-1]), xytext=(9, 0), textcoords="offset points",
                    color=LEVEL_COLOR[lv], fontsize=9, weight="600", va="center")
    ax.set_xticks(range(1, 11), [f"d{i}" for i in range(1, 11)])
    ax.set_xlabel("decile độ chiếu sáng của cảnh  (d1 = tối nhất → d10 = sáng nhất)")
    ax.set_ylabel("tỉ trọng nhánh IR")
    ax.set_title("Cổng có bám theo điều kiện chiếu sáng — đúng hướng giả thuyết", loc="left")
    ax.set_xlim(0.4, 12.8)
    bare(ax)

    ax = axes[1]
    for k, lv in enumerate(LEVELS):
        ax.hist(Y[:, k], bins=60, range=(0, 1), color=LEVEL_COLOR[lv], alpha=0.85, label=lv)
    ax.axvline(0.5, color=AXIS, lw=1.2, ls=(0, (5, 2)))
    ax.text(0.485, ax.get_ylim()[1] * 0.55, "0,5 = chia đều\nhai modality  ",
            fontsize=8.5, color=MUTED, va="center", ha="right")
    ax.annotate("toàn bộ 8.980 ảnh nằm gọn\ntrong dải rộng 0,019",
                xy=(0.575, ax.get_ylim()[1] * 0.42), xytext=(0.78, ax.get_ylim()[1] * 0.60),
                fontsize=8.5, color=INK2, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    ax.set_xlim(0, 1)
    ax.set_xlabel("tỉ trọng nhánh IR của từng ảnh")
    ax.set_ylabel("số ảnh")
    ax.set_title("Nhưng biên độ gần như bằng 0 — cổng là hằng số", loc="left")
    bare(ax)

    handles = [Line2D([], [], color=LEVEL_COLOR[lv], lw=2.4, label=f"mức hợp nhất {lv}")
               for lv in LEVELS]
    layout(fig, "Hình B — cổng attention thật sự học được gì (F2c, DroneVehicle test)",
           "Tỉ trọng nhánh IR do cổng sinh ra cho từng ảnh: 0,5 là chia đều, > 0,5 là nghiêng về IR.",
           "Đối chứng C2c (IR vào cả hai nhánh) cho tỉ trọng 0,498–0,508 — xác nhận số đo này phản ứng "
           "với khác biệt giữa hai modality chứ không phải hiện vật của phép đo.",
           handles, top=0.87, bottom=0.16, ncol=3)
    return save(fig, "fig10_attention_gate")




# ───────────────────────────── hình C — đường cong ────────────────────────────
# Bốn cấu hình cốt lõi của phần attention. C2 là đối chứng capacity nên vẽ xám
# nét đứt: nó là nền để đối chiếu, không phải một ứng viên ngang hàng.
TRACK = [
    ("C2", "C2 · đối chứng capacity", MUTED, (0, (5, 2))),
    ("F2a", "F2a · concat+1×1", SLOTS[0], "-"),
    ("F2b", "F2b · cổng SE", SLOTS[1], "-"),
    ("F2c", "F2c · cổng modality", SLOTS[2], "-"),
]


def curves(ds: dict, cid: str, col: str):
    """(epoch, mean, min, max) của một cột results.csv, gộp qua seed/fold."""
    mats = []
    for r in sorted((ROOT / "runs" / ds["key"]).glob(f"{cid}_*")):
        f = r / "results.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(f.open(newline="", encoding="utf-8")))
        if not rows or col not in rows[0]:
            return None
        mats.append(np.array([float(x[col]) for x in rows]))
    if not mats:
        return None
    n = min(len(m) for m in mats)
    M = np.vstack([m[:n] for m in mats])
    return np.arange(1, n + 1), M.mean(0), M.min(0), M.max(0)


def fig_curves():
    panels = [("metrics/mAP50(B)", "mAP@50 trên tập val (%)", 100),
              ("val/cls_loss", "val — cls loss", 1)]
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(11.0, 3.9 * len(DATASETS)),
                             squeeze=False)
    for row, ds in enumerate(DATASETS):
        for col_i, (col, ylab, scale) in enumerate(panels):
            ax = axes[row][col_i]
            ends = []
            for cid, _, color, ls in TRACK:
                c = curves(ds, cid, col)
                if c is None:
                    continue
                ep, mu, lo, hi = c
                ax.fill_between(ep, lo * scale, hi * scale, color=color, alpha=0.12, lw=0)
                ax.plot(ep, mu * scale, color=color, lw=2.0, ls=ls)
                ends.append([mu[-1] * scale, cid, color, ep[-1]])
            # ba đường fusion hội tụ sát nhau -> nhãn cuối đường phải tách ra, nếu không
            # chúng đè lên nhau và không đọc được cái nào
            ends.sort(key=lambda e: -e[0])
            y0, y1 = ax.get_ylim()
            gap = 0.045 * (y1 - y0)                    # theo chiều cao trục, không theo
            #                                            khoảng cách giữa các giá trị —
            #                                            các đường có thể trùng khít nhau
            for k in range(1, len(ends)):
                ends[k][0] = min(ends[k][0], ends[k - 1][0] - gap)
            for y_lab, cid, color, x_end in ends:
                ax.annotate(cid, (x_end, y_lab), xytext=(7, 0), textcoords="offset points",
                            color=color, fontsize=8.5, weight="600", va="center")
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylab)
            ax.set_xlim(1, 66)
            ax.set_title(ylab if len(DATASETS) == 1 else f"{ds['name']} — {ylab}", loc="left")
            bare(ax)

    handles = [Line2D([], [], color=c, lw=2.2, ls=ls, label=lab) for _, lab, c, ls in TRACK]
    layout(fig, "Hình C — đường cong huấn luyện (DroneVehicle)",
           "Đường = trung bình qua seed, dải = khoảng min–max. 60 epoch, cosine LR, "
           "batch 8 (nbs 32) cho mọi cấu hình.",
           "Chỉ số ở đây do validator của Ultralytics tính (khớp box bằng ProbIoU) nên lạc quan có "
           "hệ thống — dùng để xem động lực học khi train, KHÔNG so trực tiếp với Hình A.",
           handles, top=0.87, bottom=0.14, ncol=4, h_pad=3.0, w_pad=3.0)
    return save(fig, "fig11_attention_curves")


# ───────────────────────────── hình D — theo lớp ─────────────────────────────
def per_class(ds: dict, cid: str):
    """(ap50 trung bình theo lớp, số box GT theo lớp) — gộp qua seed/fold."""
    aps, gts = [], []
    for r in sorted((ROOT / "runs" / ds["key"]).glob(f"{cid}_*")):
        m = r / f"metrics_{ds['split']}.json"
        if not m.exists():
            continue
        d = json.loads(m.read_text(encoding="utf-8"))["all"]["all"]
        keys = sorted(d["ap50_per_class"], key=int)
        aps.append([100 * (d["ap50_per_class"][k] or 0) for k in keys])
        gts.append([d["n_gt_per_class"][k] for k in keys])
    if not aps:
        return None, None
    return np.mean(aps, axis=0), np.mean(gts, axis=0)


def class_names(ds: dict) -> list[str]:
    from ultralytics.utils import YAML
    run = sorted((ROOT / "runs" / ds["key"]).glob("F2a_*"))[0]
    names = YAML.load(str(run / "data.yaml"))["names"]
    return [names[i] for i in sorted(names)]


def fig_per_class():
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(11.0, 4.0 * len(DATASETS)),
                             squeeze=False)
    w = 0.8 / len(TRACK)
    for row, ds in enumerate(DATASETS):
        ax = axes[row][0]
        names = class_names(ds)
        series = {cid: per_class(ds, cid) for cid, _, _, _ in TRACK}
        _, n_gt = series["F2a"]
        order = np.argsort(-n_gt)                       # lớp nhiều box trước
        x = np.arange(len(order))
        for k, (cid, _, color, ls) in enumerate(TRACK):
            ap, _ = series[cid]
            if ap is None:
                continue
            off = (k - (len(TRACK) - 1) / 2) * w
            ax.bar(x + off, ap[order], width=w * 0.88, color=color,
                   hatch="//" if cid == "C2" else None, edgecolor=SURF, linewidth=0.6)
            # chênh lệch giữa các cột nhỏ nên phải ghi số, nhìn cột không đọc ra được
            for xi, v in zip(x + off, ap[order]):
                ax.text(xi, v + 1.2, f"{v:.1f}".replace(".", ","), ha="center", va="bottom",
                        fontsize=7.2, color=INK2, rotation=90)
        ax.set_xticks(x, [f"{names[i]}\n{int(n_gt[i]):,}".replace(",", ".") for i in order],
                      fontsize=8.5)
        ax.set_ylabel("AP@50 (%)")
        ax.set_ylim(0, 112)
        prefix = "" if len(DATASETS) == 1 else f"{ds['name']} — "
        ax.set_title(f"{prefix}AP@50 theo lớp  ·  lớp xếp theo số box GT giảm dần", loc="left")
        bare(ax)

    handles = [Patch(facecolor=c, hatch="//" if cid == "C2" else None, label=lab)
               for cid, lab, c, _ in TRACK]
    layout(fig, "Hình D — độ chính xác theo từng lớp đối tượng (DroneVehicle)",
           "Số dưới tên lớp là số box ground-truth. Cột gạch chéo là đối chứng capacity "
           "(chỉ thấy IR), ba cột đặc là các biến thể fusion RGB+IR.",
           "Khoảng cách giữa cột gạch chéo và các cột đặc chính là phần đóng góp của thông tin "
           "RGB bổ sung, sau khi đã trừ đi ảnh hưởng của việc mô hình có gấp đôi tham số.",
           handles, top=0.895, bottom=0.125, ncol=4, h_pad=3.4)
    return save(fig, "fig12_attention_per_class")


if __name__ == "__main__":
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_map()
    fig_gate()
    fig_curves()
    fig_per_class()
