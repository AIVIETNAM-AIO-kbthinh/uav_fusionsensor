"""Sinh biểu đồ so sánh kết quả từ toàn bộ run trong `runs/`.

Mọi con số đọc từ `metrics_{split}.json`, `results.csv`, `result.json` và
`preds_{split}.npz` — không có số nào gõ tay, không đọc từ stdout.

    .venv/Scripts/python.exe scripts/make_figures.py            # tất cả hình
    .venv/Scripts/python.exe scripts/make_figures.py --only 1 4 # chỉ hình 1 và 4

Ghi chú quan trọng về tính so sánh được (plan.md mục 0.4, phát hiện #2):
`results.csv` chứa chỉ số của validator Ultralytics (khớp box bằng ProbIoU, lạc
quan có hệ thống). Các hình dùng `results.csv` (đường cong huấn luyện) VÌ VẬY chỉ
để xem động lực học khi train, KHÔNG so trực tiếp với các hình dùng
`metrics_{split}.json` (IoU đa giác chính xác của harness đề tài).
"""
from __future__ import annotations

import argparse
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

OUT = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

# ─────────────────────── bảng màu (dataviz skill, đã validate) ────────────────
SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
POS = "#2a78d6"   # cực dương của cặp phân kỳ
NEG = "#d03b3b"   # cực âm
NS = "#898781"    # không có ý nghĩa thống kê

# thứ tự slot cố định của bảng màu định danh — KHÔNG xoay vòng, không đổi thứ tự
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

CONFIGS = [
    ("S1",  "S1 · RGB đơn",            "baseline"),
    ("S2",  "S2 · IR đơn",             "baseline"),
    ("C1",  "C1 · RGB×2 (đối chứng)",  "control"),
    ("C2",  "C2 · IR×2 (đối chứng)",   "control"),
    ("F1",  "F1 · fusion sớm",         "fusion"),
    ("F2a", "F2a · fusion giữa",       "fusion"),
]
CID = [c for c, _, _ in CONFIGS]
LABEL = {c: l for c, l, _ in CONFIGS}
COLOR = {c: SLOTS[i] for i, c in enumerate(CID)}
# mã hoá phụ: hai cấu hình đối chứng vẽ nét đứt. S1 và C1 gần như trùng nhau
# (đúng như kỳ vọng của thiết kế), nếu chỉ phân biệt bằng màu thì đường sau che
# hẳn đường trước.
DASH = (0, (5, 2))
LS = {"S1": "-", "S2": "-", "C1": DASH, "C2": DASH, "F1": "-", "F2a": "-"}

# hình 8 — mỗi đường nối hai mức capacity của CÙNG một nguồn thông tin
CAPACITY_PAIRS = [
    ("S1", "C1", "chỉ RGB", SLOTS[0]),
    ("S2", "C2", "chỉ IR", SLOTS[1]),
    ("F1", "F2a", "RGB + IR (fusion)", SLOTS[2]),
]

DATASETS = [
    dict(key="dronevehicle", name="DroneVehicle", split="test",
         sub="test 8.980 ảnh · 2 seed", tag="seed"),
    dict(key="vedai", name="VEDAI", split="val",
         sub="3 fold (01/03/05) · 121 ảnh/fold", tag="fold"),
]

COMPARISONS = [
    ("F2a", "C2", "RQ1 trung thực — fusion vs đối chứng capacity"),
    ("F2a", "S2", "RQ1 ngây thơ — fusion vs modality đơn"),
    ("F2a", "F1", "RQ2 — fusion giữa vs fusion sớm"),
    ("C2", "S2", "chẩn đoán — lợi ích thuần từ capacity (IR)"),
    ("C1", "S1", "chẩn đoán — lợi ích thuần từ capacity (RGB)"),
    ("F2a", "F3", "RQ2 — fusion giữa vs fusion muộn"),
    ("F2b", "F1", "RQ2 — two-stream + cổng vs early fusion 4 kênh"),
    ("F2b", "C2b", "RQ1 trung thực (attn) — F2b vs đối chứng capacity của chính nó"),
    ("F2b", "F2a", "RQ2 — attention gate vs concat+1×1"),
    ("C2b", "C2", "chẩn đoán — cổng attn khi KHÔNG có tín hiệu bổ sung"),
]

FOOT_HARNESS = ("Chỉ số của harness đề tài: IoU đa giác chính xác, AP nội suy 101 điểm "
                "(pycocotools). Không so trực tiếp với số của model.val() Ultralytics.")


# ─────────────────────────────────── nền chung ────────────────────────────────
def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlesize": 10.5, "axes.titleweight": "600", "axes.titlecolor": INK,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def bare(ax, x_grid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.xaxis.grid(x_grid)
    ax.yaxis.grid(not x_grid)


def layout(fig, title, subtitle, note, handles, top=0.90, bottom=0.11, ncol=6, **pad):
    """Tiêu đề / chú giải / ghi chú đặt trong toạ độ figure, vùng vẽ co lại vừa đủ.

    Đặt mọi thứ BÊN TRONG figure (thay vì y âm / y > 1) để `bbox_inches="tight"`
    không đẩy tiêu đề đè lên tiêu đề của từng panel.
    """
    fig.tight_layout(rect=(0.0, bottom, 1.0, top), **pad)
    h = fig.get_size_inches()[1]
    fig.text(0.008, 0.995, title, ha="left", va="top", fontsize=13.5, weight="600", color=INK)
    fig.text(0.008, 0.995 - 0.30 / h, subtitle, ha="left", va="top", fontsize=9, color=INK2)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=ncol,
                   bbox_to_anchor=(0.5, 0.24 / h))
    fig.text(0.008, 0.16 / h, note, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def bar_handles():
    return [Patch(facecolor=COLOR[c], label=LABEL[c]) for c in CID]


def line_handles():
    return [Line2D([], [], color=COLOR[c], lw=2.4, ls=LS[c], label=LABEL[c]) for c in CID]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  -> {p.relative_to(ROOT)}")
    return p


# ──────────────────────────────── đọc dữ liệu ─────────────────────────────────
def project(ds) -> Path:
    return ROOT / "runs" / ds["key"]


def run_dirs(ds, cid) -> list[Path]:
    return sorted(d for d in project(ds).glob(f"{cid}_*") if (d / "result.json").exists())


def metrics(run: Path, split: str) -> dict | None:
    p = run / f"metrics_{split}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


_NAMES: dict[str, list[str]] = {}


def class_names(ds) -> list[str]:
    if ds["key"] not in _NAMES:
        from ultralytics.utils import YAML
        names = YAML.load(str(run_dirs(ds, "S1")[0] / "data.yaml"))["names"]
        _NAMES[ds["key"]] = [names[i] for i in sorted(names)]
    return _NAMES[ds["key"]]


def stat(vals):
    """mean, std, n — bỏ NaN."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    return (float(v.mean()), float(v.std()), len(v)) if len(v) else (np.nan, np.nan, 0)


def series(ds, dim, group, key):
    """{cid: (mean, std, n, [giá trị từng run])} cho một ô của metrics json."""
    out = {}
    for cid in CID:
        vals = []
        for r in run_dirs(ds, cid):
            m = metrics(r, ds["split"])
            if m and dim in m and group in m.get(dim, {}) and key in m[dim][group]:
                vals.append(m[dim][group][key])
        out[cid] = (*stat(vals), vals)
    return out


def read_csv_run(run: Path) -> dict[str, np.ndarray]:
    with open(run / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k.strip(): np.array([float(r[k]) for r in rows]) for k in rows[0]}


def curves(ds, cid, col):
    """(epochs, mean, min, max) của một cột results.csv, gộp qua seed/fold."""
    mats = []
    for r in run_dirs(ds, cid):
        d = read_csv_run(r)
        if col not in d:
            return None
        mats.append(d[col])
    n = min(len(m) for m in mats)
    M = np.vstack([m[:n] for m in mats])
    return np.arange(1, n + 1), M.mean(0), M.min(0), M.max(0)


def params_of(ds, cid) -> float:
    r = json.loads((run_dirs(ds, cid)[0] / "result.json").read_text(encoding="utf-8"))
    return r["params"] / 1e6


# ───────────────────── precision / recall / F1 theo lớp ───────────────────────
CACHE = OUT / "_prf_cache.json"
THRS = np.linspace(0.005, 0.995, 199)


def prf_run(run: Path, split: str, nc: int) -> dict:
    """P/R/F1 theo lớp ở IoU 0,5, quét theo ngưỡng confidence.

    Ngưỡng báo cáo là ngưỡng cực đại hoá macro-F1 của chính run đó (cùng cách
    Ultralytics chọn), nhưng khớp box bằng IoU đa giác chính xác của harness.
    """
    from src.eval.harness import load_dump, match_all

    per_image = match_all(load_dump(run / f"preds_{split}.npz"))
    tp, conf, pcls, gcls = [], [], [], []
    for r in per_image.values():
        if len(r["conf"]):
            tp.append(np.asarray(r["tp"])[:, 0])
            conf.append(r["conf"])
            pcls.append(r["pred_cls"])
        if len(r["gt_cls"]):
            gcls.append(r["gt_cls"])
    tp = np.concatenate(tp) if tp else np.zeros(0, bool)
    conf = np.concatenate(conf) if conf else np.zeros(0)
    pcls = np.concatenate(pcls) if pcls else np.zeros(0, int)
    gcls = np.concatenate(gcls) if gcls else np.zeros(0, int)
    n_gt = np.bincount(gcls.astype(int), minlength=nc)

    P = np.full((nc, len(THRS)), np.nan)
    R = np.full((nc, len(THRS)), np.nan)
    F = np.full((nc, len(THRS)), np.nan)
    for c in range(nc):
        if n_gt[c] == 0:
            continue
        m = pcls == c
        cc, tt = conf[m], tp[m]
        order = np.argsort(-cc)
        cc, tt = cc[order], tt[order]
        cum = np.concatenate([[0.0], np.cumsum(tt.astype(float))])
        k = np.searchsorted(-cc, -THRS, side="right")     # số dự đoán có conf >= thr
        tpk = cum[k]
        P[c] = np.where(k > 0, tpk / np.maximum(k, 1), 0.0)
        R[c] = tpk / n_gt[c]
        F[c] = np.where(P[c] + R[c] > 0,
                        2 * P[c] * R[c] / np.maximum(P[c] + R[c], 1e-12), 0.0)

    present = n_gt > 0
    macro = F[present].mean(axis=0) if present.any() else np.zeros(len(THRS))
    b = int(np.argmax(macro))
    return {
        "thrs": THRS.tolist(), "macro_f1": macro.tolist(),
        "best_thr": float(THRS[b]), "best_macro_f1": float(macro[b]),
        "f1": F[:, b].tolist(), "p": P[:, b].tolist(), "r": R[:, b].tolist(),
        "n_gt": n_gt.tolist(),
    }


def prf_all(force=False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() and not force else {}
    changed = False
    for ds in DATASETS:
        nc = len(class_names(ds))
        for cid in CID:
            for r in run_dirs(ds, cid):
                npz = r / f"preds_{ds['split']}.npz"
                key = f"{ds['key']}/{r.name}"
                stamp = str(int(npz.stat().st_mtime))
                if cache.get(key, {}).get("_stamp") == stamp:
                    continue
                print(f"  tính P/R/F1: {key} ...", flush=True)
                cache[key] = {**prf_run(r, ds["split"], nc), "_stamp": stamp}
                changed = True
    if changed:
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


# ═══════════════════════════════════ hình 1 ═══════════════════════════════════
def fig01(cache):
    # thang y cố định theo từng chỉ số (không theo dữ liệu) để hai dataset đọc
    # được cạnh nhau và cột luôn bắt đầu từ 0
    metrics_cols = [("map50", "mAP@50", 100.0), ("map5095", "mAP@50-95", 70.0)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for row, ds in enumerate(DATASETS):
        for col, (key, mname, ymax) in enumerate(metrics_cols):
            ax = axes[row][col]
            bare(ax)
            s = series(ds, "all", "all", key)
            x = np.arange(len(CID))
            mu = np.array([s[c][0] for c in CID]) * 100
            sd = np.array([s[c][1] for c in CID]) * 100
            ax.bar(x, mu, width=0.62, color=[COLOR[c] for c in CID], zorder=2)
            ax.errorbar(x, mu, yerr=sd, fmt="none", ecolor=INK2, elinewidth=1.1,
                        capsize=3, capthick=1.1, zorder=4)
            for i, c in enumerate(CID):                       # từng seed/fold
                v = np.array(s[c][3]) * 100
                ax.scatter(np.full(len(v), i), v, s=11, color=SURF, edgecolor=INK2,
                           linewidth=0.9, zorder=5)
                ax.text(i, mu[i] + sd[i] + ymax * 0.02, f"{mu[i]:.1f}", ha="center",
                        va="bottom", fontsize=8.5, color=INK, weight="600")
            ax.set_xticks(x, CID, color=INK2)
            ax.set_ylim(0, ymax)
            ax.set_ylabel(f"{mname} (%)")
            ax.set_title(f"{ds['name']} — {mname}  ·  {ds['sub']}", loc="left", fontsize=10)
    layout(fig, "So sánh cấu hình — mAP trên tập đánh giá",
           "Cột = trung bình qua seed (DroneVehicle) / fold (VEDAI); thanh sai số = độ lệch chuẩn; "
           "chấm trắng = từng lần chạy.",
           "So sánh để kết luận RQ1 là F2a vs C2 (cùng số tham số), không phải F2a vs S1/S2. "
           + FOOT_HARNESS,
           bar_handles(), top=0.905, bottom=0.105, h_pad=3.2, w_pad=3.0)
    return save(fig, "fig01_map_comparison")


# ═══════════════════════════════ hình 2 & 3 ═══════════════════════════════════
def _grouped_per_class(ax, labels, order, vals_by_cid, ylabel):
    x = np.arange(len(order))
    w = 0.84 / len(CID)
    for i, cid in enumerate(CID):
        pos = x + (i - (len(CID) - 1) / 2) * w
        v = np.array([vals_by_cid[cid][0][j] for j in order]) * 100
        e = np.array([vals_by_cid[cid][1][j] for j in order]) * 100
        ax.bar(pos, v, width=w * 0.86, color=COLOR[cid], zorder=2)
        ax.errorbar(pos, v, yerr=e, fmt="none", ecolor=INK2, elinewidth=0.7,
                    capsize=1.6, capthick=0.7, alpha=0.75, zorder=4)
    ax.set_xticks(x, [labels[j] for j in order], color=INK2)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    bare(ax)


def _per_class_from_metrics(ds, key):
    """{cid: (mean_per_class, std_per_class)} + n_gt trung bình mỗi lần chạy."""
    nc = len(class_names(ds))
    out, G = {}, []
    for cid in CID:
        M = []
        for r in run_dirs(ds, cid):
            m = metrics(r, ds["split"])
            if not m:
                continue
            d = m["all"]["all"]
            M.append([d[key].get(str(c), np.nan) for c in range(nc)])
            G.append([d["n_gt_per_class"].get(str(c), 0) for c in range(nc)])
        A = np.array(M, float)
        out[cid] = (np.nanmean(A, 0), np.nanstd(A, 0))
    return out, np.array(G, float).mean(0)


def _per_class_f1(ds, cache):
    nc = len(class_names(ds))
    out, G, thr = {}, [], []
    for cid in CID:
        M = []
        for r in run_dirs(ds, cid):
            e = cache[f"{ds['key']}/{r.name}"]
            M.append(e["f1"])
            G.append(e["n_gt"])
            thr.append(e["best_thr"])
        A = np.array(M, float)
        out[cid] = (np.nanmean(A, 0), np.nanstd(A, 0))
    return out, np.array(G, float).mean(0), float(np.mean(thr))


def _class_labels(ds, names, n_gt):
    # DroneVehicle: mọi run dùng chung một tập test. VEDAI: mỗi fold một tập val
    # khác nhau, nên số box GT là trung bình mỗi fold.
    unit = "box" if ds["key"] == "dronevehicle" else "box/fold"
    return [f"{names[j]}\n{int(round(n_gt[j])):,} {unit}".replace(",", ".")
            for j in range(len(names))]


def fig02(cache):
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4))
    notes = []
    for row, ds in enumerate(DATASETS):
        ax = axes[row]
        names = class_names(ds)
        vals, n_gt, thr = _per_class_f1(ds, cache)
        notes.append(f"{ds['name']}: conf≈{thr:.2f}")
        order = list(np.argsort(-n_gt))
        _grouped_per_class(ax, _class_labels(ds, names, n_gt), order, vals, "F1 @ IoU 0,5 (%)")
        ax.set_title(f"{ds['name']} — F1 theo lớp  ·  {ds['sub']}", loc="left")
    layout(fig, "F1 theo từng lớp",
           "Lớp xếp theo số box GT giảm dần. Thanh sai số = độ lệch chuẩn qua seed/fold.",
           "F1 đo ở IoU 0,5, tại ngưỡng confidence cực đại hoá macro-F1 của từng lần chạy ("
           + "; ".join(notes) + "). " + FOOT_HARNESS,
           bar_handles(), top=0.895, bottom=0.115, h_pad=3.6)
    return save(fig, "fig02_per_class_f1")


def fig03(cache):
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4))
    for row, ds in enumerate(DATASETS):
        ax = axes[row]
        names = class_names(ds)
        vals, n_gt = _per_class_from_metrics(ds, "ap50_per_class")
        order = list(np.argsort(-n_gt))
        _grouped_per_class(ax, _class_labels(ds, names, n_gt), order, vals, "AP@50 (%)")
        ax.set_title(f"{ds['name']} — AP@50 theo lớp  ·  {ds['sub']}", loc="left")
    layout(fig, "AP@50 theo từng lớp",
           "Bổ sung cho hình 2: AP không phụ thuộc việc chọn ngưỡng confidence.",
           FOOT_HARNESS, bar_handles(), top=0.895, bottom=0.115, h_pad=3.6)
    return save(fig, "fig03_per_class_ap50")


# ═══════════════════════════════════ hình 4 ═══════════════════════════════════
def fig04(cache):
    panels = [("metrics/mAP50(B)", "mAP@50 trên tập val (%)", 100),
              ("metrics/mAP50-95(B)", "mAP@50-95 trên tập val (%)", 100),
              ("train/cls_loss", "train — cls loss", 1),
              ("val/cls_loss", "val — cls loss", 1)]
    outs = []
    for ds in DATASETS:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
        for ax, (col, ylab, scale) in zip(axes.ravel(), panels):
            bare(ax)
            for cid in CID:
                c = curves(ds, cid, col)
                if c is None:
                    continue
                ep, mu, lo, hi = c
                ax.fill_between(ep, lo * scale, hi * scale, color=COLOR[cid], alpha=0.13, lw=0)
                ax.plot(ep, mu * scale, color=COLOR[cid], lw=2.0, ls=LS[cid],
                        solid_capstyle="round", dash_capstyle="round")
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylab)
            ax.set_title(ylab, loc="left")
            ax.set_xlim(1, None)
        layout(fig, f"Đường cong huấn luyện — {ds['name']}",
               f"Đường = trung bình qua {ds['tag']}; dải = khoảng min–max. 60 epoch, cosine LR, "
               "batch 8 (nbs 32) cho mọi cấu hình. Nét đứt = hai cấu hình đối chứng.",
               "Chỉ số trên hình này do validator của Ultralytics tính (khớp box bằng ProbIoU) nên "
               "lạc quan có hệ thống — chỉ dùng để xem động lực học khi train, KHÔNG so trực tiếp "
               "với hình 1–3 (IoU đa giác chính xác).",
               line_handles(), top=0.895, bottom=0.115, h_pad=3.0, w_pad=3.0)
        outs.append(save(fig, f"fig04_training_curves_{ds['key']}"))
    return outs


# ═══════════════════════════════════ hình 5 ═══════════════════════════════════
def fig05(cache):
    ds = DATASETS[0]                                   # chỉ DroneVehicle có nhãn chiếu sáng
    bins = ["lowlight", "midlight", "bright"]
    bin_lab = {"lowlight": "thiếu sáng", "midlight": "trung gian", "bright": "đủ sáng"}
    ref = metrics(run_dirs(ds, "S1")[0], ds["split"])["illum"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    bare(ax)
    x = np.arange(len(bins))
    w = 0.84 / len(CID)
    for i, cid in enumerate(CID):
        s = [series(ds, "illum", b, "map50")[cid] for b in bins]
        pos = x + (i - (len(CID) - 1) / 2) * w
        ax.bar(pos, [v[0] * 100 for v in s], width=w * 0.86, color=COLOR[cid], zorder=2)
        ax.errorbar(pos, [v[0] * 100 for v in s], yerr=[v[1] * 100 for v in s], fmt="none",
                    ecolor=INK2, elinewidth=0.7, capsize=1.6, capthick=0.7, alpha=0.75, zorder=4)
    ax.set_xticks(x, [f"{bin_lab[b]}\n{ref[b]['n_images']:,} ảnh".replace(",", ".") for b in bins],
                  color=INK2)
    ax.set_ylabel("mAP@50 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Theo tầng chiếu sáng", loc="left")

    ax = axes[1]
    bare(ax)
    dec = [f"d{i:02d}" for i in range(1, 11)]
    for cid in CID:
        mu = [series(ds, "illum_decile", d, "map50")[cid][0] * 100 for d in dec]
        ax.plot(np.arange(1, 11), mu, color=COLOR[cid], lw=2.0, ls=LS[cid], marker="o", ms=4.5,
                mfc=SURF, mew=1.4, mec=COLOR[cid], solid_capstyle="round", dash_capstyle="round")
    ax.set_xticks(np.arange(1, 11), [str(i) for i in range(1, 11)], color=INK2)
    ax.set_xlabel("decile độ sáng (1 = tối nhất → 10 = sáng nhất)")
    ax.set_ylabel("mAP@50 (%)")
    ax.set_title("Theo decile độ sáng — độ mịn cao hơn", loc="left")

    layout(fig, "RQ3 — kết quả phân tầng theo điều kiện chiếu sáng (DroneVehicle)",
           "Nếu fusion có giá trị, chênh lệch phải lớn nhất ở tầng thiếu sáng, nơi RGB kém nhất. "
           "Nét đứt = hai cấu hình đối chứng.",
           "Cảnh báo diễn giải (plan.md 3.1): GT của DroneVehicle lấy từ ảnh IR, nên ở tầng thiếu "
           "sáng mọi cấu hình có IR đều được lợi từ chính giao thức chọn GT. " + FOOT_HARNESS,
           bar_handles(), top=0.845, bottom=0.185, w_pad=3.0)
    return save(fig, "fig05_illumination_strata")


# ═══════════════════════════════════ hình 6 ═══════════════════════════════════
def _pooled_per_image(ds, cid):
    """Gộp dự đoán theo ảnh. VEDAI gộp cả 3 fold (tập val rời nhau) để đủ cỡ mẫu."""
    from src.eval.harness import load_dump, match_all
    runs = run_dirs(ds, cid)[:1] if ds["key"] == "dronevehicle" else run_dirs(ds, cid)
    out = {}
    for r in runs:
        pi = match_all(load_dump(r / f"preds_{ds['split']}.npz"))
        pref = r.name.split("_", 1)[1]
        out.update({f"{pref}/{k}": v for k, v in pi.items()})
    return out


EFFECTS = OUT / "_effect_sizes.json"


def _effect_rows(n_boot, reuse):
    """Bootstrap tốn ~15 phút; `reuse` đọc lại kết quả đã lưu nếu khớp cấu hình."""
    # Cap nao thuc su chay duoc: ca hai phia phai co run da xong. Cau hinh chua
    # train (vd F2b tren DroneVehicle) bi bo qua thay vi lam vo ca ham. CID la
    # danh sach de VE HINH (chi 6 mau trong SLOTS) nen khong dung o day.
    avail = {(d["name"], a, b) for d in DATASETS for a, b, _ in COMPARISONS
             if run_dirs(d, a) and run_dirs(d, b)}

    if reuse and EFFECTS.exists():
        rows = json.loads(EFFECTS.read_text(encoding="utf-8"))
        same_pairs = {(r["ds"], r["a"], r["b"]) for r in rows} == avail
        if same_pairs and all(r["n_boot"] == n_boot for r in rows):
            print("  dùng lại _effect_sizes.json đã có")
            return rows
        print("  _effect_sizes.json không khớp cấu hình hiện tại — tính lại")

    from src.eval.bootstrap import bootstrap_delta
    rows = []
    for ds in DATASETS:
        nc = len(class_names(ds))
        # chi nap cau hinh xuat hien trong mot cap chay duoc cua dataset nay
        need = sorted({c for (dsn, x, y) in avail if dsn == ds["name"] for c in (x, y)})
        pi = {}
        for cid in need:
            print(f"  bootstrap: nạp {ds['key']}/{cid} ...", flush=True)
            pi[cid] = _pooled_per_image(ds, cid)
        for A, B, lab in COMPARISONS:
            if (ds["name"], A, B) not in avail:
                print(f"    {ds['name']:13s} {A} vs {B}: bo qua (chua co run)")
                continue
            ids = sorted(set(pi[A]) & set(pi[B]))
            r = bootstrap_delta(pi[A], pi[B], ids, nc=nc, n_boot=n_boot, seed=0)
            # `bootstrap_delta` cũng trả về khoá "a"/"b" (giá trị mAP của từng
            # cấu hình) — đổi tên để không đè lên tên cấu hình.
            r["map_a"], r["map_b"] = r.pop("a"), r.pop("b")
            rows.append(dict(ds=ds["name"], a=A, b=B, label=lab, **r))
            print(f"    {ds['name']:13s} {A} vs {B}: {100*r['delta']:+.2f} "
                  f"[{100*r['ci95_low']:+.2f},{100*r['ci95_high']:+.2f}] p={r['p_value']:.3f}",
                  flush=True)
    EFFECTS.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return rows


def fig06(cache, n_boot=2000, reuse=False):
    rows = _effect_rows(n_boot, reuse)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    for ax, ds in zip(axes, DATASETS):
        bare(ax, x_grid=True)
        sub = [r for r in rows if r["ds"] == ds["name"]]
        y = np.arange(len(sub))[::-1]
        ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
        lo_all = min(100 * r["ci95_low"] for r in sub)
        hi_all = max(100 * r["ci95_high"] for r in sub)
        span = max(hi_all - lo_all, 1e-6)
        ax.set_xlim(lo_all - span * 0.10, hi_all + span * 0.55)
        for yi, r in zip(y, sub):
            col = NS if not r["significant"] else (POS if r["delta"] > 0 else NEG)
            lo, hi = 100 * r["ci95_low"], 100 * r["ci95_high"]
            ax.plot([lo, hi], [yi, yi], color=col, lw=2.4, solid_capstyle="round", zorder=3)
            ax.plot([r["delta"] * 100], [yi], "o", ms=8, color=col, mec=SURF, mew=1.6, zorder=4)
            ax.text(hi + span * 0.04, yi, f"{100*r['delta']:+.1f}  p={r['p_value']:.3f}",
                    va="center", fontsize=8, color=INK if r["significant"] else MUTED,
                    weight="600" if r["significant"] else "normal")
        ax.set_yticks(y, [f"{r['a']} − {r['b']}" for r in sub], color=INK)
        ax.set_ylim(-0.7, len(sub) - 0.3)
        ax.set_xlabel("Δ mAP@50 (điểm phần trăm)")
        ax.set_title(f"{ds['name']} · {ds['sub']}", loc="left")
    handles = [
        Line2D([], [], color=POS, lw=2.4, marker="o", ms=7, mec=SURF,
               label="A tốt hơn — CI 95% không chứa 0"),
        Line2D([], [], color=NEG, lw=2.4, marker="o", ms=7, mec=SURF,
               label="A kém hơn — CI 95% không chứa 0"),
        Line2D([], [], color=NS, lw=2.4, marker="o", ms=7, mec=SURF,
               label="không có ý nghĩa thống kê")]
    layout(fig, "Kích thước hiệu ứng — bootstrap ghép cặp theo ảnh (CI 95%)",
           "Mỗi dòng là hiệu số mAP@50 giữa hai cấu hình trên cùng một tập ảnh; "
           f"{n_boot} lần lấy mẫu lại.",
           "Dòng để kết luận RQ1 là F2a − C2 (đã kiểm soát capacity). CI chứa 0 nghĩa là \"không có "
           "bằng chứng về lợi ích của fusion sau khi kiểm soát capacity\" — một kết quả hợp lệ "
           "(plan.md 6.3, rủi ro R6). Bootstrap lấy mẫu lại theo ẢNH (không theo seed): "
           "DroneVehicle dùng seed 0, VEDAI gộp cả 3 fold. " + FOOT_HARNESS,
           handles, ncol=3, top=0.835, bottom=0.20, w_pad=6.0)
    return save(fig, "fig06_effect_sizes")


# ═══════════════════════════════════ hình 7 ═══════════════════════════════════
def fig07(cache):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, ds in zip(axes, DATASETS):
        bare(ax)
        for cid in CID:
            A = np.array([cache[f"{ds['key']}/{r.name}"]["macro_f1"]
                          for r in run_dirs(ds, cid)], float) * 100
            mu = A.mean(0)
            ax.fill_between(THRS, A.min(0), A.max(0), color=COLOR[cid], alpha=0.13, lw=0)
            ax.plot(THRS, mu, color=COLOR[cid], lw=2.0, ls=LS[cid], solid_capstyle="round",
                    dash_capstyle="round")
            b = int(np.argmax(mu))
            ax.plot([THRS[b]], [mu[b]], "o", ms=6, color=COLOR[cid], mec=SURF, mew=1.4, zorder=5)
        ax.set_xlabel("ngưỡng confidence")
        ax.set_ylabel("macro-F1 @ IoU 0,5 (%)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.set_title(f"{ds['name']} · {ds['sub']}", loc="left")
    layout(fig, "Đường cong F1 theo ngưỡng confidence",
           "Chấm tròn = ngưỡng tối ưu, chính là ngưỡng dùng cho hình 2. Đường = trung bình; "
           "dải = min–max qua seed/fold. Nét đứt = hai cấu hình đối chứng.",
           FOOT_HARNESS, line_handles(), top=0.845, bottom=0.185, w_pad=3.0)
    return save(fig, "fig07_f1_confidence_curve")


# ═══════════════════════════════════ hình 8 ═══════════════════════════════════
def _spread(vals, gap):
    """Đẩy các nhãn trùng nhau ra xa tối thiểu `gap`, giữ nguyên thứ tự."""
    out = np.array(vals, float)
    order = np.argsort(out)
    for k in range(1, len(order)):
        i, j = order[k], order[k - 1]
        if out[i] - out[j] < gap:
            out[i] = out[j] + gap
    return out


def fig08(cache):
    """Slope chart: chỉ có ĐÚNG HAI mức capacity, nên nối cặp đọc rõ hơn scatter.

    Độ dốc của mỗi đường = phần do tăng dung lượng mô hình (thông tin giữ nguyên).
    Khoảng cách giữa các đường = phần do nguồn thông tin.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for ax, ds in zip(axes, DATASETS):
        bare(ax)
        s = series(ds, "all", "all", "map50")
        left, right = [], []
        for lo_id, hi_id, name, col in CAPACITY_PAIRS:
            mu = [s[lo_id][0] * 100, s[hi_id][0] * 100]
            sd = [s[lo_id][1] * 100, s[hi_id][1] * 100]
            ax.plot([0, 1], mu, color=col, lw=2.2, zorder=3, solid_capstyle="round")
            ax.errorbar([0, 1], mu, yerr=sd, fmt="o", ms=8, color=col, mec=SURF, mew=1.6,
                        ecolor=INK2, elinewidth=1.0, capsize=3, zorder=4)
            left.append((mu[0], f"{lo_id}  {mu[0]:.1f}"))
            right.append((mu[1], f"{hi_id}  {mu[1]:.1f}   ({name}, Δ {mu[1] - mu[0]:+.1f})"))
        p_lo = params_of(ds, CAPACITY_PAIRS[0][0])
        p_hi = params_of(ds, CAPACITY_PAIRS[0][1])
        ax.set_xticks([0, 1], [f"1 luồng\n{p_lo:.2f} M tham số".replace(".", ","),
                               f"2 luồng\n{p_hi:.2f} M tham số".replace(".", ",")], color=INK2)
        ax.set_xlim(-0.45, 2.15)
        ax.set_ylabel("mAP@50 (%)")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - (hi - lo) * 0.10, hi + (hi - lo) * 0.10)
        for x, ha, items in ((-0.07, "right", left), (1.07, "left", right)):
            for y, txt in zip(_spread([v for v, _ in items], np.diff(ax.get_ylim())[0] * 0.055),
                              [t for _, t in items]):
                ax.text(x, y, txt, ha=ha, va="center", fontsize=8.5, color=INK)
        ax.set_title(f"{ds['name']} · {ds['sub']}", loc="left")
    handles = [Line2D([], [], color=c, lw=2.4, marker="o", ms=7, mec=SURF, label=n)
               for _, _, n, c in CAPACITY_PAIRS]
    layout(fig, "Kiểm soát capacity — hiệu ứng của dung lượng tách khỏi hiệu ứng của cảm biến",
           "Độ dốc mỗi đường = lợi ích thuần khi nhân đôi tham số mà KHÔNG thêm thông tin; khoảng "
           "cách giữa các đường ở cùng một cột = lợi ích của nguồn thông tin.",
           "C1, C2 và F2a dùng chung một arch_spec nên số tham số bằng nhau tuyệt đối "
           "(tests/test_models.py::test_capacity_control_exact_match). " + FOOT_HARNESS,
           handles, ncol=3, top=0.845, bottom=0.175, w_pad=5.0)
    return save(fig, "fig08_capacity_vs_map")


# ═════════════════════════════════ bảng số liệu ═══════════════════════════════
def tables(cache):
    TABLES.mkdir(parents=True, exist_ok=True)
    L = ["# Số liệu đằng sau các biểu đồ", "",
         "Sinh bởi `scripts/make_figures.py`. Mọi giá trị tính từ `metrics_*.json` / `preds_*.npz`.",
         ""]
    for ds in DATASETS:
        names = class_names(ds)
        m50 = series(ds, "all", "all", "map50")
        m95 = series(ds, "all", "all", "map5095")
        L += [f"## {ds['name']} ({ds['split']} — {ds['sub']})", "",
              "### Tổng thể — mean ± std (%)", "",
              "| cấu hình | n run | tham số (M) | mAP@50 | mAP@50-95 | macro-F1 @IoU0,5 | ngưỡng conf |",
              "|---|---|---|---|---|---|---|"]
        for cid in CID:
            e = [cache[f"{ds['key']}/{r.name}"] for r in run_dirs(ds, cid)]
            f1 = stat([x["best_macro_f1"] for x in e])
            L.append(f"| {cid} | {m50[cid][2]} | {params_of(ds, cid):.3f} | "
                     f"{100*m50[cid][0]:.2f} ± {100*m50[cid][1]:.2f} | "
                     f"{100*m95[cid][0]:.2f} ± {100*m95[cid][1]:.2f} | "
                     f"{100*f1[0]:.2f} ± {100*f1[1]:.2f} | "
                     f"{np.mean([x['best_thr'] for x in e]):.2f} |")
        vals, n_gt, _ = _per_class_f1(ds, cache)
        L += ["", "### F1 theo lớp (%) — tại ngưỡng cực đại macro-F1", "",
              "| cấu hình | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for cid in CID:
            L.append(f"| {cid} | " + " | ".join(f"{100*v:.1f}" for v in vals[cid][0]) + " |")
        L.append("| **n box GT** | "
                 + " | ".join(f"{int(v):,}".replace(",", ".") for v in n_gt) + " |")
        ap, _ = _per_class_from_metrics(ds, "ap50_per_class")
        L += ["", "### AP@50 theo lớp (%)", "",
              "| cấu hình | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for cid in CID:
            L.append(f"| {cid} | " + " | ".join(f"{100*v:.1f}" for v in ap[cid][0]) + " |")
        L.append("")
    if EFFECTS.exists():
        L += ["## Kích thước hiệu ứng (bootstrap ghép cặp, mAP@50)", "",
              "| dataset | so sánh | Δ (điểm %) | CI 95% | p | có ý nghĩa |",
              "|---|---|---|---|---|---|"]
        for r in json.loads(EFFECTS.read_text(encoding="utf-8")):
            L.append(f"| {r['ds']} | {r['a']} − {r['b']}<br><sub>{r['label']}</sub> | "
                     f"{100*r['delta']:+.2f} | "
                     f"[{100*r['ci95_low']:+.2f}, {100*r['ci95_high']:+.2f}] | "
                     f"{r['p_value']:.3f} | {'**có**' if r['significant'] else 'không'} |")
        L.append("")
    p = TABLES / "figure_data.md"
    p.write_text("\n".join(L), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT)}")
    return p


FIGS = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06, 7: fig07, 8: fig08}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--reuse-effects", action="store_true",
                    help="hình 6: dùng lại _effect_sizes.json thay vì chạy lại bootstrap")
    a = ap.parse_args()

    style()
    cache = prf_all(force=a.refresh_cache)
    for k in a.only or sorted(FIGS):
        print(f"[hình {k}]")
        FIGS[k](cache, **({"n_boot": a.n_boot, "reuse": a.reuse_effects} if k == 6 else {}))
    if not a.no_tables:
        print("[bảng số liệu]")
        tables(cache)


if __name__ == "__main__":
    main()
