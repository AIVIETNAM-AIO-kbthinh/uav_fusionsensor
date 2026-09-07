"""Tổng hợp toàn bộ run thành bảng kết quả + khoảng tin cậy bootstrap.

Đây là nơi các câu hỏi nghiên cứu được trả lời bằng số:

    RQ1 (ngây thơ)   max(F1,F2a,F2b,F3) vs max(S1,S2)
    RQ1 (trung thực) F2a vs C2, F2b vs C2b   <- ĐÃ KIỂM SOÁT CAPACITY, con số để claim
    chẩn đoán        C2  vs S2      <- lợi ích thuần từ tăng capacity
    RQ2              F1 vs F2a vs F2b vs F3
    RQ3              các so sánh trên, tách theo tầng chiếu sáng

Không đọc số từ stdout của bất cứ đâu — mọi thứ lấy từ `metrics_*.json` và
`preds_*.npz` mà scripts/evaluate_run.py sinh ra.

    python scripts/make_tables.py --project runs/dronevehicle --split test
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.bootstrap import bootstrap_delta  # noqa: E402
from src.eval.harness import load_dump, match_all  # noqa: E402
from src.eval.stratify import load_illum  # noqa: E402

COMPARISONS = [
    ("F2a", "C2", "RQ1 trung thuc — fusion vs doi chung capacity"),
    ("C2", "S2", "chan doan — loi ich thuan tu tang capacity (IR)"),
    ("C1", "S1", "chan doan — loi ich thuan tu tang capacity (RGB)"),
    ("F2a", "S2", "RQ1 ngay tho — fusion vs modality don le"),
    ("F2a", "F1", "RQ2 — mid vs early"),
    ("F2b", "F1", "RQ2 — two-stream + cong vs early fusion 4 kenh"),
    ("F2a", "F3", "RQ2 — mid vs late"),
    ("F2b", "C2b", "RQ1 trung thuc (attn) — F2b vs doi chung capacity cua chinh no"),
    ("F2b", "F2a", "RQ2 — attention vs concat"),
    ("C2b", "C2", "chan doan — chi phi/loi ich thuan cua cong attn (khong co tin hieu bo sung)"),
]
ILLUM_BINS = ["lowlight", "midlight", "bright"]


def collect(project: Path, split: str) -> dict:
    out = defaultdict(dict)
    for m in sorted(project.glob(f"*/metrics_{split}.json")):
        run = m.parent
        rj = run / "result.json"
        rid = json.loads(rj.read_text(encoding="utf-8")).get("id", run.name) if rj.exists() else run.name
        out[rid][run.name] = json.loads(m.read_text(encoding="utf-8"))
    return dict(out)


def agg(runs: dict, dim: str, group: str, key: str):
    vals = [r[dim][group][key] for r in runs.values()
            if dim in r and group in r.get(dim, {}) and key in r[dim][group]]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None


def fmt(a):
    return "—" if a is None else f"{100 * a[0]:.2f} ± {100 * a[1]:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--illum", default=None, help="meta/{split}_illum.csv de bootstrap theo tang")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    project = Path(a.project)
    data = collect(project, a.split)
    if not data:
        sys.exit(f"khong tim thay metrics_{a.split}.json trong {project}; "
                 f"chay scripts/evaluate_run.py truoc.")

    cols = [("all", "all")] + [("illum", b) for b in ILLUM_BINS]
    L = [f"# Ket qua — {project.name} ({a.split})", "",
         "## mAP50 (%) — mean ± std qua cac seed/fold", "",
         "| cau hinh | n run | " + " | ".join(g for _, g in cols) + " |",
         "|---|---|" + "---|" * len(cols)]
    for rid in sorted(data):
        runs = data[rid]
        row = [rid, str(len(runs))] + [fmt(agg(runs, d, g, "map50")) for d, g in cols]
        L.append("| " + " | ".join(row) + " |")

    L += ["", "## mAP50-95 (%)", "",
          "| cau hinh | n run | " + " | ".join(g for _, g in cols) + " |",
          "|---|---|" + "---|" * len(cols)]
    for rid in sorted(data):
        runs = data[rid]
        row = [rid, str(len(runs))] + [fmt(agg(runs, d, g, "map5095")) for d, g in cols]
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    if not a.no_bootstrap:
        cache: dict[str, dict] = {}

        nc_cache: dict[str, int] = {}

        def per_image(rid):
            if rid not in cache:
                runs = sorted(data.get(rid, {}))
                p = project / runs[0] / f"preds_{a.split}.npz" if runs else None
                cache[rid] = match_all(load_dump(p)) if p and p.exists() else None
                if runs:
                    from ultralytics.utils import YAML
                    dy = project / runs[0] / "data.yaml"
                    if dy.exists():
                        nc_cache[rid] = len(YAML.load(str(dy))["names"])
            return cache[rid]

        groups = {"all": None}
        if a.illum and Path(a.illum).exists():
            illum = load_illum(a.illum)
            for b in ILLUM_BINS:
                groups[b] = {i for i, v in illum.items() if v["illum_bin"] == b}

        L += ["## So sanh chinh (bootstrap ghep cap theo anh, CI 95%, mAP50)", "",
              "| so sanh | tang | delta (%) | CI95 | p | co y nghia |",
              "|---|---|---|---|---|---|"]
        for A, B, label in COMPARISONS:
            pa, pb = per_image(A), per_image(B)
            if pa is None or pb is None:
                continue
            common = sorted(set(pa) & set(pb))
            # nc lay tu data.yaml, khong suy tu nhan co mat (lop hiem co the vang mat)
            fallback = 1 + max((int(c) for r in pa.values() for c in r["gt_cls"]), default=0)
            nc = nc_cache.get(A) or nc_cache.get(B) or fallback
            for gname, gset in groups.items():
                ids = common if gset is None else [i for i in common if i in gset]
                if len(ids) < 20:
                    continue
                r = bootstrap_delta(pa, pb, ids, nc=nc, n_boot=a.n_boot, seed=0)
                L.append(f"| {A} vs {B}<br><sub>{label}</sub> | {gname} | {100*r['delta']:+.2f} | "
                         f"[{100*r['ci95_low']:+.2f}, {100*r['ci95_high']:+.2f}] | "
                         f"{r['p_value']:.3f} | {'**co**' if r['significant'] else 'khong'} |")
        L += ["", "> Con so de ket luan RQ1 la dong `F2a vs C2`. Neu CI chua 0 thi ket luan la",
              "> \"khong co bang chung ve loi ich cua fusion sau khi kiem soat capacity\" —",
              "> mot ket qua hop le, xem plan.md muc 6.3 va rui ro R6.",
              "> Dong `F2a vs S2` la gioi han tren: no con chua phan dong gop cua capacity,",
              "> va o tang thieu sang con bi thoi phong boi giao thuc chon GT (plan.md 3.1).", ""]

    out = Path(a.out or project / f"tables_{a.split}.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
