"""Gop toan bo ket qua da co thanh mot bo bang bieu san sang dua vao bao cao.

Khac voi scripts/make_tables.py (tu chay bootstrap, moi lan mot dataset), script
nay *khong tinh lai gi ca* — no doc cac artifact da sinh ra va sap xep lai:

    runs/{ds}/{run}/metrics_{split}.json   mAP tong the + theo lop + theo tang sang
    runs/{ds}/{run}/result.json            params, do phu pretrained, val_metrics
    runs/{ds}/{run}/results.csv            duong cong train
    runs/{ds}/matrix_summary.json          GPU-hours
    results/figures/_effect_sizes.json     bootstrap CI 95% (make_figures.py)
    results/figures/_prf_cache.json        P/R/F1 theo lop tai nguong toi uu

    python scripts/make_report_tables.py
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DATASETS = [("dronevehicle", "DroneVehicle", "test"), ("vedai", "VEDAI", "val")]
ORDER = ["S1", "S2", "C1", "C2", "C1b", "C2b", "C1c", "C2c", "F1", "F2a", "F2b", "F2c", "F3"]
ROLE = {
    "S1": ("RGB", "1 luong", "baseline"),
    "S2": ("IR", "1 luong", "baseline"),
    "C1": ("RGB vao ca 2 luong", "2 luong", "**doi chung capacity**"),
    "C2": ("IR vao ca 2 luong", "2 luong", "**doi chung capacity**"),
    "C1b": ("RGB vao ca 2 luong", "2 luong + attn", "**doi chung capacity (attn)**"),
    "C2b": ("IR vao ca 2 luong", "2 luong + attn", "**doi chung capacity (attn)**"),
    "C1c": ("RGB vao ca 2 luong", "2 luong + mgate", "**doi chung capacity (mgate)**"),
    "C2c": ("IR vao ca 2 luong", "2 luong + mgate", "**doi chung capacity (mgate)**"),
    "F1": ("RGB + IR", "1 luong, 4 kenh", "fusion som"),
    "F2a": ("RGB + IR", "2 luong", "fusion giua (concat+1x1)"),
    "F2b": ("RGB + IR", "2 luong + attn", "fusion giua (attention)"),
    "F2c": ("RGB + IR", "2 luong + mgate", "fusion giua (modality gate)"),
    "F3": ("RGB + IR", "2 x 1 luong", "fusion muon (WBF)"),
}
BINS = ["lowlight", "midlight", "bright"]
BIN_VN = {"lowlight": "thieu sang", "midlight": "trung binh", "bright": "sang"}


def dot(n):
    return f"{int(n):,}".replace(",", ".")


def load(ds, split):
    out = defaultdict(dict)
    proj = ROOT / "runs" / ds
    for mf in sorted(proj.glob(f"*/metrics_{split}.json")):
        run = mf.parent
        rj = run / "result.json"
        res = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
        rid = res.get("id", run.name)
        last = None
        cf = run / "results.csv"
        if cf.exists():
            rows = list(csv.DictReader(cf.read_text(encoding="utf-8").splitlines()))
            last = rows[-1] if rows else None
        out[rid][run.name] = {
            "m": json.loads(mf.read_text(encoding="utf-8")),
            "r": res,
            "last": last,
        }
    return dict(out)


def names(ds):
    from ultralytics.utils import YAML
    for dy in sorted((ROOT / "runs" / ds).glob("*/data.yaml")):
        return YAML.load(str(dy))["names"]
    return {}


def ms(vals, nd=2):
    if not vals:
        return "—"
    a = np.asarray(vals, float) * 100.0
    return f"{a.mean():.{nd}f} ± {a.std():.{nd}f}"


def pick(runs, path):
    out = []
    for v in runs.values():
        cur = v["m"]
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                cur = None
                break
            cur = cur[k]
        if cur is not None:
            out.append(cur)
    return out


def per_class(entry, key, nm):
    if isinstance(entry, dict):
        return [entry[str(i)] for i in sorted(nm)]
    return [entry[i] for i in sorted(nm)]


def tbl(header, rows):
    L = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "tables" / "report_tables.md"))
    a = ap.parse_args()

    eff = json.loads((ROOT / "results" / "figures" / "_effect_sizes.json").read_text(encoding="utf-8"))
    prf_p = ROOT / "results" / "figures" / "_prf_cache.json"
    prf = json.loads(prf_p.read_text(encoding="utf-8")) if prf_p.exists() else {}

    data = {ds: load(ds, sp) for ds, _, sp in DATASETS}
    L = ["# Bang bieu ket qua", "",
         "Sinh tu dong bang `python scripts/make_report_tables.py`. Moi con so doc tu "
         "artifact trong `runs/` va `results/figures/` — khong gia tri nao go tay.", "",
         "Don vi mac dinh: **%**. Chi tieu chinh: **mAP50 (OBB)**, do bang harness cua de "
         "tai (IoU da giac chinh xac, AP 101 diem chuan COCO).", ""]

    # ---------- B1 ----------
    L += ["## Bang 1 — Ma tran thi nghiem va trang thai", ""]
    rows = []
    for rid in ORDER:
        inp, arch, role = ROLE[rid]
        dv, vd = data["dronevehicle"].get(rid, {}), data["vedai"].get(rid, {})
        prm = next((v["r"].get("params") for v in list(dv.values()) + list(vd.values())
                    if v["r"].get("params")), None)
        rows.append([rid, inp, arch, role, dot(prm) if prm else "—",
                     len(dv) or "—", len(vd) or "—",
                     "da chay" if (dv or vd) else "**chua chay**"])
    L += tbl(["ID", "Dau vao", "Kien truc", "Vai tro", "Tham so",
              "n run DV", "n run VEDAI", "Trang thai"], rows)
    L += ["", "> Hai nhom co **so tham so bang nhau tuyet doi** trong noi bo nhom — cung file "
          "kien truc, chi khac truong `modalities`: **{C1, C2, F2a}**, **{C1b, C2b, F2b}** va "
          "**{C1c, C2c, F2c}**. "
          "Day la dieu kien de `F2a - C2` va `F2b - C2b` doc duoc la dong gop cua *thong tin "
          "bo sung*, khong phai cua *dung luong mo hinh*.", "",
          "> Cot *Tham so* lay tu run dau tien tim duoc nen phu thuoc so lop `nc` cua dataset do "
          "(DroneVehicle 5 lop, VEDAI 8 lop) — bat bien duoc canh la *bang nhau trong cung mot "
          "nhom tren cung mot dataset*, xem `scripts/gen_experiment_configs.py`.", "",
          "> Cong attn cua F2b them **198.784 tham so** so voi F2a (khong phu thuoc `nc`), nen "
          "**`F2b - C2` khong phai so sanh co kiem soat** — doi chung dung cua F2b la C2b.", "",
          "> Cong mgate cua F2c co **dung bang so tham so** cong attn cua F2b (cung hinh dang MLP); "
          "chi khac khoi tao dong nhat, mo ta avg+max va softmax giua hai modality. "
          "Doi chung dung cua F2c la C2c.", ""]

    # ---------- B2 / B3 ----------
    for idx, (ds, disp, split) in enumerate(DATASETS):
        d = data[ds]
        if not d:
            continue
        unit = "seed" if ds == "dronevehicle" else "fold"
        n = len(next(iter(d.values())))
        L += [f"## Bang {2 + idx} — {disp} ({split}): ket qua tong the", "",
              f"mean ± std qua {n} {unit}.", ""]
        rows = []
        for rid in ORDER:
            if rid not in d:
                continue
            runs = d[rid]
            rows.append([rid, ROLE[rid][2], len(runs),
                         ms(pick(runs, ["all", "all", "map50"])),
                         ms(pick(runs, ["all", "all", "map5095"]))])
        L += tbl(["cau hinh", "vai tro", f"n {unit}", "mAP50", "mAP50-95"], rows)
        L += [""]

    # ---------- B4 ----------
    L += ["## Bang 4 — Kiem dinh hieu so (bootstrap ghep cap theo anh, 2.000 lan lap, CI 95%)", "",
          "Chi tieu: mAP50. Ghep cap theo anh nen nhieu do do kho de anh gay ra bi triet tieu.", ""]
    rows = []
    for e in eff:
        rows.append([e["ds"], f"{e['a']} vs {e['b']}", e["label"],
                     dot(e["n_images"]),
                     f"{100 * e['delta']:+.2f}",
                     f"[{100 * e['ci95_low']:+.2f}, {100 * e['ci95_high']:+.2f}]",
                     f"{e['p_value']:.3f}",
                     "**co**" if e["significant"] else "khong"])
    L += tbl(["dataset", "so sanh", "y nghia", "n anh", "delta mAP50", "CI 95%", "p", "co y nghia"], rows)
    L += ["", "> Bootstrap lay mau lai theo **anh**, khong theo seed. DroneVehicle dung seed 0; "
          "VEDAI gop ca 3 fold (cac tap val roi nhau) de du co mau — do la ly do n = 363.", ""]

    # ---------- B5 ----------
    L += ["## Bang 5 — Phan ra: bao nhieu la do cam bien, bao nhieu la do tham so?", ""]
    rows = []
    for ds in ["DroneVehicle", "VEDAI"]:
        g = {(e["a"], e["b"]): e for e in eff if e["ds"] == ds}
        tot, cap, net = g.get(("F2a", "S2")), g.get(("C2", "S2")), g.get(("F2a", "C2"))
        if not (tot and cap and net):
            continue
        share = 100 * net["delta"] / tot["delta"] if tot["delta"] else float("nan")
        rows += [
            [ds, "F2a − S2", "tong chenh lech (cach bao cao pho bien)",
             f"{100 * tot['delta']:+.2f}", "100%"],
            [ds, "C2 − S2", "phan do **tang so tham so** (cung modality, 2 luong)",
             f"{100 * cap['delta']:+.2f}", f"{100 - share:.0f}%"],
            [ds, "**F2a − C2**", "phan do **thong tin bo sung tu RGB** — con so de claim",
             f"**{100 * net['delta']:+.2f}**", f"**{share:.0f}%**"],
        ]
    L += tbl(["dataset", "hieu so", "dien giai", "delta mAP50", "ty trong"], rows)
    L += [""]

    # ---------- B6 ----------
    d = data["dronevehicle"]
    L += ["## Bang 6 — RQ3 (DroneVehicle): ket qua theo tang chieu sang", "",
          "mAP50, mean ± std qua 2 seed.", ""]
    rows = []
    for rid in ORDER:
        if rid not in d:
            continue
        runs = d[rid]
        rows.append([rid, ms(pick(runs, ["all", "all", "map50"]))] +
                    [ms(pick(runs, ["illum", b, "map50"])) for b in BINS])
    L += tbl(["cau hinh", "tat ca"] + [BIN_VN[b] for b in BINS], rows)
    ref = d.get("F2a", {})
    ni = {b: (pick(ref, ["illum", b, "n_images"]) or [0])[0] for b in BINS}
    nt = (pick(ref, ["all", "all", "n_images"]) or [0])[0]
    L += ["", "So anh moi tang: " + " / ".join(f"{BIN_VN[b]} {dot(ni[b])}" for b in BINS) +
          f" — tong {dot(nt)}.", ""]

    L += ["### Bang 6b — Chenh lech theo tang (diem uoc luong, trung binh 2 seed)", ""]
    rows = []
    for b in ["all"] + BINS:
        key = ["all", "all"] if b == "all" else ["illum", b]

        def mean_of(rid):
            v = pick(d.get(rid, {}), key + ["map50"])
            return float(np.mean(v)) if v else None

        f2a, c2, s2 = mean_of("F2a"), mean_of("C2"), mean_of("S2")
        if None in (f2a, c2, s2):
            continue
        rows.append([BIN_VN.get(b, "tat ca"), f"{100 * f2a:.2f}", f"{100 * c2:.2f}",
                     f"{100 * s2:.2f}", f"{100 * (f2a - c2):+.2f}", f"{100 * (f2a - s2):+.2f}"])
    L += tbl(["tang", "F2a", "C2", "S2", "F2a − C2", "F2a − S2"], rows)
    L += ["", "> Day la **hieu cua trung binh 2 seed**, chua co CI. CI theo tang nam trong "
          "`runs/dronevehicle/tables_test.md` (sinh boi `scripts/make_tables.py --illum`).", ""]

    # ---------- B7: AP50 theo lop, DroneVehicle ----------
    def ap50_block(tno, ds, disp):
        d2, nm2 = data[ds], names(ds)
        if not d2 or not nm2:
            return []
        out = [f"## Bang {tno} — {disp}: AP50 theo tung lop", "",
               f"mean qua cac {'seed' if ds == 'dronevehicle' else 'fold'}.", ""]
        rows, ngt = [], None
        for rid in ORDER:
            if rid not in d2:
                continue
            per = pick(d2[rid], ["all", "all", "ap50_per_class"])
            if not per:
                continue
            arr = np.array([per_class(p, "ap50", nm2) for p in per], float)
            rows.append([rid] + [f"{v:.1f}" for v in 100 * arr.mean(0)])
            if ngt is None:
                g = pick(d2[rid], ["all", "all", "n_gt_per_class"])
                if g:
                    ngt = [int(x) for x in per_class(g[0], "n_gt", nm2)]
        if ngt:
            rows.append(["*so box GT*"] + [f"*{dot(v)}*" for v in ngt])
        out += tbl(["cau hinh"] + [nm2[i] for i in sorted(nm2)], rows)
        out += ["", "> VEDAI: so box GT la cua **mot fold** (fold01); ba fold co tap val roi "
                "nhau.", ""] if ds == "vedai" else [""]
        return out

    L += ap50_block(7, "dronevehicle", "DroneVehicle")

    # ---------- B8 ----------
    nm = names("dronevehicle")
    have = [k for k in prf if k.startswith("dronevehicle/")]
    if have and nm:
        L += ["## Bang 8 — DroneVehicle: Precision / Recall / F1 theo lop tai nguong toi uu", ""]
        rows, thr = [], None
        for rid in ORDER:
            ks = sorted(k for k in have if k.split("/")[1].startswith(rid + "_"))
            if not ks:
                continue
            e = prf[ks[0]]
            thr = thr or e.get("best_thr")
            for lab, arr in (("P", e["p"]), ("R", e["r"]), ("F1-score", e["f1"])):
                rows.append([f"{rid} · {lab}"] + [f"{100 * v:.1f}" for v in arr] +
                            [f"**{100 * e['best_macro_f1']:.1f}**" if lab == "F1-score" else "—"])
        L += tbl(["cau hinh"] + [nm[i] for i in sorted(nm)] + ["macro-F1"], rows)
        L += ["", f"> Nguong confidence toi uu hoa macro-F1: **{thr}**. Lay tu run dau tien "
              "cua moi cau hinh (seed0).", ""]

    # ---------- B9 ----------
    L += ap50_block(9, "vedai", "VEDAI")

    # ---------- B10 ----------
    L += ["## Bang 10 — Chi phi tinh toan va cau hinh train", "",
          "Gio train = cot `time` cua epoch cuoi trong `results.csv`, cong don qua cac run "
          "(khong dung `hours` trong `matrix_summary.json` — truong do chi ghi lan goi "
          "`run_matrix.py` gan nhat, khong phai tong cong). Cot *run hoan tat* dem tu chinh cac "
          "thu muc run, khong tu truong `done` vi truong do co cung khiem khuyet. Cot *run loi* thi "
          "van la cua lan goi gan nhat.", ""]
    rows = []
    for ds, disp, split in DATASETS:
        sp = ROOT / "runs" / ds / "matrix_summary.json"
        s = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
        ep, hrs, nrun = None, 0.0, 0
        for runs in data[ds].values():
            for v in runs.values():
                if v["last"]:
                    ep = ep or int(float(v["last"]["epoch"]))
                    hrs += float(v["last"]["time"]) / 3600
                    nrun += 1
        rows.append([disp, nrun, len(s.get("failed", [])), ep or "—",
                     f"{hrs:.1f}", f"{hrs / nrun:.1f}" if nrun else "—"])
    L += tbl(["dataset", "run hoan tat", "run loi", "epoch", "tong gio train", "gio/run (TB)"], rows)

    L += ["", "### Bang 10b — Gio train theo cau hinh (trung binh moi run)", ""]
    rows = []
    for rid in ORDER:
        cells = []
        for ds, disp, split in DATASETS:
            hs = [float(v["last"]["time"]) / 3600
                  for v in data[ds].get(rid, {}).values() if v["last"]]
            cells.append(f"{np.mean(hs):.2f}" if hs else "—")
        if any(c != "—" for c in cells):
            rows.append([rid, ROLE[rid][1]] + cells)
    L += tbl(["cau hinh", "kien truc"] + [d for _, d, _ in DATASETS], rows)
    prov = json.loads((ROOT / "runs" / "dronevehicle" / "F2a_seed0" /
                       "provenance.json").read_text(encoding="utf-8"))
    ar = prov["config"]["args"]
    L += ["", "Sieu tham so dung chung cho **moi** cau hinh (dieu kien so sanh cong bang):", "",
          f"`epochs={ar['epochs']}` · `batch={ar['batch']}` · `nbs={ar['nbs']}` · "
          f"`imgsz={ar['imgsz']}` · `optimizer={ar['optimizer']}` · `lr0={ar['lr0']}` · "
          f"`cos_lr={ar['cos_lr']}` · `mosaic={ar['mosaic']}` · "
          f"`close_mosaic={ar['close_mosaic']}` · `hsv_h/s/v=0` · `augmentations=[]` · "
          f"`amp={ar['amp']}` · `deterministic={ar['deterministic']}`", ""]

    # ---------- B11 ----------
    L += ["## Bang 11 — Canh bao doc so: validator Ultralytics vs harness cua de tai", "",
          "Cung mot bo trong so, hai cach do cho hai con so khac han.", ""]
    rows = []
    for rid in ORDER:
        runs = data["dronevehicle"].get(rid, {})
        if not runs:
            continue
        def um(k):
            v = [r["r"].get("val_metrics", {}).get(k) for r in runs.values()]
            v = [x for x in v if x is not None]
            return float(np.mean(v)) if v else None
        u50, u95 = um("metrics/mAP50(B)"), um("metrics/mAP50-95(B)")
        h50 = pick(runs, ["all", "all", "map50"])
        h95 = pick(runs, ["all", "all", "map5095"])
        if u50 is None or not h50:
            continue
        rows.append([rid, f"{100 * u50:.2f}", f"{100 * np.mean(h50):.2f}",
                     f"{100 * u95:.2f}", f"{100 * np.mean(h95):.2f}",
                     f"{100 * (u95 - np.mean(h95)):+.2f}"])
    L += tbl(["cau hinh", "mAP50 Ultralytics (val)", "mAP50 harness (test)",
              "mAP50-95 Ultralytics (val)", "mAP50-95 harness (test)", "chenh 50-95"], rows)
    L += ["", "> Hai cot **khong so truc tiep duoc**: khac ca split (val vs test) lan cach do. "
          "`OBBValidator` khop box bang ProbIoU — xap xi kha vi von dung cho *ham mat mat*, "
          "lac quan co he thong; harness dung IoU da giac chinh xac + AP 101 diem. Bang nay "
          "chi de cho thay do lon cua khoang cach — **moi ket luan trong bao cao phai lay tu "
          "cot harness**.", ""]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
