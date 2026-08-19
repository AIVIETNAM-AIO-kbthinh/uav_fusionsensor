"""RQ4 — độ nhạy của từng chiến lược fusion với sai lệch đồng đăng ký.

Cố tình dịch ảnh IR theo trục x một lượng δ ∈ {0, 2, 5, 10, 20} px **tại thời điểm
test** (model train ở δ=0), rồi đo mức suy giảm mAP.

Dải δ này có căn cứ đo đạc chứ không tuỳ tiện: đã đo trên 300 cặp val của
DroneVehicle, lệch trung vị 1,8 px, p90 7,2 px, 20,9% số cặp lệch quá 5 px
(xem DATA_REPORT.md mục 2.6).

Chỉ chạy suy luận nên rất rẻ so với train.

    python scripts/run_rq4_ablation.py --runs runs/dronevehicle/F1_seed0 \
        runs/dronevehicle/F2a_seed0 --deltas 0 2 5 10 20 --split test
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML  # noqa: E402

from src.eval.harness import evaluate_strata, load_dump, match_all, run_inference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--deltas", nargs="*", type=int, default=[0, 2, 5, 10, 20])
    ap.add_argument("--axis", choices=["x", "y", "xy"], default="x")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="results/rq4_misalignment.json")
    a = ap.parse_args()

    table = {}
    for run in a.runs:
        run_dir = Path(run)
        base_data = YAML.load(str(run_dir / "data.yaml"))
        nc = len(base_data["names"])
        rid = json.loads((run_dir / "result.json").read_text(encoding="utf-8")).get("id", run_dir.name)
        table[run_dir.name] = {"id": rid, "deltas": {}}

        for d in a.deltas:
            shift = {"x": [d, 0], "y": [0, d], "xy": [d, d]}[a.axis]
            dcfg = dict(base_data)
            dcfg["ir_shift"] = shift
            dpath = run_dir / f"data_shift{d}{a.axis}.yaml"
            YAML.save(str(dpath), dcfg)

            out_name = f"preds_{a.split}_shift{d}{a.axis}.npz"
            p = run_dir / out_name
            if not p.exists():
                p = run_inference(run_dir, split=a.split, imgsz=a.imgsz, batch=a.batch,
                                  device=a.device, out_name=out_name, data_override=dpath)
            per = match_all(load_dump(p))
            res = evaluate_strata(per, {"all": {"all": list(per)}}, nc=nc)["all"]["all"]
            table[run_dir.name]["deltas"][d] = {"map50": res["map50"], "map5095": res["map5095"]}
            print(f"[{run_dir.name}] delta={d:3d}px  mAP50={res['map50']:.4f}  "
                  f"mAP50-95={res['map5095']:.4f}", flush=True)

    # quy về phần trăm suy giảm so với delta=0
    for name, rec in table.items():
        base = rec["deltas"].get(0, {}).get("map50")
        if base:
            rec["retention"] = {d: round(v["map50"] / base, 4) for d, v in rec["deltas"].items()}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1), encoding="utf-8")

    print(f"\n{'run':24s} " + " ".join(f"d={d:<7d}" for d in a.deltas))
    for name, rec in table.items():
        cells = " ".join(f"{rec['retention'].get(d, float('nan')):<9.3f}" for d in a.deltas)
        print(f"{name:24s} {cells}   (ty le mAP50 giu duoc so voi delta=0)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
