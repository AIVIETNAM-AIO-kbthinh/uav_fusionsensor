"""Chạy suy luận + tính chỉ số phân tầng cho MỘT run đã train.

    python scripts/evaluate_run.py --run runs/dv/F2a_seed0 --split test \
        --illum data/dronevehicle/meta/test_illum.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.utils import YAML  # noqa: E402

from src.eval.harness import evaluate_strata, load_dump, match_all, run_inference  # noqa: E402
from src.eval.stratify import strata_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--illum", default=None, help="meta/{split}_illum.csv de phan tang RQ3")
    ap.add_argument("--skip-inference", action="store_true")
    a = ap.parse_args()

    run_dir = Path(a.run)
    dump_path = run_dir / f"preds_{a.split}.npz"
    if not a.skip_inference or not dump_path.exists():
        dump_path = run_inference(run_dir, split=a.split, imgsz=a.imgsz, batch=a.batch,
                                  device=a.device, out_name=f"preds_{a.split}.npz")

    dump = load_dump(dump_path)
    per_image = match_all(dump)
    data = YAML.load(str(run_dir / "data.yaml"))
    nc = len(data["names"])

    if a.illum and Path(a.illum).exists():
        from src.eval.stratify import load_illum
        strata = strata_spec(list(per_image), load_illum(a.illum))
    else:
        strata = {"all": {"all": list(per_image)}}

    res = evaluate_strata(per_image, strata, nc=nc)
    out = run_dir / f"metrics_{a.split}.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    overall = res["all"]["all"]
    print(f"[{run_dir.name}] anh={overall['n_images']} gt={overall['n_gt']}  "
          f"mAP50={overall['map50']:.4f}  mAP50-95={overall['map5095']:.4f}")
    for dim in ("illum",):
        if dim in res:
            for g, r in sorted(res[dim].items()):
                print(f"    {dim:6s} {g:10s} n={r['n_images']:5d} gt={r['n_gt']:6d} "
                      f"mAP50={r['map50']:.4f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
