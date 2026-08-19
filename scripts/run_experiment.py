"""Chạy một ô của ma trận thí nghiệm."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics.utils import YAML
from src.train import run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="configs/experiments/*.yaml")
    ap.add_argument("--dataset", required=True, help="configs/datasets/*.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="ghi de arg Ultralytics, vd epochs=1 batch=4")
    a = ap.parse_args()

    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        overrides[k] = v

    exp = YAML.load(a.exp)
    r = run_experiment(exp, a.dataset, seed=a.seed, project=a.project, name=a.name,
                       overrides=overrides)
    print(json.dumps(r, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
