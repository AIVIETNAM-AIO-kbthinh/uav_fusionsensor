"""Kiểm thử tích hợp toàn bộ pipeline trên VEDAI với số epoch nhỏ.

Chạy đúng chuỗi thao tác của thí nghiệm thật — train ma trận, đánh giá, late
fusion, sinh bảng — nhưng rẻ. Mục đích là bắt lỗi lắp ghép TRƯỚC khi tiêu
hàng chục giờ GPU cho DroneVehicle.

    python scripts/smoke_pipeline.py --epochs 3
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw)
    if r.returncode != 0:
        sys.exit(f"THAT BAI: {cmd}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="runs/smoke_pipeline")
    a = ap.parse_args()
    P = Path(a.project)

    py = sys.executable
    run([py, "scripts/run_matrix.py", "--dataset", "vedai", "--folds", "01",
         "--only", "S1", "S2", "C2", "F2a", "--project", a.project,
         "--set", f"epochs={a.epochs}", f"imgsz={a.imgsz}", f"batch={a.batch}", "workers=4"])

    for name in ("S1_fold01", "S2_fold01", "C2_fold01", "F2a_fold01"):
        run([py, "scripts/evaluate_run.py", "--run", P / name, "--split", "val",
             "--imgsz", a.imgsz, "--batch", a.batch])

    run([py, "scripts/run_late_fusion.py", "--s1", P / "S1_fold01", "--s2", P / "S2_fold01",
         "--out", P / "F3_fold01", "--split", "val", "--imgsz", a.imgsz, "--batch", a.batch])
    run([py, "scripts/evaluate_run.py", "--run", P / "F3_fold01", "--split", "val",
         "--skip-inference"])

    run([py, "scripts/run_rq4_ablation.py", "--runs", P / "F2a_fold01", P / "F1_fold01"
         if (P / "F1_fold01").exists() else P / "F2a_fold01",
         "--deltas", "0", "5", "--split", "val", "--imgsz", a.imgsz, "--batch", a.batch,
         "--out", str(P / "rq4.json")])

    run([py, "scripts/make_tables.py", "--project", a.project, "--split", "val",
         "--n-boot", "300"])
    print("\n=== PIPELINE SMOKE TEST HOAN TAT ===")


if __name__ == "__main__":
    main()
