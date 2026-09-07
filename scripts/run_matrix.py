"""Chạy toàn bộ ma trận thí nghiệm theo cấu hình đã chốt ở plan.md mục 0.1.

    # DroneVehicle, 2 seed (mặc định)
    python scripts/run_matrix.py --dataset dronevehicle

    # VEDAI, 3 fold
    python scripts/run_matrix.py --dataset vedai --folds 01 03 05

    # chạy thử nhanh
    python scripts/run_matrix.py --dataset vedai --folds 01 --only S1 F2a --set epochs=2

Thiết kế: nối tiếp, một GPU, **dừng và chạy lại được ở hai mức**.

  * mức run   — run đã có `result.json` thì bỏ qua hoàn toàn
  * mức epoch — run dở dang có `weights/last.pt` thì tiếp tục đúng epoch bị ngắt,
                khôi phục cả optimizer, EMA và lịch learning rate

Nên khi bị ngắt giữa chừng chỉ cần chạy lại **đúng lệnh cũ**. Dùng `--no-resume`
nếu muốn train lại từ đầu.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML  # noqa: E402

# thứ tự chạy: baseline trước để phát hiện sớm lỗi pipeline, rồi đối chứng, rồi fusion
ORDER = ["S2_ir", "S1_rgb", "F1_early", "C2_dup_ir", "C1_dup_rgb", "F2a_mid_concat",
         "C2b_dup_ir_attn", "C1b_dup_rgb_attn", "F2b_mid_attn"]


def fold_dataset_yaml(base_yaml: Path, fold: str, out_dir: Path) -> Path:
    """Sinh dataset yaml cho một fold của VEDAI."""
    d = YAML.load(str(base_yaml))
    d["train"] = f"fold{fold}_train.txt"
    d["val"] = f"fold{fold}_test.txt"
    out = out_dir / f"vedai_fold{fold}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(str(out), d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dronevehicle", "vedai"], required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--folds", nargs="*", default=["01", "03", "05"])
    ap.add_argument("--only", nargs="*", default=None, help="loc theo id, vd S1 F2a")
    ap.add_argument("--skip", nargs="*", default=["F2b", "C1b", "C2b"], help="mac dinh hoan nhanh attn: F2b + doi chung C1b/C2b")
    ap.add_argument("--project", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="ghi de arg Ultralytics")
    ap.add_argument("--no-resume", action="store_true",
                    help="train lai tu dau moi run do dang thay vi tiep tuc")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    project = Path(a.project or f"runs/{a.dataset}")
    exps = []
    for stem in ORDER:
        cfg = ROOT / "configs" / "experiments" / f"{stem}.yaml"
        e = YAML.load(str(cfg))
        if a.only and e["id"] not in a.only:
            continue
        if e["id"] in (a.skip or []):
            continue
        exps.append((cfg, e))

    jobs = []
    if a.dataset == "dronevehicle":
        base = ROOT / "configs/datasets/dronevehicle_ir.yaml"
        for cfg, e in exps:
            for seed in a.seeds:
                jobs.append((cfg, e, base, seed, f"{e['id']}_seed{seed}"))
    else:
        base = ROOT / "configs/datasets/vedai_rgb.yaml"
        for cfg, e in exps:
            for fold in a.folds:
                dy = fold_dataset_yaml(base, fold, project / "_datasets")
                jobs.append((cfg, e, dy, 0, f"{e['id']}_fold{fold}"))

    print(f"Ma tran: {len(jobs)} run -> {project}")
    for _, e, dy, seed, name in jobs:
        print(f"  {name:22s} modalities={e['modalities']} data={Path(dy).name}")
    if a.dry_run:
        return

    done, failed = [], []
    t_all = time.time()
    for k, (cfg, e, dy, seed, name) in enumerate(jobs, 1):
        run_dir = project / name
        if (run_dir / "result.json").exists():
            print(f"[{k}/{len(jobs)}] {name}: da co ket qua, bo qua")
            done.append(name)
            continue
        from src.train import resumable_checkpoint
        last = None if a.no_resume else resumable_checkpoint(run_dir)
        cmd = [sys.executable, str(ROOT / "scripts/run_experiment.py"),
               "--exp", str(cfg), "--dataset", str(dy), "--seed", str(seed),
               "--project", str(project), "--name", name]
        if a.no_resume:
            cmd.append("--no-resume")
        if a.set:
            cmd += ["--set", *a.set]
        tag = f"  (tiep tuc tu epoch da luu trong {last.name})" if last else ""
        print(f"\n[{k}/{len(jobs)}] {name}{tag}\n  $ {' '.join(cmd)}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        dt = (time.time() - t0) / 3600
        if r.returncode == 0:
            done.append(name)
            print(f"  -> xong sau {dt:.2f} h")
        else:
            failed.append(name)
            print(f"  -> THAT BAI (exit {r.returncode}) sau {dt:.2f} h")

    summary = {"project": str(project), "done": done, "failed": failed,
               "hours": round((time.time() - t_all) / 3600, 2)}
    (project / "matrix_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nXong {len(done)}/{len(jobs)} run trong {summary['hours']} h. That bai: {failed or 'khong'}")


if __name__ == "__main__":
    main()
