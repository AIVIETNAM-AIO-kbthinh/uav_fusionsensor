"""Test resume: ngắt giữa chừng rồi chạy lại phải tiếp tục, không train lại từ đầu.

Đây là test tốn thời gian nhất (train thật vài epoch) nên đánh dấu `slow`:

    python -m pytest tests/test_resume.py -q -m slow
"""

import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
HAS_DATA = (ROOT / "data/vedai/rgb/images").is_dir()
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not HAS_DATA, reason="chua co du lieu VEDAI da xu ly")]


def _ckpt_epoch(run_dir: Path) -> int:
    ck = torch.load(run_dir / "weights" / "last.pt", map_location="cpu", weights_only=False)
    return ck.get("epoch", -1)


def _cmd(name, project, extra=()):
    return [sys.executable, str(ROOT / "scripts/run_experiment.py"),
            "--exp", str(ROOT / "configs/experiments/S1_rgb.yaml"),
            "--dataset", str(ROOT / "configs/datasets/vedai_rgb.yaml"),
            "--seed", "0", "--project", str(project), "--name", name,
            "--set", "epochs=6", "imgsz=320", "batch=8", "workers=2", *extra]


def test_resume_continues_from_interruption(tmp_path):
    project = tmp_path / "resume_test"
    name = "S1_interrupt"
    run_dir = project / name

    # --- chạy rồi ngắt sau khi đã có ít nhất 2 epoch -----------------------
    p = subprocess.Popen(_cmd(name, project), cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 600
    epoch_at_kill = -1
    try:
        while time.time() < deadline:
            if p.poll() is not None:
                pytest.fail("tien trinh ket thuc truoc khi kip ngat")
            last = run_dir / "weights" / "last.pt"
            if last.exists():
                try:
                    e = _ckpt_epoch(run_dir)
                except Exception:
                    e = -1
                if e >= 1:
                    epoch_at_kill = e
                    break
            time.sleep(2)
    finally:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=60)

    assert epoch_at_kill >= 1, "khong kip luu checkpoint truoc khi ngat"
    assert not (run_dir / "result.json").exists(), "run da hoan tat, khong con gi de resume"

    # --- phát hiện được là resume được -------------------------------------
    sys.path.insert(0, str(ROOT))
    from src.train import resumable_checkpoint
    assert resumable_checkpoint(run_dir) is not None

    # --- chạy lại đúng lệnh cũ --------------------------------------------
    r = subprocess.run(_cmd(name, project), cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out[-4000:]
    assert "[resume]" in out, "khong thay dau hieu resume trong log"
    assert "khong nap lai pretrained" in out, "resume nhung van nap de pretrained ImageNet"
    # Ultralytics in ra so epoch bat dau khi resume
    # Ultralytics bao ro so epoch bat dau khi resume
    assert "Resuming training" in out, out[-2000:]
    start = [ln for ln in out.splitlines() if "Resuming training" in ln]
    assert start, out[-2000:]
    print("dong resume cua Ultralytics:", start[0].strip())

    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert res["val_metrics"], "khong co metric sau khi resume"

    prov = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert prov["config"]["resumed_from"], "provenance khong ghi lai viec da resume"


def test_finished_run_is_not_resumable(tmp_path):
    """Run đã xong (`epoch = -1` sau strip_optimizer) không được coi là resume được."""
    sys.path.insert(0, str(ROOT))
    from src.train import resumable_checkpoint

    src = ROOT / "runs/smoke_pipeline/S1_fold01/weights/last.pt"
    if not src.exists():
        pytest.skip("chua co run mau da hoan tat")
    d = tmp_path / "finished" / "weights"
    d.mkdir(parents=True)
    shutil.copy2(src, d / "last.pt")
    assert resumable_checkpoint(tmp_path / "finished") is None
