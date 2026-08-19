"""Cố định seed và ghi lại dấu vết để tái lập được sau nhiều tháng."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Cố định seed cho random / numpy / torch / cudnn."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def config_hash(cfg: dict) -> str:
    """Hash ổn định của một dict cấu hình (dùng để phát hiện chạy lệch cấu hình)."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def git_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def provenance(cfg: dict) -> dict:
    """Toàn bộ thông tin cần để tái lập một run."""
    import ultralytics

    return {
        "config_hash": config_hash(cfg),
        "git_hash": git_hash(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def dump_provenance(path: str | Path, cfg: dict) -> dict:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"config": cfg, "provenance": provenance(cfg)}
    p.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return rec
