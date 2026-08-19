"""Dựng model cho từng cấu hình thí nghiệm và nạp trọng số pretrained công bằng.

Vấn đề công bằng mà file này giải quyết
---------------------------------------
S1/S2 là kiến trúc chuẩn nên nạp thẳng được `yolo11s-obb.pt`. F1 (4 kênh) và các
cấu hình two-stream thì tên/hình dạng tham số khác, `model.load()` của Ultralytics
sẽ **âm thầm bỏ qua** phần lớn trọng số → chúng train from scratch trong khi
S1/S2 có pretrained.

Chênh lệch đó sẽ bị quy nhầm cho fusion và làm hỏng toàn bộ so sánh. Đây đúng là
loại lỗi mà đối chứng capacity sinh ra để bắt, nhưng ở đây ta chặn từ gốc:

* **Early fusion (4 kênh)**: nong stem — sao chép trọng số 3 kênh RGB, kênh IR
  khởi tạo bằng trung bình 3 kênh đó, rồi chia lại tỉ lệ để giữ nguyên độ lớn
  đáp ứng (thủ thuật inflation chuẩn của I3D).
* **Two-stream**: ánh xạ chỉ số lớp. Backbone gốc lớp k -> lớp `off_a + k`
  (nhánh A) VÀ `off_b + k` (nhánh B); head gốc lớp `nb + h` -> `head_start + h`.
  Cả hai nhánh cùng nhận một bộ trọng số pretrained — đối xứng, nên C1/C2 và
  F2a xuất phát từ đúng cùng một điểm.

Các lớp hợp nhất (Concat/Conv 1x1/GatedFusion) không có đối ứng pretrained nên
khởi tạo ngẫu nhiên — điều này giống nhau ở mọi cấu hình two-stream.
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics.nn.tasks import OBBModel

from src import patches
from src.models import fusion_ops
from src.models.two_stream import build_two_stream_yaml


def _ensure_patches() -> None:
    patches.apply_index_patch()
    fusion_ops.register()


def build_arch(spec: dict, nc: int, ch: int, verbose: bool = False) -> tuple[OBBModel, dict | str]:
    """Dựng model theo mô tả kiến trúc.

    Args:
        spec: {"arch": "single"|"two_stream", "base": ..., "scale": ..., "fusion": ...}
        nc: số lớp.
        ch: số kênh đầu vào (phải khớp modalities).

    Returns:
        (model, cfg) — cfg là dict (two-stream) hoặc đường dẫn yaml (single).
    """
    _ensure_patches()
    arch = spec.get("arch", "single")
    scale = spec.get("scale", "s")
    base = spec.get("base", "yolo11-obb.yaml")

    if arch == "single":
        cfg = base.replace("yolo11", f"yolo11{scale}") if "yolo11-" in base else base
        model = OBBModel(cfg, ch=ch, nc=nc, verbose=verbose)
    elif arch == "two_stream":
        split = tuple(spec.get("split", (3, 3)))
        if sum(split) != ch:
            raise ValueError(f"split={split} khong khop ch={ch}")
        cfg = build_two_stream_yaml(base=base, scale=scale, split=split,
                                    fusion=spec.get("fusion", "concat1x1"), nc=nc)
        model = OBBModel(cfg, ch=ch, nc=nc, verbose=verbose)
    else:
        raise ValueError(f"arch khong hop le: {arch!r}")
    return model, cfg


# --------------------------------------------------------------------------- #
#  Nạp pretrained
# --------------------------------------------------------------------------- #

def _inflate_stem(w: torch.Tensor, ch: int) -> torch.Tensor:
    """Nong trọng số conv đầu tiên từ 3 kênh lên `ch` kênh, giữ độ lớn đáp ứng."""
    c_out, c_in, kh, kw = w.shape
    if c_in == ch:
        return w
    if ch < c_in:
        return w[:, :ch] * (c_in / ch)
    extra = w.mean(dim=1, keepdim=True).repeat(1, ch - c_in, 1, 1)
    out = torch.cat([w, extra], dim=1)
    return out * (c_in / ch)


def load_pretrained(model: OBBModel, weights: str | Path, cfg: dict | str,
                    verbose: bool = True) -> dict:
    """Nạp trọng số pretrained vào model, xử lý cả 4 kênh lẫn two-stream.

    Returns:
        dict thống kê: số tensor nạp được / tổng, để ghi vào log run.
    """
    from ultralytics.nn.tasks import load_checkpoint

    src_model, _ = load_checkpoint(str(weights))
    src = src_model.float().state_dict()
    dst = model.state_dict()

    meta = cfg.get("_two_stream") if isinstance(cfg, dict) else None
    mapped: dict[str, torch.Tensor] = {}

    if meta is None:
        mapped = dict(src)
    else:
        nb, off_a, off_b, head_start = meta["nb"], meta["off_a"], meta["off_b"], meta["head_start"]
        for k, v in src.items():
            if not k.startswith("model."):
                continue
            parts = k.split(".")
            try:
                idx = int(parts[1])
            except (IndexError, ValueError):
                continue
            rest = ".".join(parts[2:])
            if idx < nb:                                  # backbone -> cả hai nhánh
                for off in (off_a, off_b):
                    mapped[f"model.{off + idx}.{rest}"] = v.clone()
            else:                                         # head -> dời chỉ số
                mapped[f"model.{head_start + (idx - nb)}.{rest}"] = v

    loaded, skipped = {}, []
    for k, v in mapped.items():
        if k not in dst:
            skipped.append(k)
            continue
        t = dst[k]
        if v.shape != t.shape:
            if v.dim() == 4 and v.shape[0] == t.shape[0] and v.shape[2:] == t.shape[2:]:
                v = _inflate_stem(v, t.shape[1])          # stem 3 -> 4 kênh
            if v.shape != t.shape:
                skipped.append(k)
                continue
        loaded[k] = v

    model.load_state_dict(loaded, strict=False)
    stats = {"loaded": len(loaded), "model_tensors": len(dst),
             "coverage": round(len(loaded) / max(len(dst), 1), 4), "skipped": len(skipped)}
    if verbose:
        print(f"[pretrained] {weights}: nap {stats['loaded']}/{stats['model_tensors']} tensor "
              f"({100*stats['coverage']:.1f}%), bo qua {stats['skipped']}")
    return stats
