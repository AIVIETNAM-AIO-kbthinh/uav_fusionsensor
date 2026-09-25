"""Sinh kiến trúc two-stream từ YAML gốc của Ultralytics.

Vì sao sinh tự động thay vì viết tay 45 dòng YAML:

1. **Đảm bảo đối chứng capacity đúng theo cấu trúc.** C1/C2 và F2a dùng CHUNG một
   file kiến trúc; chỉ khác dữ liệu nạp vào. Số tham số bằng nhau tuyệt đối,
   không phải "bằng nhau trong sai số 1%" như plan mục 7.3 dự phòng.
2. Nhánh B là bản sao chính xác của nhánh A, không thể gõ nhầm.
3. Đổi scale (n/s/m) hay đổi sang YOLO khác chỉ cần đổi tham số.

Bố cục sinh ra (nb = số lớp backbone gốc, với YOLO11 là 11):

    0                   nn.Identity          giữ lại ảnh N kênh gốc để nhánh B dùng
    1                   Index [3, 0, 3]      nhánh A = 3 kênh đầu
    2 .. nb+1           backbone A           bản sao backbone gốc
    nb+2                Index [3, 3, 6]      nhánh B = 3 kênh sau, lấy `from: 0`
    nb+3 .. 2*nb+2      backbone B           bản sao backbone gốc
    2*nb+3 ..           Concat + Conv 1x1    hợp nhất tại từng mức P mà head cần
    ...                 head gốc             chỉ đổi lại chỉ số tham chiếu

Toán tử hợp nhất `Concat` -> `Conv 1x1` (biến thể F2a) dùng module CÓ SẴN của
Ultralytics, không cần module tự viết nào. Biến thể F2b/F2c (cổng chú ý) thêm một
module trong src/models/fusion_ops.py.
"""

from __future__ import annotations

import copy
from pathlib import Path

from ultralytics.utils import YAML


def _shift(f, offset: int):
    """Dời chỉ số `from` của một lớp backbone được sao chép."""
    if isinstance(f, int):
        return f if f == -1 else f + offset
    return [x if x == -1 else x + offset for x in f]


def _scaled(c: int, d: dict, scale: str) -> int:
    """Số kênh sau khi áp width/max_channels — đúng công thức của `parse_model`."""
    from ultralytics.utils.ops import make_divisible

    _, width, max_channels = d["scales"][scale]
    return make_divisible(min(c, max_channels) * width, 8)


def build_two_stream_yaml(
    base: str | Path = "yolo11-obb.yaml",
    scale: str = "s",
    split: tuple[int, int] = (3, 3),
    fusion: str = "concat1x1",
    nc: int | None = None,
) -> dict:
    """Dựng dict kiến trúc two-stream.

    Args:
        base: YAML gốc của Ultralytics (mọi model dạng backbone/head đều hợp lệ).
        scale: n/s/m/l/x.
        split: số kênh mỗi nhánh, ví dụ (3, 3) cho RGB + IR.
        fusion: "concat1x1" (F2a), "attn" (F2b) hoặc "mgate" (F2c).
        nc: số lớp; None thì giữ của file gốc.
    """
    from ultralytics.utils.checks import check_yaml

    d = YAML.load(check_yaml(str(base)))
    backbone = copy.deepcopy(d["backbone"])
    head = copy.deepcopy(d["head"])
    nb = len(backbone)
    ca, cb = split

    # ---- các mức P mà head lấy từ backbone -----------------------------------
    fused = set()
    for f, *_ in head:
        for x in ([f] if isinstance(f, int) else f):
            if 0 <= x < nb:
                fused.add(x)
    fused.add(nb - 1)                     # lớp cuối backbone: head nhận qua `-1`
    fused = sorted(fused)

    off_a = 2
    off_b = 3 + nb
    layers = [
        [-1, 1, "nn.Identity", []],                       # 0
        [-1, 1, "Index", [ca, 0, ca]],                    # 1: nhánh A
    ]
    layers += [[_shift(f, off_a), n, m, copy.deepcopy(a)] for f, n, m, a in backbone]
    layers.append([0, 1, "Index", [cb, ca, ca + cb]])     # nhánh B, lấy từ ảnh gốc
    layers += [[_shift(f, off_b), n, m, copy.deepcopy(a)] for f, n, m, a in backbone]

    # ---- hợp nhất tại từng mức P --------------------------------------------
    fused_map: dict[int, int] = {}
    for k in fused:
        a_idx, b_idx = off_a + k, off_b + k
        c_out = backbone[k][3][0]                         # số kênh CHƯA scale của lớp gốc
        layers.append([[a_idx, b_idx], 1, "Concat", [1]])
        if fusion == "concat1x1":
            layers.append([-1, 1, "Conv", [c_out, 1]])
        elif fusion in ("attn", "mgate"):
            # Cổng giữ nguyên số kênh -> cần số kênh ĐÃ scale của tensor sau Concat
            gate = "GatedFusion" if fusion == "attn" else "ModalityGate"
            layers.append([-1, 1, gate, [2 * _scaled(c_out, d, scale)]])
            layers.append([-1, 1, "Conv", [c_out, 1]])
        else:
            raise ValueError(f"fusion khong hop le: {fusion!r}")
        fused_map[k] = len(layers) - 1

    head_start = len(layers)

    def remap(f):
        def one(x):
            if x == -1:
                return -1
            if x < nb:                                    # tham chiếu backbone -> lớp đã hợp nhất
                return fused_map[x]
            return head_start + (x - nb)                  # tham chiếu trong head -> dời
        return one(f) if isinstance(f, int) else [one(x) for x in f]

    layers += [[remap(f), n, m, copy.deepcopy(a)] for f, n, m, a in head]

    out = {k: v for k, v in d.items() if k not in ("backbone", "head", "yaml_file")}
    out["backbone"] = layers[:head_start]
    out["head"] = layers[head_start:]
    out["scale"] = scale
    if nc is not None:
        out["nc"] = nc
    out["_two_stream"] = {
        "base": str(base), "split": list(split), "fusion": fusion,
        "fused_levels": fused, "fused_map": fused_map,
        # đủ để ánh xạ ngược trọng số pretrained (xem src/models/build.py)
        "nb": nb, "off_a": off_a, "off_b": off_b, "head_start": head_start,
    }
    return out


def save_two_stream_yaml(path: str | Path, **kwargs) -> dict:
    """Dựng và ghi ra đĩa (để tái lập được và soi bằng mắt)."""
    d = build_two_stream_yaml(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(str(path), d)
    return d
