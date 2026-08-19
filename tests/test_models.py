"""Test kiến trúc: bản vá, đối chứng capacity, nạp pretrained."""

import numpy as np
import pytest
import torch

from src import patches
from src.models import fusion_ops
from src.models.build import _inflate_stem, build_arch, load_pretrained
from src.models.two_stream import build_two_stream_yaml


@pytest.fixture(scope="module", autouse=True)
def _patched():
    patches.apply_all()
    fusion_ops.register()


def test_index_patch_backward_compatible():
    """Lớp Index thay thế phải giữ nguyên hành vi cũ (chọn phần tử của list)."""
    from ultralytics.nn import tasks
    assert tasks.Index is patches.Index
    m = patches.Index(1)
    xs = [torch.zeros(1), torch.ones(1), torch.full((1,), 2.0)]
    assert m(xs).item() == 1.0


def test_index_patch_channel_slice():
    m = patches.Index(3, 6)
    x = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1)
    assert m(x).flatten().tolist() == [3.0, 4.0, 5.0]


def test_patches_idempotent():
    before = patches.status()
    patches.apply_all()
    patches.apply_all()
    assert patches.status() == before


def test_two_stream_indices_consistent():
    """Chỉ số tham chiếu sinh ra phải hợp lệ: không trỏ tới lớp phía sau."""
    d = build_two_stream_yaml(scale="s", nc=5)
    layers = d["backbone"] + d["head"]
    for i, (f, *_rest) in enumerate(layers):
        for x in ([f] if isinstance(f, int) else f):
            assert x == -1 or 0 <= x < i, f"lop {i} tham chieu {x} khong hop le"


def test_two_stream_forward_and_stride():
    d = build_two_stream_yaml(scale="s", nc=5)
    from ultralytics.nn.tasks import OBBModel
    m = OBBModel(d, ch=6, nc=5, verbose=False).eval()
    with torch.no_grad():
        m(torch.randn(1, 6, 320, 320))
    assert m.stride.tolist() == [8.0, 16.0, 32.0]


def test_two_stream_branches_are_independent():
    """Hai nhánh phải có trọng số RIÊNG, không chia sẻ."""
    d = build_two_stream_yaml(scale="s", nc=5)
    meta = d["_two_stream"]
    from ultralytics.nn.tasks import OBBModel
    m = OBBModel(d, ch=6, nc=5, verbose=False)
    a = m.model[meta["off_a"]]
    b = m.model[meta["off_b"]]
    pa = next(a.parameters())
    pb = next(b.parameters())
    assert pa.data_ptr() != pb.data_ptr(), "hai nhanh dung chung tensor"
    assert pa.shape == pb.shape


def test_capacity_control_exact_match():
    """BẤT BIẾN CỐT LÕI CỦA ĐỀ TÀI.

    C1, C2 và F2a phải có SỐ THAM SỐ GIỐNG HỆT nhau — chúng dùng chung một
    arch_spec, chỉ khác dữ liệu nạp vào. Nếu test này hỏng thì so sánh
    `F2a vs C2` mất hết ý nghĩa và toàn bộ đóng góp phương pháp luận sụp đổ.
    """
    from ultralytics.utils import YAML
    ids = ["C1_dup_rgb", "C2_dup_ir", "F2a_mid_concat"]
    specs = [YAML.load(f"configs/experiments/{i}.yaml")["arch_spec"] for i in ids]
    assert all(s == specs[0] for s in specs), "C1/C2/F2a khong dung chung arch_spec"

    counts = []
    for s in specs:
        m, _ = build_arch(s, nc=5, ch=6)
        counts.append(sum(p.numel() for p in m.parameters()))
    assert len(set(counts)) == 1, dict(zip(ids, counts))


def test_capacity_control_larger_than_single():
    """Đối chứng capacity phải thật sự lớn hơn single-stream, nếu không thì vô nghĩa."""
    m1, _ = build_arch({"arch": "single", "scale": "s"}, nc=5, ch=3)
    m2, _ = build_arch({"arch": "two_stream", "scale": "s", "fusion": "concat1x1"}, nc=5, ch=6)
    n1 = sum(p.numel() for p in m1.parameters())
    n2 = sum(p.numel() for p in m2.parameters())
    assert n2 > 1.5 * n1, (n1, n2)


def test_inflate_stem_preserves_response():
    """Nong stem phải giữ độ lớn đáp ứng trên đầu vào hằng số."""
    w = torch.randn(16, 3, 3, 3)
    w4 = _inflate_stem(w.clone(), 4)
    assert w4.shape == (16, 4, 3, 3)
    x3 = torch.ones(1, 3, 8, 8)
    x4 = torch.ones(1, 4, 8, 8)
    y3 = torch.nn.functional.conv2d(x3, w, padding=1)
    y4 = torch.nn.functional.conv2d(x4, w4, padding=1)
    assert torch.allclose(y3, y4, atol=1e-4), (y3.mean().item(), y4.mean().item())


@pytest.mark.parametrize("spec,ch,min_cov", [
    ({"arch": "single", "scale": "s"}, 3, 0.95),
    ({"arch": "single", "scale": "s"}, 4, 0.95),
    ({"arch": "two_stream", "scale": "s", "fusion": "concat1x1"}, 6, 0.90),
])
def test_pretrained_coverage(spec, ch, min_cov):
    """Mọi cấu hình phải nạp được phần lớn trọng số pretrained.

    Nếu two-stream chỉ nạp được vài phần trăm thì nó train from scratch trong khi
    S1/S2 có pretrained -> so sánh mất công bằng ngay từ điểm xuất phát.
    """
    import os
    if not os.path.exists("yolo11s-obb.pt"):
        pytest.skip("chua tai yolo11s-obb.pt")
    m, cfg = build_arch(spec, nc=5, ch=ch)
    st = load_pretrained(m, "yolo11s-obb.pt", cfg, verbose=False)
    assert st["coverage"] >= min_cov, st


def test_pretrained_symmetric_between_streams():
    """Hai nhánh phải nhận CÙNG trọng số pretrained -> C1/C2/F2a xuất phát như nhau."""
    import os
    if not os.path.exists("yolo11s-obb.pt"):
        pytest.skip("chua tai yolo11s-obb.pt")
    spec = {"arch": "two_stream", "scale": "s", "fusion": "concat1x1"}
    m, cfg = build_arch(spec, nc=5, ch=6)
    load_pretrained(m, "yolo11s-obb.pt", cfg, verbose=False)
    meta = cfg["_two_stream"]
    for k in range(meta["nb"]):
        a = m.model[meta["off_a"] + k]
        b = m.model[meta["off_b"] + k]
        pa = list(a.parameters())
        pb = list(b.parameters())
        for x, y in zip(pa, pb):
            assert torch.allclose(x, y), f"lop backbone {k} khac nhau giua hai nhanh"
