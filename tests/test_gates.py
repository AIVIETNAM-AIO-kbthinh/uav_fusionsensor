"""Test cổng ModalityGate (F2c) và phần ghi tỉ trọng cổng của harness."""

import pytest
import torch

from src import patches
from src.models import fusion_ops
from src.models.build import build_arch
from src.models.fusion_ops import GatedFusion, ModalityGate


@pytest.fixture(scope="module", autouse=True)
def _patched():
    patches.apply_all()
    fusion_ops.register()


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


def _spec(fusion):
    return {"arch": "two_stream", "scale": "s", "base": "yolo11-obb.yaml",
            "split": [3, 3], "fusion": fusion}


def test_modality_gate_identity_at_init():
    """Cổng phải là phép đồng nhất CHÍNH XÁC lúc khởi tạo -> F2c xuất phát bằng F2a."""
    g = ModalityGate(64)
    x = torch.randn(2, 64, 9, 7)
    assert torch.equal(g(x), x)


def test_modality_gate_is_identity_inside_full_model():
    """Cả model F2c lúc khởi tạo phải cho đúng đầu ra như khi bỏ hẳn cổng."""
    m, _ = build_arch(_spec("mgate"), nc=5, ch=6)
    m.eval()
    x = torch.randn(1, 6, 320, 320)
    with torch.no_grad():
        y = m(x)[0]
        gates = [g for g in m.modules() if isinstance(g, ModalityGate)]
        assert len(gates) == 3
        for g in gates:
            g.forward = lambda t: t
        y_id = m(x)[0]
    assert torch.equal(y, y_id)


def test_modality_gate_not_dead_at_init():
    """Conv cuối khởi tạo 0 nhưng vẫn phải nhận gradient, nếu không cổng chết cứng ở 1."""
    g = ModalityGate(32)
    x = torch.randn(4, 32, 5, 5)
    (g(x) * torch.randn_like(x)).sum().backward()
    assert g.fc[-1].weight.grad.abs().sum() > 0


def test_modality_gate_competition_and_share():
    """Trọng số kênh i của A và của B cộng lại bằng 2; tỉ trọng ghi lại nằm trong [0, 1]."""
    torch.manual_seed(0)
    g = ModalityGate(32)
    torch.nn.init.normal_(g.fc[-1].weight, std=1.0)
    g.record = True
    x = torch.rand(3, 32, 4, 4) + 0.1
    y = g(x)
    w = (y / x).mean((2, 3))                                   # (3, 32), cổng là hằng theo không gian
    assert torch.allclose(w[:, :16] + w[:, 16:], torch.full((3, 16), 2.0), atol=1e-5)
    assert g.last_share.shape == (3,)
    assert ((g.last_share >= 0) & (g.last_share <= 1)).all()
    assert torch.allclose(g.last_share, w[:, 16:].mean(1) / 2, atol=1e-5)


def test_modality_gate_no_record_by_default():
    """Không ghi gì khi train (EMA/val trong lúc train) để không tốn chuyển dữ liệu về CPU."""
    g = ModalityGate(16)
    g(torch.randn(2, 16, 3, 3))
    assert g.last_share is None


def test_modality_gate_rejects_odd_channels():
    with pytest.raises(ValueError):
        ModalityGate(15)


def test_mgate_same_params_as_attn():
    """F2c chỉ đổi cách dùng tham số, không đổi số tham số: F2c == F2b == F2a + 198.784."""
    n = {f: _n_params(build_arch(_spec(f), nc=5, ch=6)[0]) for f in ("concat1x1", "attn", "mgate")}
    assert n["mgate"] == n["attn"], n
    assert n["mgate"] - n["concat1x1"] == 198_784, n


def test_attn_still_builds_old_gate():
    """F2b đã train phải dựng lại được ĐÚNG kiến trúc cũ, nếu không harness đánh giá sai."""
    m, _ = build_arch(_spec("attn"), nc=5, ch=6)
    kinds = {type(x) for x in m.modules()}
    assert GatedFusion in kinds and ModalityGate not in kinds


def test_mgate_capacity_control_exact_match():
    """C1c, C2c, F2c dùng chung arch_spec -> F2c vs C2c là so sánh có kiểm soát."""
    from ultralytics.utils import YAML
    ids = ["C1c_dup_rgb_mgate", "C2c_dup_ir_mgate", "F2c_mid_mgate"]
    specs = [YAML.load(f"configs/experiments/{i}.yaml")["arch_spec"] for i in ids]
    assert all(s == specs[0] for s in specs)
    assert specs[0]["fusion"] == "mgate"


def test_gate_share_roundtrip_and_summary(tmp_path):
    from src.eval.harness import save_gate_share, summarize_gate_share
    share = {"a": [0.2, 0.4, 0.6], "b": [0.4, 0.6, 0.8], "c": [0.9, 0.9, 0.9]}
    p = tmp_path / "gates_test.csv"
    save_gate_share(share, p)
    s = summarize_gate_share(p, {"all": {"all": ["a", "b", "c"]},
                                 "illum": {"dark": ["a", "b"], "none": ["zz"]}})
    assert s["all"]["all"]["n_images"] == 3
    assert s["illum"]["dark"] == pytest.approx({"n_images": 2, "P3": 0.3, "P4": 0.5, "P5": 0.7})
    assert "none" not in s["illum"]
