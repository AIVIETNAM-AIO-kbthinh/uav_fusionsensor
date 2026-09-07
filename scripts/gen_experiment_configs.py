"""Sinh 8 file config thí nghiệm của ma trận.

Sinh tự động thay vì viết tay để đảm bảo điều quan trọng nhất của đề tài:
**C1, C2 và F2a dùng CHUNG một `arch_spec`** — cùng kiến trúc, cùng số tham số,
cùng FLOPs. Chỉ khác đúng một trường `modalities`. Nếu viết tay thì không có gì
bảo đảm điều đó vẫn đúng sau vài lần sửa.
"""

from pathlib import Path

from ultralytics.utils import YAML

SCALE = "s"
BASE_ARCH = "yolo11-obb.yaml"
PRETRAINED = f"yolo11{SCALE}-obb.pt"

SINGLE = {"arch": "single", "base": BASE_ARCH, "scale": SCALE}
EARLY = {"arch": "single", "base": BASE_ARCH, "scale": SCALE}
TWO_CONCAT = {"arch": "two_stream", "base": BASE_ARCH, "scale": SCALE,
              "split": [3, 3], "fusion": "concat1x1"}
TWO_ATTN = {"arch": "two_stream", "base": BASE_ARCH, "scale": SCALE,
            "split": [3, 3], "fusion": "attn"}

EXPERIMENTS = [
    dict(id="S1", name="RGB single-stream", role="baseline",
         modalities=["rgb"], arch_spec=SINGLE),
    dict(id="S2", name="IR single-stream", role="baseline",
         modalities=["ir"], arch_spec=SINGLE),
    dict(id="C1", name="doi chung capacity - RGB vao ca hai luong",
         role="capacity_control", modalities=["rgb", "rgb"], arch_spec=TWO_CONCAT),
    dict(id="C2", name="doi chung capacity - IR vao ca hai luong",
         role="capacity_control", modalities=["ir", "ir"], arch_spec=TWO_CONCAT),
    dict(id="C1b", name="doi chung capacity attn - RGB vao ca hai luong",
         role="capacity_control", modalities=["rgb", "rgb"], arch_spec=TWO_ATTN),
    dict(id="C2b", name="doi chung capacity attn - IR vao ca hai luong",
         role="capacity_control", modalities=["ir", "ir"], arch_spec=TWO_ATTN),
    dict(id="F1", name="early fusion 4 kenh", role="fusion",
         modalities=["rgb", "ir1"], arch_spec=EARLY),
    dict(id="F2a", name="mid fusion concat+1x1", role="fusion",
         modalities=["rgb", "ir"], arch_spec=TWO_CONCAT),
    dict(id="F2b", name="mid fusion attention gate", role="fusion",
         modalities=["rgb", "ir"], arch_spec=TWO_ATTN),
    dict(id="F3", name="late fusion WBF tu S1+S2", role="fusion_posthoc",
         modalities=["rgb"], arch_spec=SINGLE,
         note="khong train; suy ra tu weight cua S1 va S2, xem src/fusion/wbf_obb.py"),
]

FILENAMES = {"S1": "S1_rgb", "S2": "S2_ir", "C1": "C1_dup_rgb", "C2": "C2_dup_ir",
             "C1b": "C1b_dup_rgb_attn", "C2b": "C2b_dup_ir_attn",
             "F1": "F1_early", "F2a": "F2a_mid_concat", "F2b": "F2b_mid_attn",
             "F3": "F3_late_wbf"}


def main():
    out = Path("configs/experiments")
    out.mkdir(parents=True, exist_ok=True)
    for e in EXPERIMENTS:
        e = dict(e)
        e["pretrained_weights"] = PRETRAINED
        e["normalize"] = "none"
        YAML.save(str(out / f"{FILENAMES[e['id']]}.yaml"), e)
        print(f"  {FILENAMES[e['id']]}.yaml  modalities={e['modalities']} arch={e['arch_spec']['arch']}")

    # Moi toan tu hop nhat phai co doi chung capacity dung CHUNG arch_spec voi
    # cau hinh fusion tuong ung - neu khong, F2b vs C2 lech 198.784 tham so.
    by_id = {e["id"]: e["arch_spec"] for e in EXPERIMENTS}
    print()
    for group in (("C1", "C2", "F2a"), ("C1b", "C2b", "F2b")):
        specs = [by_id[i] for i in group]
        assert all(x == specs[0] for x in specs), f"{'/'.join(group)} PHAI dung chung arch_spec"
        print(f"[OK] {', '.join(group)} dung chung arch_spec -> doi chung capacity hop le")
    assert by_id["F2a"] != by_id["F2b"], "F2a va F2b phai khac toan tu hop nhat"


if __name__ == "__main__":
    main()
