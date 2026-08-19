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

    # bất biến quan trọng nhất của đề tài
    same = [e["arch_spec"] for e in EXPERIMENTS if e["id"] in ("C1", "C2", "F2a")]
    assert all(s == same[0] for s in same), "C1/C2/F2a PHAI dung chung arch_spec"
    print("\n[OK] C1, C2, F2a dung chung arch_spec -> doi chung capacity hop le")


if __name__ == "__main__":
    main()
