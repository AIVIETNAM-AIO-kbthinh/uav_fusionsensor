"""Test dữ liệu: ghép cặp, đồng bộ augmentation, cắt viền, chuẩn hoá.

Đây là nhóm test quan trọng nhất của dự án. Các lỗi ở đây KHÔNG báo lỗi lúc chạy
mà chỉ làm kết quả sai một cách âm thầm — plan mục 7.3 xếp `test_pairing` là
"bug có thể huỷ toàn bộ kết quả".
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src import patches
from src.data.paired_dataset import MODALITY_SPEC, modality_channels, swap_view

ROOT = Path("data/dronevehicle")
HAS_DATA = (ROOT / "rgb" / "images" / "val").is_dir()
pytestmark = pytest.mark.skipif(not HAS_DATA, reason="chua co du lieu DroneVehicle da xu ly")


def _ids(split="val", n=30):
    files = sorted((ROOT / "rgb" / "images" / split).glob("*.jpg"))[:n]
    return [f.stem for f in files]


def test_swap_view():
    p = "data/dronevehicle/rgb/images/val/00001.jpg"
    assert swap_view(p, "ir").replace("\\", "/") == "data/dronevehicle/ir/images/val/00001.jpg"
    with pytest.raises(ValueError):
        swap_view("data/foo/bar.jpg", "ir")


def test_modality_channels():
    assert modality_channels(["rgb"]) == 3
    assert modality_channels(["ir"]) == 3
    assert modality_channels(["rgb", "ir"]) == 6
    assert modality_channels(["rgb", "ir1"]) == 4
    assert modality_channels(["ir", "ir"]) == 6


def test_border_removed():
    """Ảnh sau tiền xử lý đúng 640x512 và không còn viền trắng."""
    for sid in _ids(n=10):
        for view in ("rgb", "ir"):
            im = cv2.imread(str(ROOT / view / "images" / "val" / f"{sid}.jpg"), cv2.IMREAD_UNCHANGED)
            assert im is not None, sid
            assert im.shape[:2] == (512, 640), f"{view}/{sid}: {im.shape}"
            g = im if im.ndim == 2 or im.shape[2] == 1 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            g = g.squeeze()
            # viền trắng cũ có giá trị 255 ở toàn bộ dải 100px; sau khi cắt không
            # được còn dải nào như vậy ở mép
            for edge in (g[0, :], g[-1, :], g[:, 0], g[:, -1]):
                assert not np.all(edge == 255), f"{view}/{sid}: con vien trang"


def test_labels_normalized():
    """Mọi toạ độ nhãn nằm trong [0,1] và đúng 8 giá trị mỗi dòng."""
    for sid in _ids(n=50):
        p = ROOT / "rgb" / "labels" / "val" / f"{sid}.txt"
        for line in p.read_text().splitlines():
            parts = line.split()
            assert len(parts) == 9, f"{sid}: {line!r}"
            v = np.array([float(x) for x in parts[1:]])
            assert (v >= -1e-6).all() and (v <= 1 + 1e-6).all(), f"{sid}: {v}"
            assert 0 <= int(parts[0]) <= 4


def test_pairing_is_same_scene():
    """RGB[i] và IR[i] phải là cùng một cảnh.

    Tên file trong XML gốc KHÔNG dùng làm khoá được (dataset có nhiều quy ước đặt
    tên, xem DATA_REPORT.md 2.5), nên kiểm chứng bằng nội dung ảnh: thông tin
    tương hỗ của cặp khớp phải cao hơn cặp ghép lệch.
    """
    from scripts.check_alignment import nmi

    ids = _ids(n=40)
    matched, shuffled = [], []
    for k, sid in enumerate(ids):
        rgb = cv2.imread(str(ROOT / "rgb" / "images" / "val" / f"{sid}.jpg"), cv2.IMREAD_GRAYSCALE)
        ir = cv2.imread(str(ROOT / "ir" / "images" / "val" / f"{sid}.jpg"), cv2.IMREAD_GRAYSCALE)
        other = ids[(k + 7) % len(ids)]
        ir_o = cv2.imread(str(ROOT / "ir" / "images" / "val" / f"{other}.jpg"), cv2.IMREAD_GRAYSCALE)
        matched.append(nmi(rgb, ir))
        shuffled.append(nmi(rgb, ir_o))
    matched, shuffled = np.array(matched), np.array(shuffled)
    assert matched.mean() > 2 * shuffled.mean(), (matched.mean(), shuffled.mean())
    assert (matched > shuffled).mean() > 0.8


def _build_dataset(modalities, augment, imgsz=640, seed=0):
    from ultralytics.cfg import get_cfg

    from src.data.paired_dataset import PairedYOLODataset

    patches.apply_all()
    data = {
        "path": str(ROOT / "rgb"), "train": "images/val", "val": "images/val",
        "names": {i: n for i, n in enumerate(["car", "truck", "bus", "van", "freight_car"])},
        "nc": 5, "channels": modality_channels(modalities), "modalities": list(modalities),
    }
    hyp = get_cfg(overrides={"task": "obb", "imgsz": imgsz, "mosaic": 0.0, "mixup": 0.0,
                             "cutmix": 0.0, "copy_paste": 0.0, "erasing": 0.0,
                             "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
                             "degrees": 0.0, "translate": 0.0, "scale": 0.0,
                             "fliplr": 1.0 if augment else 0.0, "flipud": 0.0})
    return PairedYOLODataset(img_path=str(ROOT / "rgb" / "images" / "val"), imgsz=imgsz,
                             batch_size=2, augment=augment, hyp=hyp, rect=False, cache=None,
                             single_cls=False, stride=32, pad=0.0, prefix="test: ",
                             task="obb", classes=None, data=data, fraction=1.0)


@pytest.mark.parametrize("modalities,ch", [(["rgb"], 3), (["ir"], 3),
                                           (["rgb", "ir"], 6), (["rgb", "ir1"], 4),
                                           (["ir", "ir"], 6)])
def test_dataset_channels(modalities, ch):
    ds = _build_dataset(modalities, augment=False)
    s = ds[0]
    assert s["img"].shape[0] == ch, (modalities, s["img"].shape)


def test_duplicate_control_streams_identical():
    """C1/C2: hai nửa kênh phải giống hệt nhau — đó là định nghĩa của đối chứng."""
    ds = _build_dataset(["ir", "ir"], augment=False)
    img = ds[0]["img"].numpy()
    assert np.array_equal(img[:3], img[3:]), "hai luong cua doi chung khong giong nhau"


def test_fusion_streams_differ():
    """F2a: hai nửa kênh phải KHÁC nhau, nếu không thì đã nạp nhầm dữ liệu."""
    ds = _build_dataset(["rgb", "ir"], augment=False)
    img = ds[0]["img"].numpy()
    assert not np.array_equal(img[:3], img[3:])


def test_augment_sync():
    """Augmentation hình học phải tác động ĐỒNG NHẤT lên mọi kênh.

    Cách kiểm chắc chắn nhất: nạp CÙNG một modality vào cả hai luồng. Nếu pipeline
    biến đổi từng modality bằng các lần rút ngẫu nhiên riêng — hoặc lấp viền khác
    nhau giữa các nhóm kênh — thì hai nửa sẽ lệch nhau. Chúng phải bằng nhau từng
    pixel sau MỌI phép augment.

    Test này đã bắt được một lỗi thật: `cv2.warpAffine` chỉ nhận `borderValue` tối
    đa 4 thành phần nên `RandomPerspective` lấp viền 114 cho 4 kênh đầu và 0 cho
    các kênh sau. Xem `src.patches.apply_warp_patch`.
    """
    ds = _build_dataset(["rgb", "rgb"], augment=True)
    for idx in (0, 3, 7, 11, 17):
        img = ds[idx]["img"].numpy()
        assert np.array_equal(img[:3], img[3:]), (
            f"idx={idx}: hai luong lech nhau sau augment "
            f"(max |hieu| = {np.abs(img[:3].astype(int) - img[3:].astype(int)).max()})"
        )


def test_augment_sync_border_fill():
    """Vùng viền lộ ra sau biến đổi hình học phải cùng giá trị ở mọi kênh."""
    ds = _build_dataset(["rgb", "ir"], augment=True)
    seen_pad = False
    for idx in range(12):
        img = ds[idx]["img"].numpy()
        pad = (img == 114).all(axis=0)
        if pad.sum() < 100:
            continue
        seen_pad = True
        for c in range(img.shape[0]):
            assert (img[c][pad] == 114).all(), f"kenh {c} co gia tri vien khac 114"
    assert seen_pad, "khong gap vung vien nao de kiem tra"


def test_ir_shift_only_affects_ir():
    """Ablation RQ4: dịch IR không được đụng tới kênh RGB."""
    from ultralytics.cfg import get_cfg

    from src.data.paired_dataset import PairedYOLODataset
    patches.apply_all()

    def build(shift):
        data = {
            "path": str(ROOT / "rgb"), "train": "images/val", "val": "images/val",
            "names": {i: str(i) for i in range(5)}, "nc": 5,
            "channels": 6, "modalities": ["rgb", "ir"],
        }
        if shift:
            data["ir_shift"] = list(shift)
        hyp = get_cfg(overrides={"task": "obb", "imgsz": 640, "mosaic": 0.0, "fliplr": 0.0,
                                 "scale": 0.0, "translate": 0.0, "erasing": 0.0})
        return PairedYOLODataset(img_path=str(ROOT / "rgb" / "images" / "val"), imgsz=640,
                                 batch_size=2, augment=False, hyp=hyp, rect=False, cache=None,
                                 single_cls=False, stride=32, pad=0.0, prefix="t: ",
                                 task="obb", classes=None, data=data, fraction=1.0)

    base = build(None)[0]["img"].numpy()
    shifted = build((10, 0))[0]["img"].numpy()
    assert np.array_equal(base[:3], shifted[:3]), "dich IR da lam doi kenh RGB"
    assert not np.array_equal(base[3:], shifted[3:]), "dich IR khong co tac dung"
