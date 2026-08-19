"""Ba bản vá tối thiểu vào Ultralytics để chạy được fusion đa kênh.

Nguyên tắc: **không fork Ultralytics**. Chỉ thay đúng ba điểm, mỗi điểm đều
idempotent và có test phủ (tests/test_models.py, tests/test_data.py).

Bản vá 3 (`apply_warp_patch`) là SỬA LỖI THẬT của Ultralytics trên ảnh >4 kênh —
xem docstring của hàm đó. Hai bản vá còn lại chỉ là điểm mở rộng.

--------------------------------------------------------------------------------
Bản vá 1 — `Index` hỗ trợ cắt kênh
--------------------------------------------------------------------------------
`parse_model` tính số kênh đầu ra `c2` cho từng lớp. Với module lạ nó dùng
`c2 = ch[f]` (giữ nguyên số kênh), nên không thể viết một module cắt 6 kênh
xuống 3 mà `parse_model` hiểu đúng.

Ultralytics có sẵn một nhánh đặc biệt:

    elif m in frozenset({TorchVision, Index}):
        c2 = args[0]; args = [*args[1:]]

tức `Index` được phép khai báo `c2` tường minh qua `args[0]`. `frozenset` này
được dựng lại mỗi lần gọi `parse_model` từ biến global của module, nên gán đè
`ultralytics.nn.tasks.Index` là đủ để lớp của ta nhận suất đó.

Lớp thay thế giữ nguyên hành vi cũ (chọn phần tử từ list) và thêm hành vi mới
(cắt kênh khi đầu vào là tensor), nên tương thích ngược hoàn toàn.

    YAML cũ:  [x, 1, Index, [c2, idx]]        -> chọn phần tử idx của list
    YAML mới: [x, 1, Index, [3, 0, 3]]        -> tensor[:, 0:3]

--------------------------------------------------------------------------------
Bản vá 2 — `build_yolo_dataset` trả về dataset ghép cặp
--------------------------------------------------------------------------------
Cả `DetectionTrainer.build_dataset` lẫn `DetectionValidator.build_dataset` đều
gọi `build_yolo_dataset(args, img_path, batch, data, ...)`. Ta chỉ cần chèn:
nếu `data["modalities"]` tồn tại thì dựng `PairedYOLODataset`, ngược lại giữ
nguyên hành vi gốc.

Nhờ đi qua `data` (dict được truyền cho cả trainer và validator) mà train và
val luôn dùng cùng một cấu hình modality — không thể lệch.
"""

from __future__ import annotations

from torch import nn

_APPLIED = {"index": False, "dataset": False, "warp": False}


def apply_warp_patch() -> None:
    """Sửa lỗi lấp viền của `RandomPerspective` trên ảnh nhiều hơn 4 kênh.

    ĐÂY LÀ BẢN VÁ SỬA LỖI THẬT, không phải mở rộng tiện nghi.

    `RandomPerspective.apply_image` gọi::

        cv2.warpAffine(img, M[:2], dsize=size, borderValue=(114, 114, 114))

    `borderValue` của OpenCV là một `Scalar`, tối đa **4 thành phần**. Với ảnh 6
    kênh, bốn kênh đầu được lấp 114 còn **kênh 5 và 6 bị lấp 0**. Phép biến đổi
    hình học vẫn đồng nhất trên mọi kênh (đúng), nhưng vùng viền lộ ra sau khi
    xoay/co giãn/tịnh tiến lại mang giá trị khác nhau giữa hai modality.

    Hậu quả nếu không sửa:
      * model học được artefact "viền IR bằng 0" thay vì học nội dung;
      * đối chứng C1 (nạp cùng RGB vào hai luồng) KHÔNG còn đối xứng thật sự,
        làm hỏng chính phép đối chứng capacity là đóng góp chính của đề tài.

    Đây đúng là loại bug âm thầm mà plan mục 7.3 cảnh báo: không có thông báo lỗi
    nào, chỉ có kết quả sai. `tests/test_data.py::test_augment_sync` bắt được nó.

    Cách sửa: warp từng nhóm tối đa 3 kênh bằng CÙNG ma trận và CÙNG giá trị viền,
    rồi ghép lại. Ảnh 3 kênh đi đúng đường code cũ nên không đổi hành vi.
    """
    if _APPLIED["warp"]:
        return
    import cv2
    import numpy as np
    from ultralytics.data.augment import RandomPerspective

    original = RandomPerspective.apply_image

    def apply_image(self, labels, params=None):
        img = labels["img"]
        if img.ndim != 3 or img.shape[2] <= 4:
            return original(self, labels, params)

        M, size = params["M"], params["size"]
        if (size[0] == img.shape[1] and size[1] == img.shape[0]) and not (M != np.eye(3)).any():
            return labels

        warp = ((lambda a: cv2.warpPerspective(a, M, dsize=size, borderValue=(114, 114, 114)))
                if self.perspective else
                (lambda a: cv2.warpAffine(a, M[:2], dsize=size, borderValue=(114, 114, 114))))
        chunks = []
        for c0 in range(0, img.shape[2], 3):
            part = warp(img[:, :, c0:c0 + 3])
            chunks.append(part[..., None] if part.ndim == 2 else part)
        out = np.concatenate(chunks, axis=2)
        labels["img"] = out
        labels["resized_shape"] = out.shape[:2]
        return labels

    RandomPerspective.apply_image = apply_image
    _APPLIED["warp"] = True


class Index(nn.Module):
    """Chọn phần tử từ list (hành vi gốc) HOẶC cắt kênh của tensor (mở rộng).

    Args:
        index: chỉ số phần tử (list) hoặc kênh bắt đầu (tensor).
        end: kênh kết thúc; chỉ dùng cho tensor. None -> giữ hành vi gốc.
    """

    def __init__(self, index: int = 0, end: int | None = None):
        super().__init__()
        self.index = index
        self.end = end

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return x[self.index]
        if self.end is None:
            return x[:, self.index]
        return x[:, self.index : self.end]

    def extra_repr(self) -> str:
        return f"index={self.index}, end={self.end}"


def apply_index_patch() -> None:
    """Gán đè `ultralytics.nn.tasks.Index`. Idempotent."""
    if _APPLIED["index"]:
        return
    from ultralytics.nn import tasks

    tasks.Index = Index
    _APPLIED["index"] = True


def apply_dataset_patch() -> None:
    """Chèn PairedYOLODataset vào `build_yolo_dataset` ở mọi namespace đã import nó."""
    if _APPLIED["dataset"]:
        return
    from ultralytics.data import build as build_mod
    from ultralytics.models.yolo.detect import train as det_train
    from ultralytics.models.yolo.detect import val as det_val

    original = build_mod.build_yolo_dataset

    def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
        if not data.get("modalities"):
            return original(cfg, img_path, batch, data, mode=mode, rect=rect, stride=stride,
                            multi_modal=multi_modal)
        from src.data.paired_dataset import PairedYOLODataset

        return PairedYOLODataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=cfg,
            rect=cfg.rect or rect,
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=cfg.task,
            classes=cfg.classes,
            data=data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )

    for mod in (build_mod, det_train, det_val):
        if hasattr(mod, "build_yolo_dataset"):
            mod.build_yolo_dataset = build_yolo_dataset
    _APPLIED["dataset"] = True


def apply_all() -> None:
    apply_index_patch()
    apply_dataset_patch()
    apply_warp_patch()


def status() -> dict:
    return dict(_APPLIED)
