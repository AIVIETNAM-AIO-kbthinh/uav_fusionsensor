"""Toán tử hợp nhất đặc trưng cho mid fusion.

F2a (`concat1x1`) chỉ dùng `Concat` + `Conv 1x1` có sẵn của Ultralytics, không cần
gì ở đây.

F2b (`attn`) thêm `GatedFusion`: cổng chú ý theo kênh kiểu SE, học trọng số động
cho từng modality. Đặt SAU `Concat` và TRƯỚC `Conv 1x1` nên **giữ nguyên số kênh**
— nhờ vậy `parse_model` suy `c2 = ch[f]` là đúng, không cần bản vá nào.

Ý đồ: khi điều kiện thay đổi (ví dụ trời tối, RGB mất tác dụng), cổng có thể hạ
trọng số nửa kênh của RGB và nâng nửa kênh của IR. Đây chính là cơ chế mà H3 giả
định làm mid fusion mạnh hơn early fusion.

F2c (`mgate`) thêm `ModalityGate` — bản sửa của `GatedFusion`, CÙNG số tham số
(cùng hình dạng MLP), khác ba điểm đều không tốn tham số nào. Xem docstring lớp.

⚠️ KHÔNG sửa `GatedFusion`. Các run F2b/C2b đã train dùng nó, và harness dựng lại
model từ config rồi nạp `state_dict`: nếu đổi forward mà giữ tên tham số, checkpoint
cũ vẫn nạp "thành công" nhưng chạy hàm khác -> số liệu F2b sai mà không báo lỗi.
"""

from __future__ import annotations

import torch
from torch import nn


class GatedFusion(nn.Module):
    """Cổng chú ý theo kênh trên đặc trưng đã ghép của hai nhánh.

    Args:
        c: tổng số kênh đầu vào (= đầu ra). Phải là số kênh ĐÃ scale.
        r: hệ số thắt cổ chai của MLP.
    """

    def __init__(self, c: int, r: int = 16):
        super().__init__()
        hidden = max(c // r, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))

    def extra_repr(self) -> str:
        return f"c={self.fc[0].in_channels}"


class ModalityGate(nn.Module):
    """Cổng chọn modality theo kênh — bản sửa của `GatedFusion`, cùng số tham số.

    Đầu vào là `Concat([A, B])`: kênh `[0:C)` của nhánh A, `[C:2C)` của nhánh B.
    Ba thay đổi so với `GatedFusion`, cả ba đều 0 tham số thêm:

    1. **Khởi tạo đồng nhất.** Conv cuối khởi tạo bằng 0 -> cổng ≡ 1 tại t=0, nên
       F2c xuất phát ĐÚNG bằng F2a về mặt số học. `GatedFusion` khởi tạo ngẫu nhiên
       nên `Sigmoid` ≈ 0,5: mọi đặc trưng hợp nhất bị co về một nửa ngay lúc
       vừa nạp pretrained đối xứng vào hai nhánh.
    2. **Mô tả avg + max, chung một MLP** (kiểu CBAM). Xe ấm trên nền lạnh trong IR
       là tín hiệu đỉnh; chỉ lấy trung bình toàn cục thì bị bôi mất.
    3. **Softmax giữa hai modality** thay cho sigmoid độc lập: kênh i của A và kênh
       i của B cạnh tranh nhau, `w_A + w_B = 2`. Việc ghép cặp theo chỉ số kênh có
       nghĩa vì hai nhánh nhận CÙNG trọng số pretrained (src/models/build.py), nên
       lúc đầu kênh i của hai nhánh là cùng một bộ lọc. Hệ quả phụ quan trọng: cổng
       đọc được thành "tỉ trọng tin vào B" — xem `record` / `last_share`.

    Args:
        c: tổng số kênh đầu vào (= đầu ra), phải chẵn và là số kênh ĐÃ scale.
        r: hệ số thắt cổ chai của MLP (giống `GatedFusion` để số tham số bằng nhau).
    """

    def __init__(self, c: int, r: int = 16):
        super().__init__()
        if c % 2:
            raise ValueError(f"ModalityGate can so kenh chan (A|B), nhan c={c}")
        hidden = max(c // r, 8)
        self.c_mod = c // 2
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)
        # Bật khi đánh giá để ghi tỉ trọng của nhánh B (IR ở F2c) cho từng ảnh.
        self.record = False
        self.last_share: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x.mean((2, 3), keepdim=True)) + self.fc(x.amax((2, 3), keepdim=True))
        b = x.shape[0]
        p = logits.view(b, 2, self.c_mod, 1, 1).softmax(dim=1)        # (b, 2, C, 1, 1)
        if self.record:
            self.last_share = p[:, 1].mean(dim=(1, 2, 3)).detach().float().cpu()
        return x * p.mul(2).view(b, 2 * self.c_mod, 1, 1)

    def extra_repr(self) -> str:
        return f"c={2 * self.c_mod}"


def register() -> None:
    """Đăng ký module vào namespace mà `parse_model` tra cứu (`globals()` của tasks)."""
    from ultralytics.nn import tasks

    tasks.GatedFusion = GatedFusion
    tasks.ModalityGate = ModalityGate
