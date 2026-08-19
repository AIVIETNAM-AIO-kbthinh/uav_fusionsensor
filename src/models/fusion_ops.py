"""Toán tử hợp nhất đặc trưng cho mid fusion.

F2a (`concat1x1`) chỉ dùng `Concat` + `Conv 1x1` có sẵn của Ultralytics, không cần
gì ở đây.

F2b (`attn`) thêm `GatedFusion`: cổng chú ý theo kênh kiểu SE, học trọng số động
cho từng modality. Đặt SAU `Concat` và TRƯỚC `Conv 1x1` nên **giữ nguyên số kênh**
— nhờ vậy `parse_model` suy `c2 = ch[f]` là đúng, không cần bản vá nào.

Ý đồ: khi điều kiện thay đổi (ví dụ trời tối, RGB mất tác dụng), cổng có thể hạ
trọng số nửa kênh của RGB và nâng nửa kênh của IR. Đây chính là cơ chế mà H3 giả
định làm mid fusion mạnh hơn early fusion.
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


def register() -> None:
    """Đăng ký module vào namespace mà `parse_model` tra cứu (`globals()` của tasks)."""
    from ultralytics.nn import tasks

    tasks.GatedFusion = GatedFusion
