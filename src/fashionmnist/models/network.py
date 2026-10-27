"""
network.py — Kiến trúc mạng nơ-ron phân loại ảnh thời trang FashionMNIST.

Bao gồm hai kiến trúc đối chứng:
1. BaselineMLP: Mạng nơ-ron kết nối đầy đủ (Multi-Layer Perceptron), làm phẳng ảnh 28x28 thành vector 784 chiều.
   - Dùng làm mốc cơ sở (baseline) so sánh.
   - Nhược điểm: Phá vỡ cấu trúc không gian 2D của ảnh, bùng nổ số lượng tham số kết nối.
2. SimpleCNN: Mạng nơ-ron tích chập (Convolutional Neural Network) với 2 khối ConvBlock + Global Average Pooling (GAP).
   - Khai thác triệt để thiên kiến cảm nạp (inductive bias / translation invariance) của ảnh.
   - Sử dụng GAP nén feature map về 1x1 thay vì flatten toàn bộ, giảm mạnh số tham số xuống ~38k và hạn chế overfitting.

Quy tắc thiết kế (RULES.md):
- Đầu ra của forward pass luôn là 10 raw logits (chưa qua Softmax) để kết hợp ổn định với nn.CrossEntropyLoss.
- Tương thích hoàn toàn với cả CPU, CUDA và Apple Silicon MPS.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Baseline Model: Multi-Layer Perceptron (MLP)
# ---------------------------------------------------------------------------

class BaselineMLP(nn.Module):
    """
    Mạng nơ-ron kết nối đầy đủ (Baseline MLP).

    Kiến trúc:
        Input (B, 1, 28, 28)
        -> Flatten -> Vector (B, 784)
        -> Linear(784, hidden_units) -> ReLU -> Dropout(p)
        -> Linear(hidden_units, 10) -> Logits (B, 10)
    """

    def __init__(self, hidden_units: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 28 * 28, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_units, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền xuôi (Forward pass).
        Args:
            x: Tensor ảnh kích thước (B, 1, 28, 28) hoặc (B, 784).
        Returns:
            Tensor logits chưa qua softmax kích thước (B, 10).
        """
        if x.dim() != 4 and x.dim() != 2:
            raise ValueError(f"BaselineMLP yêu cầu input 4D (B, 1, 28, 28) hoặc 2D (B, 784), nhận được: {x.shape}")
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. Main Model: Simple Convolutional Neural Network (CNN)
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    Mạng nơ-ron tích chập gọn nhẹ cho FashionMNIST (SimpleCNN).

    Kiến trúc phân lớp:
        [Khối Conv 1]: Conv2d(1 -> 32, 3x3, pad=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2x2) [28x28 -> 14x14]
        [Khối Conv 2]: Conv2d(32 -> 64, 3x3, pad=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2x2) [14x14 -> 7x7]
        [Global Average Pooling]: AdaptiveAvgPool2d(1x1) nén (B, 64, 7, 7) thành (B, 64, 1, 1)
        [Bộ phân loại Classifier]: Flatten -> Linear(64, fc_hidden) -> ReLU -> Dropout(0.3) -> Linear(fc_hidden, 10)

    Lý do thiết kế:
    - ConvBlock 1: Trích xuất các đặc trưng biên cạnh, đường vân cơ bản của trang phục.
    - ConvBlock 2: Trích xuất các đặc trưng hình học kết hợp bậc cao (cổ áo, phom dáng, quai dép).
    - BatchNorm: Chuẩn hóa phân phối kích hoạt giữa các batch, ổn định gradient và tăng tốc hội tụ.
    - Global Average Pooling (GAP): Thay thế lớp Flatten truyền thống, giảm số tham số nối vào FC từ 64*7*7=3136 xuống 64,
      giúp mô hình chỉ khoảng ~38k tham số nhưng chống overfitting cực kỳ mạnh mẽ.
    - Không áp dụng Softmax ở tầng cuối: nn.CrossEntropyLoss trong PyTorch đã tích hợp sẵn LogSoftmax
      giúp đảm bảo tính ổn định số học (numerical stability).
    """

    def __init__(
        self,
        hidden_channels: list[int] | None = None,
        fc_hidden: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64]
        c1, c2 = hidden_channels[0], hidden_channels[1]

        # Khối Conv 1: 1 kênh xám -> c1 kênh đặc trưng
        # Dùng bias=False vì lớp BatchNorm2d ngay sau đã có tham số dịch chuyển (beta)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
        )

        # Khối Conv 2: c1 kênh -> c2 kênh đặc trưng
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
        )

        # Global Average Pooling: Ép feature map 7x7 về 1x1 cho từng kênh
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)  # (B, c2, 1, 1)

        # Bộ phân loại (Classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền xuôi (Forward pass).
        Input: (B, 1, 28, 28) -> Output: (B, 10)
        """
        if x.dim() != 4:
            raise ValueError(f"SimpleCNN yêu cầu input 4D (B, 1, 28, 28), nhận được: {x.shape}")
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.global_avg_pool(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 3. Model Factory & Utilities
# ---------------------------------------------------------------------------

def build_model(cfg: dict[str, Any]) -> nn.Module:
    """
    Khởi tạo mô hình dựa trên từ điển cấu hình hệ thống.

    Args:
        cfg: Từ điển cấu hình (có trường 'model').

    Returns:
        Instance nn.Module tương ứng (BaselineMLP hoặc SimpleCNN).
    """
    model_cfg = cfg.get("model", {})
    arch = model_cfg.get("architecture", "SimpleCNN")

    if arch == "BaselineMLP":
        return BaselineMLP(
            hidden_units=model_cfg.get("hidden_units", 256),
            dropout=model_cfg.get("dropout", 0.0),
        )
    elif arch == "SimpleCNN":
        return SimpleCNN(
            hidden_channels=model_cfg.get("hidden_channels", [32, 64]),
            fc_hidden=model_cfg.get("fc_hidden", 256),
            dropout=model_cfg.get("dropout", 0.3),
        )
    else:
        raise ValueError(
            f"Kiến trúc không xác định: '{arch}'. Các kiến trúc hợp lệ: ['BaselineMLP', 'SimpleCNN']."
        )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Đếm tổng số lượng tham số của mô hình.

    Args:
        model: Mô hình PyTorch.
        trainable_only: Nếu True, chỉ đếm các tham số cần cập nhật gradient (requires_grad=True).

    Returns:
        Tổng số tham số nguyên.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
