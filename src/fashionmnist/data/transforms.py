"""
transforms.py — Torchvision transforms cho FashionMNIST.

Quy tắc chống rò rỉ dữ liệu (RULES.md):
- Tập Train: ToTensor() -> Normalize(mean, std) -> Data Augmentation (RandomHorizontalFlip, RandomCrop).
- Tập Validation & Test: ToTensor() -> Normalize(mean, std). Tuyệt đối KHÔNG áp dụng Data Augmentation.
- Tham số chuẩn hóa Z-score: mean = 0.2860, std = 0.3530 được ước lượng duy nhất trên tập Train.
"""
from __future__ import annotations

from typing import Any
import torchvision.transforms as T


def get_train_transform(cfg: dict[str, Any]) -> T.Compose:
    """
    Tạo pipeline tiền xử lý cho tập Train.

    Bao gồm:
    1. ToTensor: Chuyển đổi ảnh PIL sang Tensor [0, 1] kích thước (1, 28, 28).
    2. Data Augmentation (nếu augment_train=True):
       - RandomHorizontalFlip(p=0.5): Lật ngang ngẫu nhiên.
       - RandomCrop(28, padding=2): Cắt và đệm biên giữ kích thước 28x28.
    3. Normalize: Chuẩn hóa Z-score với mean và std tính từ tập Train.
    """
    mean = cfg.get("data", {}).get("normalize_mean", 0.2860)
    std = cfg.get("data", {}).get("normalize_std", 0.3530)
    augment = cfg.get("data", {}).get("augment_train", True)

    transforms_list: list[Any] = []
    if augment:
        transforms_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(28, padding=2),
        ])
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=(mean,), std=(std,)),
    ])
    return T.Compose(transforms_list)


def get_eval_transform(cfg: dict[str, Any]) -> T.Compose:
    """
    Tạo pipeline tiền xử lý tất định cho tập Validation và Test.

    Chỉ bao gồm ToTensor và Normalize với tham số (mean, std) của tập Train.
    Tuyệt đối không áp dụng các phép biến đổi ngẫu nhiên nhằm giữ nguyên phân phối kiểm thử.
    """
    mean = cfg.get("data", {}).get("normalize_mean", 0.2860)
    std = cfg.get("data", {}).get("normalize_std", 0.3530)
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(mean,), std=(std,)),
    ])
