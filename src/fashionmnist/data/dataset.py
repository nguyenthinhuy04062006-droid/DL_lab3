"""
dataset.py — Tải dữ liệu FashionMNIST, phân chia phân tầng (Stratified Split) và tạo DataLoader.

Tuân thủ nghiêm ngặt RULES.md:
- Stratified Split 80/20 trên 60.000 mẫu train (Train: 48.000, Val: 12.000).
- Tập Test (10.000 mẫu) được cô lập hoàn toàn.
- Kiểm tra tính không chồng lấn (No-overlap) giữa tập Train và Validation.
- Áp dụng augmentation riêng cho Train Subset và eval transform cho Val/Test Subsets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as T

from src.fashionmnist.data.transforms import get_eval_transform, get_train_transform


def load_raw_dataset(
    data_root: str | Path,
    train: bool = True,
    download: bool = True,
) -> datasets.FashionMNIST:
    """
    Nạp dữ liệu FashionMNIST với ToTensor() cơ bản (chưa qua chuẩn hóa), phục vụ EDA.

    Args:
        data_root: Thư mục chứa dữ liệu gốc.
        train: True để nạp tập train (60.000 mẫu), False cho tập test (10.000 mẫu).
        download: Tự động tải nếu chưa tồn tại.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    return datasets.FashionMNIST(
        root=str(data_root),
        train=train,
        transform=T.ToTensor(),
        download=download,
    )


def load_dataset_with_transform(
    data_root: str | Path,
    train: bool,
    transform: T.Compose,
    download: bool = True,
) -> datasets.FashionMNIST:
    """Nạp FashionMNIST với pipeline transforms chỉ định."""
    data_root = Path(data_root)
    return datasets.FashionMNIST(
        root=str(data_root),
        train=train,
        transform=transform,
        download=download,
    )


def stratified_split(
    full_train_ds: datasets.FashionMNIST,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Phân chia chỉ số tập Train/Validation theo chiến lược phân tầng (Stratified Split).

    Bảo tồn chính xác tỷ lệ 10 lớp cân bằng giữa hai tập:
    Ví dụ với 60.000 mẫu (mỗi lớp 6.000 mẫu), tỷ lệ val_fraction = 0.20:
    - Train: 48.000 mẫu (mỗi lớp 4.800 mẫu).
    - Validation: 12.000 mẫu (mỗi lớp 1.200 mẫu).

    Args:
        full_train_ds: Dataset 60.000 mẫu gốc.
        val_fraction: Tỷ lệ tập validation (mặc định 0.20 tức 20%).
        seed: Hạt giống ngẫu nhiên đảm bảo tính tái lập.

    Returns:
        Tuple (train_indices, val_indices).
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction phải nằm trong khoảng (0, 1), nhận được: {val_fraction}")

    labels = full_train_ds.targets
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    labels_np = labels.numpy()

    rng = np.random.default_rng(seed=seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for cls in range(10):
        cls_idx = np.where(labels_np == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_fraction))
        val_indices.extend(cls_idx[:n_val].tolist())
        train_indices.extend(cls_idx[n_val:].tolist())

    return train_indices, val_indices


def verify_no_overlap(train_indices: list[int], val_indices: list[int]) -> bool:
    """
    Kiểm tra tính không chồng lấn giữa tập Train và Validation.

    Returns:
        True nếu không có bất kỳ mẫu nào trùng nhau, ngược lại False.
    """
    overlap = set(train_indices).intersection(set(val_indices))
    return len(overlap) == 0


def build_dataloaders(
    cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Xây dựng 3 DataLoader hoàn chỉnh: train_loader, val_loader, test_loader.

    Quy trình:
    1. Tải dữ liệu thô để trích xuất nhãn và tính toán chỉ số Stratified Split.
    2. Khởi tạo dataset với train_transform cho tập Train, eval_transform cho Val và Test.
    3. Tạo Subset độc lập cho Train và Validation.
    4. Bọc các tập dữ liệu trong PyTorch DataLoader theo batch_size từ config.

    Args:
        cfg: Từ điển cấu hình hệ thống.

    Returns:
        Tuple (train_loader, val_loader, test_loader).
    """
    data_root = cfg["paths"]["data_root"]
    seed = cfg["project"]["seed"]
    val_fraction = cfg["data"]["val_fraction"]
    batch_size = cfg["dataloader"]["batch_size"]
    num_workers = cfg["dataloader"].get("num_workers", 0)
    pin_memory = cfg["dataloader"].get("pin_memory", False)

    # 1. Nạp dataset để trích xuất chỉ số phân tầng
    raw_train_ds = load_raw_dataset(data_root=data_root, train=True, download=True)
    train_indices, val_indices = stratified_split(
        raw_train_ds, val_fraction=val_fraction, seed=seed
    )

    if not verify_no_overlap(train_indices, val_indices):
        raise RuntimeError("Phát hiện chồng lấn chỉ số giữa tập Train và Validation (Data Leakage)!")

    # 2. Khởi tạo dataset với transforms tương ứng
    train_transform = get_train_transform(cfg)
    eval_transform = get_eval_transform(cfg)

    train_base_ds = load_dataset_with_transform(
        data_root=data_root, train=True, transform=train_transform, download=False
    )
    val_base_ds = load_dataset_with_transform(
        data_root=data_root, train=True, transform=eval_transform, download=False
    )
    test_ds = load_dataset_with_transform(
        data_root=data_root, train=False, transform=eval_transform, download=True
    )

    # 3. Tạo Subsets
    train_ds = Subset(train_base_ds, train_indices)
    val_ds = Subset(val_base_ds, val_indices)

    # 4. Bọc DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Alias thuận tiện cho build_dataloaders theo interface trong project_design.md."""
    return build_dataloaders(cfg)
