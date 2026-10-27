"""
tests/test_dataset.py
Kiểm thử đơn vị toàn diện cho Module M2: Data Pipeline.
Tương thích hoàn toàn với unittest và pytest.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Đảm bảo in ấn tiếng Việt trơn tru trên Windows console (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from PIL import Image

from src.fashionmnist.config import load_config
from src.fashionmnist.data.dataset import (
    build_dataloaders,
    load_raw_dataset,
    stratified_split,
    verify_no_overlap,
)
from src.fashionmnist.data.transforms import get_eval_transform, get_train_transform


class TestDataPipeline(unittest.TestCase):
    """Kiểm thử quy trình nạp dữ liệu, biến đổi không rò rỉ, và phân chia phân tầng."""

    @classmethod
    def setUpClass(cls) -> None:
        """Nạp cấu hình và dataset thô dùng chung cho toàn bộ test suite."""
        cls.cfg = load_config()
        cls.data_root = cls.cfg["paths"]["data_root"]
        cls.raw_train_ds = load_raw_dataset(cls.data_root, train=True, download=True)
        cls.raw_test_ds = load_raw_dataset(cls.data_root, train=False, download=True)

    def test_raw_dataset_sizes(self) -> None:
        """Kiểm tra kích thước tiêu chuẩn của FashionMNIST (60.000 train, 10.000 test)."""
        self.assertEqual(len(self.raw_train_ds), 60000)
        self.assertEqual(len(self.raw_test_ds), 10000)

    def test_raw_image_and_label_properties(self) -> None:
        """Kiểm tra shape, dtype và khoảng giá trị của ảnh và nhãn."""
        img, label = self.raw_train_ds[0]
        # Shape (1, 28, 28)
        self.assertEqual(img.shape, (1, 28, 28))
        self.assertEqual(img.dtype, torch.float32)
        # Giá trị ảnh sau ToTensor() nằm trong [0.0, 1.0]
        self.assertGreaterEqual(float(img.min()), 0.0)
        self.assertLessEqual(float(img.max()), 1.0)
        # Nhãn là số nguyên từ 0 đến 9
        labels = self.raw_train_ds.targets
        self.assertEqual(int(labels.min()), 0)
        self.assertEqual(int(labels.max()), 9)

    def test_transforms_properties(self) -> None:
        """Kiểm tra các pipeline transforms trả về đúng shape và chuẩn hóa Z-score."""
        train_tf = get_train_transform(self.cfg)
        eval_tf = get_eval_transform(self.cfg)

        sample_pil = Image.fromarray(self.raw_train_ds.data[0].numpy())
        
        train_out = train_tf(sample_pil)
        eval_out = eval_tf(sample_pil)

        self.assertEqual(train_out.shape, (1, 28, 28))
        self.assertEqual(eval_out.shape, (1, 28, 28))
        self.assertEqual(train_out.dtype, torch.float32)
        self.assertEqual(eval_out.dtype, torch.float32)

        # Chuẩn hóa Z-score với mean=0.2860 sẽ tạo ra giá trị âm tại các vùng nền đen (pixel 0)
        self.assertLess(float(eval_out.min()), 0.0)

    def test_stratified_split_and_no_overlap(self) -> None:
        """Kiểm tra phân chia phân tầng 80/20 bảo toàn 10 lớp và không rò rỉ chồng lấn."""
        train_idx, val_idx = stratified_split(
            self.raw_train_ds,
            val_fraction=self.cfg["data"]["val_fraction"],
            seed=self.cfg["project"]["seed"],
        )

        # 80/20 của 60.000 mẫu
        self.assertEqual(len(train_idx), 48000)
        self.assertEqual(len(val_idx), 12000)

        # Kiểm tra không chồng lấn (zero overlap)
        self.assertTrue(verify_no_overlap(train_idx, val_idx))

        # Kiểm tra tính phân tầng (mỗi lớp có đúng 4.800 train và 1.200 val)
        targets = self.raw_train_ds.targets.numpy()
        for cls in range(10):
            train_cls_count = sum(1 for idx in train_idx if targets[idx] == cls)
            val_cls_count = sum(1 for idx in val_idx if targets[idx] == cls)
            self.assertEqual(train_cls_count, 4800)
            self.assertEqual(val_cls_count, 1200)

    def test_dataloaders_batch_and_sizes(self) -> None:
        """Kiểm tra DataLoader khởi tạo thành công và xuất đúng kích thước batch."""
        train_loader, val_loader, test_loader = build_dataloaders(self.cfg)

        self.assertEqual(len(train_loader.dataset), 48000)
        self.assertEqual(len(val_loader.dataset), 12000)
        self.assertEqual(len(test_loader.dataset), 10000)

        # Lấy 1 batch thử nghiệm
        batch_x, batch_y = next(iter(train_loader))
        batch_size = self.cfg["dataloader"]["batch_size"]
        self.assertEqual(batch_x.shape, (batch_size, 1, 28, 28))
        self.assertEqual(batch_y.shape, (batch_size,))
        self.assertEqual(batch_x.dtype, torch.float32)
        self.assertEqual(batch_y.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
