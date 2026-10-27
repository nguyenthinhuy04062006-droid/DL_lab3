"""
test_checkpoint.py — Unit tests chuyên sâu cho Module M5: Checkpoint Management.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim

from src.fashionmnist.models.network import BaselineMLP, SimpleCNN
from src.fashionmnist.training.checkpoint import (
    load_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
)


class TestCheckpointManagement(unittest.TestCase):
    """Bộ kiểm thử tính toàn vẹn và bảo mật của cơ chế Checkpointing."""

    def setUp(self) -> None:
        self.test_dir = Path("tests/scratch_m5_checkpoints")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cpu")

    def tearDown(self) -> None:
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_save_checkpoint_structure(self) -> None:
        """Kiểm tra cấu trúc và các trường dữ liệu được lưu trong checkpoint."""
        model = BaselineMLP()
        opt = optim.Adam(model.parameters(), lr=0.001)
        sched = optim.lr_scheduler.StepLR(opt, step_size=5)
        cfg = {"model": {"architecture": "baseline_mlp"}}

        saved_path = save_checkpoint(
            checkpoint_dir=self.test_dir,
            model=model,
            optimizer=opt,
            scheduler=sched,
            epoch=4,
            best_metric=0.875,
            cfg=cfg,
            filename="test_ckpt.pt",
        )

        self.assertTrue(saved_path.exists())
        self.assertEqual(saved_path.name, "test_ckpt.pt")

        # Nạp trực tiếp để kiểm tra các trường siêu dữ liệu
        raw_data = torch.load(saved_path, map_location=self.device, weights_only=True)
        self.assertIn("epoch", raw_data)
        self.assertIn("best_metric", raw_data)
        self.assertIn("model_state_dict", raw_data)
        self.assertIn("optimizer_state_dict", raw_data)
        self.assertIn("scheduler_state_dict", raw_data)
        self.assertIn("architecture", raw_data)
        self.assertIn("cfg", raw_data)
        self.assertIn("saved_at", raw_data)
        self.assertEqual(raw_data["epoch"], 4)
        self.assertEqual(raw_data["best_metric"], 0.875)

    def test_load_checkpoint_weights_only(self) -> None:
        """Kiểm tra khôi phục trọng số mô hình đảm bảo cờ an toàn weights_only=True."""
        model_a = SimpleCNN()
        opt_a = optim.SGD(model_a.parameters(), lr=0.05, momentum=0.9)
        cfg = {"model": {"architecture": "simple_cnn"}}

        saved_path = save_checkpoint(
            checkpoint_dir=self.test_dir,
            model=model_a,
            optimizer=opt_a,
            scheduler=None,
            epoch=1,
            best_metric=0.90,
            cfg=cfg,
        )

        model_b = SimpleCNN()
        opt_b = optim.SGD(model_b.parameters(), lr=0.01)

        # Trọng số trước khi load phải khác nhau do khởi tạo ngẫu nhiên
        w_a = next(model_a.parameters()).detach()
        w_b_before = next(model_b.parameters()).detach()
        self.assertFalse(torch.allclose(w_a, w_b_before))

        # Thực hiện khôi phục
        metadata = load_checkpoint(
            checkpoint_path=saved_path,
            model=model_b,
            optimizer=opt_b,
            device=self.device,
        )

        # Trọng số sau khi nạp phải trùng khớp từng bit
        w_b_after = next(model_b.parameters()).detach()
        self.assertTrue(torch.allclose(w_a, w_b_after))
        self.assertEqual(metadata["best_metric"], 0.90)

    def test_save_best_checkpoint_naming(self) -> None:
        """Kiểm tra tên file khi lưu mô hình tốt nhất với các thực nghiệm khác nhau."""
        model = BaselineMLP()
        opt = optim.Adam(model.parameters())
        cfg = {"model": {"architecture": "baseline_mlp"}}

        path_default = save_best_checkpoint(
            checkpoint_dir=self.test_dir,
            model=model,
            optimizer=opt,
            scheduler=None,
            epoch=2,
            val_acc=0.85,
            cfg=cfg,
            experiment_name="default",
        )
        self.assertEqual(path_default.name, "best_model.pt")

        path_e3 = save_best_checkpoint(
            checkpoint_dir=self.test_dir,
            model=model,
            optimizer=opt,
            scheduler=None,
            epoch=5,
            val_acc=0.92,
            cfg=cfg,
            experiment_name="exp_cnn_small",
        )
        self.assertEqual(path_e3.name, "best_model_exp_cnn_small.pt")

    def test_load_nonexistent_checkpoint_raises_error(self) -> None:
        """Kiểm tra báo lỗi chuẩn xác khi đường dẫn file không tồn tại."""
        model = BaselineMLP()
        non_existent = self.test_dir / "does_not_exist.pt"
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(non_existent, model)


if __name__ == "__main__":
    unittest.main()
