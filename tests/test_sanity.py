"""
test_sanity.py — Unit tests cho Module M6: Sanity Checks (Gradient Flow & Micro-batch Overfitting).
"""
from __future__ import annotations

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from src.fashionmnist.models.network import BaselineMLP, SimpleCNN
from src.fashionmnist.training.sanity_checks import (
    check_gradient_flow,
    check_initial_loss,
    overfit_micro_batch,
    run_sanity_checks,
)


class TestSanityChecks(unittest.TestCase):
    """Kiểm thử tính toàn vẹn giải thuật của mô hình trước huấn luyện."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        # Tạo 1 micro-batch 16 ảnh (1, 28, 28) với 10 lớp ngẫu nhiên
        self.images = torch.randn(16, 1, 28, 28)
        self.labels = torch.randint(0, 10, (16,))
        self.batch = (self.images, self.labels)

    def test_initial_loss_near_theoretical_value_mlp(self) -> None:
        """BaselineMLP: Loss ban đầu phải nằm quanh ln(10) ≈ 2.302."""
        model = BaselineMLP()
        init_loss = check_initial_loss(model, self.batch)
        self.assertGreater(init_loss, 1.8)
        self.assertLess(init_loss, 2.8)

    def test_initial_loss_near_theoretical_value_cnn(self) -> None:
        """SimpleCNN: Loss ban đầu phải nằm quanh ln(10) ≈ 2.302."""
        model = SimpleCNN()
        init_loss = check_initial_loss(model, self.batch)
        self.assertGreater(init_loss, 1.8)
        self.assertLess(init_loss, 2.8)

    def test_gradient_flow_mlp(self) -> None:
        """BaselineMLP: Mọi tầng tham số đều phải nhận được gradient hợp lệ (không NaN, không 0)."""
        model = BaselineMLP()
        norms = check_gradient_flow(model, self.batch)
        self.assertTrue(len(norms) > 0)
        for name, norm in norms.items():
            self.assertGreater(norm, 1e-6, f"Tầng {name} bị vanishing gradient.")
            self.assertLess(norm, 100.0, f"Tầng {name} bị exploding gradient.")

    def test_gradient_flow_cnn(self) -> None:
        """SimpleCNN: Mọi tầng tham số đều phải nhận được gradient hợp lệ."""
        model = SimpleCNN()
        norms = check_gradient_flow(model, self.batch)
        self.assertTrue(len(norms) > 0)
        for name, norm in norms.items():
            self.assertGreater(norm, 1e-6, f"Tầng {name} bị vanishing gradient.")
            self.assertLess(norm, 100.0, f"Tầng {name} bị exploding gradient.")

    def test_overfit_micro_batch_mlp(self) -> None:
        """BaselineMLP: Phải overfit hoàn toàn 16 mẫu đạt Accuracy 100% và Loss < 0.05."""
        model = BaselineMLP()
        final_loss, final_acc = overfit_micro_batch(model, self.batch, max_steps=80, lr=0.01)
        self.assertEqual(final_acc, 1.0)
        self.assertLess(final_loss, 0.05)

    def test_overfit_micro_batch_cnn(self) -> None:
        """SimpleCNN: Phải overfit hoàn toàn 16 mẫu đạt Accuracy 100% và Loss < 0.05."""
        model = SimpleCNN()
        final_loss, final_acc = overfit_micro_batch(model, self.batch, max_steps=80, lr=0.01)
        self.assertEqual(final_acc, 1.0)
        self.assertLess(final_loss, 0.05)

    def test_run_sanity_checks_wrapper(self) -> None:
        """Kiểm tra hàm bao run_sanity_checks trả về True cho cả 2 mô hình."""
        model_mlp = BaselineMLP()
        self.assertTrue(run_sanity_checks(model_mlp, self.batch, verbose=False))

        model_cnn = SimpleCNN()
        self.assertTrue(run_sanity_checks(model_cnn, self.batch, verbose=False))


if __name__ == "__main__":
    unittest.main()
