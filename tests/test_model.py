"""
tests/test_model.py
Kiểm thử đơn vị cho Module M3: Models Architecture (BaselineMLP và SimpleCNN).
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

import torch

from src.fashionmnist.config import load_config
from src.fashionmnist.models.network import (
    BaselineMLP,
    SimpleCNN,
    build_model,
    count_parameters,
)


class TestModelsArchitecture(unittest.TestCase):
    """Kiểm thử cấu trúc, shape đầu ra, số lượng tham số và tính toán gradient của mô hình."""

    def setUp(self) -> None:
        self.dummy_batch = torch.randn(8, 1, 28, 28)
        self.dummy_single = torch.randn(1, 1, 28, 28)

    def test_baseline_mlp_forward_shape(self) -> None:
        """Kiểm tra shape đầu ra của BaselineMLP với batch và single sample."""
        model = BaselineMLP(hidden_units=256, dropout=0.0)
        model.eval()
        with torch.no_grad():
            out_batch = model(self.dummy_batch)
            out_single = model(self.dummy_single)

        self.assertEqual(out_batch.shape, (8, 10))
        self.assertEqual(out_single.shape, (1, 10))
        self.assertTrue(torch.isfinite(out_batch).all())

    def test_baseline_mlp_parameter_count(self) -> None:
        """Kiểm tra số lượng tham số của BaselineMLP (~203.530 tham số)."""
        model = BaselineMLP(hidden_units=256, dropout=0.0)
        n_params = count_parameters(model)
        # 784*256 + 256 + 256*10 + 10 = 203,530
        self.assertEqual(n_params, 203530)

    def test_simple_cnn_forward_shape(self) -> None:
        """Kiểm tra shape đầu ra của SimpleCNN với batch và single sample."""
        model = SimpleCNN(hidden_channels=[32, 64], fc_hidden=256, dropout=0.3)
        model.eval()
        with torch.no_grad():
            out_batch = model(self.dummy_batch)
            out_single = model(self.dummy_single)

        self.assertEqual(out_batch.shape, (8, 10))
        self.assertEqual(out_single.shape, (1, 10))
        self.assertTrue(torch.isfinite(out_batch).all())

    def test_simple_cnn_parameter_efficiency(self) -> None:
        """Kiểm tra tính hiệu quả tham số của SimpleCNN (khoảng ~38k tham số, giảm >80% so với MLP)."""
        mlp = BaselineMLP(hidden_units=256)
        cnn = SimpleCNN(hidden_channels=[32, 64], fc_hidden=256)

        mlp_params = count_parameters(mlp)
        cnn_params = count_parameters(cnn)

        # CNN chỉ có khoảng ~38k tham số
        self.assertLess(cnn_params, 50000)
        self.assertGreater(cnn_params, 30000)
        # Giảm hơn 75-80% số tham số so với MLP
        self.assertLess(cnn_params, mlp_params * 0.25)

    def test_no_softmax_in_outputs(self) -> None:
        """Kiểm tra đầu ra của mô hình là raw logits (không áp dụng Softmax trước)."""
        model = SimpleCNN()
        model.eval()
        with torch.no_grad():
            logits = model(self.dummy_batch)

        # Nếu đã qua softmax, tổng mỗi hàng sẽ bằng 1.0
        row_sums = logits.sum(dim=1)
        self.assertFalse(torch.allclose(row_sums, torch.ones(8), atol=0.1))

    def test_build_model_factory(self) -> None:
        """Kiểm tra hàm khởi tạo mô hình từ tệp cấu hình."""
        cfg = load_config()

        # SimpleCNN mặc định
        model_cnn = build_model(cfg)
        self.assertIsInstance(model_cnn, SimpleCNN)

        # BaselineMLP từ experiment
        cfg_mlp = load_config(experiment_name="baseline_mlp")
        model_mlp = build_model(cfg_mlp)
        self.assertIsInstance(model_mlp, BaselineMLP)

        # Báo lỗi khi tên kiến trúc không hợp lệ
        with self.assertRaises(ValueError):
            build_model({"model": {"architecture": "TransformerViT_XYZ"}})

    def test_invalid_input_dimension_raises(self) -> None:
        """Kiểm tra ném ngoại lệ khi tensor đầu vào sai số chiều."""
        cnn = SimpleCNN()
        invalid_tensor = torch.randn(8, 28, 28)  # Thiếu kênh channel (3D thay vì 4D)
        with self.assertRaises(ValueError):
            cnn(invalid_tensor)


if __name__ == "__main__":
    unittest.main()
