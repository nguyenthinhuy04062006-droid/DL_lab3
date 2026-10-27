"""
tests/test_config.py
Kiểm thử đơn vị cho Module M1: Config & Utils.
Tương thích hoàn toàn với cả pytest và python -m unittest.
"""
from __future__ import annotations

import random
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

from src.fashionmnist.config import (
    PROJECT_ROOT,
    get_class_names,
    get_paths,
    load_config,
)
from src.fashionmnist.utils.reproducibility import get_device, set_seed


class TestConfigAndUtils(unittest.TestCase):
    """Kiểm thử nạp cấu hình YAML, ghi đè thực nghiệm, và tính tái lập."""

    def test_load_default_config(self) -> None:
        """Kiểm tra nạp cấu hình mặc định default.yaml thành công."""
        cfg = load_config()
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["project"]["name"], "fashionmnist-deep-learning")
        self.assertEqual(cfg["project"]["seed"], 42)
        self.assertEqual(cfg["data"]["num_classes"], 10)
        self.assertEqual(len(get_class_names(cfg)), 10)
        self.assertEqual(cfg["data"]["val_fraction"], 0.20)
        
        paths = get_paths(cfg)
        self.assertIn("data_root", paths)
        self.assertIn("checkpoints_dir", paths)
        self.assertTrue(isinstance(paths["data_root"], Path))

    def test_load_all_experiments(self) -> None:
        """Kiểm tra ghi đè cấu hình cho toàn bộ 5 thực nghiệm E1 đến E5."""
        # E1: baseline_mlp
        cfg_e1 = load_config(experiment_name="baseline_mlp")
        self.assertEqual(cfg_e1["model"]["architecture"], "BaselineMLP")
        self.assertEqual(cfg_e1["training"]["optimizer"], "sgd")
        self.assertEqual(cfg_e1["training"]["epochs"], 10)
        self.assertEqual(cfg_e1["dataloader"]["batch_size"], 128)

        # E2: mlp_adam
        cfg_e2 = load_config(experiment_name="mlp_adam")
        self.assertEqual(cfg_e2["model"]["architecture"], "BaselineMLP")
        self.assertEqual(cfg_e2["training"]["optimizer"], "adam")
        self.assertEqual(cfg_e2["training"]["epochs"], 15)

        # E3: exp_cnn_small
        cfg_e3 = load_config(experiment_name="exp_cnn_small")
        self.assertEqual(cfg_e3["model"]["architecture"], "SimpleCNN")
        self.assertEqual(cfg_e3["training"]["optimizer"], "adam")
        self.assertEqual(cfg_e3["training"]["learning_rate"], 0.001)

        # E4: exp_lr_01
        cfg_e4 = load_config(experiment_name="exp_lr_01")
        self.assertEqual(cfg_e4["model"]["architecture"], "SimpleCNN")
        self.assertEqual(cfg_e4["training"]["learning_rate"], 0.01)

        # E5: exp_batch_256
        cfg_e5 = load_config(experiment_name="exp_batch_256")
        self.assertEqual(cfg_e5["model"]["architecture"], "SimpleCNN")
        self.assertEqual(cfg_e5["dataloader"]["batch_size"], 256)

    def test_load_invalid_experiment_raises(self) -> None:
        """Kiểm tra ném KeyError khi tên thực nghiệm không tồn tại."""
        with self.assertRaises(KeyError):
            load_config(experiment_name="non_existent_experiment_xyz")

    def test_seed_reproducibility(self) -> None:
        """Kiểm tra tính tái lập khi cố định seed cho random, numpy và PyTorch."""
        set_seed(42)
        r1 = random.random()
        n1 = np.random.rand(5)
        t1 = torch.rand(3, 3)

        set_seed(42)
        r2 = random.random()
        n2 = np.random.rand(5)
        t2 = torch.rand(3, 3)

        self.assertEqual(r1, r2)
        self.assertTrue(np.allclose(n1, n2))
        self.assertTrue(torch.equal(t1, t2))

    def test_get_device(self) -> None:
        """Kiểm tra get_device trả về đối tượng torch.device hợp lệ."""
        dev = get_device()
        self.assertIsInstance(dev, torch.device)
        self.assertIn(dev.type, ["cuda", "mps", "cpu"])


if __name__ == "__main__":
    unittest.main()
