"""
test_training.py — Unit tests cho Module M4: Train Engine & Checkpoint.
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
from torch.utils.data import DataLoader, TensorDataset

from src.fashionmnist.models.network import BaselineMLP, SimpleCNN
from src.fashionmnist.training.checkpoint import (
    load_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
)
from src.fashionmnist.training.trainer import (
    build_optimizer,
    build_scheduler,
    evaluate_one_epoch,
    train,
    train_one_epoch,
)


class TestOptimizerScheduler(unittest.TestCase):
    """Kiểm tra khởi tạo optimizer và scheduler từ cấu hình."""

    def setUp(self) -> None:
        self.model = BaselineMLP()

    def test_build_optimizer_adam(self) -> None:
        cfg = {"training": {"optimizer": "adam", "learning_rate": 0.002, "weight_decay": 1e-4}}
        opt = build_optimizer(self.model, cfg)
        self.assertIsInstance(opt, torch.optim.Adam)
        self.assertEqual(opt.param_groups[0]["lr"], 0.002)
        self.assertEqual(opt.param_groups[0]["weight_decay"], 1e-4)

    def test_build_optimizer_sgd(self) -> None:
        cfg = {"training": {"optimizer": "sgd", "learning_rate": 0.01, "weight_decay": 0.0}}
        opt = build_optimizer(self.model, cfg)
        self.assertIsInstance(opt, torch.optim.SGD)
        self.assertEqual(opt.param_groups[0]["lr"], 0.01)
        self.assertEqual(opt.param_groups[0]["momentum"], 0.9)

    def test_build_optimizer_invalid(self) -> None:
        cfg = {"training": {"optimizer": "rmsprop_unsupported"}}
        with self.assertRaises(ValueError):
            build_optimizer(self.model, cfg)

    def test_build_scheduler_cosine_and_step(self) -> None:
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)

        cfg_cosine = {"training": {"lr_scheduler": "cosine", "epochs": 10}}
        sched_cos = build_scheduler(opt, cfg_cosine)
        self.assertIsInstance(sched_cos, torch.optim.lr_scheduler.CosineAnnealingLR)

        cfg_step = {"training": {"lr_scheduler": "step", "epochs": 9}}
        sched_step = build_scheduler(opt, cfg_step)
        self.assertIsInstance(sched_step, torch.optim.lr_scheduler.StepLR)

        cfg_none = {"training": {"lr_scheduler": None, "epochs": 10}}
        sched_none = build_scheduler(opt, cfg_none)
        self.assertIsNone(sched_none)


class TestTrainEvaluateStep(unittest.TestCase):
    """Kiểm tra train_one_epoch và evaluate_one_epoch với dữ liệu giả lập."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        # Tạo mini dataset 16 ảnh (1, 28, 28) với 10 nhãn
        x = torch.randn(16, 1, 28, 28)
        y = torch.randint(0, 10, (16,))
        dataset = TensorDataset(x, y)
        self.loader = DataLoader(dataset, batch_size=8, shuffle=False)
        self.device = torch.device("cpu")
        self.criterion = nn.CrossEntropyLoss()

    def test_train_one_epoch_updates_weights(self) -> None:
        model = BaselineMLP()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        # Lưu lại bản sao trọng số trước khi train
        initial_param = next(model.parameters()).clone()

        loss, acc = train_one_epoch(model, self.loader, self.criterion, opt, self.device)

        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

        # Trọng số phải thay đổi sau bước tối ưu
        updated_param = next(model.parameters())
        self.assertFalse(torch.allclose(initial_param, updated_param))

    def test_evaluate_one_epoch_does_not_update_weights(self) -> None:
        model = BaselineMLP()
        initial_param = next(model.parameters()).clone()

        loss, acc = evaluate_one_epoch(model, self.loader, self.criterion, self.device)

        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

        # Trọng số không được thay đổi trong eval
        current_param = next(model.parameters())
        self.assertTrue(torch.allclose(initial_param, current_param))


class TestCheckpointing(unittest.TestCase):
    """Kiểm tra cơ chế lưu và nạp an toàn checkpoint (weights_only=True)."""

    def setUp(self) -> None:
        self.tmp_dir = Path("tests/scratch_checkpoints")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.model = SimpleCNN()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def tearDown(self) -> None:
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    def test_save_and_load_checkpoint_weights_only(self) -> None:
        cfg = {"model": {"architecture": "simple_cnn"}}
        saved_path = save_checkpoint(
            checkpoint_dir=self.tmp_dir,
            model=self.model,
            optimizer=self.opt,
            scheduler=None,
            epoch=3,
            best_metric=0.885,
            cfg=cfg,
        )
        self.assertTrue(saved_path.exists())

        # Tạo một mô hình mới cùng kiến trúc
        new_model = SimpleCNN()
        new_opt = torch.optim.Adam(new_model.parameters(), lr=0.001)

        # Trước khi nạp, trọng số khác nhau
        old_weight = next(self.model.parameters())
        new_weight_before = next(new_model.parameters())
        self.assertFalse(torch.allclose(old_weight, new_weight_before))

        # Nạp an toàn bằng weights_only=True
        ckpt = load_checkpoint(saved_path, new_model, optimizer=new_opt)

        self.assertEqual(ckpt["epoch"], 3)
        self.assertAlmostEqual(ckpt["best_metric"], 0.885)
        # Sau khi nạp, trọng số phải trùng khớp hoàn toàn
        new_weight_after = next(new_model.parameters())
        self.assertTrue(torch.allclose(old_weight, new_weight_after))

    def test_save_best_checkpoint(self) -> None:
        cfg = {"model": {"architecture": "simple_cnn"}}
        best_path = save_best_checkpoint(
            checkpoint_dir=self.tmp_dir,
            model=self.model,
            optimizer=self.opt,
            scheduler=None,
            epoch=5,
            val_acc=0.912,
            cfg=cfg,
            experiment_name="cnn_test",
        )
        self.assertTrue(best_path.exists())
        self.assertEqual(best_path.name, "best_model_cnn_test.pt")


class TestTrainLoop(unittest.TestCase):
    """Kiểm tra toàn bộ hàm train() nhiều epoch và dừng sớm."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        x_tr = torch.randn(32, 1, 28, 28)
        y_tr = torch.randint(0, 10, (32,))
        x_val = torch.randn(16, 1, 28, 28)
        y_val = torch.randint(0, 10, (16,))

        self.train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=8)
        self.val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=8)
        self.tmp_dir = Path("tests/scratch_train_loop")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    def test_train_loop_completes_and_returns_history(self) -> None:
        model = BaselineMLP()
        cfg = {
            "model": {"architecture": "baseline_mlp"},
            "training": {
                "epochs": 2,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "lr_scheduler": "step",
                "early_stopping_patience": 5,
            },
        }
        device = torch.device("cpu")
        history = train(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            cfg=cfg,
            device=device,
            checkpoint_dir=self.tmp_dir,
            experiment_name="test_exp",
            verbose=False,
        )

        expected_keys = {"train_loss", "train_acc", "val_loss", "val_acc", "lr", "best_val_acc", "best_epoch", "total_time_sec"}
        self.assertTrue(expected_keys.issubset(history.keys()))
        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(len(history["val_loss"]), 2)
        self.assertGreater(history["total_time_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
