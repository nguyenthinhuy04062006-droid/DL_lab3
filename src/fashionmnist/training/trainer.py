"""
trainer.py — Vòng lặp huấn luyện và kiểm định chuẩn mực cho FashionMNIST.

Các khâu cốt lõi (RULES.md):
- model.train(): Kích hoạt Dropout và running stats của BatchNorm.
- optimizer.zero_grad(): Xóa gradient tích lũy từ bước trước trước khi lan truyền ngược.
- loss.backward(): Tự động tính gradient qua giải thuật backpropagation dựa trên đồ thị tính toán động.
- optimizer.step(): Cập nhật trọng số theo giải thuật tối ưu (SGD/Adam).
- model.eval() & torch.no_grad(): Chuyển sang chế độ kiểm định và ngắt tính toán gradient để tiết kiệm bộ nhớ.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.fashionmnist.training.checkpoint import save_best_checkpoint, save_checkpoint


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> optim.Optimizer:
    """
    Khởi tạo bộ tối ưu hóa (Optimizer) từ cấu hình.

    Hỗ trợ: 'adam' (mặc định) và 'sgd' (có momentum 0.9).
    """
    training_cfg = cfg.get("training", {})
    name = training_cfg.get("optimizer", "adam").lower()
    lr = training_cfg.get("learning_rate", 0.001)
    wd = training_cfg.get("weight_decay", 0.0)

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Không hỗ trợ optimizer: '{name}'. Chọn 'adam' hoặc 'sgd'.")


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: dict[str, Any],
) -> Any | None:
    """
    Khởi tạo bộ điều chỉnh tốc độ học (LR Scheduler) từ cấu hình.

    Hỗ trợ: 'cosine' (CosineAnnealingLR), 'step' (StepLR) hoặc None.
    """
    training_cfg = cfg.get("training", {})
    name = training_cfg.get("lr_scheduler", None)
    epochs = training_cfg.get("epochs", 15)

    if name is None or str(name).lower() in ("null", "none", ""):
        return None
    elif str(name).lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif str(name).lower() == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    else:
        raise ValueError(f"Không hỗ trợ lr_scheduler: '{name}'. Chọn 'cosine', 'step', hoặc None.")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Thực hiện 1 epoch huấn luyện (Training Loop).

    Args:
        model: Mô hình nơ-ron cần cập nhật trọng số.
        loader: DataLoader chứa tập dữ liệu Train.
        criterion: Hàm mất mát (CrossEntropyLoss).
        optimizer: Bộ tối ưu hóa.
        device: Thiết bị tính toán (CPU/CUDA/MPS).

    Returns:
        Tuple (train_loss_trung_bình, train_accuracy).
    """
    # 1. Bật chế độ huấn luyện: Cho phép Dropout hoạt động ngẫu nhiên và BatchNorm cập nhật running mean/var
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 2. XÓA GRADIENT CŨ: Trong PyTorch, gradient được cộng dồn sau mỗi lần backward().
        # Bắt buộc phải gọi zero_grad() để không bị cộng dồn gradient của batch trước ngoài ý muốn.
        optimizer.zero_grad()

        # 3. LAN TRUYỀN XUÔI (Forward Pass): Tính logits từ ảnh đầu vào
        logits = model(images)

        # 4. TÍNH TOÁN HÀM MẤT MÁT (Loss Calculation): Đo lường sai lệch giữa logits và nhãn thực tế
        loss = criterion(logits, labels)

        # 5. LAN TRUYỀN NGƯỢC (Backward Pass): Tính đạo hàm d(loss)/d(w) cho toàn bộ tham số qua Autograd
        loss.backward()

        # 6. CẬP NHẬT TRỌNG SỐ: Cập nhật tham số theo hướng giảm gradient: w = w - lr * grad
        optimizer.step()

        # Thống kê số liệu
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Thực hiện 1 epoch kiểm định (Validation Loop).

    Áp dụng @torch.no_grad() và model.eval() để ngắt tính toán gradient và chuyển BatchNorm/Dropout
    sang chế độ suy luận tất định, giúp tiết kiệm bộ nhớ và đảm bảo tính khách quan.

    Args:
        model: Mô hình cần đánh giá.
        loader: DataLoader tập Validation hoặc Test.
        criterion: Hàm mất mát.
        device: Thiết bị tính toán.

    Returns:
        Tuple (val_loss_trung_bình, val_accuracy).
    """
    # 1. Chuyển sang chế độ đánh giá: Tắt Dropout, cố định running stats của BatchNorm
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path | str,
    experiment_name: str = "default",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Toàn bộ quy trình huấn luyện nhiều epoch kèm Validation, LR Scheduler, Checkpointing và Early Stopping.

    Returns:
        Từ điển history chứa:
        - train_loss, train_acc
        - val_loss, val_acc
        - lr, best_val_acc, best_epoch, total_time_sec
    """
    checkpoint_dir = Path(checkpoint_dir)
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"].get("early_stopping_patience", 5)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    history: dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "best_val_acc": 0.0,
        "best_epoch": 0,
        "total_time_sec": 0.0,
    }

    best_val_acc = -1.0
    epochs_no_improve = 0
    t0 = time.time()

    if verbose:
        print(f"\n--- BẮT ĐẦU HUẤN LUYỆN: {experiment_name.upper()} ({epochs} Epochs) ---")

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        
        # 1. Huấn luyện 1 epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 2. Kiểm định 1 epoch
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion, device)

        # 3. Lấy learning rate hiện tại và cập nhật scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # 4. Ghi nhận nhật ký
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_duration = time.time() - epoch_t0

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} ({epoch_duration:.1f}s)"
            )

        # 5. Theo dõi Checkpoint tốt nhất (dựa trên val_accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_val_acc"] = best_val_acc
            history["best_epoch"] = epoch
            epochs_no_improve = 0
            save_best_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_acc=val_acc,
                cfg=cfg,
                experiment_name=experiment_name,
            )
            if verbose:
                print(f"  ⭐ Đã lưu best checkpoint (Val Acc: {val_acc:.4f})")
        else:
            epochs_no_improve += 1

        # 6. Cơ chế Dừng sớm (Early Stopping)
        if patience > 0 and epochs_no_improve >= patience:
            if verbose:
                print(f"  🛑 Kích hoạt Early Stopping: Val Acc không cải thiện sau {patience} epochs liên tiếp.")
            break

    history["total_time_sec"] = time.time() - t0
    if verbose:
        print(f"--- HOÀN TẤT: Best Val Acc = {history['best_val_acc']:.4f} tại Epoch {history['best_epoch']} (Tổng thời gian: {history['total_time_sec']:.1f}s) ---\n")

    return history
