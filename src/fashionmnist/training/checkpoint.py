"""
checkpoint.py — Cơ chế lưu và phục hồi checkpoint an toàn cho FashionMNIST.

Quy chuẩn bảo mật (RULES.md):
- Bắt buộc sử dụng weights_only=True khi nạp checkpoint bằng torch.load.
- Lưu đầy đủ model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, best_metric, và thông số cấu hình.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    best_metric: float,
    cfg: dict[str, Any],
    filename: str | None = None,
) -> Path:
    """
    Lưu checkpoint trạng thái huấn luyện ra tệp đĩa.

    Args:
        checkpoint_dir: Thư mục lưu tệp checkpoint.
        model: Mô hình PyTorch đang huấn luyện.
        optimizer: Bộ tối ưu hóa.
        scheduler: Bộ điều chỉnh tốc độ học (hoặc None).
        epoch: Chỉ số epoch hiện tại (1-indexed).
        best_metric: Giá trị metric tốt nhất đạt được.
        cfg: Từ điển cấu hình thực nghiệm.
        filename: Tên tệp lưu (mặc định 'checkpoint_epoch_{epoch:03d}.pt').

    Returns:
        Đường dẫn Path tới tệp checkpoint vừa lưu.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch_{epoch:03d}.pt"

    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "architecture": cfg.get("model", {}).get("architecture", "unknown"),
        "cfg": cfg,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = checkpoint_dir / filename
    torch.save(payload, out_path)
    return out_path


def save_best_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    val_acc: float,
    cfg: dict[str, Any],
    experiment_name: str = "default",
) -> Path:
    """Lưu checkpoint mô hình có validation accuracy cao nhất."""
    filename = f"best_model_{experiment_name}.pt" if experiment_name != "default" else "best_model.pt"
    return save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        best_metric=val_acc,
        cfg=cfg,
        filename=filename,
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Nạp an toàn trọng số mô hình từ tệp checkpoint (.pt).

    Tuân thủ cờ bảo mật weights_only=True để ngăn ngừa rủi ro thực thi mã độc qua pickle.

    Args:
        checkpoint_path: Đường dẫn tới file checkpoint.
        model: Đối tượng nn.Module nhận trọng số.
        optimizer: Đối tượng optimizer cần khôi phục trạng thái (tùy chọn).
        scheduler: Đối tượng scheduler cần khôi phục trạng thái (tùy chọn).
        device: Thiết bị cần map tensor (mặc định CPU).

    Returns:
        Từ điển checkpoint chứa siêu dữ liệu (epoch, best_metric, cfg...).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint không tồn tại: {checkpoint_path}")

    if device is None:
        device = torch.device("cpu")

    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
