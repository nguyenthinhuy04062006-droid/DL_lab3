"""
sanity_checks.py — Bộ kiểm tra tính toàn vẹn (Sanity Checks) trước khi huấn luyện.

Mục tiêu (RULES.md):
1. Kiểm tra Loss ban đầu: ~ ln(10) ≈ 2.302 cho bài toán 10 lớp phân loại ngẫu nhiên.
2. Kiểm tra Gradient Flow: Không tầng nào bị gradient bằng 0 (vanishing) hoặc bùng nổ (exploding).
3. Overfit trên 1 micro-batch: Mô hình phải ép loss về gần 0 và đạt accuracy 100% trên 16 mẫu.
"""
from __future__ import annotations

import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim


def check_initial_loss(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device | None = None,
) -> float:
    """
    Kiểm tra giá trị hàm mất mát trước khi huấn luyện.
    
    Với 10 lớp đồng đều, xác suất dự đoán ngẫu nhiên p = 0.1, CrossEntropyLoss kỳ vọng:
    -ln(0.1) = ln(10) ≈ 2.3025.
    Nếu loss ban đầu quá xa khoảng [1.5, 3.2], mô hình có thể gặp lỗi khởi tạo hoặc nhãn bị sai lệch.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(images)
        loss = criterion(logits, labels).item()

    return loss


def check_gradient_flow(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device | None = None,
) -> dict[str, float]:
    """
    Kiểm tra dòng gradient truyền qua các tầng tham số.

    Returns:
        Từ điển ánh xạ tên tham số -> chuẩn L2 (norm) của gradient.
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()

    grad_norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                raise RuntimeError(f"Lỗi Gradient: Tầng '{name}' không nhận được gradient (grad is None).")
            norm = param.grad.data.norm(2).item()
            if torch.isnan(param.grad).any():
                raise RuntimeError(f"Lỗi Gradient: Tầng '{name}' xuất hiện giá trị NaN.")
            if torch.isinf(param.grad).any():
                raise RuntimeError(f"Lỗi Gradient: Tầng '{name}' xuất hiện giá trị Inf.")
            grad_norms[name] = norm

    return grad_norms


def overfit_micro_batch(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    max_steps: int = 100,
    target_acc: float = 1.0,
    target_loss: float = 0.05,
    lr: float = 0.01,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """
    Kiểm tra năng lực biểu diễn của mô hình bằng cách overfit 1 micro-batch.

    Returns:
        Tuple (final_loss, final_accuracy).
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    criterion = nn.CrossEntropyLoss()
    # Sử dụng Adam với lr đủ lớn để overfit nhanh trong ít bước
    optimizer = optim.Adam(model.parameters(), lr=lr)

    final_loss = 999.0
    final_acc = 0.0

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        preds = logits.argmax(dim=1)
        final_acc = (preds == labels).float().mean().item()

        if final_acc >= target_acc and final_loss <= target_loss:
            break

    return final_loss, final_acc


def run_sanity_checks(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    model_name: str = "Model",
    verbose: bool = True,
) -> bool:
    """
    Thực thi toàn bộ bộ kiểm tra Sanity Checks cho mô hình.

    Returns:
        True nếu vượt qua tất cả bài kiểm tra, ngược lại False.
    """
    if verbose:
        print(f"\n{'='*55}")
        print(f" SANITY CHECKS: {model_name}")
        print(f"{'='*55}")

    # 1. Kiểm tra Loss ban đầu
    init_loss = check_initial_loss(model, batch)
    loss_pass = 1.5 <= init_loss <= 3.2
    if verbose:
        status = "✅ PASS" if loss_pass else "❌ FAIL"
        print(f"[1/3] Loss ban đầu (kỳ vọng ≈ 2.302): {init_loss:.4f} -> {status}")

    # 2. Kiểm tra Gradient Flow
    try:
        grad_norms = check_gradient_flow(model, batch)
        min_norm = min(grad_norms.values())
        max_norm = max(grad_norms.values())
        grad_pass = min_norm > 1e-6 and max_norm < 100.0
        if verbose:
            status = "✅ PASS" if grad_pass else "❌ FAIL"
            print(f"[2/3] Gradient Flow: min_norm={min_norm:.6f}, max_norm={max_norm:.4f} -> {status}")
    except Exception as e:
        grad_pass = False
        if verbose:
            print(f"[2/3] Gradient Flow: ❌ Lỗi ({e})")

    # 3. Overfit 1 micro-batch 16 mẫu
    final_loss, final_acc = overfit_micro_batch(model, batch)
    overfit_pass = final_acc >= 0.99 and final_loss <= 0.08
    if verbose:
        status = "✅ PASS" if overfit_pass else "❌ FAIL"
        print(f"[3/3] Overfit Micro-batch (16 mẫu): Acc={final_acc*100:.1f}%, Loss={final_loss:.4f} -> {status}")

    all_pass = loss_pass and grad_pass and overfit_pass
    if verbose:
        if all_pass:
            print(f"🎉 {model_name} VƯỢT QUA SANITY CHECKS HOÀN TOÀN!\n")
        else:
            print(f"❌ {model_name} KHÔNG ĐẠT TIÊU CHUẨN SANITY CHECKS!\n")

    return all_pass
