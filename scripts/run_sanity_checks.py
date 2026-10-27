"""
run_sanity_checks.py — Script kiểm thử tính toàn vẹn (Sanity Checks) trước khi chạy thực nghiệm.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Thêm PROJECT_ROOT vào sys.path để import chuẩn mực
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from src.fashionmnist.config import load_config
from src.fashionmnist.data.dataset import build_dataloaders
from src.fashionmnist.models.network import BaselineMLP, SimpleCNN
from src.fashionmnist.training.sanity_checks import run_sanity_checks
from src.fashionmnist.utils.reproducibility import get_device, set_seed


def main() -> int:
    print("=" * 60)
    print(" BẮT ĐẦU KIỂM TRA SANITY CHECKS (MODULE M6)")
    print("=" * 60)

    cfg = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device()

    # Tải 1 batch mẫu từ pipeline dữ liệu thực tế
    train_loader, _, _ = build_dataloaders(cfg)
    images, labels = next(iter(train_loader))

    # Cắt 1 micro-batch đúng 16 mẫu
    micro_batch = (images[:16].to(device), labels[:16].to(device))
    print(f"[*] Đã trích xuất 1 micro-batch: {micro_batch[0].shape}, nhãn: {micro_batch[1].shape}\n")

    # 1. Kiểm tra BaselineMLP
    mlp = BaselineMLP().to(device)
    mlp_pass = run_sanity_checks(mlp, micro_batch, model_name="BaselineMLP (203k params)")

    # 2. Kiểm tra SimpleCNN
    cnn = SimpleCNN().to(device)
    cnn_pass = run_sanity_checks(cnn, micro_batch, model_name="SimpleCNN (~38k params)")

    if mlp_pass and cnn_pass:
        print("=" * 60)
        print("🎉 TẤT CẢ KIẾN TRÚC ĐỀU ĐẠT TIÊU CHUẨN SANITY CHECKS!")
        print("   Sẵn sàng thực thi 5 thực nghiệm có đối chứng (Module M7).")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("❌ PHÁT HIỆN LỖI TRONG QUÁ TRÌNH SANITY CHECKS.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
