"""
reproducibility.py — Quản lý hạt giống ngẫu nhiên (seed) và nhận diện thiết bị tính toán.

Cung cấp:
- set_seed: Cố định hạt giống ngẫu nhiên cho Python, NumPy, PyTorch (CPU & CUDA) và cuDNN.
- get_device: Tự động phát hiện và lựa chọn phần cứng tối ưu (CUDA -> MPS -> CPU).
- reproducibility_info: Thu thập thông số phiên bản và phần cứng để ghi log thực nghiệm.
"""
from __future__ import annotations

import platform
import random
import sys
from typing import Any

# Đảm bảo in ấn tiếng Việt trơn tru trên Windows console (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Thiết lập hạt giống ngẫu nhiên (seed) đảm bảo tính tái lập thực nghiệm.

    Args:
        seed: Giá trị hạt giống nguyên (mặc định 42 theo quy chuẩn RULES.md).
    """
    # 1. Cố định seed cho thư viện chuẩn Python
    random.seed(seed)
    
    # 2. Cố định seed cho mảng NumPy
    np.random.seed(seed)
    
    # 3. Cố định seed cho PyTorch trên CPU
    torch.manual_seed(seed)
    
    # 4. Cố định seed cho PyTorch trên GPU CUDA (nếu có)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # 5. Cấu hình cuDNN đảm bảo thuật toán tất định (deterministic)
    # Lưu ý: Bật deterministic có thể làm giảm nhẹ thông lượng nhưng đảm bảo kết quả nhất quán
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Tự động lựa chọn thiết bị tính toán tối ưu: CUDA -> MPS -> CPU.

    Args:
        prefer_gpu: Nếu True, ưu tiên sử dụng GPU (NVIDIA CUDA hoặc Apple MPS) nếu có.

    Returns:
        Đối tượng torch.device tương ứng.
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[Thiết bị] Sử dụng NVIDIA GPU (CUDA): {torch.cuda.get_device_name(0)}")
            return device
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[Thiết bị] Sử dụng Apple Silicon GPU (MPS).")
            return device

    device = torch.device("cpu")
    print("[Thiết bị] Sử dụng CPU.")
    return device


def reproducibility_info(seed: int = 42) -> dict[str, Any]:
    """Trả về từ điển chứa thông số môi trường phục vụ ghi chép báo cáo học thuật."""
    return {
        "seed": seed,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
