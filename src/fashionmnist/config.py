"""
config.py — Centralized configuration loader for FashionMNIST.

Tự động nạp default.yaml và ghi đè (merge) các khóa thực nghiệm từ experiments.yaml.
Mọi đường dẫn tương đối đều được chuẩn hóa thành Path tuyệt đối dựa trên PROJECT_ROOT.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _find_project_root() -> Path:
    """Xác định thư mục gốc của dự án bằng cách tìm tệp marker (RULES.md hoặc .git)."""
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "RULES.md").exists() or (parent / ".git").exists():
            return parent
    # Fallback: 2 cấp thư mục phía trên src/fashionmnist/
    return here.parent.parent


PROJECT_ROOT: Path = _find_project_root()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Hợp nhất đệ quy từ điển override vào bản sao của base.
    Giữ nguyên các giá trị mặc định của base nếu override không chỉ định.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    config_path: str | Path | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """
    Nạp và trả về từ điển cấu hình hoàn chỉnh.

    Cơ chế hoạt động:
    1. Đọc cấu hình gốc default.yaml (siêu tham số chuẩn: seed=42, batch_size=128, lr=0.001...).
    2. Nếu có experiment_name, đọc experiments.yaml và ghi đè (override) các tham số tương ứng.
    3. Chuẩn hóa mọi đường dẫn tương đối thành tuyệt đối dựa theo PROJECT_ROOT.
    4. Kiểm tra hợp lệ (validate) các trường bắt buộc (10 lớp, lr > 0, epochs > 0).

    Args:
        config_path: Đường dẫn tệp default.yaml. Mặc định là configs/default.yaml.
        experiment_name: Tên thực nghiệm cần ghi đè (ví dụ: baseline_mlp, exp_cnn_small).

    Returns:
        Từ điển cấu hình hợp nhất với toàn bộ đường dẫn đã được resolve.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "default.yaml"
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    if experiment_name is not None:
        exp_path = PROJECT_ROOT / "configs" / "experiments.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiments config not found: {exp_path}")
        with exp_path.open("r", encoding="utf-8") as f:
            experiments = yaml.safe_load(f)
        if experiment_name not in experiments.get("experiments", {}):
            available = list(experiments.get("experiments", {}).keys())
            raise KeyError(
                f"Experiment '{experiment_name}' not found. Available: {available}"
            )
        cfg = _deep_merge(cfg, experiments["experiments"][experiment_name])
        cfg["experiment_name"] = experiment_name

    cfg = _resolve_paths(cfg)
    _validate_config(cfg)
    return cfg


def _resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Chuyển đổi các đường dẫn tương đối trong trường 'paths' sang đối tượng Path tuyệt đối."""
    cfg = copy.deepcopy(cfg)
    paths = cfg.get("paths", {})
    for key, value in paths.items():
        if isinstance(value, (str, Path)):
            resolved = (PROJECT_ROOT / value).resolve()
            paths[key] = resolved
    cfg["paths"] = paths
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """Xác thực tính hợp lệ của cấu hình theo ràng buộc học thuật."""
    num_classes = cfg.get("data", {}).get("num_classes", 0)
    if num_classes != 10:
        raise ValueError(f"FashionMNIST yêu cầu num_classes=10, nhận được: {num_classes}")

    class_names = cfg.get("data", {}).get("class_names", [])
    if len(class_names) != 10:
        raise ValueError(f"FashionMNIST yêu cầu 10 class_names, nhận được {len(class_names)}")

    lr = cfg.get("training", {}).get("learning_rate", 0)
    if lr <= 0:
        raise ValueError(f"Learning rate phải > 0, nhận được: {lr}")

    epochs = cfg.get("training", {}).get("epochs", 0)
    if epochs <= 0:
        raise ValueError(f"Epochs phải > 0, nhận được: {epochs}")


def get_class_names(cfg: dict[str, Any]) -> list[str]:
    """Trả về danh sách 10 tên lớp của FashionMNIST."""
    return cfg.get("data", {}).get("class_names", [])


def get_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Trả về từ điển các đường dẫn đã được chuẩn hóa."""
    return cfg.get("paths", {})
