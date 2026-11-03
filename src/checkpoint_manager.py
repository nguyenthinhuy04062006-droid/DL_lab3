"""
checkpoint_manager.py — Hệ thống Quản lý và Giám sát Checkpoint theo Epoch.

Cung cấp:
1. EpochCheckpointCallback: Hugging Face TrainerCallback tự động ghi nhận metadata,
   tính toán kích thước, phát hiện checkpoint tốt nhất và tạo tóm tắt sau mỗi epoch.
2. CheckpointManager: Bộ công cụ trực quan quản lý, liệt kê, kiểm tra và nạp checkpoint.
3. Giao diện CLI độc lập để tra cứu danh sách checkpoint và trạng thái mô hình.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# Đường dẫn mặc định trong dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
DEFAULT_FINAL_MODEL_DIR = PROJECT_ROOT / "outputs" / "final_model"


def get_dir_size_mb(path: Path) -> float:
    """Tính tổng dung lượng (MB) của một thư mục."""
    if not path.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total_bytes / (1024 * 1024)


class EpochCheckpointCallback(TrainerCallback):
    """
    Hugging Face TrainerCallback chuyên biệt cho Epoch Checkpointing:
    - Bắt sự kiện on_save để ghi metadata chi tiết (epoch, step, metrics, thời gian).
    - Tạo tệp `epoch_metadata.json` trong mỗi thư mục checkpoint.
    - Cập nhật file chỉ mục `checkpoints_index.json` tại thư mục gốc checkpoints.
    - In thông báo tiến độ trực quan, rõ ràng trên console.
    """

    def __init__(self, checkpoints_dir: Optional[Path] = None):
        super().__init__()
        self.checkpoints_dir = checkpoints_dir

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        epoch_num = int(round(state.epoch)) if state.epoch is not None else -1
        is_best = (
            state.best_model_checkpoint is not None
            and Path(state.best_model_checkpoint).resolve() == ckpt_dir.resolve()
        )

        # Lấy bản ghi log gần nhất cho epoch hiện tại
        recent_log = {}
        for log in reversed(state.log_history):
            if "eval_accuracy" in log or "eval_loss" in log or "loss" in log:
                recent_log = log
                break

        eval_acc = recent_log.get("eval_accuracy")
        eval_loss = recent_log.get("eval_loss")
        train_loss = recent_log.get("loss")

        # 1. Ghi file metadata chi tiết vào chính thư mục checkpoint
        metadata = {
            "epoch": epoch_num,
            "global_step": state.global_step,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "eval_accuracy": eval_acc,
            "eval_loss": eval_loss,
            "train_loss": train_loss,
            "best_metric": state.best_metric,
            "metric_for_best_model": args.metric_for_best_model,
            "is_best_so_far": is_best,
            "dir_size_mb": round(get_dir_size_mb(ckpt_dir), 2),
        }

        try:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with open(ckpt_dir / "epoch_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"   [CẢNH BÁO] Không thể ghi metadata cho {ckpt_dir.name}: {e}")

        # 2. Cập nhật chỉ mục toàn cục `checkpoints_index.json`
        try:
            index_path = Path(args.output_dir) / "checkpoints_index.json"
            index_data: Dict[str, Any] = {}
            if index_path.exists():
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)

            index_data[f"epoch_{epoch_num}"] = {
                "checkpoint_dir": str(ckpt_dir.name),
                "step": state.global_step,
                "epoch": epoch_num,
                "eval_accuracy": eval_acc,
                "eval_loss": eval_loss,
                "updated_at": metadata["timestamp"],
            }

            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # 3. Hiển thị thông báo tiến độ trực quan
        status_tag = "[BEST]" if is_best else "[OK]"
        acc_text = f"{eval_acc * 100:.2f}%" if eval_acc is not None else "N/A"
        loss_text = f"{eval_loss:.4f}" if eval_loss is not None else "N/A"

        print("\n" + "-" * 70)
        print(f"[CHECKPOINT EPOCH {epoch_num}] Đã lưu thành công: {ckpt_dir.name}")
        print(f"   • Global Step     : {state.global_step}")
        print(f"   • Val Accuracy    : {acc_text}")
        print(f"   • Val Loss        : {loss_text}")
        print(f"   • Trạng thái      : [{status_tag}] (Dung lượng: {metadata['dir_size_mb']} MB)")
        print("-" * 70 + "\n")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        print("\n" + "=" * 70)
        print("QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT — TỔNG KẾT CHECKPOINT EPOCH")
        print("=" * 70)
        if state.best_model_checkpoint:
            print(f"Checkpoint tốt nhất: {Path(state.best_model_checkpoint).name}")
            print(f"   • Điểm số ({args.metric_for_best_model}): {state.best_metric}")
        print(f"Toàn bộ checkpoint được lưu tại: {args.output_dir}")
        print("=" * 70 + "\n")


class CheckpointManager:
    """Bộ công cụ tra cứu, quản lý và khôi phục checkpoint theo epoch."""

    def __init__(self, checkpoints_dir: Optional[Path] = None):
        self.checkpoints_dir = Path(checkpoints_dir or DEFAULT_CHECKPOINTS_DIR)

    def scan_checkpoints(self) -> List[Dict[str, Any]]:
        """Quét và thu thập thông tin của tất cả các checkpoint hiện có."""
        if not self.checkpoints_dir.exists():
            return []

        # Kiểm tra xem có file trainer_state.json toàn cục không
        best_ckpt_name: Optional[str] = None
        for ckpt in self.checkpoints_dir.glob("checkpoint-*"):
            state_file = ckpt / "trainer_state.json"
            if state_file.exists():
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
                        ts = json.load(f)
                        if ts.get("best_model_checkpoint"):
                            best_ckpt_name = Path(ts["best_model_checkpoint"]).name
                except Exception:
                    pass

        results: List[Dict[str, Any]] = []
        for ckpt_dir in sorted(self.checkpoints_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime):
            if not ckpt_dir.is_dir():
                continue

            metadata_file = ckpt_dir / "epoch_metadata.json"
            meta: Dict[str, Any] = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass

            step = meta.get("global_step")
            if step is None:
                try:
                    step = int(ckpt_dir.name.split("-")[-1])
                except ValueError:
                    step = 0

            epoch = meta.get("epoch")
            is_best = (ckpt_dir.name == best_ckpt_name) or meta.get("is_best_so_far", False)
            size_mb = meta.get("dir_size_mb", round(get_dir_size_mb(ckpt_dir), 2))

            results.append({
                "name": ckpt_dir.name,
                "path": str(ckpt_dir),
                "epoch": epoch,
                "step": step,
                "val_accuracy": meta.get("eval_accuracy"),
                "val_loss": meta.get("eval_loss"),
                "is_best": is_best,
                "size_mb": size_mb,
                "timestamp": meta.get("timestamp", datetime.fromtimestamp(ckpt_dir.stat().st_mtime).isoformat(timespec="seconds")),
            })

        # Sắp xếp theo step
        results.sort(key=lambda x: x["step"])
        return results

    def print_dashboard(self) -> None:
        """In bảng tóm tắt trực quan tình trạng hệ thống checkpoint."""
        ckpts = self.scan_checkpoints()

        print("\n" + "=" * 78)
        print("HỆ THỐNG QUẢN LÝ CHECKPOINT THEO EPOCH (HUGGING FACE TRANSFORMERS)")
        print("=" * 78)
        print(f"Thư mục lưu trữ: {self.checkpoints_dir.resolve()}")
        print("-" * 78)

        if not ckpts:
            print("[CẢNH BÁO]  Chưa có checkpoint nào được tạo trong thư mục!")
            print(f"- Vui lòng chạy huấn luyện bằng lệnh: python src/02_train_finetune.py")
            print("=" * 78 + "\n")
            return

        header = f"{'Checkpoint':<18} | {'Epoch':<7} | {'Step':<6} | {'Val Acc':<9} | {'Val Loss':<9} | {'Dung lượng':<10} | {'Đánh giá':<8}"
        print(header)
        print("-" * 78)

        for c in ckpts:
            epoch_str = str(c['epoch']) if c['epoch'] is not None else "-"
            acc_str = f"{c['val_accuracy']*100:.2f}%" if c['val_accuracy'] is not None else "-"
            loss_str = f"{c['val_loss']:.4f}" if c['val_loss'] is not None else "-"
            best_tag = "[BEST]" if c['is_best'] else "[OK]"
            row = (
                f"{c['name']:<18} | "
                f"{epoch_str:<7} | "
                f"{c['step']:<6} | "
                f"{acc_str:<9} | "
                f"{loss_str:<9} | "
                f"{c['size_mb']:>7.2f} MB | "
                f"{best_tag:<8}"
            )
            print(row)

        print("-" * 78)
        print("Ghi chú:")
        print("   • [BEST]: Checkpoint có Validation Accuracy cao nhất được tự động nạp.")
        print("   • Để nạp checkpoint suy luận: AutoModelForSequenceClassification.from_pretrained(path)")
        print("=" * 78 + "\n")

    def load_checkpoint(
        self,
        checkpoint_name_or_step: str | int,
        device: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        """Nạp mô hình và Tokenizer từ một checkpoint cụ thể."""
        if isinstance(checkpoint_name_or_step, int):
            target_path = self.checkpoints_dir / f"checkpoint-{checkpoint_name_or_step}"
        else:
            target_path = self.checkpoints_dir / str(checkpoint_name_or_step)

        if not target_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {target_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Đang nạp checkpoint từ: {target_path} (Thiết bị: {device})...")
        tokenizer = AutoTokenizer.from_pretrained(str(target_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(target_path))
        model.to(device)
        model.eval()
        return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Công cụ Giám sát & Quản lý Checkpoint theo Epoch.")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_CHECKPOINTS_DIR), help="Thư mục chứa checkpoints.")
    parser.add_argument("--list", action="store_true", help="Hiển thị bảng danh sách các checkpoint.")
    parser.add_argument("--clean", action="store_true", help="Dọn dẹp các checkpoint phụ, chỉ giữ lại best checkpoint.")
    args = parser.parse_args()

    mgr = CheckpointManager(Path(args.dir))
    mgr.print_dashboard()


if __name__ == "__main__":
    main()

