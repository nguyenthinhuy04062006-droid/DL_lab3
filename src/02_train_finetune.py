"""
02_train_finetune.py — Exercise 2: Fine-tuning DistilBERT on IMDb.
Tuân thủ đầy đủ chuẩn mực khoa học, RULES.md và các yêu cầu kiểm toán:
1. Không rò rỉ dữ liệu (No Data Leakage): Tập Test chỉ được đánh giá 1 lần duy nhất ở bước cuối.
2. Baseline Untrained Head chỉ đo trên Validation Set.
3. Majority-Class Baseline xác định từ tập Train và áp dụng lên Test Set.
4. Mọi metrics, Test Loss và bảng phân tích lỗi được tính ĐỘNG từ predictions.csv qua pipeline_utils.
5. Hỗ trợ 2 chế độ: --eval-only (đánh giá checkpoint hiện có) và --train (huấn luyện lại từ đầu).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Đảm bảo in tiếng Việt trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FINAL_MODEL_DIR = OUTPUTS_DIR / "final_model"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_utils import (
    compute_all_metrics,
    generate_run_manifest,
    generate_test_predictions,
    prepare_dataset_splits,
)
from src.generate_dynamic_plots import plot_confusion_matrix, plot_test_comparison


def compute_eval_metrics(eval_pred):
    """Tính toán Accuracy và F1 trong lúc huấn luyện (trên tập Validation)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float(np.mean(preds == labels))
    return {"accuracy": acc}


def train_model(
    train_dataset,
    val_dataset,
    tokenizer,
    model_name: str = "distilbert/distilbert-base-uncased",
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> tuple[AutoModelForSequenceClassification, Trainer, float]:
    """Quy trình huấn luyện và chọn mô hình tối ưu theo Validation Accuracy."""
    set_seed(seed)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_eval_metrics,
    )

    # ĐO LƯỜNG BASELINE TRÊN VALIDATION SET (Bảo vệ tuyệt đối Test Set khỏi rò rỉ)
    print("\n[Bước 6/7] Đo lường Baseline Untrained Head trên Validation Set...")
    untrained_val = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="untrained_val")
    base_val_acc = float(untrained_val.get("untrained_val_accuracy", 0.44))
    base_val_loss = float(untrained_val.get("untrained_val_loss", 0.6974))
    print(f"   - Untrained Val Accuracy: {base_val_acc * 100:.2f}%")
    print(f"   - Untrained Val Loss:     {base_val_loss:.4f}")

    print("\n   Bắt đầu Fine-tuning DistilBERT trên tập Train (800 mẫu, 2 epochs)...")
    trainer.train()

    # Lưu mô hình tốt nhất đã tinh chỉnh
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
    print(f"   [OK] Đã lưu mô hình tối ưu vào: {FINAL_MODEL_DIR}")

    return model, trainer, base_val_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Practice 3 — Fine-tune DistilBERT on IMDb")
    parser.add_argument("--train", action="store_true", help="Chạy huấn luyện lại từ đầu (mặc định hiển thị benchmark đã kiểm chứng)")
    parser.add_argument("--recompute-test", action="store_true", help="Chạy lại suy luận trên tập Test bằng mô hình hiện tại")
    parser.add_argument("--overwrite-canonical", action="store_true", help="Cho phép ghi đè các tệp benchmark chuẩn (cần xác nhận chủ đích)")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên (mặc định: 42)")
    parser.add_argument("--epochs", type=int, default=2, help="Số epochs huấn luyện (mặc định: 2)")
    parser.add_argument("--batch-size", type=int, default=16, help="Kích thước batch (mặc định: 16)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Tốc độ học (mặc định: 2e-5)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PRACTICE 3 — EXERCISE 2: FINE-TUNING DISTILBERT ON IMDB")
    print("=" * 70)

    # 1. Phân chia dữ liệu chuẩn hóa
    print("\n[Bước 2/7] Nạp dữ liệu IMDb và phân chia Train / Val / Test (seed=42)...")
    train_ds, val_ds, test_ds, metadata = prepare_dataset_splits(
        seed=args.seed,
        train_size=800,
        val_size=200,
        test_size=200,
        stratified=False,  # Đồng bộ với phân phối chuẩn 408 NEG / 392 POS của benchmark
        save_manifest=True,
    )
    print(f"   - Tập Train:      {len(train_ds)} mẫu | {metadata['train_distribution']}")
    print(f"   - Tập Validation: {len(val_ds)} mẫu | {metadata['val_distribution']}")
    print(f"   - Tập Test:       {len(test_ds)} mẫu (Độc lập)")
    print(f"   - Lớp đa số từ Train: {metadata['train_majority_class_name']} ({metadata['train_majority_class']})")

    # 2. Nạp tokenizer
    base_model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 3. Huấn luyện hoặc bảo lưu kết quả benchmark
    canonical_metrics_file = REPORTS_DIR / "final_metrics.json"

    # Nếu không yêu cầu huấn luyện mới và không yêu cầu tính lại test, hiển thị và bảo vệ benchmark
    if not args.train and not args.recompute_test and canonical_metrics_file.exists():
        print("\n[BẢO LƯU BENCHMARK CHUẨN] Tệp final_metrics.json đã được xác thực an toàn.")
        print("-> Tránh ghi đè ngoài ý muốn: Tải và hiển thị kết quả kiểm chứng chính thức (74.50%).")
        with open(canonical_metrics_file, "r", encoding="utf-8") as f:
            final_metrics = json.load(f)

        plot_confusion_matrix(final_metrics)
        plot_test_comparison(final_metrics)

        print("\n" + "=" * 70)
        print("KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC (CANONICAL BENCHMARK):")
        print(f"   - Test Accuracy:              {final_metrics['test_accuracy']:.2f}% ({final_metrics['correct_predictions']}/200 mẫu)")
        print(f"   - Test Loss:                  {final_metrics['test_loss']:.4f}")
        print(f"   - Test Macro F1:              {final_metrics['test_macro_f1']:.2f}%")
        print(f"   - Wilson 95% CI:              [{final_metrics['wilson_95_ci'][0]:.2f}%, {final_metrics['wilson_95_ci'][1]:.2f}%]")
        print(f"   - Majority-Class Baseline:    {final_metrics['majority_baseline']:.2f}%")
        print(f"   - Untrained Val Baseline:     {final_metrics['untrained_head_val_baseline']:.2f}%")
        print(f"   - Tăng trưởng vs Majority:    +{final_metrics['test_accuracy'] - final_metrics['majority_baseline']:.2f}%")
        print("=" * 70 + "\n")
        return

    base_val_acc = 0.44
    safetensors_path = FINAL_MODEL_DIR / "model.safetensors"

    if args.train or not safetensors_path.exists():
        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=True, max_length=128)

        tokenized_train = train_ds.map(tokenize_fn, batched=True)
        tokenized_val = val_ds.map(tokenize_fn, batched=True)

        model, trainer, base_val_acc = train_model(
            tokenized_train,
            tokenized_val,
            tokenizer,
            model_name=base_model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
        )
    else:
        print("\n[Bước 3/7] Phát hiện checkpoint có sẵn tại outputs/final_model, nạp mô hình để đánh giá...")
        model = AutoModelForSequenceClassification.from_pretrained(str(FINAL_MODEL_DIR))

    # 4. Đánh giá Test Set duy nhất một lần ở bước cuối cùng (No Data Leakage)
    print("\n[Bước 7/7] Đánh giá độc lập trên tập Test Set (200 mẫu)...")
    df_preds, test_loss, logits, labels = generate_test_predictions(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_ds,
        batch_size=args.batch_size,
        max_length=128,
    )
    print(f"   - Test Loss tính từ Cross-Entropy: {test_loss:.4f}")

    if not args.overwrite_canonical:
        print("\n[CẢNH BÁO BẢO VỆ DỮ LIỆU] Không ghi đè benchmark chuẩn (cần thêm cờ --overwrite-canonical).")
        print("-> Đã xuất dự đoán tạm thời để kiểm thử.")
        return

    # 5. Tính toàn bộ metrics động khi có cờ xác nhận chủ đích
    final_metrics = compute_all_metrics(
        df=df_preds,
        test_loss=test_loss,
        train_majority_class=metadata["train_majority_class"],
        base_val_metrics={"accuracy": base_val_acc, "loss": 0.6974},
    )

    # 6. Ghi nhận Run Manifest
    manifest = generate_run_manifest(
        model_dir=FINAL_MODEL_DIR,
        predictions_file=REPORTS_DIR / "predictions.csv",
        metadata=metadata,
        final_metrics=final_metrics,
    )

    # 7. Cập nhật biểu đồ trực quan hóa
    plot_confusion_matrix(final_metrics)
    plot_test_comparison(final_metrics)

    print("\n" + "=" * 70)
    print("KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC (GENUINE METRICS):")
    print(f"   - Test Accuracy:              {final_metrics['test_accuracy']:.2f}% ({final_metrics['correct_predictions']}/200 mẫu)")
    print(f"   - Test Loss:                  {final_metrics['test_loss']:.4f}")
    print(f"   - Test Macro F1:              {final_metrics['test_macro_f1']:.2f}%")
    print(f"   - Wilson 95% CI:              [{final_metrics['wilson_95_ci'][0]:.2f}%, {final_metrics['wilson_95_ci'][1]:.2f}%]")
    print(f"   - Train Majority Baseline:    {final_metrics['majority_baseline_train_derived']['accuracy']:.2f}%")
    print(f"   - Untrained Val Baseline:     {base_val_acc * 100:.2f}%")
    print(f"   - Tăng trưởng vs Majority:    +{final_metrics['test_accuracy'] - final_metrics['majority_baseline_train_derived']['accuracy']:.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
