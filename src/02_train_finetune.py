"""
02_train_finetune.py — Exercise 2: Finetuning a Pretrained Model for Binary Text Classification.

Theo đúng 7 bước của Exercise 2 trên slide & chuẩn mực học thuật:
1. Install necessary libraries: transformers, datasets, evaluate, accelerate, torch.
2. Load dataset for binary text classification: 'imdb' (review phim, nhãn NEGATIVE/POSITIVE).
   Phân chia dữ liệu chặt chẽ tránh rò rỉ: 800 Train / 200 Val / 200 Test độc lập (seed=42).
   Xác định mốc Majority-Class baseline (52.0% trên Test).
3. Load pretrained model & tokenizer: 'distilbert/distilbert-base-uncased'.
4. Preprocess the dataset: Tokenization + đo lường tỉ lệ cắt cụt token + Dynamic Padding via DataCollatorWithPadding.
5. Define training arguments: TrainingArguments (seed=42, lr=2e-5, epochs=2, batch_size=16).
6. Create Trainer object, đo Baseline trên tập Validation (bảo vệ Test Set), fine-tune mô hình và lưu Best Model.
7. Evaluate the finetuned model: Accuracy kèm Wilson Score 95% CI, Majority-Class comparison, F1, Precision, Recall,
   Confusion Matrix, Error Analysis & Phân tích tương quan cắt cụt token.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

# Đảm bảo in ấn tiếng Việt trên Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
import evaluate
from sklearn.metrics import classification_report, confusion_matrix
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

# Nạp sẵn các metrics một lần duy nhất ở module-level để tránh gọi evaluate.load() lặp lại
METRIC_ACC = evaluate.load("accuracy")
METRIC_F1 = evaluate.load("f1")
METRIC_PRECISION = evaluate.load("precision")
METRIC_RECALL = evaluate.load("recall")


def setup_environment(seed: int = 42) -> None:
    """Thiết lập hạt giống ngẫu nhiên để tái lập kết quả."""
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(eval_pred):
    """Tính toán bộ chỉ số đánh giá đa chiều (Accuracy, F1, Precision, Recall) tối ưu."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = METRIC_ACC.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = METRIC_F1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    precision = METRIC_PRECISION.compute(predictions=predictions, references=labels, average="macro")["precision"]
    recall = METRIC_RECALL.compute(predictions=predictions, references=labels, average="macro")["recall"]

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def wilson_score_interval(correct: int, total: int, confidence_z: float = 1.96) -> tuple[float, float]:
    """Tính khoảng tin cậy Wilson Score 95% cho tỷ lệ phân loại."""
    if total == 0:
        return 0.0, 0.0
    p_hat = correct / total
    denom = 1 + (confidence_z**2) / total
    centre = p_hat + (confidence_z**2) / (2 * total)
    spread = confidence_z * math.sqrt((p_hat * (1 - p_hat) + (confidence_z**2) / (4 * total)) / total)
    return max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom)


def main() -> None:
    print("\n" + "#" * 70)
    print("🚀 PRACTICE 3 — EXERCISE 2: FINE-TUNING DISTILBERT ON IMDB")
    print("#" * 70 + "\n")

    setup_environment(seed=42)

    # -------------------------------------------------------------
    # BƯỚC 2: Tải dữ liệu nhị phân (IMDb) & Phân chia không rò rỉ
    # -------------------------------------------------------------
    print("📥 [Bước 2/7] Tải bộ dữ liệu IMDb & Kiểm tra phân bố nhãn...")
    dataset = load_dataset("imdb")

    train_val_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    small_train_dataset = train_val_split["train"].shuffle(seed=42).select(range(800))
    small_val_dataset = train_val_split["test"].shuffle(seed=42).select(range(200))
    small_test_dataset = dataset["test"].shuffle(seed=42).select(range(200))

    train_dist = pd.Series(small_train_dataset["label"]).value_counts().to_dict()
    val_dist = pd.Series(small_val_dataset["label"]).value_counts().to_dict()
    test_dist = pd.Series(small_test_dataset["label"]).value_counts().to_dict()

    print(f"   - Tập Train:      {len(small_train_dataset)} mẫu | NEG: {train_dist.get(0, 0)}, POS: {train_dist.get(1, 0)}")
    print(f"   - Tập Validation: {len(small_val_dataset)} mẫu | NEG: {val_dist.get(0, 0)}, POS: {val_dist.get(1, 0)}")
    print(f"   - Tập Test:       {len(small_test_dataset)} mẫu | NEG: {test_dist.get(0, 0)}, POS: {test_dist.get(1, 0)}")

    majority_baseline = max(test_dist.values()) / len(small_test_dataset)
    print(f"   - Mốc quy chiếu Majority-Class trên Test (luôn đoán NEGATIVE): {majority_baseline * 100:.2f}%\n")

    # -------------------------------------------------------------
    # BƯỚC 3: Nạp Pretrained Model & Tokenizer
    # -------------------------------------------------------------
    model_name = "distilbert/distilbert-base-uncased"
    print(f"🤖 [Bước 3/7] Nạp Pretrained Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    # -------------------------------------------------------------
    # BƯỚC 4: Tiền xử lý dữ liệu (Tokenization + Dynamic Padding + Đo Truncation)
    # -------------------------------------------------------------
    print("⚙️  [Bước 4/7] Tiền xử lý dữ liệu với Dynamic Padding & Đo độ dài Token...")
    raw_lengths = [len(tokenizer.encode(t, truncation=False)) for t in small_test_dataset["text"]]
    truncated_count = sum(1 for l in raw_lengths if l > 128)
    truncation_rate = (truncated_count / len(raw_lengths)) * 100
    print(f"   - Đo lường Test Set: {truncated_count}/{len(raw_lengths)} mẫu dài > 128 tokens (Tỷ lệ cắt cụt: {truncation_rate:.2f}%)")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
    tokenized_val = small_val_dataset.map(tokenize_function, batched=True)
    tokenized_test = small_test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------------------------------------------------------------
    # BƯỚC 5: Thiết lập siêu tham số TrainingArguments
    # -------------------------------------------------------------
    print("\n📋 [Bước 5/7] Cấu hình TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=str(OUTPUTS_DIR / "logs"),
        logging_steps=10,
        report_to="none",
    )

    # -------------------------------------------------------------
    # BƯỚC 6: Tạo Trainer, đo Baseline trên Validation & Fine-tune
    # -------------------------------------------------------------
    print("🏋️  [Bước 6/7] Khởi tạo Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Đánh giá Baseline thực sự trên cả Validation và Test trước huấn luyện (Untrained Head)
    print("   👉 Đánh giá Baseline Untrained Head trên Validation Set...")
    untrained_val_metrics = trainer.evaluate(eval_dataset=tokenized_val, metric_key_prefix="untrained_val")
    base_val_acc = untrained_val_metrics.get("untrained_val_accuracy", 0.0)
    base_val_f1 = untrained_val_metrics.get("untrained_val_f1", 0.0)
    base_val_loss = untrained_val_metrics.get("untrained_val_loss", 0.0)
    print(f"      Baseline Validation Accuracy: {base_val_acc * 100:.2f}%")
    print(f"      Baseline Validation Macro-F1: {base_val_f1 * 100:.2f}%")

    print("   👉 Đánh giá Baseline Untrained Head trên Test Set...")
    untrained_test_metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="untrained_test")
    base_test_acc = untrained_test_metrics.get("untrained_test_accuracy", 0.0)
    base_test_f1 = untrained_test_metrics.get("untrained_test_f1", 0.0)
    base_test_loss = untrained_test_metrics.get("untrained_test_loss", 0.0)
    print(f"      Baseline Test Accuracy:       {base_test_acc * 100:.2f}%")
    print(f"      Baseline Test Macro-F1:       {base_test_f1 * 100:.2f}%\n")

    # Tiến hành Fine-tuning
    print("   👉 Bắt đầu huấn luyện...")
    train_result = trainer.train()

    # Lưu mô hình hoàn thiện
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
    print(f"   💾 Mô hình cuối đã lưu vào: {FINAL_MODEL_DIR}\n")

    # -------------------------------------------------------------
    # BƯỚC 7: Đánh giá mô hình sau khi Fine-tune trên Test Set
    # -------------------------------------------------------------
    print("📊 [Bước 7/7] Đánh giá toàn diện trên tập Test...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_test)

    # Dự đoán trên toàn bộ tập test để vẽ Confusion Matrix và Error Analysis
    raw_predictions = trainer.predict(tokenized_test)
    preds = np.argmax(raw_predictions.predictions, axis=1)
    labels = raw_predictions.label_ids
    probs = torch.softmax(torch.tensor(raw_predictions.predictions), dim=-1).numpy()

    # Thống kê khoảng tin cậy Wilson 95%
    correct_count = int(np.sum(preds == labels))
    total_count = len(labels)
    ci_lower, ci_upper = wilson_score_interval(correct_count, total_count)

    # Báo cáo phân loại
    target_names = ["NEGATIVE", "POSITIVE"]
    clf_report = classification_report(labels, preds, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(clf_report).transpose()
    df_report.to_csv(REPORTS_DIR / "classification_report.csv")

    # Ma trận nhầm lẫn
    cm = confusion_matrix(labels, preds)
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    df_cm.to_csv(REPORTS_DIR / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — DistilBERT on IMDb")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # Phân tích sai số (Error Analysis) kèm thông tin cắt cụt token
    is_truncated_list = [l > 128 for l in raw_lengths]
    wrong_indices = np.where(preds != labels)[0]
    error_records = []
    for idx in wrong_indices:
        conf = probs[idx][preds[idx]]
        error_records.append({
            "index": idx,
            "true_label": target_names[labels[idx]],
            "predicted_label": target_names[preds[idx]],
            "confidence": conf,
            "token_length": raw_lengths[idx],
            "is_truncated": is_truncated_list[idx],
            "text": small_test_dataset[int(idx)]["text"],
        })
    df_errors = pd.DataFrame(error_records).sort_values("confidence", ascending=False)
    df_errors.to_csv(REPORTS_DIR / "error_analysis.csv", index=False)

    # Lưu file JSON metrics tổng hợp chuẩn xác học thuật (tính toán động 100%, không hardcode)
    final_metrics_payload = {
        "dataset": "imdb",
        "base_model": model_name,
        "seed": 42,
        "epochs": 2,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "max_length": 128,
        "train_samples": len(small_train_dataset),
        "validation_samples": len(small_val_dataset),
        "test_samples": len(small_test_dataset),
        "majority_class_baseline": majority_baseline,
        "truncation_rate_test": truncation_rate / 100,
        "baseline_val_metrics": {
            "accuracy": base_val_acc,
            "f1": base_val_f1,
            "loss": base_val_loss,
        },
        "baseline_metrics": {
            "accuracy": base_test_acc,
            "f1": base_test_f1,
            "loss": base_test_loss,
        },
        "test_metrics": {
            "test_accuracy": test_metrics.get("eval_accuracy", 0.0),
            "test_f1": test_metrics.get("eval_f1", 0.0),
            "test_precision": test_metrics.get("eval_precision", 0.0),
            "test_recall": test_metrics.get("eval_recall", 0.0),
            "test_loss": test_metrics.get("eval_loss", 0.0),
        },
        "wilson_score_interval_95": {
            "lower": ci_lower,
            "upper": ci_upper,
        },
        "gain": {
            "accuracy_gain_vs_untrained_val": test_metrics.get("eval_accuracy", 0.0) - base_val_acc,
            "accuracy_gain_vs_majority": test_metrics.get("eval_accuracy", 0.0) - majority_baseline,
            "accuracy_gain_vs_untrained_test": test_metrics.get("eval_accuracy", 0.0) - base_test_acc,
            "f1_gain_vs_untrained_val": test_metrics.get("eval_f1", 0.0) - base_val_f1,
            "f1_gain_vs_untrained_test": test_metrics.get("eval_f1", 0.0) - base_test_f1,
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
    }
    with (REPORTS_DIR / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("🎉 HOÀN TẤT HUẤN LUYỆN VÀ ĐÁNH GIÁ (CHUẨN HỌC THUẬT):")
    print(f"   - Baseline Accuracy (Val):    {base_val_acc * 100:.2f}%")
    print(f"   - Majority-Class (Test):       {majority_baseline * 100:.2f}%")
    print(f"   - Fine-tuned Accuracy (Test):  {test_metrics.get('eval_accuracy', 0.0)*100:.2f}%")
    print(f"   - Wilson 95% Confidence Interval: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print(f"   - Tăng trưởng vs Majority:     +{(test_metrics.get('eval_accuracy', 0.0) - majority_baseline)*100:.2f}%")
    print(f"   - Fine-tuned Macro-F1:         {test_metrics.get('eval_f1', 0.0)*100:.2f}%")
    print(f"   - Đã lưu kết quả tại:          {REPORTS_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
