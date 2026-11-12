"""
pipeline_utils.py — Bộ công cụ chuẩn hóa dữ liệu, suy luận, tính metrics và quản lý manifest.
Phục vụ thống nhất giữa src/ và notebooks/.
Tuân thủ tuyệt đối RULES.md (Zero Hallucination, No Leakage, Reproducibility).
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FINAL_MODEL_DIR = OUTPUTS_DIR / "final_model"


def compute_sha256(file_path: Path | str) -> str:
    """Tính mã băm SHA-256 của một tệp tin."""
    p = Path(file_path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def wilson_score_interval(correct: int, total: int, confidence_z: float = 1.96) -> tuple[float, float]:
    """Tính khoảng tin cậy Wilson Score 95% cho tỷ lệ phân loại nhị phân."""
    if total == 0:
        return 0.0, 0.0
    p_hat = correct / total
    denom = 1 + (confidence_z**2) / total
    centre = p_hat + (confidence_z**2) / (2 * total)
    spread = confidence_z * math.sqrt((p_hat * (1 - p_hat) + (confidence_z**2) / (4 * total)) / total)
    return max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom)


def prepare_dataset_splits(
    seed: int = 42,
    train_size: int = 800,
    val_size: int = 200,
    test_size: int = 200,
    stratified: bool = True,
    save_manifest: bool = True,
) -> tuple[Dataset, Dataset, Dataset, dict]:
    """
    Phân chia dữ liệu Train, Validation và Test tuân thủ RULES.md.
    Sử dụng train_test_split(seed=42) có phân tầng (stratified) để cân bằng lớp hoàn hảo.
    """
    dataset = load_dataset("imdb")

    if stratified:
        # Tách Train (800) và Validation (200) từ dataset['train']
        train_val_split = dataset["train"].train_test_split(
            train_size=train_size,
            test_size=val_size,
            seed=seed,
            stratify_by_column="label",
        )
        small_train = train_val_split["train"]
        small_val = train_val_split["test"]

        # Tách Test (200) từ dataset['test']
        test_split = dataset["test"].train_test_split(
            test_size=test_size,
            seed=seed,
            stratify_by_column="label",
        )
        small_test = test_split["test"]
    else:
        raw_train = dataset["train"].shuffle(seed=seed)
        raw_test = dataset["test"].shuffle(seed=seed)
        small_train = raw_train.select(range(train_size))
        small_val = raw_train.select(range(train_size, train_size + val_size))
        small_test = raw_test.select(range(test_size))

    # Xác định lớp đa số từ tập Train (Không nhìn trước tập Test)
    train_labels = list(small_train["label"])
    train_counts = Counter(train_labels)
    majority_class_train = 0 if train_counts[0] >= train_counts[1] else 1
    majority_class_name = "NEGATIVE" if majority_class_train == 0 else "POSITIVE"

    metadata = {
        "dataset_name": "imdb",
        "dataset_version": "plain_text",
        "seed": seed,
        "stratified": stratified,
        "train_samples": len(small_train),
        "val_samples": len(small_val),
        "test_samples": len(small_test),
        "train_distribution": {int(k): int(v) for k, v in train_counts.items()},
        "val_distribution": {int(k): int(v) for k, v in Counter(small_val["label"]).items()},
        "test_distribution": {int(k): int(v) for k, v in Counter(small_test["label"]).items()},
        "train_majority_class": majority_class_train,
        "train_majority_class_name": majority_class_name,
    }

    if save_manifest:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        manifest_records = []
        for split_name, split_ds in [("train", small_train), ("val", small_val), ("test", small_test)]:
            for i, item in enumerate(split_ds):
                text_content = item["text"]
                item_hash = hashlib.md5(text_content.encode("utf-8")).hexdigest()
                manifest_records.append({
                    "sample_id": f"{split_name}_{i:04d}",
                    "split": split_name,
                    "split_index": i,
                    "label": int(item["label"]),
                    "label_name": "NEGATIVE" if item["label"] == 0 else "POSITIVE",
                    "text_char_length": len(text_content),
                    "text_md5": item_hash,
                })
        df_manifest = pd.DataFrame(manifest_records)
        df_manifest.to_csv(REPORTS_DIR / "data_manifest.csv", index=False)
        with open(REPORTS_DIR / "data_manifest.json", "w", encoding="utf-8") as f:
            json.dump({
                "metadata": metadata,
                "total_records": len(df_manifest),
                "manifest_preview": manifest_records[:5],
            }, f, indent=2, ensure_ascii=False)

    return small_train, small_val, small_test, metadata


def generate_test_predictions(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    test_dataset: Dataset,
    batch_size: int = 16,
    max_length: int = 128,
    device: Optional[torch.device] = None,
) -> tuple[pd.DataFrame, float, torch.Tensor, torch.Tensor]:
    """
    Quy trình suy luận chuẩn hóa trên tập Test:
    - model.eval(), torch.inference_mode().
    - Giữ nguyên thứ tự mẫu, không shuffle.
    - Tính logits, softmax probabilities, predicted labels, confidence.
    - Tính Test Loss chuẩn xác từ chính logits và labels của lần chạy này.
    - Xuất DataFrame với cấu trúc chuẩn và kiểm tra assertion toán học.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []
    raw_lengths = []
    sample_records = []

    total_samples = len(test_dataset)

    with torch.inference_mode():
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_samples = [test_dataset[i] for i in range(start_idx, end_idx)]
            batch_texts = [s["text"] for s in batch_samples]
            batch_labels = [s["label"] for s in batch_samples]

            # Đo độ dài token thô trước khi cắt cụt
            batch_raw_lens = [len(tokenizer.encode(t, truncation=False)) for t in batch_texts]
            raw_lengths.extend(batch_raw_lens)

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**encoded)
            logits = outputs.logits
            all_logits.append(logits.cpu())
            all_labels.extend(batch_labels)

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # Tính Test Loss từ CrossEntropyLoss trên đúng logits và labels của lần này
    test_loss = F.cross_entropy(logits_tensor, labels_tensor).item()

    probs_tensor = F.softmax(logits_tensor, dim=-1)
    preds_tensor = torch.argmax(probs_tensor, dim=-1)
    confs_tensor = torch.max(probs_tensor, dim=-1).values

    probs_np = probs_tensor.numpy()
    logits_np = logits_tensor.numpy()
    preds_np = preds_tensor.numpy()
    labels_np = labels_tensor.numpy()
    confs_np = confs_tensor.numpy()

    target_names = {0: "NEGATIVE", 1: "POSITIVE"}

    records = []
    for i in range(total_samples):
        records.append({
            "sample_id": f"test_{i:04d}",
            "dataset_index": i,
            "true_label": int(labels_np[i]),
            "true_label_name": target_names[labels_np[i]],
            "predicted_label": int(preds_np[i]),
            "predicted_label_name": target_names[preds_np[i]],
            "logit_negative": float(logits_np[i, 0]),
            "logit_positive": float(logits_np[i, 1]),
            "prob_negative": float(probs_np[i, 0]),
            "prob_positive": float(probs_np[i, 1]),
            "confidence": float(confs_np[i]),
            "correct": bool(preds_np[i] == labels_np[i]),
            "text_length": int(raw_lengths[i]),
            "was_truncated": bool(raw_lengths[i] > max_length),
            "text": test_dataset[i]["text"],
        })

    df = pd.DataFrame(records)

    # Kiểm tra tính toàn vẹn toán học
    assert (df["predicted_label"] == (df["logit_positive"] > df["logit_negative"]).astype(int)).all(), (
        "LỖI: predicted_label không khớp với argmax(logits)!"
    )
    assert np.allclose(df["prob_negative"] + df["prob_positive"], 1.0, atol=1e-5), (
        "LỖI: Tổng prob_negative + prob_positive != 1.0!"
    )
    assert np.allclose(df["confidence"], np.maximum(df["prob_negative"], df["prob_positive"]), atol=1e-5), (
        "LỖI: confidence != max(prob_negative, prob_positive)!"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS_DIR / "predictions.csv", index=False)

    return df, test_loss, logits_tensor, labels_tensor


def compute_all_metrics(
    df: pd.DataFrame,
    test_loss: float,
    train_majority_class: int,
    base_val_metrics: Optional[dict] = None,
    training_log: Optional[list] = None,
) -> dict:
    """
    Tính toàn bộ metrics động từ duy nhất file predictions.csv:
    - Accuracy, Macro/Weighted F1, Precision, Recall.
    - Confusion Matrix chuẩn.
    - Wilson Score 95% CI.
    - Truncation Error Analysis & Fisher Exact Test.
    - Xuất classification_report.csv, confusion_matrix.csv, error_analysis.csv, final_metrics.json.
    """
    labels = df["true_label"].to_numpy()
    preds = df["predicted_label"].to_numpy()
    total_samples = len(df)
    correct_count = int(np.sum(labels == preds))
    accuracy = correct_count / total_samples

    # Majority Baseline tính từ Train: nếu luôn đoán train_majority_class trên Test
    majority_preds = np.full_like(labels, train_majority_class)
    majority_accuracy = float(np.mean(labels == majority_preds))

    # Classification Report
    target_names = ["NEGATIVE", "POSITIVE"]
    clf_dict = classification_report(labels, preds, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(clf_dict).transpose()
    df_report.to_csv(REPORTS_DIR / "classification_report.csv")

    # Confusion Matrix: TN, FP, FN, TP
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    df_cm.to_csv(REPORTS_DIR / "confusion_matrix.csv")

    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    assert tn + fp + fn + tp == total_samples, f"LỖI: Tổng CM {tn+fp+fn+tp} != {total_samples}"

    # Wilson Score 95% CI
    ci_lower, ci_upper = wilson_score_interval(correct_count, total_samples)

    # Phân tích Truncation & Fisher's Exact Test
    truncated = df["was_truncated"].to_numpy()
    trunc_correct = int(np.sum((truncated == True) & (labels == preds)))
    trunc_wrong = int(np.sum((truncated == True) & (labels != preds)))
    nontrunc_correct = int(np.sum((truncated == False) & (labels == preds)))
    nontrunc_wrong = int(np.sum((truncated == False) & (labels != preds)))

    trunc_total = trunc_correct + trunc_wrong
    nontrunc_total = nontrunc_correct + nontrunc_wrong

    trunc_err_rate = (trunc_wrong / trunc_total * 100) if trunc_total > 0 else 0.0
    nontrunc_err_rate = (nontrunc_wrong / nontrunc_total * 100) if nontrunc_total > 0 else 0.0

    table_fisher = [[nontrunc_wrong, nontrunc_correct], [trunc_wrong, trunc_correct]]
    fisher_odds, fisher_p = fisher_exact(table_fisher)

    # Xuất error_analysis.csv
    df_errors = df[df["correct"] == False].sort_values("confidence", ascending=False)
    df_errors.to_csv(REPORTS_DIR / "error_analysis.csv", index=False)

    payload = {
        "test_accuracy": round(accuracy * 100, 4),
        "test_macro_f1": round(clf_dict["macro avg"]["f1-score"] * 100, 4),
        "test_macro_precision": round(clf_dict["macro avg"]["precision"] * 100, 4),
        "test_macro_recall": round(clf_dict["macro avg"]["recall"] * 100, 4),
        "test_weighted_f1": round(clf_dict["weighted avg"]["f1-score"] * 100, 4),
        "test_loss": round(test_loss, 4),
        "correct_predictions": correct_count,
        "total_samples": total_samples,
        "wilson_95_ci": [round(ci_lower * 100, 4), round(ci_upper * 100, 4)],
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": cm.tolist(),
        },
        "majority_baseline_train_derived": {
            "majority_class": train_majority_class,
            "majority_class_name": "NEGATIVE" if train_majority_class == 0 else "POSITIVE",
            "accuracy": round(majority_accuracy * 100, 4),
        },
        "truncation_analysis": {
            "max_length": 128,
            "total_truncated": trunc_total,
            "total_non_truncated": nontrunc_total,
            "truncated_error_count": trunc_wrong,
            "truncated_error_rate_pct": round(trunc_err_rate, 2),
            "non_truncated_error_count": nontrunc_wrong,
            "non_truncated_error_rate_pct": round(nontrunc_err_rate, 2),
            "fisher_odds_ratio": round(float(fisher_odds), 4),
            "fisher_p_value": round(float(fisher_p), 4),
        },
        "baseline_val_metrics": base_val_metrics or {},
        "classification_report": clf_dict,
    }

    with open(REPORTS_DIR / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


def generate_run_manifest(
    model_dir: Path,
    predictions_file: Path,
    metadata: dict,
    final_metrics: dict,
) -> dict:
    """Tạo tệp run_manifest.json ghi nhận toàn bộ nguồn gốc thực nghiệm."""
    import transformers
    import datasets
    import evaluate
    import sklearn

    # Lấy git commit hash hiện tại nếu có
    git_commit = "unknown"
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        git_commit = res.stdout.strip()
    except Exception:
        pass

    safetensors_path = model_dir / "model.safetensors"
    ckpt_hash = compute_sha256(safetensors_path) if safetensors_path.exists() else ""
    preds_hash = compute_sha256(predictions_file) if predictions_file.exists() else ""

    manifest = {
        "run_id": f"run_{metadata.get('seed', 42)}_{git_commit[:7]}",
        "seed": metadata.get("seed", 42),
        "model_id": "distilbert/distilbert-base-uncased",
        "dataset": "imdb",
        "train_samples": metadata.get("train_samples", 800),
        "validation_samples": metadata.get("val_samples", 200),
        "test_samples": metadata.get("test_samples", 200),
        "stratified_split": metadata.get("stratified", True),
        "test_accuracy": final_metrics.get("test_accuracy"),
        "test_loss": final_metrics.get("test_loss"),
        "test_macro_f1": final_metrics.get("test_macro_f1"),
        "checkpoint_safetensors_sha256": ckpt_hash,
        "predictions_csv_sha256": preds_hash,
        "library_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "evaluate": evaluate.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "git_commit": git_commit,
    }

    with open(REPORTS_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest

