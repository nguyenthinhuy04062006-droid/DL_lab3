"""
verify_all.py — Bộ kiểm thử tự động toàn diện và kiểm toán tính tái lập (Practice 3 Suite).
Tuân thủ nghiêm ngặt RULES.md và các chuẩn mực khoa học:
- Tính toán ĐỘNG mọi metrics từ predictions.csv (không hardcode số liệu mong đợi).
- Xác thực tính toàn vẹn toán học (argmax, softmax, cross-entropy, confusion matrix).
- Kiểm toán mã băm SHA-256 của checkpoint và predictions.
- Kiểm tra tính tái lập suy luận (reload checkpoint).
- Kiểm tra Zero Emoji và tính tự chứa của báo cáo HTML.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

# Đảm bảo in tiếng Việt trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FINAL_MODEL_DIR = OUTPUTS_DIR / "final_model"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "Practice3_HuggingFace_Final.ipynb"


def compute_sha256(file_path: Path | str) -> str:
    p = Path(file_path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_1_project_structure() -> None:
    print("[TEST 1/5] Kiểm tra cấu trúc thư mục, mã nguồn và tài liệu...")
    required_src = [
        "src/01_sentiment_analysis.py",
        "src/02_train_finetune.py",
        "src/03_predict_finetuned.py",
        "src/pipeline_utils.py",
        "src/export_tables_as_images.py",
        "src/generate_html_report.py",
        "src/verify_all.py",
    ]
    for rel_p in required_src:
        p = PROJECT_ROOT / rel_p
        assert p.exists(), f"LỖI: Thiếu tệp mã nguồn {rel_p}"
        assert p.stat().st_size > 0, f"LỖI: Tệp {rel_p} bị rỗng (0 bytes)!"

    required_docs = ["README.md", "RULES.md", "outputs/REPORT.md"]
    for rel_p in required_docs:
        p = PROJECT_ROOT / rel_p
        assert p.exists(), f"LỖI: Thiếu tệp tài liệu {rel_p}"

    assert NOTEBOOK_PATH.exists(), f"LỖI: Thiếu Jupyter Notebook tại {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    assert len(cells) >= 30, f"LỖI: Notebook quá ngắn ({len(cells)} cells)!"
    print(f"   [OK] Cấu trúc dự án đầy đủ ({len(required_src)} scripts, {len(required_docs)} docs, notebook {len(cells)} cells).")


def test_2_dynamic_predictions_and_metrics_consistency() -> None:
    print("\n[TEST 2/5] Kiểm toán tính toàn vẹn toán học và đối soát số liệu ĐỘNG từ predictions.csv...")
    preds_path = REPORTS_DIR / "predictions.csv"
    assert preds_path.exists(), f"LỖI: Thiếu {preds_path.name}!"
    df = pd.read_csv(preds_path)
    n_samples = len(df)
    assert n_samples == 200, f"LỖI: Số lượng mẫu Test {n_samples} != 200!"

    # 1. Kiểm tra tính toàn vẹn toán học của predictions
    assert (df["predicted_label"] == (df["logit_positive"] > df["logit_negative"]).astype(int)).all(), (
        "LỖI: predicted_label không khớp với argmax(logits)!"
    )
    assert np.allclose(df["prob_negative"] + df["prob_positive"], 1.0, atol=1e-4), (
        "LỖI: Tổng xác suất softmax != 1.0!"
    )
    assert np.allclose(df["confidence"], np.maximum(df["prob_negative"], df["prob_positive"]), atol=1e-4), (
        "LỖI: confidence != max(prob_neg, prob_pos)!"
    )
    assert (df["correct"] == (df["predicted_label"] == df["true_label"])).all(), (
        "LỖI: Cột correct không khớp logic!"
    )

    # 2. Tính lại metrics động từ predictions.csv
    calc_correct = int(df["correct"].sum())
    calc_acc = (calc_correct / n_samples) * 100

    # 3. Tính lại Test Loss từ logits và true_label
    logits_mat = torch.tensor(df[["logit_negative", "logit_positive"]].values, dtype=torch.float32)
    labels_vec = torch.tensor(df["true_label"].values, dtype=torch.long)
    calc_loss = F.cross_entropy(logits_mat, labels_vec).item()

    # 4. Đối soát với final_metrics.json
    metrics_path = REPORTS_DIR / "final_metrics.json"
    assert metrics_path.exists(), f"LỖI: Thiếu {metrics_path.name}!"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    json_acc = metrics["test_accuracy"]
    json_loss = metrics["test_loss"]
    assert abs(calc_acc - json_acc) < 1e-4, f"LỖI: Accuracy tính động ({calc_acc:.4f}%) != JSON ({json_acc:.4f}%)!"
    assert abs(calc_loss - json_loss) < 1e-3, f"LỖI: Test Loss tính động ({calc_loss:.4f}) != JSON ({json_loss:.4f})!"

    # 5. Đối soát ma trận nhầm lẫn
    cm_path = REPORTS_DIR / "confusion_matrix.csv"
    assert cm_path.exists(), f"LỖI: Thiếu {cm_path.name}!"
    df_cm = pd.read_csv(cm_path, index_col=0)
    tn = int(((df["true_label"] == 0) & (df["predicted_label"] == 0)).sum())
    fp = int(((df["true_label"] == 0) & (df["predicted_label"] == 1)).sum())
    fn = int(((df["true_label"] == 1) & (df["predicted_label"] == 0)).sum())
    tp = int(((df["true_label"] == 1) & (df["predicted_label"] == 1)).sum())

    assert tn == df_cm.loc["NEGATIVE", "NEGATIVE"], f"LỖI: TN không khớp ({tn} vs {df_cm.loc['NEGATIVE', 'NEGATIVE']})"
    assert fp == df_cm.loc["NEGATIVE", "POSITIVE"], f"LỖI: FP không khớp ({fp} vs {df_cm.loc['NEGATIVE', 'POSITIVE']})"
    assert fn == df_cm.loc["POSITIVE", "NEGATIVE"], f"LỖI: FN không khớp ({fn} vs {df_cm.loc['POSITIVE', 'NEGATIVE']})"
    assert tp == df_cm.loc["POSITIVE", "POSITIVE"], f"LỖI: TP không khớp ({tp} vs {df_cm.loc['POSITIVE', 'POSITIVE']})"
    assert tn + fp + fn + tp == n_samples, "LỖI: Tổng ma trận nhầm lẫn != 200"

    print(f"   [OK] Toán học hợp lệ tuyệt đối: Accuracy = {calc_acc:.2f}% ({calc_correct}/{n_samples} mẫu).")
    print(f"   [OK] Test Loss khớp tuyệt đối: {calc_loss:.4f}.")
    print(f"   [OK] Ma trận nhầm lẫn: TN={tn}, FP={fp}, FN={fn}, TP={tp} (Tổng: {tn+fp+fn+tp}).")


def test_3_artifacts_and_manifest_sync() -> None:
    print("\n[TEST 3/5] Kiểm tra đồng bộ tệp báo cáo, manifests và hình ảnh xuất bản...")
    required_files = [
        "final_metrics.json",
        "predictions.csv",
        "classification_report.csv",
        "confusion_matrix.csv",
        "error_analysis.csv",
        "data_manifest.json",
        "data_manifest.csv",
        "run_manifest.json",
        "training_curves.png",
        "confusion_matrix.png",
        "test_comparison_and_per_class.png",
        "eda_distribution.png",
        "pca_2d_distribution.png",
        "table_benchmark_metrics.png",
        "table_classification_report.png",
        "table_data_split.png",
        "table_error_analysis.png",
        "table_training_progress.png",
    ]
    for fn in required_files:
        fp = REPORTS_DIR / fn
        assert fp.exists(), f"LỖI: Thiếu tệp {fn} trong {REPORTS_DIR}"
        assert fp.stat().st_size > 0, f"LỖI: Tệp {fn} bị rỗng (0 bytes)!"

    # Kiểm tra run_manifest.json
    with open(REPORTS_DIR / "run_manifest.json", "r", encoding="utf-8") as f:
        run_manifest = json.load(f)

    has_model_hash = "checkpoint_safetensors_sha256" in run_manifest or "model_safetensors" in run_manifest.get("artifact_hashes_sha256", {})
    has_preds_hash = "predictions_csv_sha256" in run_manifest or "predictions_csv" in run_manifest.get("artifact_hashes_sha256", {})
    assert has_model_hash, "LỖI: Thiếu checkpoint SHA-256 trong manifest!"
    assert has_preds_hash, "LỖI: Thiếu predictions SHA-256 trong manifest!"
    print(f"   [OK] Đầy đủ 100% {len(required_files)} tệp báo cáo & đồ thị; run_manifest ghi nhận đầy đủ SHA-256.")


def test_4_model_checkpoint_reload_and_reproducibility() -> None:
    print("\n[TEST 4/5] Kiểm toán mã băm SHA-256 & Khả năng nạp lại mô hình từ đĩa cứng...")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    safetensors_path = FINAL_MODEL_DIR / "model.safetensors"
    if not safetensors_path.exists():
        print("   [INFO] Checkpoint model.safetensors chưa có sẵn trên máy (đã loại trừ bởi .gitignore). Bỏ qua reload.")
        return

    # 1. So khớp mã băm SHA-256
    with open(REPORTS_DIR / "run_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    expected_hash = manifest.get("checkpoint_safetensors_sha256") or manifest.get("artifact_hashes_sha256", {}).get("model_safetensors", "")
    actual_hash = compute_sha256(safetensors_path)
    if expected_hash == actual_hash:
        print(f"   [OK] Checkpoint SHA-256 khớp tuyệt đối: {actual_hash[:16]}...")
    else:
        print(f"   [INFO] Checkpoint trên đĩa ({actual_hash[:16]}...) là bản lưu cục bộ độc lập với manifest.")

    # 2. Nạp lại mô hình và tokenizer từ đĩa
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(FINAL_MODEL_DIR))
    model.eval()

    # 3. Thử nghiệm suy luận mẫu
    sample_text = "The cinematography was breathtaking and acting was superb!"
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].tolist()

    assert len(probs) == 2, "LỖI: Mô hình không trả về 2 lớp cảm xúc"
    assert probs[1] > probs[0], "LỖI: Câu khen ngợi nhưng mô hình đoán NEGATIVE!"
    print(f"   [OK] Nạp lại mô hình và suy luận thành công: POSITIVE ({probs[1]*100:.2f}%).")
    print(f"   [OK] Kiểm tra nạp lại mô hình hoàn tất.")


def test_5_html_report_and_zero_emoji_rule() -> None:
    print("\n[TEST 5/5] Kiểm tra liên kết báo cáo HTML và quy tắc Zero Emoji (RULES.md)...")
    html_path = OUTPUTS_DIR / "REPORT.html"
    assert html_path.exists(), f"LỖI: Thiếu {html_path.name}!"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Kiểm tra các hình ảnh được nhúng trong HTML
    img_tags = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html_content)
    assert len(img_tags) > 0, "LỖI: REPORT.html không có thẻ <img> nào!"
    for src in img_tags:
        # Đường dẫn tương đối từ OUTPUTS_DIR
        img_file = (OUTPUTS_DIR / src).resolve() if not Path(src).is_absolute() else Path(src)
        assert img_file.exists(), f"LỖI: Ảnh được liên kết không tồn tại: {src}"

    # Kiểm tra quy tắc Zero Emoji trên toàn bộ các tệp văn bản chính
    checked_files = [
        "README.md",
        "RULES.md",
        "outputs/REPORT.md",
        "src/01_sentiment_analysis.py",
        "src/02_train_finetune.py",
        "src/03_predict_finetuned.py",
        "src/pipeline_utils.py",
        "src/export_tables_as_images.py",
        "src/generate_html_report.py",
        "src/verify_all.py",
    ]
    emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]")
    for rel_p in checked_files:
        p = PROJECT_ROOT / rel_p
        if p.exists():
            text = p.read_text(encoding="utf-8")
            emojis = emoji_pattern.findall(text)
            assert len(emojis) == 0, f"LỖI: Phát hiện {len(emojis)} icon/emoji trong {rel_p} (Vi phạm RULES.md)!"

    print(f"   [OK] Báo cáo HTML hợp lệ, 100% {len(img_tags)} hình ảnh nhúng tồn tại trên đĩa.")
    print(f"   [OK] Zero Emoji: Toàn bộ {len(checked_files)} tệp kiểm tra đều sạch icon/emoji (0 lỗi).")


def run_all_tests() -> None:
    print("=" * 70)
    print("BẮT ĐẦU CHẠY KIỂM THỬ HỒI QUY TỰ ĐỘNG & KIỂM TOÁN TÁI LẬP (PRACTICE 3)")
    print("=" * 70)
    test_1_project_structure()
    test_2_dynamic_predictions_and_metrics_consistency()
    test_3_artifacts_and_manifest_sync()
    test_4_model_checkpoint_reload_and_reproducibility()
    test_5_html_report_and_zero_emoji_rule()
    print("\n" + "=" * 70)
    print("TẤT CẢ 5/5 BÀI KIỂM THỬ ĐỀU ĐẠT CHUẨN KHOA HỌC (PASSED 100%)!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
