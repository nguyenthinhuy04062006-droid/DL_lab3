"""
03_predict_finetuned.py — Công cụ dự đoán cảm xúc câu văn bản sử dụng mô hình đã fine-tune (outputs/final_model).

Cách dùng:
    python src/03_predict_finetuned.py --text "This movie was absolutely brilliant and touching."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_MODEL_DIR = PROJECT_ROOT / "outputs" / "final_model"


def predict(text: str) -> None:
    if not FINAL_MODEL_DIR.exists() or not (FINAL_MODEL_DIR / "config.json").exists():
        print(f"❌ Chưa tìm thấy mô hình fine-tune tại: {FINAL_MODEL_DIR}")
        print("   Vui lòng chạy 'python src/02_train_finetune.py' trước!")
        return

    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(FINAL_MODEL_DIR))
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    id2label = model.config.id2label
    pred_idx = torch.argmax(probs).item()
    pred_label = id2label.get(pred_idx, str(pred_idx))
    confidence = probs[pred_idx].item() * 100

    icon = "🟢" if pred_label == "POSITIVE" else "🔴"

    print("\n" + "=" * 60)
    print("🎯 KẾT QUẢ DỰ ĐOÁN CẢM XÚC (FINE-TUNED DISTILBERT)")
    print("=" * 60)
    print(f"💬 Câu đầu vào: \"{text}\"")
    print(f"{icon} Cảm xúc dự đoán: {pred_label} (Độ tự tin: {confidence:.2f}%)")
    print("\nChi tiết xác suất từng nhãn:")
    for idx, prob in enumerate(probs):
        lbl = id2label.get(idx, str(idx))
        bar = "█" * int(prob.item() * 30)
        print(f"  - {lbl:<10}: {prob.item() * 100:>6.2f}% | {bar}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment using fine-tuned DistilBERT")
    parser.add_argument(
        "--text",
        type=str,
        default="I thoroughly enjoyed this film, the cinematography and direction were top notch!",
        help="Câu văn bản cần phân tích cảm xúc",
    )
    args = parser.parse_args()
    predict(args.text)


if __name__ == "__main__":
    main()
