"""
01_sentiment_analysis.py — Exercise 1: Sentiment Analysis with Hugging Face.

Theo đúng 4 yêu cầu của Exercise 1:
1. Install and import Hugging Face transformers.
2. Use a pre-trained sentiment analysis model from the Hugging Face Hub.
3. Tokenize a sample sentence (inspect tokens, input_ids, attention_mask).
4. Perform sentiment analysis on the sentence (via pipeline and manual forward pass with Softmax).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Đảm bảo hiển thị tiếng Việt trên Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Khắc phục tương thích lỗi 404 additional_chat_templates trên Hugging Face Hub
try:
    import transformers.tokenization_utils_base
    transformers.tokenization_utils_base.list_repo_templates = lambda *a, **k: []
except Exception:
    pass

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def run_pipeline_demo(model_name: str, sentence: str) -> None:
    """
    Bước 2 & 4: Thực thi suy luận nhanh bằng Hugging Face pipeline().
    """
    print("=" * 70)
    print("1. SUY LUẬN NHANH VỚI HUGGING FACE PIPELINE")
    print("=" * 70)
    print(f"📦 Mô hình Pretrained: {model_name}")
    print(f"💬 Câu đầu vào: \"{sentence}\"")

    classifier = pipeline("sentiment-analysis", model=model_name)
    result = classifier(sentence)[0]

    label = result["label"]
    score = result["score"] * 100
    print(f"👉 Kết quả dự đoán: {label} (Độ tự tin: {score:.2f}%)\n")


def run_tokenizer_deep_dive(model_name: str, sentence: str) -> None:
    """
    Bước 3: Mổ xẻ chi tiết cơ chế hoạt động của Tokenizer (Sub-word WordPiece).
    """
    print("=" * 70)
    print("2. KHÁM PHÁ CHI TIẾT TOKENIZER (TOKENIZATION DEEP DIVE)")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded = tokenizer(sentence, return_tensors="pt")

    tokens = tokenizer.tokenize(sentence)
    input_ids = encoded["input_ids"][0].tolist()
    attention_mask = encoded["attention_mask"][0].tolist()

    print(f"💬 Câu gốc: \"{sentence}\"")
    print(f"🔤 Sub-word Tokens ({len(tokens)} tokens): {tokens}")
    print(f"🔢 Input IDs ({len(input_ids)} IDs kèm [CLS], [SEP]): {input_ids}")
    print(f"🎭 Attention Mask: {attention_mask}")

    # Bảng ánh xạ từng token với ID tương ứng
    print("\n📋 Bảng ánh xạ Token -> ID:")
    for token_id in input_ids:
        token_str = tokenizer.decode([token_id])
        print(f"   ID {token_id:>6}  --->  '{token_str}'")
    print()


def run_manual_forward_pass(model_name: str, sentence: str) -> None:
    """
    Bước 4: Suy luận thủ công — Trích xuất Logits và tính Softmax.
    Giúp sinh viên hiểu rõ bản chất toán học bên dưới của Hugging Face pipeline.
    """
    print("=" * 70)
    print("3. SUY LUẬN THỦ CÔNG: TÍNH TOÁN LOGITS VÀ SOFTMAX")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Kích thước: (1, 2)
        probabilities = F.softmax(logits, dim=-1)[0]  # Chuẩn hóa về [0, 1]

    id2label = model.config.id2label
    pred_idx = torch.argmax(probabilities).item()
    pred_label = id2label[pred_idx]
    confidence = probabilities[pred_idx].item() * 100

    print(f"⚡ Raw Logits từ mô hình: {logits[0].tolist()}")
    print("📊 Phân phối xác suất Softmax:")
    for idx, prob in enumerate(probabilities):
        label_name = id2label[idx]
        print(f"   - Lớp {idx} ({label_name:<8}): {prob.item() * 100:>6.2f}%")

    print(f"\n⭐ Dự đoán cuối cùng: {pred_label} ({confidence:.2f}%)")

    # Kiểm chứng tính nhất quán toán học với pipeline trên cùng câu văn bản
    classifier = pipeline("sentiment-analysis", model=model_name)
    pipe_res = classifier(sentence)[0]
    assert pipe_res["label"] == pred_label, f"Lệch nhãn: {pipe_res['label']} vs {pred_label}"
    diff = abs(pipe_res["score"] - probabilities[pred_idx].item())
    assert diff < 1e-4, f"Lệch xác suất quá ngưỡng: {diff}"
    print(f"✅ [KIỂM CHỨNG TOÁN HỌC: assert Passed] Sai số giữa Pipeline và Thủ công: {diff:.2e}\n")


def test_diverse_sentences(model_name: str) -> None:
    """
    Thử nghiệm trên các câu văn phong phú (tích cực, tiêu cực, phủ định kép, mỉa mai).
    """
    print("=" * 70)
    print("4. THỬ NGHIỆM TRÊN TẬP CÂU ĐA DẠNG NGỮ CẢNH")
    print("=" * 70)

    classifier = pipeline("sentiment-analysis", model=model_name)
    sample_sentences = [
        "This product exceeded all my expectations, absolutely phenomenal!",
        "The delivery was horribly delayed and customer service was rude.",
        "It was not bad at all, quite surprisingly pleasant.",
        "Oh wonderful, another package arrived completely broken. Just what I needed.",
        "The movie was an absolute masterpiece with breathtaking visuals and profound acting.",
    ]

    for s in sample_sentences:
        res = classifier(s)[0]
        label = res["label"]
        score = res["score"] * 100
        icon = "🟢" if label == "POSITIVE" else "🔴"
        print(f"{icon} [{label:<8} | {score:>5.1f}%]  \"{s}\"")
    print("=" * 70 + "\n")


def main() -> None:
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    sample_sentence = "I absolutely love using Hugging Face transformers for natural language processing!"

    print("\n" + "#" * 70)
    print("🚀 PRACTICE 3 — EXERCISE 1: SENTIMENT ANALYSIS WITH HUGGING FACE")
    print("#" * 70 + "\n")

    run_pipeline_demo(model_name, sample_sentence)
    run_tokenizer_deep_dive(model_name, sample_sentence)
    run_manual_forward_pass(model_name, sample_sentence)
    test_diverse_sentences(model_name)


if __name__ == "__main__":
    main()
