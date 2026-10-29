"""
04_realtime_stream_simulation.py — Mô phỏng phân tích cảm xúc theo luồng thời gian thực (Real-time Stream Inference).

Ứng dụng thực tiễn: Giám sát phản hồi khách hàng theo thời gian thực cho hệ thống Smart Logistics & E-commerce, đo độ trễ suy luận (latency ms) và kích hoạt cảnh báo sớm.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Khắc phục tương thích lỗi 404 additional_chat_templates trên Hugging Face Hub
try:
    import transformers.tokenization_utils_base
    transformers.tokenization_utils_base.list_repo_templates = lambda *a, **k: []
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_MODEL_DIR = PROJECT_ROOT / "outputs" / "final_model"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


FEEDBACK_STREAM = [
    "Package arrived 2 days earlier than scheduled, excellent service!",
    "Delivery driver was polite and called ahead before arriving.",
    "The box was crushed and water damaged upon arrival, completely unacceptable.",
    "Driver dropped off the package at the wrong building entrance.",
    "Items were packed with great care and plenty of bubble wrap.",
    "Tracking status hasn't updated for four straight days, where is my order?",
    "Quick, courteous, and hassle-free return pickup at my doorstep.",
    "Customer service rep was rude and hung up on me when I asked for a refund.",
    "Great communication throughout the entire shipping process, highly recommend!",
    "Late delivery again for the third time this month, taking my business elsewhere.",
]


def run_stream_simulation() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = str(FINAL_MODEL_DIR) if (FINAL_MODEL_DIR / "config.json").exists() else "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    print("\n" + "=" * 70)
    print("🚚 MÔ PHỎNG PHÂN TÍCH CẢM XÚC LUỒNG THỜI GIAN THỰC (SMART LOGISTICS)")
    print("=" * 70)
    print(f"📦 Mô hình sử dụng: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    records = []
    print(f"{'STT':>3} | {'Cảm xúc':<8} | {'Độ tin cậy':>10} | {'Độ trễ':>9} | {'Nội dung phản hồi':<50}")
    print("-" * 90)

    for idx, text in enumerate(FEEDBACK_STREAM, 1):
        start_t = time.perf_counter()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
        latency_ms = (time.perf_counter() - start_t) * 1000

        pred_idx = torch.argmax(probs).item()
        label = model.config.id2label.get(pred_idx, str(pred_idx))
        confidence = probs[pred_idx].item() * 100
        icon = "🟢" if label == "POSITIVE" else "🔴"

        records.append({
            "step": idx,
            "text": text,
            "label": label,
            "confidence": confidence,
            "latency_ms": latency_ms,
        })

        preview = text[:48] + "..." if len(text) > 48 else text
        print(f"{idx:>3} | {icon} {label:<6} | {confidence:>9.2f}% | {latency_ms:>7.2f}ms | {preview}")

    df_stream = pd.DataFrame(records)
    csv_path = REPORTS_DIR / "realtime_logistics_simulation.csv"
    df_stream.to_csv(csv_path, index=False)

    # Vẽ biểu đồ giám sát luồng
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Smart Logistics — Real-time Sentiment & Latency Monitoring", fontsize=13, fontweight="bold")

    # Đồ thị độ trễ suy luận
    axes[0].plot(df_stream["step"], df_stream["latency_ms"], marker="o", color="#3498DB", linewidth=2)
    avg_latency = df_stream["latency_ms"].mean()
    axes[0].axhline(y=avg_latency, color="#E74C3C", linestyle="--", label=f"Trung bình: {avg_latency:.2f} ms")
    axes[0].set_xlabel("Thứ tự phản hồi (Event Stream)")
    axes[0].set_ylabel("Độ trễ suy luận (ms)")
    axes[0].set_title("Inference Latency per Event")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Biểu đồ tỷ lệ cảm xúc
    counts = df_stream["label"].value_counts()
    colors = ["#27AE60" if k == "POSITIVE" else "#E74C3C" for k in counts.index]
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=140)
    axes[1].set_title("Customer Sentiment Distribution")

    plt.tight_layout()
    png_path = REPORTS_DIR / "realtime_logistics_simulation.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    print("-" * 90)
    print(f"💾 Đã lưu kết quả: {csv_path.name} & {png_path.name}")
    print(f"⚡ Độ trễ trung bình: {avg_latency:.2f} ms/sự kiện\n")


if __name__ == "__main__":
    run_stream_simulation()
