# Hướng Dẫn Chạy & Thử Nghiệm Dự Án (Demo & Quickstart)

---

## 1. Yêu Cầu Môi Trường & Cài Đặt

Yêu cầu môi trường: **Python $\ge 3.10$**.

```bash
# 1. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

Danh mục thư viện chính trong [requirements.txt](../requirements.txt):
- `torch >= 2.0.0`
- `transformers >= 4.36.0`
- `datasets >= 2.16.0`
- `evaluate >= 0.4.1`
- `accelerate >= 0.25.0`
- `scikit-learn >= 1.3.0`
- `matplotlib >= 3.8.0`
- `pandas >= 2.0.0`

---

## 2. Demo 1: Suy Luận Với Model Pretrained (Exercise 1)

Lệnh thực thi:
```bash
python src/01_sentiment_analysis.py
```

### Quy trình chạy:
1. **Pipeline Inference:** Nạp model `distilbert/distilbert-base-uncased-finetuned-sst-2-english` và dự đoán trực tiếp câu mẫu.
2. **Kiểm tra Tokenizer:** Tách từ qua WordPiece, in danh sách `input_ids` cùng các mã đặc biệt `[CLS]` (101), `[SEP]` (102) và vector `attention_mask`.
3. **Tính toán Softmax thủ công:** Chạy forward pass lấy logits thô, dùng Softmax chuyển đổi thành xác suất phần trăm và so khớp với kết quả từ pipeline.
4. **Kiểm thử 5 mẫu câu:**
   - Câu tích cực: *"The cinematography in this film was breathtaking and poetic."*
   - Câu tiêu cực: *"The plot was predictable and the pacing felt excruciatingly slow."*
   - Câu phủ định: *"I did not dislike the film, but it was not extraordinary either."*
   - Câu mỉa mai: *"Oh brilliant, another two hours of my life wasted on sheer nonsense!"*
   - Câu hỗn hợp: *"The visual effects were stunning, although the script was weak."*

---

## 3. Demo 2: Huấn Luyện Fine-tuning Model (Exercise 2)

Lệnh thực thi:
```bash
python src/02_train_finetune.py
```

### Tiến trình tự động:
1. Tải và chia tập dữ liệu `stanfordnlp/imdb` (800 Train / 200 Val / 200 Test).
2. Đo mốc Majority Baseline (52.00%).
3. Đo mốc Untrained Baseline trên tập Validation (44.00%).
4. Chạy `Trainer` qua 2 epochs (100 steps) trên CPU (~4.5 – 7 phút).
5. Tự động lưu checkpoint tốt nhất vào: `outputs/final_model/model.safetensors`.
6. Tự động xuất báo cáo và biểu đồ vào thư mục `outputs/reports/`.

---

## 4. Demo 3: Dự Đoán Câu Bất Kỳ Bằng CLI (CLI Prediction)

Dự đoán câu mới với model đã huấn luyện:

### Ví dụ 1: Câu khen ngợi
```bash
python src/03_predict_finetuned.py --text "The cinematography was breathtaking and acting was superb!"
```
**Kết quả hiển thị trên Terminal:**
```text
============================================================
KẾT QUẢ DỰ ĐOÁN CẢM XÚC (FINE-TUNED DISTILBERT)
============================================================
Câu đầu vào: "The cinematography was breathtaking and acting was superb!"
Cảm xúc dự đoán: POSITIVE (Độ tự tin: 88.42%)

Chi tiết xác suất từng nhãn:
  - NEGATIVE  :  11.58% | ███
  - POSITIVE  :  88.42% | ██████████████████████████
============================================================
```

### Ví dụ 2: Câu chê bai
```bash
python src/03_predict_finetuned.py --text "Terrible screenplay, bad acting and total waste of time."
```
**Kết quả hiển thị trên Terminal:**
```text
============================================================
KẾT QUẢ DỰ ĐOÁN CẢM XÚC (FINE-TUNED DISTILBERT)
============================================================
Câu đầu vào: "Terrible screenplay, bad acting and total waste of time."
Cảm xúc dự đoán: NEGATIVE (Độ tự tin: 93.15%)

Chi tiết xác suất từng nhãn:
  - NEGATIVE  :  93.15% | ████████████████████████████
  - POSITIVE  :   6.85% | █
============================================================
```

---

## 5. Demo 4: Kiểm Thử Toàn Bộ Hệ Thống (Regression Verification)

Lệnh thực thi kiểm thử tự động 5/5 bài test:
```bash
python src/verify_all.py
```

**Kết quả tiêu chuẩn:**
```text
=================================================================
BẮT ĐẦU CHẠY KIỂM THỬ HỒI QUY TỰ ĐỘNG (PRACTICE 3 SUITE)
=================================================================
[TEST 1/5] Kiểm tra cấu trúc & biểu đồ trực quan hóa của Notebook...
   [OK] Notebook hợp lệ: 41 cells, chứa đầy đủ 4 biểu đồ trực quan hóa (Cells 15, 27, 33, 34).

[TEST 2/5] Kiểm tra các tệp báo cáo & biểu đồ đồng bộ...
   [OK] Tệp hợp lệ: final_metrics.json (1.95 KB)
   [OK] Tệp hợp lệ: classification_report.csv (0.22 KB)
   [OK] Tệp hợp lệ: confusion_matrix.csv (0.05 KB)
   [OK] Tệp hợp lệ: error_analysis.csv (75.02 KB)
   [OK] Tệp hợp lệ: training_curves.png (120.11 KB)
   [OK] Tệp hợp lệ: confusion_matrix.png (41.10 KB)
   [OK] Tệp hợp lệ: test_comparison_and_per_class.png (75.28 KB)
   [OK] Tệp hợp lệ: eda_distribution.png (195.40 KB)
   [OK] Tệp hợp lệ: pca_2d_distribution.png (141.00 KB)

[TEST 3/5] Kiểm tra tính toán vẹn & logic số liệu thực nghiệm...
   [OK] Số liệu khớp tuyệt đối: Accuracy = 74.5% (149/200 mẫu).
   [OK] Ma trận nhầm lẫn: TN=84, FP=20, FN=31, TP=65.

[TEST 4/5] Kiểm tra khả năng nạp và suy luận của Fine-Tuned Model...
   [OK] Mô hình suy luận chuẩn xác: POSITIVE (78.23%) cho câu test.

[TEST 5/5] Kiểm tra liên kết và tính tự chứa của REPORT.html...
   [OK] Báo cáo HTML nhúng đầy đủ 5 biểu đồ chuẩn xác.

=================================================================
TẤT CẢ 5/5 BÀI TEST ĐỀU ĐẠT CHUẨN (PASSED 100%)!
=================================================================
```
