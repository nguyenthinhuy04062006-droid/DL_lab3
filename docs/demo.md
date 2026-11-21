# Hướng Dẫn Chạy & Thử Nghiệm Dự Án (Demo & Quickstart)

---

## 1. Yêu Cầu Môi Trường & Cài Đặt

Yêu cầu môi trường: **Python $\ge 3.10$** (đã xác thực trên Python 3.13, Windows 11).

```bash
# Cài đặt toàn bộ các thư viện phụ thuộc
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
3. **Tính toán Softmax thủ công:** Chạy forward pass lấy logits thô, dùng Softmax chuyển đổi thành xác suất phần trăm và so khớp với kết quả từ pipeline (sai số tuyệt đối $= 0.00$).
4. **Kiểm thử 5 mẫu câu đa dạng sắc thái:**
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
1. Tải và chia tập dữ liệu `stanfordnlp/imdb` (800 Train / 200 Val / 200 Test, `seed=42`).
2. Đo mốc Majority Baseline trên Test Set (52.00%).
3. Đo mốc Untrained Baseline trên tập Validation (44.00%).
4. Chạy `Trainer` qua 2 epochs (100 steps) trên CPU (391.42s ~6.52 phút).
5. Tự động lưu checkpoint tốt nhất vào: `outputs/final_model/model.safetensors`.
6. Tự động xuất 18 báo cáo số liệu và biểu đồ vào thư mục `outputs/reports/`.

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
Cảm xúc dự đoán: POSITIVE (Độ tự tin: 78.23%)

Chi tiết xác suất từng nhãn:
  - NEGATIVE  :  21.77% | ██████
  - POSITIVE  :  78.23% | ███████████████████████
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
Cảm xúc dự đoán: NEGATIVE (Độ tự tin: 89.45%)

Chi tiết xác suất từng nhãn:
  - NEGATIVE  :  89.45% | ██████████████████████████
  - POSITIVE  :  10.55% | ███
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
======================================================================
BẮT ĐẦU CHẠY KIỂM THỬ HỒI QUY TỰ ĐỘNG & KIỂM TOÁN TÁI LẬP (PRACTICE 3)
======================================================================
[TEST 1/5] Kiểm tra cấu trúc thư mục, mã nguồn và tài liệu...
   [OK] Cấu trúc dự án đầy đủ (7 scripts, 3 docs, notebook 53 cells).

[TEST 2/5] Kiểm toán tính toàn vẹn toán học và đối soát số liệu ĐỘNG từ predictions.csv...
   [OK] Toán học hợp lệ tuyệt đối: Accuracy = 74.50% (149/200 mẫu).
   [OK] Test Loss khớp tuyệt đối: 0.5565.
   [OK] Ma trận nhầm lẫn: TN=84, FP=20, FN=31, TP=65 (Tổng: 200).

[TEST 3/5] Kiểm tra đồng bộ tệp báo cáo, manifests và hình ảnh xuất bản...
   [OK] Đầy đủ 100% 18 tệp báo cáo & đồ thị; run_manifest ghi nhận đầy đủ SHA-256.

[TEST 4/5] Kiểm toán mã băm SHA-256 & Khả năng nạp lại mô hình từ đĩa cứng...
   [OK] Nạp lại mô hình và suy luận thành công: POSITIVE (78.23%).
   [OK] Kiểm tra nạp lại mô hình hoàn tất.

[TEST 5/5] Kiểm tra liên kết báo cáo HTML và quy tắc Zero Emoji (RULES.md)...
   [OK] Báo cáo HTML hợp lệ, 100% 10 hình ảnh nhúng tồn tại trên đĩa.
   [OK] Zero Emoji: Toàn bộ 10 tệp kiểm tra đều sạch icon/emoji (0 lỗi).

======================================================================
TẤT CẢ 5/5 BÀI KIỂM THỬ ĐỀU ĐẠT CHUẨN KHOA HỌC (PASSED 100%)!
======================================================================
```
