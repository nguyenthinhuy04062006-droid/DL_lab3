# BÁO CÁO DỰ ÁN
## PRACTICE 3: GET STARTED WITH HUGGING FACE
### Phân loại cảm xúc (Sentiment Analysis) & Fine-tuning DistilBERT trên IMDb

---

> **Thông tin thực nghiệm**
> - **Mô hình nền tảng:** DistilBERT (`distilbert/distilbert-base-uncased`, 6 layers, 66M tham số).
> - **Bộ dữ liệu:** IMDb Movie Reviews (`stanfordnlp/imdb`), 800 Train / 200 Val / 200 Test (`seed=42`).
> - **Môi trường & Thời gian:** Thực thi CPU, tổng thời gian huấn luyện 391.42s (~6.52 phút, 100 steps).
> - **Đầu ra chính:** `outputs/final_model/model.safetensors`, `outputs/reports/final_metrics.json`.

---

## 1. Mục Tiêu Thực Hành

1. **Exercise 1 — Suy luận Pretrained:** Sử dụng pipeline Hugging Face (`distilbert-base-uncased-finetuned-sst-2-english`), phân tích cơ chế Tokenizer (WordPiece sub-words, input IDs, attention mask) và tính toán thủ công xác suất từ logits qua hàm Softmax.
2. **Exercise 2 — Tinh chỉnh DistilBERT:** Triển khai quy trình 7 bước huấn luyện trên tập dữ liệu IMDb, đối chiếu đa tầng với mốc Baseline và phân tích nguyên nhân sai số chuyên sâu.

---

## 2. Cơ Sở Lý Thuyết

### 2.1 Kiến trúc DistilBERT & Sequence Classification Head
- **DistilBERT:** Bản nén tri thức (Knowledge Distillation) từ BERT-base, giảm 40% tham số (66M so với 110M), tăng tốc độ suy luận 60% mà vẫn bảo lưu hơn 97% năng lực hiểu ngôn ngữ.
- **Classification Head:** Vector ẩn của token `[CLS]` ($\mathbf{h}_{\text{CLS}} \in \mathbb{R}^{768}$) đi qua tầng `pre_classifier` ($\text{Linear}(768 \to 768) + \text{ReLU}$), tầng `Dropout(p=0.2)` và tầng `classifier` ($\text{Linear}(768 \to 2)$) để tạo ra logits thô $\mathbf{z} = [z_0, z_1]$.
- **Xác suất Softmax:** Chuyển đổi logits sang phân phối xác suất hợp lệ:
  $$P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=0}^{1} e^{z_j}}, \quad c \in \{0, 1\}$$

### 2.2 Tokenizer WordPiece & Đệm động (Dynamic Padding)
- **WordPiece Sub-words:** Tách từ thành các đơn vị con với tiền tố `##` (ví dụ: `"horrifically"` $\to$ `['horrific', '##ally']`), khắc phục hoàn toàn bài toán Out-Of-Vocabulary (OOV).
- **Mặt nạ chú ý (Attention Mask):** Phân biệt token văn bản thật (giá trị 1) và token đệm `[PAD]` (giá trị 0) để triệt tiêu tính toán thừa trong cơ chế Self-Attention.
- **Đệm động (`DataCollatorWithPadding`):** Đệm theo câu dài nhất trong từng mini-batch thay vì toàn tập dữ liệu, giảm độ phức tạp tính toán $\mathcal{O}(L^2)$ và tiết kiệm 35–50% thời gian trên CPU.

---

## 3. Thiết Kế và Triển Khai Thực Nghiệm

### 3.1 Phân chia dữ liệu (No Data Leakage)
- Do IMDb gốc sắp xếp 12,500 mẫu NEGATIVE liên tục, hệ thống bắt buộc thực hiện `shuffle(seed=42)` trước khi trích chọn.
- **Tập Train (800 mẫu):** 408 NEG (51.0%) / 392 POS (49.0%) — dùng tối ưu gradient.
- **Tập Validation (200 mẫu):** 104 NEG (52.0%) / 96 POS (48.0%) — dùng chọn Checkpoint tối ưu.
- **Tập Test (200 mẫu):** 104 NEG (52.0%) / 96 POS (48.0%) — tách biệt nghiêm ngặt, chỉ đánh giá một lần duy nhất sau cùng.

### 3.2 Cấu hình huấn luyện & Tiến trình tối ưu hóa
- Cấu hình: `learning_rate = 2e-5`, `batch_size = 16`, `epochs = 2` (100 steps), `weight_decay = 0.01`, `max_length = 128`.

| Vòng lặp (Epoch) | Bước (Step) | Training Loss | Validation Loss | Validation Accuracy (%) | Trạng thái mô hình |
|:---:|:---:|---:|---:|---:|:---|
| 0 (Untrained) | 0 | — | 0.6974 | 44.00% | Phân loại ngẫu nhiên |
| Epoch 1 | 50 | 0.6155 | 0.5998 | 69.50% | Hội tụ nhanh |
| **Epoch 2 (Best)** | **100** | **0.5032** | **0.5118** | **76.50%** | **Đạt đỉnh Val Acc $\to$ Khóa lưu Checkpoint** |

---

## 4. Kết Quả Định Lượng

### 4.1 Đối chiếu Baseline vs. Fine-tuned (Test Set N=200)

| Chỉ số kiểm thử | Untrained Baseline (Val) | Majority-Class (Test) | DistilBERT Fine-tuned (Test) | Mức tăng trưởng (+ Gain) |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy (Độ chính xác)** | 44.00% | 52.00% | **74.50%** (149/200) | **+22.50% vs Majority (Test)**<br>**+32.50% Val-to-Val Gain** |
| **Khoảng tin cậy 95% (Wilson)** | — | — | **[68.04%, 80.04%]** | Đảm bảo tính vững thống kê |
| **Macro F1-Score** | — | — | **74.27%** | Cân bằng giữa hai lớp |
| **Macro Precision / Recall** | — | — | **74.76% / 74.24%** | Phản ánh chất lượng đồng đều |
| **Cross-Entropy Test Loss** | — | — | **0.5565** | Hội tụ ổn định |

### 4.2 Báo cáo phân loại chi tiết & Ma trận nhầm lẫn

| Lớp nhãn | Precision | Recall | F1-Score | Support | Thực tế \ Dự đoán | Dự đoán NEG (0) | Dự đoán POS (1) |
|:---:|:---:|:---:|:---:|:---:|---|:---:|:---:|
| **NEGATIVE (`0`)** | 73.04% | 80.77% | **76.71%** | 104 | **Thực tế NEG (104)** | **TN = 84** (42.0%) | FP = 20 (10.0%) |
| **POSITIVE (`1`)** | 76.47% | 67.71% | **71.82%** | 96 | **Thực tế POS (96)** | FN = 31 (15.5%) | **TP = 65** (32.5%) |
| **Macro Average** | **74.76%** | **74.24%** | **74.27%** | **200** | **Tổng số dự đoán** | **115 NEG** | **85 POS** |

---

## 5. Phân Tích Sai Số Chuyên Sâu (Error Analysis)

Khảo sát 51 ca dự đoán sai trên tập Test (`outputs/reports/error_analysis.csv`) chỉ ra 3 nhóm nguyên nhân chính:
1. **Cảm xúc trộn lẫn (Mixed Sentiment):** Các đánh giá có nhiều từ ngữ u ám miêu tả phim kinh dị (*weird, horrifically, disturbing, dark*) lấn át kết luận khen ngợi ở cuối (*"worthwhile viewing"*), ví dụ Index 180 (đoán sai với độ tự tin 80.23%).
2. **Nhiễu ngữ cảnh cá nhân (Contextual Confounding):** Người xem kể về tâm trạng xấu trong ngày (*"horrid day"*) trước khi khen phim (*"movie grabbed me"*), ví dụ Index 88 (đoán NEG với độ tự tin 82.04%).
3. **Khen ngợi theo lối mỉa mai (Sarcasm & Cult-movie):** Người xem khen phim hạng B bằng cách liệt kê các khuyết điểm hài hước (*"bad hair, arthritis, good bad dino-flick"*), ví dụ Index 103 (đoán NEG 76.10%).
4. **Tác động cắt ngắn token:** Nhóm không bị cắt ($\le 128$) sai 21.43% (6/28), nhóm bị cắt ($> 128$) sai 26.16% (45/172). Kiểm định Fisher cho $p = 0.8153 > 0.05$, chứng minh cắt cụt 128 không tạo ra sai lệch thống kê đáng kể.

---

## 6. Thảo Luận Kỹ Thuật & Hạn Chế

| Khía cạnh kỹ thuật | Hiện trạng thực nghiệm | Hạn chế ghi nhận | Hướng giải quyết đề xuất |
|:---|:---|:---|:---|
| **Dữ liệu huấn luyện** | 800 mẫu IMDb, 2 epochs | Cỡ mẫu nhỏ, phương sai phụ thuộc seed | Mở rộng 25,000 mẫu toàn tập trên GPU |
| **Chiều dài chuỗi** | Giới hạn 128 tokens | Mất ngữ cảnh đoạn kết ở 86% bài viết | Tăng `max_length=512` hoặc dùng Longformer |
| **Hội tụ tối ưu** | Tốc độ học cố định 2e-5 | Chưa có giai đoạn khởi động (warmup) | Bổ sung `warmup_ratio=0.1` và Cosine Decay |

---

## 7. Kết Luận

Dự án Practice 3 đã hoàn thành xuất sắc các mục tiêu đề ra:
- Nắm vững cơ chế vận hành từ Tokenizer, cơ chế Attention đến suy luận thủ công Softmax.
- Fine-tune DistilBERT thành công trên CPU, nâng độ chính xác từ **44.00% (Untrained) lên 74.50% (Test)**, vượt mốc đa số **+22.50%**, F1 đạt **74.27%**.
- Hệ thống kiểm thử tự động đạt chuẩn tuyệt đối **5/5 tests PASSED (100%)**.

---

## 8. Tài Liệu Tham Khảo

1. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762
2. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*. https://arxiv.org/abs/1910.01108
3. Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL-HLT 2019*. https://arxiv.org/abs/1810.04805
4. Maas, A. L., et al. (2011). Learning word vectors for sentiment analysis. *ACL 2011*. https://aclanthology.org/P11-1015
5. Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *EMNLP 2020*. https://arxiv.org/abs/1910.03771
