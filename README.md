# Practice 3 — Get Started with Hugging Face: Sentiment Analysis & Fine-tuning DistilBERT

Dự án thực hành chuyên sâu về Xử lý Ngôn ngữ Tự nhiên (NLP) hiện đại với hệ sinh thái Hugging Face (`transformers`, `datasets`, `evaluate`). Dự án bao gồm toàn bộ nội dung từ lý thuyết nền tảng, thực thi suy luận với mô hình Pretrained, quy trình 7 bước chuẩn Fine-tuning trên IMDb, và đánh giá khoa học có đối chứng (Baseline vs Fine-tuned, Ma trận nhầm lẫn, Phân tích lỗi).

---

## 1. Mục tiêu học tập

- **Làm chủ quy trình xử lý nội tại của Transformer:** Hiểu sâu về Tokenizer (WordPiece sub-words, token IDs, attention mask, dynamic padding) và cơ chế suy luận từ Logits thô qua Softmax.
- **Phân biệt rành mạch Inference và Fine-tuning:** Nắm rõ khi nào chỉ cần dùng checkpoint có sẵn và khi nào bắt buộc phải tinh chỉnh trọng số mạng.
- **Quy trình 7 bước Fine-tuning chuẩn mực:** Tải dữ liệu công khai, chia tập tránh rò rỉ dữ liệu (`train_test_split`), thiết lập siêu tham số (`TrainingArguments`), sử dụng `Trainer`, và tính toán bộ chỉ số đa chiều (Accuracy, Precision, Recall, Macro-F1).
- **Đánh giá có đối chứng khoa học:** Đánh giá Baseline trước huấn luyện trên tập Validation (44.00% do classification head ngẫu nhiên) so sánh với kết quả kiểm thử sau huấn luyện (74.50%).
- **Phân tích lỗi (Error Analysis):** Truy vết các mẫu dự đoán sai mà mô hình tự tin nhất để hiểu rõ giới hạn ngữ nghĩa.

---

## 2. Nguồn dữ liệu và Mô hình

- **Bộ dữ liệu:** [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) — Large Movie Review Dataset phân loại văn bản nhị phân:
  - `0 = NEGATIVE` (tiêu cực)
  - `1 = POSITIVE` (tích cực)
- **Base Model cho Fine-tuning:** [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) (6 layers, 66 triệu tham số, nhanh hơn 60% và nhẹ hơn 40% so với BERT).
- **Mô hình Pretrained cho Bài 1:** [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) (đã fine-tune sẵn trên tập Stanford Sentiment Treebank).

---

## 3. Cấu trúc thư mục dự án

```text
lab3_DL/
├── notebooks/
│   └── Practice3_HuggingFace_Final.ipynb # Notebook Jupyter chuẩn hoá duy nhất hoàn chỉnh cả 2 bài tập
├── src/
│   ├── 01_sentiment_analysis.py         # Bài 1: Pipeline, Tokenizer Deep Dive, Logits & Softmax thủ công
│   ├── 02_train_finetune.py             # Bài 2: Fine-tuning 7 bước, Baseline, Training Curves & Error Analysis
│   └── 03_predict_finetuned.py          # Suy luận dự đoán câu mới với mô hình đã fine-tune
├── outputs/                             # Thư mục chứa kết quả thực nghiệm thực tế
│   ├── final_model/                     # Trọng số mô hình và tokenizer sau fine-tuning (model.safetensors)
│   ├── checkpoints/                     # Checkpoints lưu trong quá trình huấn luyện
│   └── reports/
│       ├── final_metrics.json           # Toàn bộ thông số và kết quả đo đạc thực tế
│       ├── classification_report.csv    # Báo cáo phân loại chi tiết (Precision/Recall/F1)
│       ├── confusion_matrix.csv         # Bảng số liệu ma trận nhầm lẫn
│       ├── confusion_matrix.png         # Đồ thị Heatmap ma trận nhầm lẫn
│       ├── training_curves.png          # Đồ thị đường cong loss và validation metrics
│       ├── test_comparison_and_per_class.png # So sánh mô hình & chỉ số từng lớp
│       ├── eda_distribution.png         # Phân phối nhãn & độ dài token
│       └── error_analysis.csv           # Bảng danh sách các mẫu dự đoán sai
├── requirements.txt                     # Danh mục thư viện phụ thuộc (transformers, datasets, evaluate...)
├── RULES.md                             # Bộ quy tắc khoa học và kỹ thuật của dự án
├── AI_Usage.md                          # Báo cáo sử dụng AI minh bạch theo chuẩn học thuật
├── REPORT.md                            # Báo cáo học thuật chi tiết bằng tiếng Việt
└── README.md                            # Hướng dẫn tổng quan dự án
```

---

## 4. Hướng Dẫn Cài Đặt & Sử Dụng

### 4.1 Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 4.2 Chạy Exercise 1: Sentiment Analysis với Pretrained Model
```bash
python src/01_sentiment_analysis.py
```
*Chương trình sẽ tự động tải mô hình SST-2, thực hiện suy luận qua pipeline, mổ xẻ Tokenizer (in bảng token -> ID), tính toán logits và xác suất Softmax thủ công, và kiểm thử trên 5 câu văn bản đa dạng ngữ cảnh.*

### 4.3 Chạy Exercise 2: Fine-tuning DistilBERT trên IMDb
```bash
python src/02_train_finetune.py
```
*Chương trình thực thi quy trình 7 bước chuẩn: nạp IMDb, tokenization kèm Dynamic Padding, đánh giá baseline trước huấn luyện, khởi tạo Trainer và fine-tune 2 epochs, lưu mô hình vào `outputs/final_model/` và xuất báo cáo vào `outputs/reports/`.*

### 4.4 Thử nghiệm dự đoán câu tùy ý với mô hình đã fine-tune
```bash
python src/03_predict_finetuned.py --text "The cinematography was breathtaking and acting was superb!"
```

---

## 5. Kết Quả Thực Nghiệm Thực Tế (Trích xuất từ `outputs/reports/final_metrics.json`)

### 5.1 So sánh Đối chứng Khoa học (Baseline vs. Fine-tuned)

| Chỉ số đánh giá | Untrained Baseline (Val) | Majority-Class (Test) | Fine-tuned DistilBERT (Test) | Mức tăng trưởng (+ Gain) |
|---|:---:|:---:|:---:|:---:|
| **Accuracy** | **44.00%** | **52.00%** | **74.50%** | **+22.50%** (vs Majority)<br>**+30.50%** (vs Untrained) |
| **Khoảng tin cậy 95% (Wilson)** | — | — | **[68.04%, 80.04%]** | — |
| **Macro F1-Score** | — | — | **74.27%** | — |
| **Macro Precision** | — | — | **74.76%** | — |
| **Macro Recall** | — | — | **74.24%** | — |
| **Test Loss** | — | — | **0.5565** | — |
| **Thời gian huấn luyện** | — | — | **~4.5 phút** | — |

> **Nhận xét khoa học:**
> 1. Trước khi huấn luyện, lớp phân loại (classification head) được khởi tạo ngẫu nhiên nên mô hình chỉ đạt 44.00% trên tập Validation (tập Test được bảo lưu nghiêm ngặt, chỉ đánh giá sau fine-tuning).
> 2. Mốc **Majority-Class Baseline** trên tập Test (104 NEG / 96 POS) đạt 52.00% (nếu luôn đoán NEGATIVE). Mô hình fine-tuned đạt **74.50%**, thực sự tăng trưởng **+22.50%** so với mốc Majority-Class.
> 3. Với cỡ mẫu kiểm thử 200, **Khoảng tin cậy Wilson 95%** là $[65.41\%, 77.76\%]$, thể hiện rõ ràng biên độ ngẫu nhiên của mẫu theo chuẩn thống kê.

### 5.2 Ma trận nhầm lẫn (Confusion Matrix trên 200 mẫu Test)

| Thực tế \ Dự đoán | Dự đoán: NEGATIVE (0) | Dự đoán: POSITIVE (1) |
|---|:---:|:---:|
| **Thực tế: NEGATIVE (104)** | **86** (True Negative) | **18** (False Positive) |
| **Thực tế: POSITIVE (96)** | **30** (False Negative) | **66** (True Positive) |

---

## 6. Phân Tích Lỗi (Error Analysis) & Tác Động Cắt Cụt Token

Trích xuất thực tế từ `outputs/reports/error_analysis.csv`:
1. **Cảm xúc phức tạp và Trộn lẫn (Mixed Sentiment):**
   - Ví dụ (Index 180, độ dài 105 tokens, **không bị cắt cụt**): *"not really sure what to make of this movie. very weird, very artsy... horrifically fascinating... nonetheless a worthwhile viewing for those interested in very dark movies."*
   - Nhãn thật là `POSITIVE` nhưng mô hình dự đoán `NEGATIVE` (tự tin 82.7%) do câu chứa dày đặc từ ngữ tiêu cực bề mặt (*weird, horrifically, disturbing, dark*) lấn át mệnh đề kết luận tích cực (*worthwhile viewing*).
2. **Cảm xúc ngoại cảnh tương phản (Contextual Confounding):**
   - Ví dụ (Index 88 & 185): *"i was having a horrid day but this movie grabbed me..."* hoặc *"I was upset to hear that it was canceled..."*. Các từ tiêu cực phản ánh hoàn cảnh cá nhân của người xem, chứ không chê chất lượng phim.
3. **Mỉa mai, Châm biếm & Tự trào (Sarcasm & Cult-movie Appreciation):**
   - Ví dụ (Ca 5 - Index 103, 77 tokens, **không bị cắt cụt**): *"Great cult flick for MST-3K types: Richard Boone is a mess -- bad hair, arthritis... about as good as a bad dino-flick can get... actually, that charging saber-toothed Styracosaurus was pretty cool..."*. Nhãn gốc `POSITIVE` nhưng mô hình dự đoán nhầm `NEGATIVE` (tự tin 75.75%) do bị chi phối bởi các từ ngữ tự trào dở/tệ.
4. **Thống kê định lượng hiện tượng cắt cụt token:**
   - 86.00% (172/200 mẫu) có độ dài token > 128 tokens.
   - Tỷ lệ lỗi ở nhóm bị cắt cụt là **23.84%** (41/172), gần tương đương nhóm không bị cắt là **25.00%** (7/28). Điều này minh chứng sai số chủ yếu xuất phát từ tính phức tạp trong ngữ nghĩa hơn là do cắt ngắn độ dài.

---

## 7. Đạo Đức Học Thuật & Cam Kết

- Toàn bộ kết quả thực nghiệm trong báo cáo được trích xuất trực tiếp từ các file số liệu thực tế tại `outputs/reports/`.
- Nhật ký sử dụng AI được kê khai minh bạch và đầy đủ trong [AI_Usage.md](AI_Usage.md).
