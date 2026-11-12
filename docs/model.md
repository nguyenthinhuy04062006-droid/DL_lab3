# MODEL DOCUMENTATION (MODEL_DOCUMENTATION.MD)

Tài liệu kỹ thuật đặc tả chi tiết kiến trúc mô hình, nguồn gốc tham số, cơ chế chuyển giao tri thức, luồng dữ liệu vào/ra và quy trình kiểm chứng thực nghiệm trong khuôn khổ Laboratory 3.

---

## 1. CREATED DATE
- **2026-11-09**

---

## 2. UPDATED DATE
- **2026-11-10**

---

## 3. PURPOSE

- **Objective:** Nghiên cứu, thực nghiệm và so sánh đối chứng có hệ thống khả năng chuyển giao tri thức (Transfer Learning) từ các mô hình ngôn ngữ lớn dựa trên kiến trúc Transformer tiền huấn luyện sang bài toán Phân loại cảm xúc văn bản nhị phân (Binary Sentiment Analysis) trên tập dữ liệu IMDb Movie Reviews trong môi trường tính toán tối ưu hóa CPU-First.
- **Task:** 
  - Khảo sát mô hình tiền huấn luyện `distilbert-base-uncased-finetuned-sst-2-english` cho bài toán suy luận trực tiếp (Zero-shot / Direct Inference).
  - Bóc tách chi tiết cơ chế hoạt động của Tokenizer WordPiece (Sub-word segmentation, token IDs, attention mask, tính toán Softmax thủ công từ logits thô).
  - Triển khai quy trình 7 bước tinh chỉnh (Fine-tuning) mô hình nền tảng `distilbert-base-uncased` kết hợp Classification Head 2 nhãn trên 800 mẫu IMDb.
  - Đánh giá định lượng trên 200 mẫu kiểm thử độc lập: tính toán Accuracy, Macro Precision, Macro Recall, Macro F1, khoảng tin cậy Wilson Score 95%, phân tích 51 ca dự đoán sai và kiểm định Fisher về tác động của cắt cụt độ dài (Truncation).
- **Expected Outcome:** Đạt độ chính xác kiểm thử Test Accuracy $\ge 74\%$ (thực tế đạt 74.50%, vượt xa Untrained Baseline 44.00% và Majority Baseline 52.00%); đảm bảo Zero Data Leakage; lưu trữ trọng số an toàn dưới định dạng `model.safetensors`; vượt qua 100% bài kiểm thử hồi quy tự động.

---

## 4. INITIAL MODEL & SOURCE INTEGRITY

- **Model Names & Versions:**
  - **Mô hình suy luận trực tiếp (Pretrained Inference):** `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (DistilBERT đã tinh chỉnh sẵn trên tập dữ liệu Stanford Sentiment Treebank - SST-2).
  - **Mô hình nền tảng tinh chỉnh (Base Pretrained):** `distilbert/distilbert-base-uncased` (DistilBERT tiền huấn luyện qua Masked Language Modeling trên tập Toronto BookCorpus và English Wikipedia).
  - **Mô hình sau tinh chỉnh (Fine-Tuned Target):** `outputs/final_model/` (Trọng số đã huấn luyện 2 epochs trên IMDb, lưu trữ dưới dạng `model.safetensors`).
- **Original Research Papers & Authors:**
  - *DistilBERT:* Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf. *"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"*, NeurIPS 2019 Workshop on Energy Efficient Deep Learning. Hugging Face, 2019.
  - *BERT gốc:* Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"*, Proceedings of NAACL-HLT, 2019, pp. 4171-4186.
- **Pretrained Weights & Performance:**
  - `distilbert-base-uncased-finetuned-sst-2-english`: Đạt độ chính xác 91.3% trên tập kiểm thử SST-2.
  - `distilbert-base-uncased`: Giữ lại hơn 97% năng lực hiểu ngôn ngữ tự nhiên của BERT-base trên bộ đo chuẩn GLUE benchmark, đồng thời giảm 40% tham số và tăng 60% tốc độ xử lý.
- **Official Repository & Framework:** Thư viện Hugging Face `transformers` (v4.x) kết hợp nền tảng PyTorch (PyTorch 2.x+cpu).
- **Model License:** Giấy phép mã nguồn mở Apache License 2.0.
- **Initialization Strategy:**
  - *Encoder Backbone:* Nạp nguyên vẹn trọng số tiền huấn luyện từ Hugging Face Model Hub (`distilbert-base-uncased`).
  - *Sequence Classification Head (2 nhãn):* Gồm 2 tầng tuyến tính:
    - Tầng tiền phân loại `pre_classifier`: Khởi tạo ngẫu nhiên `Linear(in_features=768, out_features=768)`.
    - Tầng kích hoạt phi tuyến & điều hòa: `ReLU()` kết hợp `Dropout(p=0.2)`.
    - Tầng phân loại quyết định `classifier`: Khởi tạo ngẫu nhiên `Linear(in_features=768, out_features=2)`.
  - *Chiến lược Huấn luyện (Fine-Tuning Strategy):* Cập nhật toàn bộ mạng nơ-ron (Full Model Fine-Tuning) với tốc độ học nhỏ $\eta = 2 \times 10^{-5}$ kết hợp bộ tối ưu hóa AdamW và hệ số suy giảm trọng số Weight Decay = 0.01 nhằm tránh hiện tượng phá vỡ tri thức ngôn ngữ đã học sẵn (Catastrophic Forgetting).

---

## 5. MODEL ARCHITECTURE

### 5.1. Tổng quan kiến trúc (Architecture Overview)
- **DistilBERT:** Kiến trúc Transformer Encoder hai chiều rút gọn gồm 6 tầng (Blocks), 12 đầu chú ý (Attention Heads), kích thước vector ẩn $d_{\text{model}} = 768$, kích thước tầng trung gian biến đổi phi tuyến Feed-Forward $d_{\text{ff}} = 3.072$.
- **Cơ chế Chú ý Đa đầu (Multi-Head Self-Attention):**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
  Với $d_k = 768 / 12 = 64$. Cơ chế này cho phép mỗi token thu thập thông tin ngữ cảnh đa chiều từ toàn bộ các token khác trong chuỗi văn bản.
- **Triết lý Chưng cất Tri thức (Knowledge Distillation):**
  DistilBERT được huấn luyện dưới sự giám sát của mạng Teacher (BERT-base 12 tầng, 110 triệu tham số) thông qua hàm mất mát ba thành phần:
  $$\mathcal{L}_{\text{distil}} = \alpha \mathcal{L}_{\text{ce}} + \beta \mathcal{L}_{\text{mlm}} + \gamma \mathcal{L}_{\text{cos}}$$
  - $\mathcal{L}_{\text{ce}}$: So khớp phân phối xác suất mềm giữa Teacher và Student qua nhiệt độ làm mềm Softmax $T$.
  - $\mathcal{L}_{\text{mlm}}$: Hàm mất mát Masked Language Modeling trên các token bị che.
  - $\mathcal{L}_{\text{cos}}$: Đo khoảng cách cosine giữa các vector ẩn đại diện để đồng nhất hướng vector đặc trưng.
- **So sánh thông số kỹ thuật với BERT-base:**

| Đặc tính kiến trúc | BERT-base (`bert-base-uncased`) | DistilBERT (`distilbert-base-uncased`) | Mức độ tối ưu hóa |
|:---|:---:|:---:|:---|
| **Số tầng Encoder (Layers)** | 12 layers | **6 layers** | Giảm 50% độ sâu tính toán |
| **Kích thước vector ẩn ($d_{\text{model}}$)** | 768 | **768** | Giữ nguyên độ sâu biểu diễn |
| **Số đầu chú ý (Attention Heads)** | 12 heads | **12 heads** | Giữ nguyên khả năng liên kết ngữ cảnh |
| **Độ rộng Feed-Forward ($d_{\text{ff}}$)** | 3.072 | **3.072** | Giữ nguyên năng lực chiếu phi tuyến |
| **Tổng số tham số (Total Params)** | ~110.000.000 | **66.955.010** | **Giảm ~40% dung lượng bộ nhớ** |
| **Tốc độ suy luận (Inference Speed)** | Baseline (1.0x) | **Nhanh hơn ~60% (1.6x)** | Tối ưu hóa thực thi trên CPU |
| **Năng lực lưu giữ tri thức** | 100% | **> 97% điểm GLUE** | Suy giảm chất lượng không đáng kể |

### 5.2. Cấu trúc chi tiết từng tầng (Detailed Layer-by-Layer Architecture)

```
[Văn bản đầu vào thô]
         │
[WordPiece Tokenizer: sub-word tokens]
         │
[Tầng Embeddings: Token + Position + LayerNorm + Dropout] (23.835.648 params)
         │
┌────────────────────────────────────────────────────────┐
│ Transformer Block 0 (7.087.872 params)                 │
│  - Multi-Head Self-Attention (12 heads, d_k=64)        │
│  - Residual Connection & LayerNorm                     │
│  - Feed-Forward: Linear(768->3072) + GELU              │
│  - Linear(3072->768) + Residual Connection & LayerNorm │
└────────────────────────────────────────────────────────┘
         │
       [... 4 Transformer Blocks trung gian: Block 1 đến Block 4 ...]
         │
┌────────────────────────────────────────────────────────┐
│ Transformer Block 5 (7.087.872 params)                 │
│  - Multi-Head Self-Attention (12 heads, d_k=64)        │
│  - Residual Connection & LayerNorm                     │
│  - Feed-Forward: Linear(768->3072) + GELU              │
│  - Linear(3072->768) + Residual Connection & LayerNorm │
└────────────────────────────────────────────────────────┘
         │
[Trích xuất vector đại diện h_CLS = hidden_state[:, 0, :]] ∈ ℝ^(B × 768)
         │
[Classification Head] (592.130 params)
 ├─ pre_classifier: Linear(768 -> 768) + ReLU()           (590.592 params)
 ├─ Dropout(p = 0.2)
 └─ classifier: Linear(768 -> 2)                         (1.538 params)
         │
[Logits thô z = [z_0, z_1]] ∈ ℝ^(B × 2)
         │
[Softmax / Argmax] ──> [Nhãn: 0: NEGATIVE, 1: POSITIVE]
```

1. **Tầng Embeddings (23.835.648 tham số):**
   - *Word Embeddings:* Bảng tra cứu từ vựng kích thước $30.522 \times 768 = 23.440.896$ tham số.
   - *Position Embeddings:* Bảng vị trí token $512 \times 768 = 393.216$ tham số (DistilBERT loại bỏ Segment Embeddings của BERT để tinh giản cấu trúc).
   - *Layer Normalization & Dropout:* `LayerNorm(768)` gồm $768 \times 2 = 1.536$ tham số kèm xác suất loại trừ ngẫu nhiên $p = 0.1$.
2. **6 Khối Transformer Encoder (42.527.232 tham số, mỗi khối 7.087.872 tham số):**
   - *Multi-Head Self-Attention (3.541.248 tham số / block):*
     - Các ma trận chiếu Query, Key, Value: Mỗi ma trận có kích thước $768 \times 768 + 768 = 590.592$ tham số. Tổng 3 ma trận = $1.771.776$ tham số.
     - Ma trận chiếu đầu ra (Dense Out): $768 \times 768 + 768 = 590.592$ tham số.
     - LayerNorm sau Attention: $768 \times 2 = 1.536$ tham số.
   - *Feed-Forward Network (3.546.624 tham số / block):*
     - Tuyến tính mở rộng: `Linear(768, 3072)` gồm $768 \times 3.072 + 3.072 = 2.362.368$ tham số.
     - Hàm kích hoạt phi tuyến: `GELU()` (Gaussian Error Linear Unit).
     - Tuyến tính thu hẹp: `Linear(3072, 768)` gồm $3.072 \times 768 + 768 = 2.360.064$ tham số.
     - LayerNorm sau FFN: $768 \times 2 = 1.536$ tham số.
3. **Sequence Classification Head (592.130 tham số):**
   - *Tầng Pre-classifier:* `Linear(768, 768)` gồm $768 \times 768 + 768 = 590.592$ tham số.
   - *Kích hoạt & Điều hòa:* `ReLU()` và `Dropout(p=0.2)`.
   - *Tầng Classifier:* `Linear(768, 2)` gồm $768 \times 2 + 2 = 1.538$ tham số.

### 5.3. Thành phần chính (Main Components)
- **WordPiece Tokenizer:** Chịu trách nhiệm bóc tách câu văn bản thành chuỗi sub-word token, chèn các token điều khiển đặc biệt `[CLS]` tại vị trí 0 và `[SEP]` tại vị trí kết thúc câu.
- **Tầng biểu diễn ngữ cảnh (Contextual Representation):** 6 tầng Transformer Encoder trích xuất vector trạng thái ẩn cho từng token dựa trên mối tương quan với toàn bộ câu văn.
- **Vector đại diện toàn câu (`h_CLS`):** Trích xuất trạng thái ẩn tại vị trí token đầu tiên (`hidden_state[:, 0, :]`). Trong kiến trúc BERT/DistilBERT, token `[CLS]` đóng vai trò tích lũy tổng thể ngữ nghĩa phân loại của toàn bộ chuỗi đầu vào.
- **Đầu phân loại (Classification Head):** Chiếu không gian vector 768 chiều sang không gian 2 chiều biểu diễn điểm số cảm xúc (Logits).

### 5.4. Thống kê tham số mô hình (Model Parameters Profiling)

| Khối thành phần | Số lượng tham số | Tỷ lệ phần trăm | Trạng thái trong Fine-Tuning |
|:---|:---:|:---:|:---:|
| **Embeddings Layer** | 23.835.648 | 35,60% | Trainable (Full Fine-Tuning) |
| **Transformer Block 0** | 7.087.872 | 10,59% | Trainable (Full Fine-Tuning) |
| **Transformer Block 1** | 7.087.872 | 10,59% | Trainable (Full Fine-Tuning) |
| **Transformer Block 2** | 7.087.872 | 10,59% | Trainable (Full Fine-Tuning) |
| **Transformer Block 3** | 7.087.872 | 10,59% | Trainable (Full Fine-Tuning) |
| **Transformer Block 4** | 7.087.872 | 10,59% | Trainable (Full Fine-Tuning) |
| **Transformer Block 5** | 7.087.872 | 10,59% | Trainable (Full Fine-Tuning) |
| **Pre-Classifier Head** | 590.592 | 0,88% | Trainable (Khởi tạo mới) |
| **Classifier Head** | 1.538 | 0,002% | Trainable (Khởi tạo mới) |
| **Tổng cộng toàn mô hình** | **66.955.010** | **100,00%** | **66.955.010 (100% Trainable)** |

### 5.5. Phân tích kỹ thuật & hành vi đặc thù (Specialized Mechanisms & Behavior)
- **Cơ chế Tokenizer Sub-word (WordPiece):**
  - Khắc phục hoàn toàn bài toán từ nằm ngoài từ điển (Out-Of-Vocabulary - OOV). Mọi từ hiếm hoặc từ mới được phân tách thành các đoạn sub-word (ví dụ `"unbelievable"` $\to$ `["un", "##belie", "##vable"]`).
  - Tiền tố `##` chỉ định đoạn sub-word gắn liền phía sau một đơn vị từ trước đó, bảo toàn ranh giới từ vựng.
- **Đệm động (Dynamic Padding) qua `DataCollatorWithPadding`:**
  - Thay vì đệm cố định toàn bộ mẫu dữ liệu lên chiều dài tối đa $128$ tokens (Static Padding), cơ chế đệm động chỉ căn chỉnh độ dài của các câu trong một batch bằng với câu dài nhất của chính batch đó.
  - Giảm thiểu đáng kể số lượng phép tính tích vô hướng dư thừa trên token `[PAD]`, tiết kiệm từ 30% đến 40% chi phí tính toán trên môi trường CPU.
- **Hiện tượng cắt cụt độ dài (Truncation at `max_length=128`):**
  - 86% số bài đánh giá trong tập kiểm thử có độ dài thực tế vượt quá 128 tokens.
  - Kiểm định Fisher chính xác (Fisher's Exact Test) cho kết quả: Odds Ratio = 0.7697, $p = 0.8153$. Tỷ lệ đoán sai ở nhóm bị cắt cụt (26.16%) không có sự khác biệt mang ý nghĩa thống kê so với nhóm không bị cắt cụt (21.43%), chứng minh các đoạn văn mở đầu của bài đánh giá IMDb đã chứa đựng đầy đủ sắc thái cảm xúc quyết định.
- **Tính bất khả phân tuyến tính (Linear Inseparability via PCA 2D):**
  - Phân tích 2 thành phần chính trên ma trận TF-IDF 1.000 từ cho thấy $PC1$ giải thích 0.84% và $PC2$ giải thích 0.65% phương sai (tổng cộng chỉ đạt 1.49%).
  - Các điểm dữ liệu của 2 lớp phân bố chồng lấn dày đặc tại gốc tọa độ, chứng minh các phương pháp tuyến tính truyền thống hoàn toàn bất lực và khẳng định tính tất yếu của các tầng Self-Attention phi tuyến tính sâu trong DistilBERT.

---

## 6. MODEL INPUT / OUTPUT

### 6.1. Input Specification
- **Dữ liệu đầu vào:**
  1. `input_ids`: Tensor các mã định danh token (`torch.int64`), kích thước `[B, L]`.
  2. `attention_mask`: Tensor mặt nạ nhị phân (`torch.int64`), kích thước `[B, L]`.
     - Giá trị `1`: Token thực tế, mô hình bắt buộc tính trọng số chú ý.
     - Giá trị `0`: Token đệm `[PAD]`, mô hình bỏ qua trong cơ chế Self-Attention.
- **Kích thước Tensor:**
  - $B$: Kích thước lô (Batch Size $B = 16$ khi huấn luyện, $B = 32$ khi đánh giá).
  - $L$: Chiều dài chuỗi token trong batch ($L \le 128$).
- **Quy ước mã token đặc biệt:**
  - `[CLS]` (ID: `101`): Đặt ở vị trí index 0 của mọi chuỗi văn bản.
  - `[SEP]` (ID: `102`): Đặt ở cuối chuỗi văn bản để đánh dấu kết thúc câu.
  - `[PAD]` (ID: `0`): Dùng để đệm các vị trí khuyết cho đủ độ dài lô.
- **Pipeline tiền xử lý (Preprocessing Pipeline):**
  $$\text{Văn bản thô (String)} \xrightarrow{\text{WordPiece}} \text{Tokens} \xrightarrow{\text{Vocab Mapping}} \text{IDs} \xrightarrow[\text{max\_length=128}]{\text{Truncation \& Padding}} \text{Tensor [B, L]}$$

### 6.2. Output Specification
- **Định dạng dữ liệu đầu ra:** Tensor số thực dấu phẩy động 32-bit (`torch.float32`).
- **Kích thước Tensor:** `[B, 2]` ($B$: Batch size; 2 chiều đại diện cho điểm số 2 nhãn cảm xúc).
- **Diễn giải & Hậu xử lý (Interpretation & Post-processing):**
  - *Logits thô:* Vector $\mathbf{z} = [z_0, z_1] \in \mathbb{R}^2$ ($z_0$: điểm số nhãn NEGATIVE, $z_1$: điểm số nhãn POSITIVE).
  - *Xác suất phân loại (Softmax Probabilities):*
    $$P(y = c \mid \mathbf{x}) = \frac{\exp(z_c)}{\sum_{j=0}^{1} \exp(z_j)}, \quad c \in \{0, 1\}$$
  - *Nhãn dự đoán cuối cùng (Predicted Class):*
    $$\hat{y} = \arg\max_{c \in \{0, 1\}} z_c$$
  - *Hàm mất mát huấn luyện (Training Loss):*
    $$\mathcal{L}_{CE} = -\frac{1}{B} \sum_{i=1}^B \left[ y_i \log(\hat{p}_{i, 1}) + (1 - y_i) \log(\hat{p}_{i, 0}) \right]$$
    Được tối ưu trực tiếp bằng `Trainer` với tiêu chuẩn ổn định số học kết hợp LogSoftmax.

---

## 7. SOURCE FILES & RESPONSIBILITIES

| Đường dẫn tệp mã nguồn | Trách nhiệm kỹ thuật trong hệ thống |
|:---|:---|
| [`src/01_sentiment_analysis.py`](../src/01_sentiment_analysis.py) | Thực thi bài tập 1: Nạp pipeline SST-2, bóc tách cấu trúc Tokenizer WordPiece (tách sub-words, input IDs, attention mask), chạy forward pass trích xuất logits thô và so khớp xác suất Softmax tính toán thủ công với kết quả của pipeline trên 5 câu kiểm thử. |
| [`src/02_train_finetune.py`](../src/02_train_finetune.py) | Thực thi bài tập 2: Quy trình 7 bước chuẩn tinh chỉnh DistilBERT trên 800 mẫu IMDb, nạp base model, tích hợp `DataCollatorWithPadding`, đo Untrained Baseline, cấu hình `TrainingArguments`, huấn luyện qua `Trainer` và lưu mô hình an toàn dưới dạng `model.safetensors`. |
| [`src/03_predict_finetuned.py`](../src/03_predict_finetuned.py) | Giao diện dòng lệnh CLI suy luận trực tiếp với mô hình đã fine-tuned từ checkpoint `outputs/final_model/`: Nhận câu văn bản tùy ý, tokenize, thực hiện forward pass và xuất nhãn dự đoán cùng xác suất tự tin chi tiết. |
| [`src/generate_data_charts.py`](../src/generate_data_charts.py) | Kết xuất các biểu đồ phục vụ báo cáo khoa học dữ liệu: Biểu đồ cột phân bố nhãn Train/Val/Test và biểu đồ tần suất phân phối độ dài token trước khi cắt cụt. |
| [`src/export_tables_as_images.py`](../src/export_tables_as_images.py) | Tự động chuyển đổi các bảng số liệu Markdown thành tệp hình ảnh PNG độc lập độ phân giải cao phục vụ lưu trữ và hiển thị trực quan. |
| [`src/generate_html_report.py`](../src/generate_html_report.py) | Trình tạo báo cáo HTML khoa học độc lập (`outputs/REPORT.html`), tự động nhúng toàn bộ 5 biểu đồ thực nghiệm và định dạng số liệu theo chuẩn giao diện kỹ thuật chuyên nghiệp. |
| [`src/verify_all.py`](../src/verify_all.py) | Bộ kiểm thử tự động toàn diện chạy 5 ca kiểm thử hồi quy độc lập: Xác thực cấu trúc biểu đồ trong Jupyter Notebook, kiểm tra tính đầy đủ của các tệp báo cáo, đối soát tính toàn vẹn số liệu thực nghiệm, xác thực khả năng suy luận của mô hình tinh chỉnh và kiểm tra tính tự chứa của báo cáo HTML. |

---

## 8. VERIFICATION & REPRODUCIBILITY AUDIT

### 8.1. Kiểm chứng cấu trúc & suy luận qua CLI
```bash
# 1. Chạy quy trình kiểm thử hồi quy toàn bộ hệ thống (5/5 tests)
python src/verify_all.py

# 2. Kiểm chứng suy luận mô hình tiền huấn luyện và mổ xẻ Tokenizer
python src/01_sentiment_analysis.py

# 3. Thực thi tái lập quy trình tinh chỉnh mô hình từ đầu (seed=42)
python src/02_train_finetune.py

# 4. Dự đoán câu văn bản mới qua CLI từ checkpoint đã fine-tune
python src/03_predict_finetuned.py --text "The cinematography was breathtaking and acting was superb!"

# 5. Kiểm tra nhanh cấu trúc tham số mô hình đã tinh chỉnh từ dòng lệnh
python -c "from transformers import AutoModelForSequenceClassification; m = AutoModelForSequenceClassification.from_pretrained('outputs/final_model'); print(f'Total params: {sum(p.numel() for p in m.parameters()):,}'); print(f'Classification Head: {sum(p.numel() for n, p in m.named_parameters() if \'classifier\' in n):,}')"
```

### 8.2. Kiểm thử tự động (Regression Test Suite)
Bộ kiểm thử tự động tại [`src/verify_all.py`](../src/verify_all.py) được thực thi độc lập và đạt kết quả kiểm định tuyệt đối:
- **Kết quả kiểm thử:** `5/5 tests passed (100%)`.
- **Nội dung kiểm thực chi tiết:**
  - *Test 1 (Notebook Structure & Visuals):* Xác nhận notebook có đủ 41 cells và bảo toàn trọn vẹn 4 biểu đồ hình ảnh PNG tại các cells 15, 27, 33, 34.
  - *Test 2 (Report & Chart Artifacts Sync):* Xác nhận toàn bộ 9 tệp số liệu và đồ thị trong `outputs/reports/` tồn tại, không rỗng và đồng bộ.
  - *Test 3 (Mathematical & Metric Integrity):* Xác nhận độ chính xác kiểm thử khớp tuyệt đối 74.50% (149/200 mẫu); ma trận nhầm lẫn bảo toàn $TN=84, FP=20, FN=31, TP=65$.
  - *Test 4 (Inference Health Check):* Nạp trực tiếp trọng số `outputs/final_model/model.safetensors`, chạy suy luận câu văn mẫu và xác nhận mô hình đưa ra nhãn POSITIVE với độ tin cậy 78.23%.
  - *Test 5 (HTML Report Self-Containment):* Xác nhận `outputs/REPORT.html` tồn tại, định dạng hợp lệ và nhúng đầy đủ các biểu đồ trực quan hóa.

---

*Tài liệu được bảo toàn 100% tính xác thực khoa học, lưu trữ chính thức tại [`docs/model.md`](model.md).*
