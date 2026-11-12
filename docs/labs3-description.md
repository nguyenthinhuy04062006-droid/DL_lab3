# BÁO CÁO KHOA HỌC: PHÂN LOẠI CẢM XÚC VĂN BẢN VỚI HỆ SINH THÁI HUGGING FACE VÀ DISTILBERT

**Dự án:** Laboratory 3 — Deep Learning Scientific Report  
**Tác giả:** Nhóm 1  
**Mã nguồn:** [`src/`](../src/) | **Notebook:** [`notebooks/Practice3_HuggingFace_Final.ipynb`](../notebooks/Practice3_HuggingFace_Final.ipynb)  
**Tiêu chuẩn:** Academic – Concise – Structured – Reproducible – Evidence-based

---

## 1. ABSTRACT

Nghiên cứu này triển khai bài toán Phân loại cảm xúc văn bản nhị phân (Binary Sentiment Analysis) trên bộ dữ liệu IMDb Movie Reviews (50.000 bài đánh giá phim) bằng hệ sinh thái Hugging Face. Hai chiến lược suy luận được đánh giá đối chứng: (1) Suy luận trực tiếp (Inference Only) với mô hình `distilbert-base-uncased-finetuned-sst-2-english` đã tinh chỉnh sẵn trên SST-2 và (2) Tinh chỉnh (Fine-tuning) mô hình nền tảng `distilbert-base-uncased` theo quy trình 7 bước chuẩn trên 800 mẫu IMDb. Kết quả thực nghiệm cho thấy Fine-tuning đạt Test Accuracy = 74.50% (149/200 mẫu), Macro F1 = 74.27% với Wilson 95% CI = [68.04%, 80.04%] — vượt hoàn toàn Majority Class Baseline (52.00%) và Untrained Model Baseline (44.00%). Phân tích PCA chứng minh dữ liệu không thể phân tách tuyến tính trong không gian TF-IDF (tổng phương sai PC1+PC2 = 1.49%), khẳng định tính tất yếu của cơ chế Self-Attention sâu trong kiến trúc Transformer.

---

## 2. INTRODUCTION

### 2.1. Bối cảnh nghiên cứu
Phân loại cảm xúc (Sentiment Analysis) là bài toán nền tảng trong Xử lý ngôn ngữ tự nhiên (NLP), đòi hỏi mô hình phải hiểu ngữ cảnh, cấu trúc phủ định, châm biếm và mối quan hệ phức tạp giữa các từ trong câu. Các mô hình tuyến tính dựa trên tần suất từ (Bag-of-Words, TF-IDF) không đáp ứng được yêu cầu này.

### 2.2. Mục tiêu nghiên cứu
1. Làm chủ hệ sinh thái Hugging Face: `pipeline`, `AutoTokenizer`, `AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`.
2. Hiểu rõ cơ chế Tokenizer WordPiece: tách sub-word, ánh xạ token ID, attention mask, chuyển đổi logits sang xác suất Softmax.
3. Phân biệt rõ hai chiến lược: Inference Only (dùng model có sẵn) và Fine-tuning (thích ứng domain mới).
4. Đánh giá định lượng toàn diện: Accuracy, Macro F1, Wilson CI, phân tích lỗi chi tiết theo nguyên nhân.

### 2.3. Vấn đề giải quyết
Ánh xạ văn bản thô tiếng Anh chiều dài biến thiên sang nhãn nhị phân (NEGATIVE/POSITIVE) thông qua cơ chế mã hóa ngữ nghĩa sâu (Contextual Embeddings) của DistilBERT, khắc phục hiện tượng chồng lấn đa tạp (Manifold Entanglement) mà các mô hình tuyến tính không giải quyết được.

---

## 3. RELATED WORK

- **Mô hình BERT gốc:** Devlin et al. (2019) giới thiệu kiến trúc Transformer Encoder hai chiều (Bidirectional Encoder Representations from Transformers), tiền huấn luyện trên corpus 3.3 tỷ từ bằng Masked Language Modeling và Next Sentence Prediction.
- **DistilBERT:** Sanh et al. (2019) chưng cất BERT-base xuống mô hình nhỏ hơn 40%, nhanh hơn 60%, giữ lại 97% năng lực của mô hình gốc — phù hợp cho môi trường tài nguyên hạn chế.
- **IMDb Benchmark:** Maas et al. (2011) công bố bộ dữ liệu chuẩn mực 50.000 bài đánh giá phim cân bằng lớp, trở thành benchmark tiêu chuẩn cho bài toán Sentiment Analysis trong nghiên cứu NLP.
- **Transfer Learning trong NLP:** Howard & Ruder (2018) chứng minh chiến lược tinh chỉnh theo tầng (Discriminative Fine-tuning) kết hợp lịch trình tốc độ học một chiều giúp thích ứng mô hình ngôn ngữ lớn sang domain mới hiệu quả với lượng dữ liệu nhỏ.

---

## 4. THEORETICAL BACKGROUND

### 4.1. Cơ chế Self-Attention của Transformer
Mỗi đầu Attention tính điểm tương quan giữa các token thông qua ba ma trận chiếu Query, Key, Value:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Với $d_k$ là chiều của không gian Key, hạng tử $\sqrt{d_k}$ chuẩn hóa độ lớn tích vô hướng ngăn hiện tượng suy biến gradient. DistilBERT triển khai 6 tầng Transformer với 12 đầu Attention và $d_k = 64$.

### 4.2. Tokenizer WordPiece
Thuật toán WordPiece phân tách từ hiếm hoặc từ mới thành các đơn vị sub-word có nghĩa (prefix `##`) để đảm bảo 100% từ vựng được biểu diễn trong không gian embedding cố định 30.522 token. Ví dụ: `"playing"` → `["play", "##ing"]`.

### 4.3. Hàm mất mát và Tối ưu hóa
- **Hàm mất mát Cross-Entropy nhị phân:**
$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i) \right]$$
- **Tối ưu hóa AdamW (Adam + Weight Decay):**
$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$
Với $\eta = 2 \times 10^{-5}$, $\lambda = 0.01$ (Weight Decay), $\beta_1 = 0.9$, $\beta_2 = 0.999$.

### 4.4. Wilson Score Confidence Interval
Khoảng tin cậy Wilson 95% cho tỷ lệ phân loại đúng:
$$CI = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
Với $z = 1.96$ (z-score 95%), $\hat{p} = 0.745$ (Test Accuracy), $n = 200$ (cỡ mẫu Test).

---

## 5. METHODOLOGY

### 5.1. Dữ liệu thực nghiệm (Dataset IMDb)
- **Quy mô gốc:** 50.000 bài đánh giá phim tiếng Anh, cân bằng tuyệt đối 50/50 (25.000 NEGATIVE + 25.000 POSITIVE).
- **Phạm vi thực nghiệm Practice 3:** Trích xuất 1.200 mẫu với `seed=42`:
  - *Tập Train (Huấn luyện):* 800 mẫu (408 NEG / 392 POS).
  - *Tập Validation (Kiểm chuẩn):* 200 mẫu (104 NEG / 96 POS).
  - *Tập Test (Kiểm thử):* 200 mẫu (104 NEG / 96 POS) — độc lập hoàn toàn.
- **Lưu ý kỹ thuật:** IMDb gốc không xáo trộn (12.500 NEGATIVE đầu tiên), bắt buộc `shuffle(seed=42)` trước `select()` để tránh thiên lệch lớp.

### 5.2. Pipeline tiền xử lý dữ liệu
1. **Tokenization:** `AutoTokenizer.from_pretrained("distilbert-base-uncased")` với `truncation=True`, `max_length=128`.
2. **Đệm động (Dynamic Padding):** `DataCollatorWithPadding` đệm theo độ dài tối đa trong từng batch, giảm 30-40% chi phí tính toán so với đệm cố định toàn bộ.
3. **Chuẩn hóa nhãn:** Nhãn gốc IMDb: 0 = NEGATIVE, 1 = POSITIVE.

### 5.3. Kiến trúc mô hình
Công cụ kiểm tra: [`src/01_sentiment_analysis.py`](../src/01_sentiment_analysis.py).

| Thành phần | Mô hình Inference | Mô hình Fine-tuned |
|:---|:---|:---|
| **Base Model** | `distilbert-base-uncased-finetuned-sst-2-english` | `distilbert-base-uncased` |
| **Encoder Layers** | 6 tầng Transformer (Frozen) | 6 tầng Transformer (Frozen) |
| **Classification Head** | `Linear(768, 2)` — tinh chỉnh sẵn SST-2 | `Linear(768, 2)` — khởi tạo ngẫu nhiên |
| **Tham số trainable** | 0 (Inference Only) | 592.130 (Head + tầng cuối) |
| **Domain gốc** | SST-2 (câu đơn, ngắn) | IMDb (đoạn văn dài, phức tạp) |

---

## 6. EXPERIMENTAL SETUP

### 6.1. Môi trường thực thi
- **Hệ điều hành:** Microsoft Windows 11 (64-bit).
- **Môi trường thực thi:** Python 3.10+ (64-bit), PyTorch 2.x+cpu, Transformers 4.x, Datasets 2.x.
- **Thiết bị tính toán:** CPU (tự động phát hiện CUDA → MPS → CPU theo thứ tự ưu tiên).
- **Bộ nhớ RAM:** 16 GB DDR4.

### 6.2. Cấu hình siêu tham số (Hyperparameters)
- **Số epoch:** 2 epochs (đủ hội tụ trên CPU với 800 mẫu).
- **Kích thước lô (Batch Size):** 16 (train), 32 (evaluation).
- **Bộ tối ưu hóa:** AdamW với Weight Decay = 0.01.
- **Tốc độ học (Learning Rate):** $\eta = 2 \times 10^{-5}$ (chuẩn tinh chỉnh Transformer).
- **Checkpoint:** Lưu tự động trạng thái tốt nhất theo `eval_loss` vào `outputs/final_model/`.
- **Hạt giống ngẫu nhiên:** `seed=42` xuyên suốt toàn pipeline.

### 6.3. Tiêu chí đánh giá (Evaluation Metrics)
- Accuracy, Macro Precision, Recall, F1-Score trên tập Test (n=200).
- Wilson Score 95% Confidence Interval cho Test Accuracy.
- Ma trận nhầm lẫn (Confusion Matrix): TN, FP, FN, TP.
- Phân tích lỗi chi tiết: phân loại nguyên nhân sai theo 4 nhóm (Sarcasm, Negation, Truncation, Mixed Sentiment).
- Fisher's Exact Test: kiểm định mối liên hệ giữa độ dài token và tỷ lệ sai.

---

## 7. RESULTS & EVALUATION

### 7.1. Bảng số liệu thực nghiệm tổng hợp
Số liệu đọc trực tiếp từ [`outputs/reports/final_metrics.json`](../outputs/reports/final_metrics.json):

| Metric | Giá trị |
|:---|:---:|
| **Test Accuracy** | **74.50%** (149/200) |
| **Macro F1-Score** | **74.27%** |
| **Macro Precision** | 74.60% |
| **Macro Recall** | 74.04% |
| **Wilson 95% CI** | [68.04%, 80.04%] |
| **True Negative (TN)** | 84 |
| **False Positive (FP)** | 20 |
| **False Negative (FN)** | 31 |
| **True Positive (TP)** | 65 |
| **Tổng lỗi** | 51 ca (25.50%) |

> [!IMPORTANT]
> **TỔNG KẾT BASELINE SO SANH:**
> - **Majority Class Baseline:** 52.00% (luôn đoán NEGATIVE — lớp đa số).
> - **Untrained Model Baseline:** 44.00% (đo trước khi huấn luyện trên Val).
> - **Fine-tuned DistilBERT:** 74.50% — vượt Majority Baseline +22.50%, vượt Untrained +30.50%.
> - **Cơ chế chọn Checkpoint:** Best Checkpoint lưu tự động theo `eval_loss` thấp nhất để tránh thiên kiến và chống Overfitting.

### 7.2. Biểu đồ đối chứng hiệu năng thực nghiệm
Tất cả biểu đồ lưu tại [`outputs/reports/`](../outputs/reports/):

1. **Đường cong huấn luyện (Training Curves):** [`outputs/reports/training_curves.png`](../outputs/reports/training_curves.png)
   - Train Loss giảm từ ~0.68 (epoch 1) xuống ~0.52 (epoch 2).
   - Val Loss giảm đồng bộ, không có dấu hiệu Overfitting trong 2 epochs.
2. **Ma trận nhầm lẫn:** [`outputs/reports/confusion_matrix.png`](../outputs/reports/confusion_matrix.png)
   - FN (31) > FP (20): mô hình có xu hướng đoán POSITIVE khi thực tế là NEGATIVE.
3. **So sánh đa chỉ số:** [`outputs/reports/test_comparison_and_per_class.png`](../outputs/reports/test_comparison_and_per_class.png)
   - Lớp POSITIVE (F1=72.47%) kém hơn NEGATIVE (F1=76.07%) do mẫu Train POS ít hơn 16 mẫu so với NEG.

---

## 8. COMPARATIVE ANALYSIS

### 8.1. So sánh Inference Only vs. Fine-tuning
- **Inference Only (SST-2):** Mô hình pretrained trên SST-2 (câu ngắn) gặp khó khăn với các bài đánh giá dài, phức tạp của IMDb. Không có số liệu Test chính thức vì chiến lược này không được đánh giá trên tập IMDb Test.
- **Fine-tuning (IMDb):** Thích ứng Classification Head với phân phối ngôn ngữ IMDb, đạt Test Accuracy 74.50% — cải thiện đáng kể so với Majority Baseline (52.00%).

### 8.2. Phân tích ảnh hưởng của Truncation
- **86% mẫu Test** vượt ngưỡng `max_length=128` và bị cắt ngắn.
- **Fisher's Exact Test:** OR = 0.7697, p = 0.8153 — không có mối liên hệ thống kê có ý nghĩa giữa độ dài token và tỷ lệ sai.
- **Tỷ lệ sai:** Mẫu không bị cắt = 21.43% (6/28), mẫu bị cắt = 26.16% (45/172) — chênh lệch nhỏ, không đủ bằng chứng kết luận truncation là nguyên nhân chính gây lỗi.

### 8.3. Phân tích lỗi theo nguyên nhân
Dựa trên [`outputs/reports/error_analysis.csv`](../outputs/reports/error_analysis.csv) (51 ca lỗi):
- **Sarcasm / Châm biếm:** Mô hình đọc văn bản bề mặt tích cực nhưng ngữ nghĩa thực tế tiêu cực.
- **Negation / Phủ định phức tạp:** Chuỗi phủ định đa tầng ("not quite as bad as") làm đảo ngữ nghĩa vượt khả năng mã hóa trong 128 tokens.
- **Mixed Sentiment / Cảm xúc hỗn hợp:** Bài đánh giá vừa khen vừa chê, nhãn IMDb thiên về cảm xúc tổng thể.
- **Truncation:** Phần kết luận cảm xúc cuối bài bị cắt mất trước khi mô hình đọc tới.

---

## 9. DISCUSSION & LIMITATIONS

### 9.1. Phân tích lỗi phân loại đặc trưng
Dựa trên ma trận nhầm lẫn:
- **FN > FP (31 > 20):** Mô hình có thiên kiến nhẹ về hướng POSITIVE, có thể do phân phối từ vựng tích cực trong corpus tiền huấn luyện của DistilBERT (Wikipedia + BookCorpus).
- **Lớp NEGATIVE dễ nhận hơn:** F1 = 76.07% so với POSITIVE F1 = 72.47%, chênh lệch 3.6% phản ánh sự mất cân bằng nhẹ trong tập Train (408 NEG vs. 392 POS).

### 9.2. Hạn chế kỹ thuật của nghiên cứu
1. **Quy mô tập nhỏ:** Chỉ dùng 800/25.000 mẫu Train IMDb để huấn luyện nhanh trên CPU (391.42s ~6.52 phút). Quy mô nhỏ tạo ra dao động Accuracy theo hạt giống ngẫu nhiên.
2. **Truncation thông tin:** `max_length=128` cắt mất phần kết luận của 86% bài đánh giá dài, có thể bỏ sót câu tóm tắt cảm xúc cuối bài.
3. **Số epoch ít:** 2 epochs chưa đủ để khảo sát đường cong hội tụ dài hạn và phát hiện điểm bão hòa.
4. **Chưa so sánh kiến trúc:** Chỉ dùng DistilBERT, chưa đối chiếu với BERT-base, RoBERTa hay mô hình lớn hơn trên cùng tập dữ liệu.

---

## 10. CONCLUSION & FUTURE WORK

### 10.1. Kết luận
1. Hệ sinh thái Hugging Face cung cấp pipeline đầy đủ từ tokenization đến fine-tuning và evaluation, giảm đáng kể chi phí triển khai mô hình ngôn ngữ lớn.
2. Fine-tuning DistilBERT đạt Test Accuracy = 74.50% (vượt Baseline +22.50%) chỉ với 800 mẫu và 2 epochs trên CPU, chứng minh sức mạnh chuyển giao tri thức trong NLP.
3. Phân tích PCA xác nhận tính phi tuyến tính tuyệt đối của dữ liệu IMDb — cơ chế Self-Attention của Transformer là giải pháp tất yếu.
4. Phân tích lỗi chỉ ra sarcasm, negation và truncation là các nguồn lỗi chính, định hướng cải tiến trong tương lai.

### 10.2. Định hướng phát triển
1. Mở rộng tập huấn luyện lên toàn bộ 25.000 mẫu IMDb Train khi có GPU để khảo sát ngưỡng bão hòa hiệu năng.
2. Tăng `max_length` lên 256 hoặc 512 để giảm thất thoát thông tin do truncation.
3. Thử nghiệm các chiến lược tối ưu hóa tốc độ học: `warmup_ratio`, `CosineDecay`, Differential Learning Rates theo tầng.
4. Đánh giá đối chứng với các kiến trúc lớn hơn: `bert-base-uncased`, `roberta-base`, `deberta-v3-small`.

---

## 11. REFERENCES

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019*, pp. 4171-4186.
2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. In *Proceedings of NeurIPS 2019 Workshop on Energy Efficient Deep Learning*.
3. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. In *Proceedings of ACL 2011*, pp. 142-150.
4. Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. In *Proceedings of ACL 2018*, pp. 328-339.
5. Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. In *Proceedings of EMNLP 2020: System Demonstrations*, pp. 38-45.

---

## 12. APPENDIX

### 12.1. Cấu trúc xuất kết quả chuẩn (Output Rule Compliance)
Mọi kết quả thực nghiệm, biểu đồ và số liệu được phân tách hoàn toàn khỏi Jupyter Notebook theo sơ đồ tổ chức độc lập:

```text
DL_lab3/
├── .gitignore                              # Quy tắc loại trừ tệp nhị phân lớn (*.safetensors, *.pt), checkpoints và cache
├── README.md                               # Tài liệu tổng quan dự án, hướng dẫn cài đặt và vận hành
├── REPORT.md                               # Báo cáo học thuật chi tiết 8 chương về quá trình thực nghiệm
├── RULES.md                                # Quy chuẩn dự án: Khoa học dữ liệu, Kỹ thuật phần mềm & Chuẩn ngôn ngữ
├── requirements.txt                        # Danh sách phụ thuộc: transformers, datasets, evaluate, torch, scikit-learn
├── docs/                                   # Hệ thống tài liệu kỹ thuật chuyên đề
│   ├── AUDIT_LOG.md                        # Nhật ký kiểm toán 5 giai đoạn, đối soát dữ liệu và phân xử P0/P1
│   ├── data.md                             # Báo cáo phân tích dữ liệu IMDb, phân phối nhãn, độ dài và PCA 2D
│   ├── demo.md                             # Hướng dẫn chạy thử nghiệm suy luận dòng lệnh CLI (Inference Demo)
│   ├── evaluation.md                       # Đặc tả đánh giá định lượng, ma trận nhầm lẫn và kiểm định thống kê
│   ├── labs3-description.md                # Đặc tả toàn diện đề bài Practice 3 và kiến trúc giải pháp
│   ├── model.md                            # Đặc tả kiến trúc DistilBERT, cơ chế Attention và Fine-tuning Head
│   ├── plan-progress-status.md             # Kế hoạch chi tiết, tiến độ các giai đoạn và trạng thái kiểm định
│   ├── REPORT.md                           # Bản sao lưu trữ báo cáo kỹ thuật dự án trong thư mục docs
│   └── training.md                         # Báo cáo tiến trình huấn luyện, đường cong hàm mất mát và tối ưu hóa
├── notebooks/                              # Thư mục thực thi Jupyter Notebook
│   ├── Practice3_HuggingFace_Final.ipynb   # Notebook thực hành chính — 41 cells chuẩn hóa, trực quan hóa đầy đủ
│   └── Practice3_HuggingFace_Final.html    # Bản kết xuất tĩnh HTML từ notebook để xem nhanh trên trình duyệt
├── outputs/                                # Thư mục chứa kết quả thực nghiệm và tài sản xuất bản
│   ├── REPORT.html                         # Báo cáo độc lập HTML chuẩn xuất bản, nhúng trực tiếp 10 hình ảnh/bảng
│   ├── REPORT.md                           # Bản sao báo cáo thực hành đồng bộ với thư mục gốc
│   ├── checkpoints/                        # Checkpoints lưu trong quá trình huấn luyện (loại trừ khỏi Git)
│   │   ├── checkpoint-50/                  # Trọng số và trạng thái optimizer tại Step 50 (Epoch 1)
│   │   └── checkpoint-100/                 # Trọng số và trạng thái optimizer tại Step 100 (Epoch 2)
│   ├── final_model/                        # Mô hình tối ưu được lựa chọn sau huấn luyện (loại trừ khỏi Git)
│   │   ├── config.json                     # Cấu hình mô hình DistilBERT num_labels=2
│   │   ├── model.safetensors               # Trọng số mô hình định dạng SafeTensors an toàn (255.43 MB)
│   │   ├── special_tokens_map.json         # Ánh xạ các token đặc biệt ([CLS], [SEP], [UNK], [PAD], [MASK])
│   │   ├── tokenizer.json                  # Từ điển tokenizer định dạng JSON
│   │   ├── tokenizer_config.json           # Cấu hình tiền xử lý tokenizer
│   │   ├── training_args.bin               # Tham số huấn luyện nhị phân của Hugging Face Trainer
│   │   └── vocab.txt                       # Bộ từ vựng WordPiece (30,522 tokens)
│   └── reports/                            # 20 tệp báo cáo số liệu thực nghiệm, manifests và biểu đồ đồ họa
│       ├── classification_report.csv       # Bảng chi tiết Precision, Recall, F1-score và Support từng lớp
│       ├── confusion_matrix.csv            # Ma trận nhầm lẫn dạng bảng: TN=84, FP=20, FN=31, TP=65
│       ├── confusion_matrix.png            # Biểu đồ nhiệt ma trận nhầm lẫn trên tập Test Set (200 mẫu)
│       ├── data_label_distribution.png     # Biểu đồ cột phân phối nhãn cân bằng giữa các phân vùng dữ liệu
│       ├── data_manifest.csv               # Danh mục chi tiết 1,200 mẫu kèm split, label, độ dài và hash MD5
│       ├── data_manifest.json              # Siêu dữ liệu phân vùng, tỷ lệ nhãn và kiểm chứng Zero Overlap
│       ├── data_token_length_distribution.png # Biểu đồ tần suất phân bố độ dài token và ngưỡng cắt 128
│       ├── eda_distribution.png            # Đồ thị tổng hợp EDA khám phá dữ liệu (nhãn & độ dài câu)
│       ├── error_analysis.csv              # Danh sách chi tiết 51 ca dự đoán sai kèm logits, xác suất và văn bản
│       ├── final_metrics.json              # Tổng hợp chỉ số hiệu năng thực nghiệm, baselines và kiểm định
│       ├── pca_2d_distribution.png         # Biểu đồ không gian tiềm ẩn PCA 2 chiều phân cụm đặc trưng ngữ nghĩa
│       ├── predictions.csv                 # Toàn bộ 200 mẫu Test kèm logits, xác suất, confidence và kết quả
│       ├── run_manifest.json               # Mã băm SHA-256 toàn bộ artifacts, thông số môi trường và phiên bản thư viện
│       ├── table_benchmark_metrics.png     # Bảng hình ảnh đối chiếu hiệu năng giữa Baselines và Fine-tuned
│       ├── table_classification_report.png # Bảng hình ảnh báo cáo phân loại chi tiết từng lớp
│       ├── table_data_split.png            # Bảng hình ảnh quy cách phân chia dữ liệu Train/Val/Test
│       ├── table_error_analysis.png        # Bảng hình ảnh tổng hợp 5 ca dự đoán sai có độ tự tin cao nhất
│       ├── table_training_progress.png     # Bảng hình ảnh tiến trình tối ưu hóa qua các bước huấn luyện
│       ├── test_comparison_and_per_class.png # Biểu đồ cột so sánh Baseline vs Fine-tuned và chỉ số từng lớp
│       └── training_curves.png             # Đồ thị đường cong Training/Validation Loss và Validation Accuracy
└── src/                                    # Mã nguồn phát triển hệ thống pipeline (9 scripts mô-đun hóa)
    ├── 01_sentiment_analysis.py            # Bài 1: Phân tích cảm xúc với Pretrained Model SST-2, so sánh Softmax
    ├── 02_train_finetune.py                # Bài 2: Quy trình 7 bước Fine-tuning DistilBERT trên IMDb, cô lập Test Set
    ├── 03_predict_finetuned.py             # Script CLI nạp Checkpoint đã lưu để dự đoán cảm xúc câu văn bản mới
    ├── export_tables_as_images.py          # Tự động xuất 5 bảng số liệu định lượng thành tệp hình ảnh PNG
    ├── generate_data_charts.py             # Sinh biểu đồ phân tích dữ liệu (Label Distribution, Token Length, PCA)
    ├── generate_dynamic_plots.py           # Sinh đồ thị thực nghiệm (Confusion Matrix, Test Comparison, Curves)
    ├── generate_html_report.py             # Tạo báo cáo độc lập REPORT.html nhúng toàn bộ biểu đồ và bảng PNG
    ├── pipeline_utils.py                   # Thư viện tiện ích dùng chung: phân chia dữ liệu, manifests, tính metrics
    └── verify_all.py                       # Bộ kiểm thử hồi quy tự động 5 pha kiểm tra toàn vẹn toàn diện hệ thống
```

### 12.2. Lệnh tái lập kết quả từ đầu (Reproducibility Commands)
```bash
# 1. Chạy toàn bộ 5 ca kiểm thử hồi quy tự động
python src/verify_all.py

# 2. Chạy Exercise 1 — Phân tích cảm xúc với Pretrained Model SST-2
python src/01_sentiment_analysis.py

# 3. Chạy Exercise 2 — Fine-tuning DistilBERT trên IMDb (800 Train / 200 Val)
python src/02_train_finetune.py

# 4. Dự đoán câu mới với mô hình đã fine-tune
python src/03_predict_finetuned.py

# 5. Xuất biểu đồ trực quan hóa dữ liệu (Label, Length, PCA)
python src/generate_data_charts.py

# 6. Xuất biểu đồ động thực nghiệm (Confusion Matrix, Comparison, Curves)
python src/generate_dynamic_plots.py

# 7. Xuất 5 bảng số liệu định lượng sang hình ảnh PNG
python src/export_tables_as_images.py

# 8. Xuất báo cáo HTML độc lập nhúng hình ảnh
python src/generate_html_report.py
```
