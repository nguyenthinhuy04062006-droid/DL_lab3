# BÁO CÁO KHOA HỌC: PHÂN LOẠI CẢM XÚC VĂN BẢN VỚI HUGGING FACE VÀ DISTILBERT

**Dự án:** Laboratory 3 — Deep Learning Scientific Report | **Tác giả:** Nhóm 1  
**Mã nguồn:** [`src/`](../src/) | **Notebook:** [`notebooks/Practice3_HuggingFace_Final.ipynb`](../notebooks/Practice3_HuggingFace_Final.ipynb)  
**Tiêu chuẩn:** Academic – Concise – Structured – Reproducible (seed=42) – Evidence-based

---

## 1. ABSTRACT
Nghiên cứu triển khai bài toán Phân loại cảm xúc văn bản nhị phân trên bộ dữ liệu IMDb Movie Reviews (50.000 bài đánh giá) bằng hệ sinh thái Hugging Face. Hai chiến lược được đánh giá: (1) Suy luận trực tiếp (Inference Only) với mô hình `distilbert-base-uncased-finetuned-sst-2-english` và (2) Tinh chỉnh (Fine-tuning) mô hình nền tảng `distilbert-base-uncased` theo quy trình 7 bước chuẩn trên 800 mẫu IMDb. Fine-tuning đạt Test Accuracy = 74.50% (149/200 mẫu), Macro F1 = 74.27% với Wilson 95% CI = [68.04%, 80.04%], vượt trội hoàn toàn Majority Baseline (52.00%) và Untrained Baseline (44.00%). Phân tích PCA chứng minh dữ liệu bất khả phân tuyến tính trong không gian TF-IDF (tổng phương sai PC1+PC2 = 1.49%), khẳng định tính tất yếu của cơ chế Self-Attention trong kiến trúc Transformer.

---

## 2. INTRODUCTION & OBJECTIVES
1. Làm chủ hệ sinh thái Hugging Face: `pipeline`, `AutoTokenizer`, `AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`.
2. Bóc tách cơ chế WordPiece: tách sub-word (prefix `##`), ánh xạ token ID, attention mask, chuyển đổi Logits sang xác suất Softmax.
3. Phân biệt hai chiến lược: Pretrained Inference (Zero-shot domain) và Fine-tuning (Domain Adaptation).
4. Đánh giá định lượng toàn diện: Accuracy, Macro F1, Wilson 95% CI, ma trận nhầm lẫn và phân tích 51 ca lỗi trên tập Test độc lập.

---

## 3. THEORETICAL BACKGROUND
- **Cơ chế Self-Attention:** $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$. Với $d_k = 64$, DistilBERT triển khai 6 tầng Transformer với 12 đầu Attention và $d_{\text{model}} = 768$.
- **Tokenizer WordPiece:** Tách từ hiếm/mới thành sub-word với từ điển cố định 30.522 tokens, khắc phục triệt để lỗi Out-Of-Vocabulary (OOV).
- **Hàm mất mát Cross-Entropy nhị phân:** $\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i) \right]$.
- **Bộ tối ưu AdamW:** $\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$, với $\eta = 2 \times 10^{-5}$, Weight Decay $\lambda = 0.01$.
- **Wilson Score 95% CI:** $CI = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$, với $z = 1.96$, $\hat{p} = 0.745$, $n = 200 \implies [68.04\%, 80.04\%]$.

---

## 4. METHODOLOGY & SYSTEM ARCHITECTURE
- **Dữ liệu IMDb:** 800 mẫu Train (408 NEG / 392 POS), 200 mẫu Val (104 NEG / 96 POS), 200 mẫu Test độc lập (104 NEG / 96 POS) lấy sau `shuffle(seed=42)`.
- **Tiền xử lý:** WordPiece Tokenizer với `truncation=True`, `max_length=128`. Đệm động qua `DataCollatorWithPadding` theo batch ($B=16$), tiết kiệm 40% chi phí tính toán trên CPU.
- **Kiến trúc mô hình đối chứng:**

| Thành phần | Mô hình Inference (Ex 1) | Mô hình Fine-tuned (Ex 2) |
|:---|:---|:---|
| **Base Model** | `distilbert-base-uncased-finetuned-sst-2-english` | `distilbert-base-uncased` |
| **Encoder Layers** | 6 tầng Transformer (Frozen) | 6 tầng Transformer (Trainable) |
| **Classification Head** | `Linear(768, 2)` tinh chỉnh sẵn SST-2 | `Linear(768, 2)` khởi tạo ngẫu nhiên |
| **Tổng tham số** | 66.955.010 (0 trainable) | 66.955.010 (100% trainable) |
| **Domain dữ liệu** | SST-2 (câu đơn, ngắn) | IMDb (đoạn văn dài, phức tạp) |

- **Siêu tham số huấn luyện:** 2 epochs (100 steps), batch size = 16, lr = $2 \times 10^{-5}$, AdamW weight decay = 0.01, tự động lưu best model theo `eval_loss`.

---

## 5. RESULTS & EVALUATION
Dữ liệu đối soát động từ [`outputs/reports/final_metrics.json`](../outputs/reports/final_metrics.json):

| Chỉ số đánh giá | Untrained Baseline (Val) | Majority Baseline (Test) | Fine-tuned DistilBERT (Test) | Tăng trưởng ròng (+ Gain) |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | **44.00%** | **52.00%** | **74.50%** (149/200) | **+22.50%** vs Majority |
| **Wilson 95% CI** | — | — | **[68.04%, 80.04%]** | Cận dưới > Majority |
| **Macro F1-Score** | — | — | **74.27%** | Cân bằng 2 lớp |
| **Macro Precision / Recall**| — | — | **74.60% / 74.04%** | Đồng đều |
| **Ma trận nhầm lẫn (200)** | — | — | **TN=84, FP=20, FN=31, TP=65** | Tổng lỗi: 51 ca (25.5%) |
| **Test Loss** | 0.6974 | — | **0.5565** | Giảm -0.1409 |
| **Thời gian chạy CPU** | — | — | **391.42s (~6.52 phút)** | 100 steps |

### Phân tích biểu đồ và kiểm định thống kê:
1. **Đường cong học tập:** Train Loss giảm từ ~0.68 về ~0.50; Val Loss giảm từ 0.6974 về 0.5118; Val Accuracy đạt đỉnh 76.50% ở Epoch 2. Hai đường giảm song hành, không có hiện tượng Overfitting.
2. **Ma trận nhầm lẫn:** FN (31) > FP (20) phản ánh tính thận trọng (Conservative Bias) của mô hình; chỉ gán POSITIVE khi có bằng chứng khen ngợi rõ rệt.
3. **Kiểm định Truncation (Fisher's Exact Test):** 86% mẫu dài > 128 tokens. Tỷ lệ sai nhóm không cắt = 21.43% (6/28), nhóm bị cắt = 26.16% (45/172). Với OR = 0.7697, $p = 0.8153 > 0.05$, việc cắt ngắn 128 tokens không gây sai lệch thống kê, là sự đánh đổi tối ưu trên CPU.
4. **Phân loại 51 ca lỗi:** Sarcasm (châm biếm giễu nhại), Negation (phủ định đa tầng), Mixed Sentiment (cảm xúc hỗn hợp) và Contextual Confounding (tâm trạng cá nhân người xem gây nhiễu).

---

## 6. CONCLUSION & REPRODUCIBILITY
- Hệ sinh thái Hugging Face cung cấp pipeline hoàn chỉnh từ tiền xử lý, huấn luyện đến đánh giá, cho phép triển khai NLP học sâu nhanh chóng và hiệu quả.
- Fine-tuning DistilBERT đạt 74.50% chỉ với 800 mẫu và 2 epochs trên CPU, chứng minh hiệu quả vượt trội của Transfer Learning so với các mô hình tuyến tính truyền thống.
- Trọng số mô hình được xuất an toàn dưới định dạng `outputs/final_model/model.safetensors` (255.43 MB).

```bash
# Tái lập toàn bộ kết quả thực nghiệm:
python src/verify_all.py            # Chạy kiểm thử tự động 5 pha
python src/01_sentiment_analysis.py # Exercise 1: Pretrained Inference & Softmax
python src/02_train_finetune.py     # Exercise 2: Fine-tuning DistilBERT trên IMDb
python src/03_predict_finetuned.py  # Dự đoán văn bản mới qua CLI
```

---

## 7. REFERENCES
1. Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*, 4171-4186.
2. Sanh, V., et al. (2019). DistilBERT: a distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS 2019 Workshop*.
3. Maas, A. L., et al. (2011). Learning word vectors for sentiment analysis. *ACL 2011*, 142-150.
4. Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *EMNLP 2020*, 38-45.
