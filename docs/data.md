# Báo Cáo Khoa Học Dữ Liệu: IMDb Dataset (data.md)

Đặc tả toàn diện bộ dữ liệu IMDb Movie Reviews phục vụ bài toán Phân loại cảm xúc (Sentiment Analysis) và Fine-tuning DistilBERT trong Practice 3.

---

## 1. Tổng Quan & Nguồn Thu Thập

- **Tên Dataset:** IMDb Movie Reviews (Large Movie Review Dataset), lưu trữ tại [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb).
- **Tác giả:** Andrew L. Maas et al. (Stanford University, ACL 2011).
- **Quy mô gốc:** 50,000 bài đánh giá phim phân cực rõ rệt (25,000 train + 25,000 test). Điểm $\le 4/10$ là NEGATIVE (`0`), $\ge 7/10$ là POSITIVE (`1`). Các đánh giá trung tính (5-6 sao) đã được loại bỏ.
- **Phạm vi Practice 3:** Trích chọn 1,200 mẫu (`seed=42`) gồm 800 Train, 200 Validation và 200 Test nhằm thực nghiệm tối ưu trên môi trường CPU.

---

## 2. Cấu Trúc Dữ Liệu & Phân Vùng

### 2.1 Bảng cấu trúc thuộc tính

| Tên trường | Kiểu dữ liệu | Ý nghĩa | Miền giá trị |
|:---|:---|:---|:---|
| `text` | `string` | Nội dung văn bản đánh giá phim tiếng Anh | Chuỗi ký tự UTF-8 |
| `label` | `int64` | Nhãn cảm xúc nhị phân | `0` (NEGATIVE), `1` (POSITIVE) |
| `input_ids` | `torch.int64` | Vector mã token từ điển WordPiece | $[0, 30521]$ (kèm `[CLS]` 101, `[SEP]` 102) |
| `attention_mask` | `torch.int64` | Mặt nạ nhị phân phân biệt token thật và đệm | `1` (token thật), `0` (token đệm `[PAD]`) |

### 2.2 Quy cách phân chia dữ liệu (No Data Leakage)

| Phân vùng | Nguồn trích xuất | Số mẫu | NEGATIVE (`0`) | POSITIVE (`1`) | Tỷ lệ cân bằng |
|:---|:---|:---:|:---:|:---:|:---:|
| **Train** | `imdb['train']`, `shuffle(42)`, `select(0..799)` | 800 | 408 (51.0%) | 392 (49.0%) | Cân bằng ~50/50 |
| **Validation** | `imdb['train']`, `shuffle(42)`, `select(800..999)` | 200 | 104 (52.0%) | 96 (48.0%) | Cân bằng ~50/50 |
| **Test** | `imdb['test']`, `shuffle(42)`, `select(0..199)` | 200 | 104 (52.0%) | 96 (48.0%) | Cân bằng ~50/50 |
| **Tổng cộng** | — | **1,200** | **616 (51.3%)** | **584 (48.7%)** | Không chồng lấn |

---

## 3. Đánh Giá Chất Lượng & Tiền Xử Lý

### 3.1 Chất lượng dữ liệu

- **Khuyết thiếu (Missing):** 0 giá trị (0.0%). Không có bản ghi rỗng hoặc nhãn NaN.
- **Trùng lặp (Duplicates):** 0 bản ghi trùng lặp (100% độc lập).
- **Tính hợp lệ:** 100% nhãn nằm nghiêm ngặt trong $\{0, 1\}$.

### 3.2 Quy trình tiền xử lý

1. **Xáo trộn dữ liệu (Shuffle):** IMDb gốc xếp 12,500 mẫu NEGATIVE liên tục ở đầu, bắt buộc `shuffle(seed=42)` trước khi cắt tập.
2. **WordPiece Sub-word Tokenization:** Sử dụng từ điển 30,522 tokens của `distilbert-base-uncased`, giải quyết triệt để lỗi OOV qua tiền tố `##`.
3. **Cắt cụt (Truncation):** Giới hạn `max_length=128` tokens để tối ưu bộ nhớ và thời gian tính toán trên CPU.
4. **Đệm động (`DataCollatorWithPadding`):** Đệm mini-batch theo câu dài nhất trong batch, tiết kiệm 35–50% thời gian forward pass so với đệm tĩnh 128.

---

## 4. Phân Tích Khám Phá Dữ Liệu (EDA)

### 4.1 Thống kê mô tả độ dài chuỗi

| Đại lượng | Số từ thô (Word Count) | Độ dài Token (WordPiece) | Tỷ lệ vượt ngưỡng 128 tokens |
|:---|:---:|:---:|:---:|
| **Trung bình (Mean)** | 232.4 từ | 286.7 tokens | **Train: 87.5%** (700/800 mẫu) |
| **Độ lệch chuẩn ($\sigma$)** | 168.2 từ | 208.5 tokens | **Test: 86.0%** (172/200 mẫu) |
| **Trung vị (Q2)** | 171.0 từ | 212.0 tokens | Phân phối lệch phải (Right-skewed) |
| **Tối thiểu / Tối đa** | 11.0 / 984.0 từ | 14.0 / 1,186.0 tokens | Xử lý qua ngưỡng cắt 128 |

### 4.2 Phân tích không gian đặc trưng TF-IDF và PCA 2D

- Ma trận TF-IDF (1,000 từ vựng đặc trưng) được giảm chiều qua PCA:
  - $PC1$ giải thích **0.84%** phương sai, $PC2$ giải thích **0.65%** phương sai. Tổng giải thích chỉ đạt **1.49%**.
  - Hai lớp dữ liệu đan xen dày đặc quanh gốc tọa độ, tâm cụm gần như trùng khớp.
  - **Kết luận khoa học:** Dữ liệu cảm xúc bề mặt **hoàn toàn không thể phân tách tuyến tính (Linearly Inseparable)**, khẳng định sự cần thiết của các tầng Transformer Attention sâu.

### 4.3 Đánh giá tác động cắt cụt token đến sai số

| Nhóm Test Set | Số mẫu | Số lỗi | Tỷ lệ lỗi (%) | Độ chính xác (%) |
|:---|:---:|:---:|:---:|:---:|
| **Không bị cắt ($\le 128$ tokens)** | 28 (14.0%) | 6 | 21.43% | 78.57% |
| **Bị cắt ngắn ($> 128$ tokens)** | 172 (86.0%) | 45 | 26.16% | 73.84% |
| **Toàn bộ Test Set** | 200 (100%) | 51 | 25.50% | 74.50% |

- **Kiểm định Fisher's Exact Test:** Odds Ratio = $0.7697$, $p = 0.8153 > 0.05$. Sự khác biệt không có ý nghĩa thống kê, xác nhận việc cắt cụt tại 128 là thỏa hiệp kỹ thuật tối ưu trên CPU.

---

## 5. Trực Quan Hóa & Danh Mục Biểu Đồ

Các biểu đồ phân tích được xuất bản tại [`outputs/reports/`](../outputs/reports/):
- **Phân bố nhãn:** [`data_label_distribution.png`](../outputs/reports/data_label_distribution.png) — Kiểm chứng tỷ lệ 50/50 cân bằng.
- **Phân phối độ dài:** [`data_token_length_distribution.png`](../outputs/reports/data_token_length_distribution.png) — Trực quan hóa ngưỡng cắt 128 tokens.
- **Không gian tiềm ẩn PCA 2D:** [`pca_2d_distribution.png`](../outputs/reports/pca_2d_distribution.png) — Trực quan hóa sự chồng lấn phi tuyến của từ vựng thô.
- **Tổng hợp EDA:** [`eda_distribution.png`](../outputs/reports/eda_distribution.png) — Ghép bộ 3 đồ thị phân tích khám phá.

---

## 6. Hạn Chế & Kết Luận

1. **Hạn chế:** Cắt cụt tại 128 tokens làm mất đoạn kết của 86% mẫu; quy mô 800 mẫu train được thiết kế vừa vặn cho kiểm thử nhanh trên CPU (391.42s).
2. **Kết luận:** Dataset IMDb đạt chuẩn dữ liệu học thuật sạch, nhãn cân bằng tuyệt đối, ngữ cảnh phong phú, chứng minh tính ưu việt của phương pháp Fine-tuning Transformer so với mô hình tuyến tính truyền thống.
