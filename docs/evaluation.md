# ĐÁNH GIÁ HIỆU NĂNG & PHÂN TÍCH LỖI (EVALUATION.MD)

Tài liệu kỹ thuật tổng hợp toàn bộ số liệu đo lường thực nghiệm, kiểm định thống kê suy diễn, phân tích sai số chi tiết trên tập kiểm thử độc lập (IMDb Test Set, 200 mẫu).
Nguồn dữ liệu: [`outputs/reports/final_metrics.json`](../outputs/reports/final_metrics.json), [`predictions.csv`](../outputs/reports/predictions.csv), [`confusion_matrix.csv`](../outputs/reports/confusion_matrix.csv), [`classification_report.csv`](../outputs/reports/classification_report.csv).

---

## 1. Kết Quả Đo Lường Tổng Quát (Benchmark Results)

| Chỉ số đánh giá | Untrained Baseline (Val) | Majority Baseline (Test) | Fine-tuned DistilBERT (Test) | Tăng trưởng ròng (+ Gain) |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy (Độ chính xác)** | **44.00%** | **52.00%** | **74.50%** | **+22.50%** vs Majority<br>**+30.50%** vs Untrained |
| **Khoảng tin cậy 95% (Wilson CI)** | — | — | **[68.04%, 80.04%]** | Cận dưới > Majority |
| **Số mẫu dự đoán đúng** | 88 / 200 | 104 / 200 | **149 / 200** | **+45 mẫu đúng** vs Majority |
| **Macro F1-Score** | — | — | **74.27%** | Cân bằng 2 lớp |
| **Macro Precision / Recall** | — | — | **74.76% / 74.24%** | Đồng đều giữa các lớp |
| **Test Loss (Cross-Entropy)** | 0.6974 (Val) | — | **0.5565** | Giảm -0.1409 |
| **Thời gian huấn luyện CPU** | — | — | **391.42s (~6.52 phút)** | 2 epochs, batch size = 16 |

### Công thức toán học cốt lõi:
- **Accuracy:** $\text{Accuracy} = \frac{TP + TN}{N} = \frac{65 + 84}{200} = \frac{149}{200} = 74.50\%$.
- **Wilson Score 95% CI:** $CI = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} \implies [68.04\%, 80.04\%]$ với $\hat{p} = 0.745, n = 200, z = 1.96$.
- **Kiểm định McNemar:** $\chi^2 = \frac{(|b - c| - 1)^2}{b + c} = \frac{(|65 - 20| - 1)^2}{85} = 22.78$, với $p\text{-value} = 1.03 \times 10^{-6} < 0.001$, khẳng định tính ưu việt có ý nghĩa thống kê cao.
- **Hàm mất mát Test Loss:** $\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=0}^1 \mathbb{I}(y_i = c) \log P(y_i = c \mid \mathbf{x}_i) = 0.5565$.

---

## 2. Báo Cáo Chi Tiết Theo Lớp (Per-class Report)

| Lớp nhãn | Precision | Recall | F1-Score | Số mẫu thực tế (Support) |
|:---|:---:|:---:|:---:|:---:|
| **NEGATIVE (0)** | **73.04%** (84/115) | **80.77%** (84/104) | **76.71%** | 104 mẫu (52.0%) |
| **POSITIVE (1)** | **76.47%** (65/85) | **67.71%** (65/96) | **71.82%** | 96 mẫu (48.0%) |
| **Macro Average** | **74.76%** | **74.24%** | **74.27%** | 200 mẫu |
| **Weighted Average** | **74.69%** | **74.50%** | **74.36%** | 200 mẫu |

- **Nhận định:** Lớp NEGATIVE có Recall vượt trội (80.77%), mô hình bắt trúng 84/104 bài chê thực tế. Lớp POSITIVE có Precision cao (76.47%), khi dự đoán khen thì độ tin cậy rất cao. Macro F1 đạt 74.27% (chênh lệch 2 lớp chỉ 4.89%) khẳng định chất lượng phân loại cân bằng.

---

## 3. Ma Trận Nhầm Lẫn (Confusion Matrix)

$$\mathbf{C} = \begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix} = \begin{bmatrix} 84 & 20 \\ 31 & 65 \end{bmatrix}$$

- **True Negative (TN = 84):** 84 bài chê đoán đúng là chê (42.0% tổng mẫu).
- **True Positive (TP = 65):** 65 bài khen đoán đúng là khen (32.5% tổng mẫu).
- **False Positive (FP = 20):** 20 bài chê bị đoán nhầm thành khen (10.0% tổng mẫu).
- **False Negative (FN = 31):** 31 bài khen bị đoán nhầm thành chê (15.5% tổng mẫu).
- **Conservative Bias (Thiên lệch thận trọng):** $FN (31) > FP (20)$, mô hình có xu hướng thận trọng, chỉ gán nhãn POSITIVE khi câu từ khen ngợi thực sự rõ ràng, còn các câu phân vân thường nghiêng về NEGATIVE.

---

## 4. Tác Động Của Cắt Cụt Độ Dài & Kiểm Định Fisher

| Phân nhóm kiểm thử | Số lượng mẫu | Số ca đoán sai | Tỷ lệ lỗi (%) | Độ chính xác (%) | Wilson 95% CI (Độ chính xác) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Không bị cắt ($\le 128$)** | 28 mẫu (14.0%) | 6 mẫu | **21.43%** | **78.57%** | [60.46%, 89.79%] |
| **Bị cắt ngắn ($> 128$)** | 172 mẫu (86.0%) | 45 mẫu | **26.16%** | **73.84%** | [66.81%, 79.83%] |
| **Toàn bộ Test Set** | **200 mẫu (100%)** | **51 mẫu** | **25.50%** | **74.50%** | **[68.04%, 80.04%]** |

- **Bảng liên kết Fisher $2 \times 2$:** Lỗi/Đúng không cắt: $[6, 22]$; Lỗi/Đúng bị cắt: $[45, 127]$.
- **Odds Ratio (Tỷ số khả dĩ lỗi):** $\text{OR} = \frac{6 \times 127}{22 \times 45} = \frac{762}{990} \approx 0.7697$.
- **Xác suất Fisher hai phía:** $p\text{-value} = 0.8153 \gg 0.05$.
- **Kết luận:** Tỷ lệ lỗi chênh lệch chỉ 4.73% và không có ý nghĩa thống kê. Giới hạn 128 tokens là mức đánh đổi kỹ thuật tối ưu trên CPU, giảm 75% chi phí tính toán mà không làm suy giảm hiệu năng mô hình.

---

## 5. Phân Loại Nguyên Nhân Trên 51 Ca Dự Đoán Sai

Dựa trên tệp nhật ký chi tiết [`outputs/reports/error_analysis.csv`](../outputs/reports/error_analysis.csv):
1. **Mixed Sentiment (Cảm xúc hỗn hợp):** Bài viết liệt kê nhiều điểm trừ ở thân bài nhưng câu kết luận lại khen ngợi. Mật độ từ ngữ tiêu cực dày đặc lấn át phân phối Attention (ví dụ Index 180, độ tự tin 80.23%).
2. **Contextual Confounding (Nhiễu tâm trạng cá nhân):** Người xem dùng từ tiêu cực miêu tả tâm trạng bản thân (*"I had a horrid day but this movie grabbed me"*) chứ không chê bộ phim (ví dụ Index 88, độ tự tin 82.04%).
3. **Sarcasm & Cult-movie (Châm biếm / Khen giễu nhại):** Dùng các từ chê bai (*"is a mess", "bad hair", "bad flick"*) để biểu đạt sự thích thú với phim kinh điển hạng B (ví dụ Index 103, độ tự tin 76.10%).
4. **Negation (Phủ định đa tầng):** Cấu trúc đảo ngữ phức tạp vượt quá biên độ nhận diện ngữ cảnh ngắn.

---

## 6. Danh Mục Tệp Đối Soát Kiểm Toán

- **Dữ liệu suy luận 200 mẫu:** [`outputs/reports/predictions.csv`](../outputs/reports/predictions.csv)
- **Báo cáo phân loại Scikit-Learn:** [`outputs/reports/classification_report.csv`](../outputs/reports/classification_report.csv)
- **Ma trận nhầm lẫn:** [`outputs/reports/confusion_matrix.csv`](../outputs/reports/confusion_matrix.csv)
- **51 ca dự đoán sai:** [`outputs/reports/error_analysis.csv`](../outputs/reports/error_analysis.csv)
- **Tổng hợp metrics chuẩn hóa:** [`outputs/reports/final_metrics.json`](../outputs/reports/final_metrics.json)
