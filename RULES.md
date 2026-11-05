# RULES.md — BỘ QUY TẮC DỰ ÁN PRACTICE 3: HUGGING FACE

Bộ quy tắc này là bắt buộc đối với toàn bộ quá trình phát triển mã nguồn, thực nghiệm và lập tài liệu cho dự án **Practice 3 — Get Started with Hugging Face (Sentiment Analysis & Fine-tuning DistilBERT)**.

---

## 1. Quy Tắc Khoa Học và Tính Trung Thực Học Thuật

1. **Tuyệt đối không bịa đặt số liệu (No Hallucination):** Không được tự tạo hoặc ước lượng số liệu độ chính xác (accuracy), hàm mất mát (loss), thời gian chạy hoặc kết quả kiểm thử. Toàn bộ số liệu trong báo cáo phải được trích xuất tự động từ các tệp nhật ký thực tế (`outputs/reports/final_metrics.json`, `classification_report.csv`, `confusion_matrix.csv`).
2. **Không chỉnh sửa số liệu:** Không sửa đổi thủ công kết quả để đạt mục tiêu định trước.
3. **Ngăn chặn rò rỉ dữ liệu (No Data Leakage):**
   - Phân chia tập Train, Validation và Test độc lập bằng `train_test_split(seed=42)`.
   - Tập Validation và Test tuyệt đối không tham gia vào quá trình tính gradient cập nhật trọng số.
4. **Sử dụng Test Set đúng chuẩn:** Tuyệt đối không dùng Test Set để chọn siêu tham số hoặc early stopping. Test Set chỉ được dùng duy nhất một lần ở bước cuối cùng để đánh giá tổng quát hóa.
5. **Tính tái lập (Reproducibility):** Cố định hạt giống ngẫu nhiên (`seed=42`) cho mọi khâu (torch, numpy, transformers `set_seed`).

---

## 2. Quy Tắc Viết Mã Nguồn (Code Style & Engineering)

1. **Phiên bản & Thư viện:** Python $\ge 3.10$. Sử dụng hệ sinh thái Hugging Face chuẩn: `transformers`, `datasets`, `evaluate`, `accelerate`, `torch`.
2. **Kiểu dữ liệu & Tài liệu hóa:** Mọi hàm và phương thức bắt buộc có type hints đầy đủ và docstrings ngắn gọn, súc tích.
3. **Tính di động (Portability):**
   - Không hard-code đường dẫn tuyệt đối của máy cá nhân. Mọi đường dẫn phải dựa trên `pathlib.Path` tương đối với thư mục gốc dự án.
   - Tự động nhận diện thiết bị tính toán: CUDA $\to$ MPS $\to$ CPU.
4. **Bảo mật và an toàn dữ liệu:**
   - Sử dụng định dạng an toàn `safetensors` cho trọng số mô hình đã fine-tune.
   - Không đưa API keys, token cá nhân hoặc file `.env` vào kho lưu trữ Git.
