# RULES.md — BỘ QUY TẮC DỰ ÁN PRACTICE 3: HUGGING FACE

Tài liệu quy định các nguyên tắc bắt buộc khi phát triển mã nguồn, thực thi pipeline và lập báo cáo dự án **Practice 3: Sentiment Analysis & Fine-tuning DistilBERT**.

---

## 1. Quy Tắc Khoa Học và Tính Toàn Vẹn Dữ Liệu

1. **Số liệu thực tế (No Hallucination):** Tuyệt đối không tự tạo hoặc ước lượng số liệu. Mọi chỉ số (accuracy, loss, ma trận nhầm lẫn) phải lấy trực tiếp từ file log thực tế (`outputs/reports/final_metrics.json`, `classification_report.csv`, `confusion_matrix.csv`).
2. **Tính trung thực:** Không chỉnh sửa kết quả đo đạc thủ công dưới mọi hình thức.
3. **Chống rò rỉ dữ liệu (No Data Leakage):**
   - Phân chia độc lập: Tách riêng Train (800 mẫu) và Validation (200 mẫu) là 2 tập con rời nhau từ `dataset['train']` qua `.shuffle(seed=42).select(...)`, còn Test Set (200 mẫu) được trích xuất độc lập từ `dataset['test']` qua `.shuffle(seed=42).select(range(200))`.
   - Độc lập huấn luyện: Tập Validation và Test không tham gia vào quá trình tính gradient cập nhật mô hình. Tập Test được bảo vệ nghiêm ngặt, không can thiệp trước khi hoàn tất huấn luyện.
4. **Đánh giá Test Set:** Chỉ dùng Test Set một lần duy nhất ở bước cuối cùng để đo độ tổng quát hóa. Tuyệt đối không dùng để chọn siêu tham số.
5. **Khả năng tái lập (Reproducibility):** Cố định ngẫu nhiên bằng `seed=42` xuyên suốt toàn bộ hệ thống (`torch`, `numpy`, `transformers.set_seed`).

---

## 2. Quy Tắc Viết Mã Nguồn (Code Style & Engineering)

1. **Môi trường & Thư viện:** Sử dụng Python $\ge 3.10$ và hệ sinh thái chuẩn: `transformers`, `datasets`, `evaluate`, `accelerate`, `torch`.
2. **Khai báo kiểu & Tài liệu:** Mọi hàm đều có type hints rõ ràng và docstring ngắn gọn mô tả đầu vào, đầu ra.
3. **Tính tương thích (Portability):**
   - Đường dẫn động: Dùng thư viện `pathlib.Path` tương đối từ thư mục gốc, không gán cứng đường dẫn máy cá nhân.
   - Thiết bị tính toán: Tự động phát hiện và gán phần cứng theo thứ tự CUDA $\to$ MPS $\to$ CPU.
4. **Bảo mật và an toàn:**
   - Định dạng mô hình: Lưu trọng số mô hình bằng chuẩn an toàn `safetensors`.
   - Bảo mật thông tin: Không đưa API token, khóa bí mật hoặc file `.env` lên Git repository.

---

## 3. Quy Tắc Giao Tiếp (Communication Rules)

1. **Phong cách giao tiếp:**
   - Ngôn từ trực tiếp: Tự nhiên, ngắn gọn, dễ hiểu, phù hợp môi trường dự án IT.
   - Loại bỏ rườm rà: Không sử dụng văn phong học thuật, nghiên cứu lý luận hay hành chính.
2. **Cấu trúc diễn đạt:**
   - Cấu trúc chuẩn: Ưu tiên áp dụng `Từ khóa: Diễn giải ngắn gọn`.
   - Độ phức tạp câu: Không sử dụng câu văn dài hoặc cấu trúc phức tạp; không diễn giải một ý bằng nhiều câu.
   - Thuật ngữ kỹ thuật: Giữ nguyên các thuật ngữ IT chuẩn xác, không thay thế tùy tiện.
3. **Hình thức trình bày:**
   - Không sử dụng icon/emoji: Đảm bảo tài liệu chuyên nghiệp, chuẩn mực và không lỗi font console.
   - Bố cục rõ ràng: Sử dụng bảng biểu, danh sách phân cấp để truyền tải thông tin nhanh chóng.

---

## 4. Quy Chuẩn Ngôn Ngữ & Kỹ Thuật Kiểm Toán (Language & Technical Terminology Rules)

### 4.1 Quy tắc ngôn ngữ (Language & Terminology Rules)
1. **Preserve Technical Terminology:**
   - Giữ nguyên các thuật ngữ kỹ thuật, industry-standard terminology và các cụm từ chuyên ngành phổ biến dưới dạng tiếng Anh gốc (`System Architecture`, `Data Flow`, `Core Features`, `Folder Structure`, `Deployment`, `Code Review`, `Fine-tuning`, `Checkpoint`, `Tokenizer`, `Pipeline`, `Inference`, `Predictions`, `Logits`, `Softmax`, `Confidence`, `Metrics`, `Accuracy`, `Precision`, `Recall`, `F1-score`, `Test Loss`, `Confusion Matrix`, `Data Leakage`, `Data Splitting`, `Reproducibility`).
   - Tuyệt đối không dịch thuật ngữ kỹ thuật sang tiếng Việt khi không cần thiết hoặc gây gượng gạo, kém tự nhiên.
   - Ưu tiên sử dụng thuật ngữ tiếng Anh chuẩn mà software engineers và developers thường dùng trong thực tế.
2. **Vietnamese Explanation:**
   - Sử dụng tiếng Việt tự nhiên, rõ ràng, mạch lạc và chuyên nghiệp cho phần giải thích, phân tích và mô tả logic.
   - Khi cần thiết, giải thích ngắn gọn ý nghĩa của thuật ngữ kỹ thuật bằng tiếng Việt nhưng vẫn giữ nguyên từ khóa tiếng Anh gốc.
   - Không lạm dụng chêm từ tiếng Anh vào các từ ngữ giao tiếp thông thường; tiếng Việt vẫn là ngôn ngữ diễn đạt chính.
3. **Consistency:**
   - Sử dụng thống nhất một thuật ngữ tiếng Anh xuyên suốt toàn bộ dự án và tài liệu.
   - Không hoán đổi ngẫu nhiên giữa từ tiếng Anh và bản dịch tiếng Việt của cùng một thuật ngữ.
   - Giữ nguyên định danh kỹ thuật, tên framework, library, API, protocol, design pattern, algorithm, variable name, class name, function name và configuration key theo code gốc.
4. **Headings & Documentation:**
   - Ưu tiên tiếng Anh cho các technical headings, section names, module names, architecture diagrams và nhãn mã nguồn trong tài liệu Markdown, tables, diagrams, và báo cáo.
   - Không dịch các tiêu đề kỹ thuật chuẩn mực đã được định hình trong ngành.
5. **Exceptions:**
   - Dịch các từ tiếng Anh thông thường khi chúng không phải là thuật ngữ kỹ thuật.
   - Nếu một thuật ngữ thực sự khó hiểu, giải thích ngắn gọn bằng tiếng Việt trong ngoặc đơn hoặc câu văn tiếp theo.
   - Bảo toàn ngữ nghĩa và ngữ cảnh gốc thay vì ưu tiên dịch sát nghĩa đen.
6. **Priority Rule:**
   - Khi phải lựa chọn giữa một bản dịch tiếng Việt gượng ép và một thuật ngữ tiếng Anh chuẩn ngành, LUÔN LUÔN ưu tiên thuật ngữ tiếng Anh gốc.

### 4.2 Quy chuẩn kỹ thuật kiểm toán (Technical Rules)
- Tuân thủ `RULES.md` và yêu cầu Practice 3.
- Kiểm tra Data Overlap, Duplicate, Near-Duplicate và Data Leakage giữa Train/Validation/Test.
- Bảo đảm Test Set không được sử dụng cho Training, Hyperparameter Tuning hoặc Model Selection.
- Xác minh Data Splitting, Random Seed, Sample IDs và Dataset Version.
- Thực hiện Independent Inference từ saved Checkpoint để xác minh Predictions.
- Recalculate Accuracy, Precision, Recall, F1-score, Test Loss và Confusion Matrix từ dữ liệu thực tế.
- Kiểm tra tính nhất quán giữa Notebook, Scripts, CSV, JSON, Training Logs, Checkpoint và Reports.
- Không Hardcode Metrics, Fabricate Results hoặc sử dụng Stale Outputs.
- Không tự ý thay đổi Dataset, Model Architecture, Hyperparameters hoặc Ground Truth Labels.
- Mọi kết luận phải có Evidence từ Source Code, Execution Logs hoặc Experimental Outputs.
- Phân loại kết quả thành `PASS`, `FAIL`, `WARNING`, `NOT VERIFIED`. Không đánh dấu `PASS` nếu chưa thực sự kiểm tra.
- Khi phát hiện lỗi, nêu chính xác **File – Line – Root Cause – Impact – Recommended Fix**.

### 4.3 Nguyên tắc cốt lõi (Final Requirement)
- **Prioritize Data Integrity, Evaluation Correctness, Reproducibility and Scientific Validity over Model Accuracy.**
- Chỉ xác nhận kết quả đáng tin cậy khi Data Splitting hợp lệ, không có Data Leakage ảnh hưởng đến Evaluation và toàn bộ Metrics được kiểm chứng từ cùng Dataset Split, Checkpoint và Experiment Run.
- **Do not fabricate, hardcode, manipulate or selectively report experimental results.**
