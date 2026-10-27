# RULES.md — BỘ QUY TẮC DỰ ÁN FASHIONMNIST

Bộ quy tắc này là bắt buộc đối với toàn bộ quá trình phát triển mã nguồn, thực nghiệm và lập tài liệu cho dự án **Practice 1 — PyTorch FashionMNIST Classification**.

---

## 1. Quy Tắc Khoa Học và Tính Trung Thực Học Thuật

1. **Tuyệt đối không bịa đặt số liệu (No Hallucination):** Không được tự tạo hoặc ước lượng số liệu độ chính xác (accuracy), hàm mất mát (loss), thời gian chạy hoặc kết quả kiểm thử. Toàn bộ số liệu trong báo cáo phải được trích xuất tự động từ các tệp nhật ký thực tế (`artifacts/experiments/*.json`, `results.csv`, `final_test.json`).
2. **Không chỉnh sửa số liệu:** Không sửa đổi thủ công kết quả để đạt mục tiêu định trước.
3. **Ngăn chặn rò rỉ dữ liệu (No Data Leakage):**
   - Mọi phép biến đổi có học tham số (fit mean, std) chỉ được thực hiện trên tập Train.
   - Tập Validation và Test tuyệt đối không được tham gia tính toán tham số chuẩn hóa.
   - Không áp dụng kỹ thuật tăng cường dữ liệu (Data Augmentation) trên tập Validation và Test.
4. **Sử dụng Test Set đúng chuẩn:** Tuyệt đối không dùng Test Set để tinh chỉnh siêu tham số, early stopping hoặc lựa chọn mô hình. Test Set chỉ được dùng duy nhất một lần tại Module M8 để đánh giá mô hình cuối cùng.
5. **Tiêu chuẩn lựa chọn mô hình:** Lựa chọn mô hình tốt nhất căn cứ vào **Validation Accuracy** cao nhất; nếu bằng nhau, chọn mô hình có **Validation Loss** thấp hơn.
6. **Tính tái lập (Reproducibility):** Cố định hạt giống ngẫu nhiên (`seed=42`) cho mọi thực nghiệm. Ghi nhận đầy đủ siêu tham số cấu hình trong tệp YAML và artifacts JSON.

---

## 2. Quy Tắc Viết Mã Nguồn (Code Style & Engineering)

1. **Phiên bản & Thư viện:** Python $\ge 3.10$. Sử dụng PyTorch thuần (`torch`, `torchvision`), `numpy`, `matplotlib`, `scikit-learn`. Không sử dụng Keras, TensorFlow, hoặc PyTorch Lightning.
2. **Kiểu dữ liệu & Tài liệu hóa:** Mọi hàm và phương thức bắt buộc có type hints đầy đủ và docstrings ngắn gọn, súc tích.
3. **Chú thích tiếng Việt:** Viết comment tiếng Việt giải thích rõ ràng tại các vị trí bản chất:
   - Cơ chế Autograd và tính toán đồ thị động.
   - Lệnh xóa gradient tích lũy: `optimizer.zero_grad()`.
   - Chế độ đánh giá và vô hiệu hóa đồ thị tính đạo hàm: `model.eval()`, `with torch.no_grad():`.
   - Quy trình phân tách Train/Validation/Test độc lập.
4. **Tính di động (Portability):**
   - Không hard-code đường dẫn tuyệt đối của máy cá nhân. Mọi đường dẫn phải dựa trên `pathlib.Path` tương đối với thư mục gốc dự án.
   - Tự động nhận diện thiết bị tính toán: CUDA $\to$ MPS $\to$ CPU.
5. **Chất lượng code:** Không tạo hàm rác, code giả hoặc khối TODO không cần thiết. Xử lý ngoại lệ đầu vào hợp lý.
6. **Matplotlib:** Luôn thiết lập backend không giao diện (`matplotlib.use('Agg')`), lưu tệp hình ảnh vào thư mục `artifacts/` và gọi `plt.close()` để giải phóng bộ nhớ RAM.
7. **Bảo mật Checkpoint:** Hàm nạp mô hình `torch.load` bắt buộc thiết lập cờ `weights_only=True` và chỉ định `map_location`.

---

## 3. Quy Tắc Kiểm Thử và Definition of Done (DoD)

Một module chỉ được coi là hoàn thành (**Done**) khi đáp ứng đủ 7 tiêu chí:
1. Đã triển khai đầy đủ tính năng trong Module Card.
2. Mã nguồn đã được chạy thực tế (Smoke test) và ghi nhận kết quả.
3. Đã viết kiểm thử đơn vị (`pytest`) và toàn bộ test liên quan đều vượt qua (`PASS`).
4. Đã rà soát code kỹ thuật (Code review).
5. Không chứa khóa bảo mật (secrets, API keys, passwords, .env).
6. Đã cập nhật tài liệu tiến độ (`docs/progress_status.md`) và nhật ký AI (`AI_Usage.md`).
7. Đã được sinh viên hiểu rõ kiến thức, trả lời câu hỏi tự kiểm tra và chấp thuận commit.

*Nghiêm cấm hạ thấp ngưỡng kiểm thử hoặc xóa assertion chỉ để làm pipeline kiểm thử vượt qua.*

---

## 4. Quy Tắc Bảo Mật và Quản Trị Phiên Bản (Git & Security)

1. Không đưa API keys, token, mật khẩu, file `.env` hoặc tệp nhị phân dữ liệu lớn vào Git.
2. Duy trì tệp `.gitignore` chuẩn hóa.
3. Bắt buộc chạy công cụ kiểm tra quy tắc `python scripts/check_rules.py` trước khi thực hiện commit.
4. Không tự động push lên remote repository khi chưa có sự xác nhận của người dùng.
5. Không báo cáo đã commit hoặc push nếu chưa thực hiện thành công trên thực tế.
