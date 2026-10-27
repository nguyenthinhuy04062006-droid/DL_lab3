# BÁO CÁO SỬ DỤNG CÔNG CỤ TRÍ TUỆ NHÂN TẠO (AI USAGE REPORT)

## 1. Mục Đích Sử Dụng AI Trong Dự Án

Việc ứng dụng công cụ Trí tuệ Nhân tạo (AI) trong dự án **Practice 1 — PyTorch FashionMNIST Classification** nhằm các mục đích sau:
- Hỗ trợ xây dựng cấu trúc thư mục dự án chuẩn kỹ thuật phần mềm (Clean Architecture, Modularization).
- Rà soát tính khoa học trong phương pháp luận thực nghiệm (ngăn chặn rò rỉ dữ liệu, đảm bảo tính tái lập).
- Tự động hóa việc sinh các khung mã nguồn mẫu (boilerplate code), bộ kiểm thử đơn vị (`pytest`) và tài liệu kỹ thuật.
- Giải thích chi tiết cơ chế hoạt động của PyTorch (Tensor, Autograd, zero_grad, dynamic graph) nhằm nâng cao hiểu biết lý thuyết của sinh viên.

---

## 2. Các Công Cụ và Mô Hình AI Đã Sử Dụng

| Công cụ / Mô hình | Nhà phát triển | Phiên bản / Môi trường | Mục đích sử dụng |
|---|---|---|---|
| **Gemini 3.8 Flash (Antigravity IDE)** | Google DeepMind | Antigravity AI Coding Assistant | Trợ lý lập trình cặp (Pair-programming), tạo tài liệu, sinh mã nguồn module |

---

## 3. Phạm Vi Hỗ Trợ của AI

AI được sử dụng trong các phạm vi cụ thể:
1. **Thiết kế kiến trúc hệ thống:** Phác thảo cấu trúc thư mục, thiết kế giao tiếp giữa các module (Config $\to$ Data $\to$ Models $\to$ Trainer $\to$ Evaluation $\to$ Visualization).
2. **Khung mã nguồn (Code Scaffolding):** Tạo các lớp mô hình `BaselineMLP`, `SimpleCNN`, cấu trúc lớp `Trainer`, các hàm trực quan hóa Matplotlib và công cụ CLI.
3. **Bộ kiểm tra quy tắc (Rule Checker):** Hỗ trợ viết script `scripts/check_rules.py` để rà soát tự động trước khi commit.
4. **Phân tích và tài liệu hóa:** Tạo dàn ý báo cáo, gợi ý bảng đối chứng thực nghiệm và giải thích cơ chế Autograd bằng tiếng Việt.

---

## 4. Sự Đóng Góp của Sinh Viên (Student Contributions)

> *Quy định bắt buộc: Mục này phải do chính sinh viên tự đánh giá và điền thông tin.*

[sinh viên điền: Liệt kê chi tiết những phần việc, ý tưởng, phân tích hoặc quyết định kỹ thuật do sinh viên trực tiếp thực hiện trong dự án]

---

## 5. Các Chỉnh Sửa và Kiểm Chứng của Sinh Viên (Student Modifications & Verification)

> *Quy định bắt buộc: Mục này phải do chính sinh viên tự đánh giá và điền thông tin.*

[sinh viên điền: Liệt kê các đoạn mã nguồn hoặc tài liệu đã được sinh viên tự tay rà soát, kiểm chứng tính đúng đắn, sửa lỗi logic hoặc tối ưu lại so với gợi ý ban đầu của AI]

---

## 6. Đánh Giá Tính Chuẩn Xác và Đạo Đức Học Thuật

1. **Kiểm chứng độc lập:** Toàn bộ code sinh bởi AI đều phải trải qua kiểm thử đơn vị tự động (`pytest`) và chạy thực nghiệm thực tế trên môi trường máy tính trước khi được chấp thuận.
2. **Tuyệt đối không làm giả dữ liệu:** Các số liệu về Loss, Accuracy, F1-score và biểu đồ trong báo cáo không được để AI ước lượng hoặc bịa đặt, mà phải được trích xuất trực tiếp từ các file nhật ký thực nghiệm (`artifacts/experiments/*.json`).
3. **Bảo mật và an toàn mã nguồn:** Đảm bảo không gửi thông tin bảo mật, khóa API hoặc dữ liệu cá nhân cho các dịch vụ AI bên ngoài; mã nạp checkpoint luôn sử dụng `weights_only=True`.

---

## 7. Bài Học Kinh Nghiệm Khi Làm Việc Với AI

- **AI là công cụ hỗ trợ, con người giữ quyền quyết định:** AI giúp tăng tốc đáng kể việc viết boilerplate và tạo khung kiểm thử, nhưng sinh viên cần phải hiểu tường tận từng dòng code và cơ chế toán học phía sau.
- **Quy trình chặt chẽ giúp kiểm soát chất lượng:** Áp dụng chu trình `Describe → Generate → Review → Run → Test → Refactor → Commit` giúp ngăn ngừa hoàn toàn các lỗi logic tiềm ẩn hoặc code thừa do AI sinh ra.
- **Tính phản biện khoa học:** Cần chủ động kiểm tra các giả định của AI (ví dụ: cách tính trung bình tiền xử lý dữ liệu) để đảm bảo không vi phạm nguyên tắc ngăn ngừa rò rỉ dữ liệu (Data Leakage).
