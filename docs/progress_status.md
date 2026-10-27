# Trạng Thái Tiến Độ Dự Án: Practice 1 — PyTorch FashionMNIST Classification

## 1. Tổng Quan Tiến Độ
- **Tên dự án:** Practice 1 — PyTorch FashionMNIST Classification
- **Giai đoạn hiện tại:** **Bước 0 — Plan (Kế hoạch hóa dự án)**
- **Ngày cập nhật:** 2026-10-27
- **Trạng thái:** ⏳ Đã hoàn thành lập kế hoạch Bước 0, chờ sinh viên phê duyệt trước khi chuyển sang Module M0.
- **Quy trình triển khai từng module:** `Describe → Generate → Review → Run (Smoke test) → Test → Refactor → Commit`

---

## 2. Tiêu Chuẩn Hoàn Thành (Definition of Done - DoD)

Một module chỉ được đánh dấu **Done [X]** khi và chỉ khi thỏa mãn đầy đủ 7 tiêu chí sau:
1. **Đáp ứng đầy đủ yêu cầu kỹ thuật:** Mã nguồn hiện thực trọn vẹn các chức năng được mô tả trong Module Card.
2. **Thực thi thực tế (Smoke Test):** Đã chạy thực tế thành công trong môi trường runtime hoặc nêu rõ lý do chính đáng nếu chưa thể chạy.
3. **Kiểm thử tự động (Unit Test):** Các test cases liên quan đã được viết và chạy vượt qua (`PASS` 100%) bằng `pytest`.
4. **Rà soát mã nguồn (Code Review):** Code rõ ràng, chuẩn PEP8, có type hints, docstrings và comment tiếng Việt ở các cơ chế cốt lõi (Autograd, mode switch).
5. **Bảo mật và An toàn:** Không chứa API key, token bí mật hay file nhạy cảm; tải checkpoint sử dụng `weights_only=True`.
6. **Cập nhật tài liệu đồng bộ:** Cập nhật bảng trạng thái `progress_status.md` và ghi nhận nhật ký vào `AI_Usage.md`.
7. **Phê duyệt từ Sinh viên:** Sinh viên xác nhận đã hiểu kiến thức bản chất, trả lời câu hỏi tự kiểm tra lý thuyết và đồng ý commit.

---

## 3. Bảng Theo Dõi Trạng Thái Các Module

| Mã Module | Tên Module | Trọng tâm công việc | Trạng thái | Artifacts Đầu ra Chính |
|:---:|---|---|:---:|---|
| **M0** | **Learn & Environment Setup** | Kiểm tra môi trường, tạo `.gitignore`, `RULES.md`, `scripts/check_rules.py`, `AI_Usage.md`, `requirements.txt` | ✅ Done | `RULES.md`, `scripts/check_rules.py`, `AI_Usage.md`, `requirements.txt` |
| **M1** | **Config & Utils** | Quản lý cấu hình YAML, tiện ích `set_seed()`, `get_device()`, logger | 📋 Todo | `configs/*.yaml`, `src/fashionmnist/utils/`, `src/fashionmnist/config.py`, `tests/test_config.py` |
| **M2** | **Data Pipeline** | Tải FashionMNIST, transforms không rò rỉ dữ liệu, stratified split 80/20, DataLoader | 📋 Todo | `src/fashionmnist/data/`, `tests/test_dataset.py` |
| **M3** | **Models Architecture** | Hiện thực `BaselineMLP` (~203k params) và `SimpleCNN` (~38k params) | 📋 Todo | `src/fashionmnist/models/`, `tests/test_model.py` |
| **M4** | **Train Engine** | Vòng lặp `train()` và `eval()`, loss, optimizer, scheduler, early stopping | 📋 Todo | `src/fashionmnist/training/trainer.py`, `test_training.py` |
| **M5** | **Checkpoint Management** | Cơ chế lưu và nạp an toàn (`weights_only=True`, `map_location`) | 📋 Todo | `src/fashionmnist/training/checkpoint.py`, `test_checkpoint.py` |
| **M6** | **Sanity Checks** | Overfit trên 1 micro-batch (16 mẫu), kiểm tra gradient flow | 📋 Todo | Script kiểm tra tính toàn vẹn |
| **M7** | **Controlled Experiments** | Thực thi 5 thực nghiệm đối chứng có kiểm soát (E1 $\to$ E5) | 📋 Todo | `artifacts/experiments/*.json` |
| **M8** | **Final Evaluation** | Đánh giá mô hình tốt nhất trên Test set duy nhất (10.000 mẫu) | 📋 Todo | `artifacts/metrics/final_test.json`, `classification_report.csv` |
| **M9** | **Visualization & Analysis** | Biểu đồ Loss/Acc, Confusion Matrix, Error Analysis, Per-class F1 | 📋 Todo | `artifacts/figures/*.png` |
| **M10**| **Prediction & CLI Tool** | Công cụ dòng lệnh dự đoán ảnh thời trang đầu vào kèm confidence | 📋 Todo | `scripts/predict.py`, `artifacts/predictions/` |
| **M11**| **Docs & Final Deliverables** | Hoàn thiện `README.md` (10 mục), `AI_Usage.md` (7 mục), `report.md`, `practice1.ipynb` | 📋 Todo | `README.md`, `AI_Usage.md`, `report.md`, `practice1.ipynb` |

---

## 4. Nhật Ký Quyết Định Kỹ Thuật (Decision Log)

| Ngày | Quyết định | Lý do / Cơ sở học thuật | Phân loại |
|---|---|---|---|
| 2026-10-27 | Sử dụng Pure PyTorch thay vì PyTorch Lightning / Keras | Yêu cầu bắt buộc của môn học để hiểu bản chất đồ thị tính toán Autograd và vòng lặp huấn luyện. | Bắt buộc |
| 2026-10-27 | Phân chia Train/Validation theo Stratified Split 80/20 | Đảm bảo tính cân bằng tuyệt đối của 10 lớp trong cả tập Train và Validation, tránh lệch phân phối. | Bắt buộc |
| 2026-10-27 | Chuẩn hóa Z-score với mean/std chỉ tính trên tập Train | Ngăn chặn triệt để hiện tượng rò rỉ dữ liệu (Data Leakage) từ tập Validation và Test. | Bắt buộc |
| 2026-10-27 | Áp dụng Data Augmentation độc quyền cho tập Train | Tăng tính khái quát hóa cho mô hình mà không làm biến dạng dữ liệu kiểm định và kiểm thử. | Bắt buộc |
| 2026-10-27 | Thiết kế SimpleCNN có Global Average Pooling | Giảm mạnh số lượng tham số kết nối đầy đủ (từ hàng trăm ngàn xuống 38k), chống overfitting hiệu quả. | Đề xuất |
| 2026-10-27 | Sử dụng Matplotlib Headless Backend (`Agg`) | Tránh lỗi treo tiến trình hoặc xung đột GUI khi chạy trong môi trường dòng lệnh/headless server. | Bắt buộc |
| 2026-10-27 | Giới hạn phạm vi ở MVP, hoãn Pruning/Quantization | Tập trung hoàn thiện kiến trúc đối chứng, pipeline và độ tin cậy khoa học trước khi tối ưu nâng cao. | Bắt buộc |

---

## 5. Các Vấn Đề Đã Biết & Biện Pháp Kiểm Soát (Known Issues & Mitigations)

1. **Hiển thị tiếng Việt trên Windows Console:**
   - *Vấn đề:* Mặc định code page của console Windows có thể gây lỗi `UnicodeEncodeError`.
   - *Biện pháp:* Khởi tạo `sys.stdout.reconfigure(encoding="utf-8")` và `sys.stderr.reconfigure(encoding="utf-8")` ở tất cả các script entry point.
2. **Khác biệt tốc độ thực thi giữa CPU và GPU:**
   - *Vấn đề:* Môi trường máy tính cục bộ nếu không có GPU rời sẽ huấn luyện trên CPU.
   - *Biện pháp:* Kiến trúc `SimpleCNN` được tinh chỉnh tối giản (~38k tham số, 15 epoch) để thời gian chạy mỗi epoch trên CPU chỉ mất dưới 15-20 giây.
3. **Cảnh báo bảo mật khi tải file checkpoint:**
   - *Vấn đề:* Hàm `torch.load` mặc định có thể tiềm ẩn nguy cơ thực thi mã độc qua pickle.
   - *Biện pháp:* Áp dụng cờ `weights_only=True` và chỉ định `map_location` rõ ràng.

---

## 6. Nhật Ký Thay Đổi Chi Tiết (Changelog)

- **2026-10-27:**
  - **Hoàn thành Bước 0 — Plan:** Soạn thảo và ban hành 3 tài liệu quy chuẩn `docs/project_description.md`, `docs/project_design.md`, `docs/progress_status.md`.
  - **Hoàn thành Module M0 — Learn & Environment Setup:**
    - Ban hành bộ quy tắc bắt buộc [RULES.md](file:///c:/Users/user/Downloads/practice3_huggingface_project/RULES.md).
    - Tạo công cụ tự động kiểm tra quy tắc [scripts/check_rules.py](file:///c:/Users/user/Downloads/practice3_huggingface_project/scripts/check_rules.py) (vượt qua 5/5 bài kiểm tra).
    - Khởi tạo báo cáo sử dụng AI [AI_Usage.md](file:///c:/Users/user/Downloads/practice3_huggingface_project/AI_Usage.md) đúng 7 mục tiêu chuẩn.
    - Cập nhật [.gitignore](file:///c:/Users/user/Downloads/practice3_huggingface_project/.gitignore) loại trừ checkpoint và file nhị phân.
    - Trạng thái Module M0: `✅ Done`. Sẵn sàng cho Module M1.
