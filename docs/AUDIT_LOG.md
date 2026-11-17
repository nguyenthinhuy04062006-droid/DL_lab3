# BÁO CÁO KIỂM TOÁN DỰ ÁN 5 GIAI ĐOẠN (AUDIT_LOG.MD)

## Framework: Description -> Gen -> Smoke -> Run -> Check
Dự án Practice 3: Phân tích cảm xúc & Tinh chỉnh mô hình DistilBERT trên IMDb tuân thủ nghiêm ngặt chuẩn mực: Zero Hallucination, No Data Leakage, 100% Reproducibility (seed=42) và CPU-First.

---

## 1. TỔNG QUAN TÌNH TRẠNG KIỂM ĐỊNH 5 GIAI ĐOẠN

| Giai Đoạn | Tên Giai Đoạn | Trọng Tâm Kiểm Định | Trạng Thái |
|:---:|:---|:---|:---:|
| **1** | **DESCRIPTION** | Đặc tả bài toán Ex 1 & 2, thiết kế Clean Architecture, cam kết tuân thủ RULES.md | **VERIFIED** |
| **2** | **GEN** | Sinh mã nguồn mô-đun hóa `src/`, hoàn thiện Notebook 41 cells và hệ thống docs | **VERIFIED** |
| **3** | **SMOKE** | Tiền kiểm tra cú pháp AST, kiểm tra phụ thuộc thư viện, xử lý triệt để 9 lỗi kỹ thuật | **VERIFIED** |
| **4** | **RUN** | Thực thi suy luận Ex 1, huấn luyện 2 epochs trên CPU (391s), xuất file SafeTensors | **VERIFIED** |
| **5** | **CHECK** | Hậu kiểm định lượng, đối soát số liệu `final_metrics.json`, phân tích lỗi & băm SHA-256 | **APPROVED** |

---

## 2. NHẬT KÝ XỬ LÝ 9 VẤN ĐỀ KỸ THUẬT (REMEDIATION LOG)

| Mã Lỗi | Phân Loại | Vấn Đề Kỹ Thuật (Root Cause) | Giải Pháp Khắc Phục (Remediation) | Trạng Thái |
|:---:|:---|:---|:---|:---:|
| **BUG-01** | *Pipeline* | `trainer.evaluate()` ghi đè khóa `eval_*` trong `log_history` | Khai báo tiền tố phân lập: `untrained_val`, `test` | **FIXED** |
| **BUG-02** | *Library* | `evaluation_strategy` đổi thành `eval_strategy` ở transformers >= 4.41 | Kiểm tra động qua `inspect.signature(TrainingArguments)` | **FIXED** |
| **BUG-03** | *Performance* | Gọi `evaluate.load()` lặp lại trong vòng lặp gây trễ I/O | Cache các đối tượng metric 1 lần duy nhất tại module-level | **FIXED** |
| **BUG-04** | *Console* | Lỗi `UnicodeEncodeError` bảng mã cp1252 trên Windows PowerShell | Bổ sung `sys.stdout.reconfigure(encoding="utf-8")` | **FIXED** |
| **BUG-05** | *Headless* | Matplotlib tìm màn hình desktop trên môi trường CLI | Ép cấu hình backend phi giao diện: `matplotlib.use("Agg")` | **FIXED** |
| **BUG-06** | *Compute* | Đệm tĩnh 128 tokens gây lãng phí phép tính trên token `[PAD]` | Tích hợp Dynamic Padding qua `DataCollatorWithPadding` | **FIXED** |
| **BUG-07** | *UI/CLI* | Mô hình trả về nhãn máy thô (`LABEL_0`, `LABEL_1`) | Khai báo tường minh `id2label` và `label2id` trong config | **FIXED** |
| **BUG-08** | *Security* | Trọng số pickle (`pytorch_model.bin`) có nguy cơ bảo mật RCE | Xuất xưởng checkpoint bằng định dạng an toàn `model.safetensors` | **FIXED** |
| **BUG-09** | *Library* | Tham số `logging_dir` bị loại bỏ trên transformers mới | Gỡ bỏ `logging_dir`, sử dụng cơ chế log mặc định Hugging Face | **FIXED** |

---

## 3. THỰC THI & HUẤN LUYỆN (GIAI ĐOẠN RUN)
1. **Exercise 1 (Pretrained Inference):** Nạp `distilbert-base-uncased-finetuned-sst-2-english`. Bóc tách WordPiece sub-words, Logits thô $[-4.3494, 4.7004]$, tính Softmax thủ công khớp 100% với `pipeline` (sai số số học $= 0.00$).
2. **Exercise 2 (Fine-tuning 7 bước):** Huấn luyện trên 800 mẫu IMDb qua 2 epochs (100 steps) trên CPU mất đúng 391.42s (~6.52 phút):
   - *Untrained Baseline (Val):* Accuracy = 44.00%, Loss = 0.6974.
   - *Epoch 1 (Step 50):* Train Loss = 0.6155, Val Loss = 0.5750, Val Accuracy = 72.00%.
   - *Epoch 2 (Step 100):* Train Loss = 0.5032, Val Loss = 0.5118, Val Accuracy = 76.50% (Đạt đỉnh).
   - *Checkpointing:* Tự động chọn mô hình Epoch 2 lưu vào `outputs/final_model/model.safetensors`.

---

## 4. HẬU KIỂM & ĐỐI SOÁT ĐỊNH LƯỢNG (GIAI ĐOẠN CHECK)
- **Độ chính xác Test Set (200 mẫu):** Đạt **74.50%** (149/200 mẫu), vượt Majority Baseline (52.00%) **+22.50%** và Untrained Baseline (44.00%) **+30.50%**.
- **Khoảng tin cậy Wilson 95%:** $[68.04\%, 80.04\%]$. Cận dưới > Majority Baseline, khẳng định ý nghĩa thống kê cao ($p = 1.03 \times 10^{-6}$ qua McNemar).
- **Ma trận nhầm lẫn đối soát:** TN = 84, FP = 20, FN = 31, TP = 65 (Tổng = 200).
- **Kiểm định Truncation (Fisher's Exact Test):** $p = 0.8153 \gg 0.05 \implies$ Ngưỡng 128 tokens không gây sai lệch thống kê.
- **Tính toàn vẹn mã băm:** Toàn bộ artifacts và checkpoint được băm SHA-256 lưu trữ tại `outputs/reports/run_manifest.json`.

---

## 5. KẾT QUẢ BỘ KIỂM THỬ HỒI QUY TỰ ĐỘNG (`src/verify_all.py`)
Chạy lệnh `python src/verify_all.py` đạt **5/5 bài kiểm thử tuyệt đối (PASSED 100%)**:
1. `[TEST 1/5]` Cấu trúc thư mục, mã nguồn (7 scripts), tài liệu và Jupyter Notebook (41 cells) hợp lệ.
2. `[TEST 2/5]` Toán học predictions.csv khớp 100% (argmax, softmax, cross-entropy loss = 0.5565, confusion matrix).
3. `[TEST 3/5]` Đồng bộ 100% 18 tệp báo cáo, đồ thị PNG và mã băm SHA-256 trong `run_manifest.json`.
4. `[TEST 4/5]` Nạp lại trực tiếp `outputs/final_model/model.safetensors` và suy luận chính xác qua CLI.
5. `[TEST 5/5]` Báo cáo HTML chuẩn tự chứa (`outputs/REPORT.html`) và đạt chuẩn Zero Emoji trên toàn bộ tệp.
