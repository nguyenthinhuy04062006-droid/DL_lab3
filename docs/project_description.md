# Mô Tả Dự Án: Practice 1 — PyTorch FashionMNIST Classification

## 1. Bài Toán và Bối Cảnh (Problem Statement & Context)

### 1.1 Bối Cảnh Học Thuật
Dự án **Practice 1 — PyTorch FashionMNIST Classification** là bài thực hành nền tảng trong khuôn khổ chương trình đào tạo Học sâu (Deep Learning) và Thị giác máy tính (Computer Vision). 

Mục tiêu trọng tâm là làm chủ thư viện **PyTorch thuần (Pure PyTorch)** mà không phụ thuộc vào các framework bọc cấp cao (như Keras hay PyTorch Lightning), giúp sinh viên nắm vững bản chất toán học và cơ chế vận hành nội tại của các cấu trúc:
- **Tensor và Autograd:** Đồ thị tính toán động (*dynamic computational graph*), cơ chế tự động tích lũy gradient (`parameter.grad`), giải thuật lan truyền ngược (*backpropagation*).
- **Dataset và DataLoader:** Quy trình chuẩn hóa dữ liệu, nạp lô (*batching*), xáo trộn (*shuffling*), và tối ưu luồng dữ liệu I/O giữa CPU và GPU.
- **Kiến trúc Neural Network:** Kế thừa `torch.nn.Module`, thiết lập các khối tầng mạng nơ-ron kết nối đầy đủ (MLP) và mạng nơ-ron tích chập (CNN).
- **Vòng lặp Huấn luyện (Training Loop):** Trình tự chuẩn mực của `model.train()`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`, cùng chế độ kiểm định `model.eval()` và ngữ cảnh vô hiệu hóa gradient `torch.no_grad()`.

### 1.2 Bài Toán Phân Loại Ảnh Thời Trang
Tập dữ liệu số viết tay MNIST truyền thống quá đơn giản (các mô hình tuyến tính đơn giản dễ dàng đạt >97% accuracy). Do đó, dự án sử dụng **FashionMNIST** (Zalando Research) nhằm mang lại độ phức tạp cao hơn, phản ánh bài toán phân loại sản phẩm thị giác thực tế.

Bài toán được định nghĩa là **Phân loại ảnh đa lớp nhị phân/rời rạc (10-class Single-label Image Classification)**:
Cho một ảnh đầu vào $x \in \mathbb{R}^{1 \times 28 \times 28}$, mô hình học ánh xạ hàm xác suất:
$$P(y = c \mid x; \theta), \quad c \in \{0, 1, \dots, 9\}$$
sao cho nhãn dự đoán $\hat{y} = \arg\max_{c} P(y = c \mid x; \theta)$ khớp với nhãn thực tế $y$.

---

## 2. Dữ Liệu Thực Nghiệm (FashionMNIST Dataset)

- **Số lượng mẫu:** Tổng cộng 70.000 ảnh grayscale đơn kênh ($1 \times 28 \times 28$ pixel).
- **Phân chia tiêu chuẩn đề bài:**
  - **Development set (60.000 mẫu):** Phân chia thành tập Huấn luyện (Train set: 48.000 mẫu, 80%) và tập Kiểm định (Validation set: 12.000 mẫu, 20%) theo phương pháp phân tầng (*Stratified Split*) nhằm đảm bảo phân phối 10 lớp hoàn toàn đồng đều giữa các tập.
  - **Test set (10.000 mẫu):** Được giữ độc lập hoàn toàn, chỉ nạp để đánh giá một lần duy nhất tại bước kiểm thử cuối cùng.
- **10 Danh mục nhãn (Classes):**
  | Mã nhãn | Tên lớp (Tiếng Anh) | Tên lớp (Tiếng Việt) |
  |:---:|---|---|
  | `0` | T-shirt/top | Áo thun / Áo cộc tay |
  | `1` | Trouser | Quần dài |
  | `2` | Pullover | Áo len chui đầu |
  | `3` | Dress | Váy liền |
  | `4` | Coat | Áo khoác ngoài |
  | `5` | Sandal | Dép quai hậu (xăng-đan) |
  | `6` | Shirt | Áo sơ mi |
  | `7` | Sneaker | Giày thể thao |
  | `8` | Bag | Túi xách |
  | `9` | Ankle boot | Ủng / Bốt cổ thấp |

---

## 3. Mục Tiêu Dự Án (Project Objectives)

1. **Làm chủ Pure PyTorch Workflow:** Thiết kế đầy đủ quy trình học sâu từ đầu bằng mã nguồn rõ ràng, có chú thích tiếng Việt giải thích cơ chế Autograd và kiểm định.
2. **Kiến trúc Đối chứng Đối sánh (Comparative Model Architectures):**
   - **BaselineMLP (Mạng Perceptron Đa tầng):** Mạng kết nối đầy đủ Flatten $\to$ Linear(784, 256) $\to$ ReLU $\to$ Linear(256, 10), khoảng ~203.530 tham số.
   - **SimpleCNN (Mạng Tích chập Gọn nhẹ):** 2 khối Convolution (Conv2d, BatchNorm, ReLU, MaxPool) $\to$ Global Average Pooling $\to$ Linear(64, 256) $\to$ Dropout(0.3) $\to$ Linear(256, 10), khoảng ~38.122 tham số.
   - **Mục tiêu khoa học:** Chứng minh tính hiệu quả của thiên kiến cảm nạp (*inductive bias - translation invariance*) trong CNN: SimpleCNN giảm hơn 80% số tham số nhưng đạt độ chính xác vượt trội so với BaselineMLP.
3. **Thực nghiệm Có Kiểm Soát (Controlled Experiments):** Thiết kế và chạy chuỗi 5 thực nghiệm đơn biến (kiến trúc MLP vs CNN, bộ tối ưu SGD vs Adam, số epoch 10 vs 15, learning rate 0.001 vs 0.01, batch size 128 vs 256) nhằm ghi nhận khách quan sự ảnh hưởng của siêu tham số.
4. **Trực quan hóa và Phân tích Sai Số (Visualization & Error Analysis):**
   - Vẽ đường cong Loss và Accuracy (Train vs Val) qua từng epoch.
   - Xuất Ma trận nhầm lẫn (Confusion Matrix) dạng số đếm và dạng chuẩn hóa tỉ lệ.
   - Phân tích định tính các ca phân loại sai (Error Analysis) nhằm làm rõ nguyên nhân nhầm lẫn thị giác giữa các lớp áo có đặc trưng hình dáng tương đồng (Shirt vs T-shirt, Shirt vs Coat, Pullover vs Coat).
5. **Đảm bảo Tính Tái Lập và Chuẩn Hóa Phần Mềm:** Cố định seed ngẫu nhiên (`seed=42`), cấu hình tập trung qua tệp YAML, lưu/tải checkpoint an toàn với `weights_only=True`, và tổ chức test suite tự động bằng `pytest`.

---

## 4. Đầu Vào và Đầu Ra Hệ Thống (Input & Output Specification)

| Thành phần | Định dạng / Kích thước | Mô tả kỹ thuật |
|---|---|---|
| **Input Dữ liệu Huấn luyện** | `torch.Tensor` $(B, 1, 28, 28)$, kiểu `torch.float32` | Batch ảnh xám đã chuẩn hóa Z-score với tham số $(\mu, \sigma)$ tính từ tập Train. |
| **Input Nhãn Mục tiêu** | `torch.Tensor` $(B,)$, kiểu `torch.int64` (Long) | Chỉ số lớp nguyên rời rạc từ $0$ đến $9$. |
| **Output Suy luận Thô** | `torch.Tensor` $(B, 10)$, kiểu `torch.float32` | Logits chưa qua chuẩn hóa xác suất của 10 lớp phân loại. |
| **Output Xác suất** | `torch.Tensor` $(B, 10)$, kiểu `torch.float32` | Phân phối xác suất sau khi áp dụng hàm `F.softmax(logits, dim=1)`. |
| **Output Dự đoán Lớp** | `torch.Tensor` $(B,)$, kiểu `torch.int64` | Nhãn có xác suất cao nhất $\hat{y} = \arg\max_{c} \text{logits}$. |
| **Artifacts Mô hình** | Checkpoint `.pt` (ví dụ `best_model.pt`) | Chứa `model_state_dict`, `optimizer_state_dict`, epoch, và validation accuracy. |
| **Artifacts Báo cáo & Đồ thị** | File ảnh `.png`, file dữ liệu `.csv`, `.json` | `training_curves.png`, `confusion_matrix.png`, `classification_report.csv`, `error_analysis.csv`. |

---

## 5. Tính Năng Chính Của Hệ Thống (Key Features)

1. **Bộ nạp dữ liệu không rò rỉ (Strict Leak-free Data Pipeline):** Tính toán trung bình ($\mu$) và độ lệch chuẩn ($\sigma$) chỉ trên tập Train; áp dụng Data Augmentation (RandomHorizontalFlip, RandomCrop) độc quyền cho tập Train; giữ nguyên vẹn tập Validation và Test.
2. **Kiểm tra tính toàn vẹn sơ bộ (Sanity Checks Engine):** Kiểm tra khả năng overfit trên một micro-batch (16 mẫu) đạt loss $\approx 0$ và kiểm tra dòng chảy gradient không bị suy biến (*vanishing/exploding gradients*).
3. **Bộ huấn luyện module hóa (Modular Trainer):** Tích hợp vòng lặp huấn luyện, cơ chế theo dõi Validation Accuracy, cơ chế Dừng sớm (*Early Stopping*), và lưu mô hình tốt nhất an toàn.
4. **Công cụ dòng lệnh suy luận (Inference CLI Tool):** Cho phép người dùng chạy dự đoán trên một ảnh thời trang bất kỳ hoặc một mẫu ngẫu nhiên từ tập Test, xuất bảng điểm số Top-3 độ tin cậy.
5. **Hệ thống cấu hình hóa thực nghiệm (YAML Configurations):** Toàn bộ tham số được tách biệt hoàn toàn khỏi mã nguồn thông qua `configs/default.yaml` và `configs/experiments.yaml`.

---

## 6. Ràng Buộc Kỹ Thuật (Constraints)

1. **Ràng buộc framework:** Sử dụng **Pure PyTorch (`torch`, `torch.nn`, `torch.optim`)** và **Torchvision (`torchvision.datasets`, `torchvision.transforms`)**. Tuyệt đối không dùng framework cấp cao như Keras, PyTorch Lightning, FastAI.
2. **Ràng buộc an toàn dữ liệu (Data Leakage Prohibition):**
   - Không được dùng tập Test trong bất kỳ khâu nào của tiền xử lý, huấn luyện hay chọn mô hình.
   - Tập Test chỉ được nạp và suy luận đúng một lần tại Module M8 (Final Evaluation).
3. **Ràng buộc an toàn Checkpoint:** Khi gọi `torch.load()`, bắt buộc dùng cờ `weights_only=True` (chuẩn PyTorch $\ge 2.0$) và chỉ định `map_location` phù hợp phần cứng.
4. **Ràng buộc hiển thị đồ thị:** Matplotlib phải chạy ở chế độ Headless Backend (`matplotlib.use('Agg')`) để không gây block hoặc treo tiến trình trên môi trường dòng lệnh/headless server.
5. **Ràng buộc tính trung thực khoa học:** Mọi số liệu trong báo cáo và bảng tổng hợp bắt buộc phải được xuất từ kết quả thực nghiệm thực tế, tuyệt đối không bịa đặt số liệu.

---

## 7. Phạm Vi Triển Khai (Scope)

### 7.1 Trong Phạm Vi (In-Scope - MVP)
- Hoàn thiện trọn vẹn quy trình 12 module từ **M0** đến **M11**.
- Hiện thực hai mô hình `BaselineMLP` và `SimpleCNN`.
- Chạy 5 thực nghiệm có kiểm soát (E1 đến E5) trên tập dữ liệu FashionMNIST.
- Trích xuất toàn bộ đồ thị, ma trận nhầm lẫn, bảng chỉ số (Accuracy, Precision, Recall, Macro/Weighted F1), và phân tích sai số (Error Analysis).
- Xây dựng CLI script (`predict.py`, `train.py`, `evaluate.py`) và notebook học thuật `notebooks/practice1.ipynb`.
- Bộ unit test tự động bằng `pytest` bao phủ Dataset, Model, Training loop và Checkpoint.

### 7.2 Ngoài Phạm Vi (Out-of-Scope - Các Đề Xuất Nâng Cao Sau MVP)
- Không áp dụng Model Pruning (cắt tỉa trọng số nơ-ron).
- Không áp dụng Model Quantization (lượng tử hóa INT8/FP16).
- Không áp dụng Knowledge Distillation (chưng cất tri thức mô hình lớn sang nhỏ).
- Không xây dựng giao diện Web (Streamlit/FastAPI) hoặc triển khai đám mây (Cloud Deployment) trong phạm vi bài thực hành này.
