# Thiết Kế Hệ Thống: Practice 1 — PyTorch FashionMNIST Classification

## 1. Kiến Trúc Tổng Thể (System Architecture) & Luồng Dữ Liệu (Data Flow)

### 1.1 Sơ Đồ Phân Lớp Module Hóa (Modular Layered Architecture)
Hệ thống được thiết kế theo các tầng chức năng độc lập, phân tách rõ ràng trách nhiệm giữa cấu hình, nạp dữ liệu, kiến trúc mô hình, huấn luyện, đánh giá và giao diện người dùng:

```text
[ configs/ ] (default.yaml, experiments.yaml)
     │
     ▼
[ src/fashionmnist/utils/ ] (seed, device, logger, timer)
     │
     ▼
[ src/fashionmnist/data/ ] (dataset, dataloader, stratified split, transforms)
     │
     ▼
[ src/fashionmnist/models/ ] (BaselineMLP, SimpleCNN, count_parameters)
     │
     ▼
[ src/fashionmnist/training/ ] (trainer, loss, optimizer, scheduler, early_stopping)
     │
     ├──► [ src/fashionmnist/training/checkpoint.py ] (safe save/load)
     │
     ▼
[ src/fashionmnist/evaluation/ ] (metrics, classification report, error analysis)
     │
     ▼
[ src/fashionmnist/visualization/ ] (headless Agg plots: curves, confusion matrix, errors)
     │
     ▼
[ scripts/ & notebooks/ ] (CLI pipelines & practice1.ipynb)
```

### 1.2 Luồng Dữ Liệu Chi Tiết (Data Flow Pipeline)
1. **Khởi tạo & Cấu hình:** `load_config()` đọc file YAML, khởi tạo `set_seed(42)` và cấp phát `get_device()`.
2. **Nạp & Phân tách Dữ liệu:** `FashionMNIST` được tải về $\to$ Phân tầng 80% Train (48.000 mẫu) và 20% Val (12.000 mẫu).
3. **Tiền xử lý (Transforms Pipeline):**
   - **Train Pipeline:** `ToTensor()` $\to$ `Normalize(mean=0.2860, std=0.3530)` $\to$ `RandomHorizontalFlip(p=0.5)` $\to$ `RandomCrop(28, padding=2)`.
   - **Val & Test Pipeline:** `ToTensor()` $\to$ `Normalize(mean=0.2860, std=0.3530)` (Không Augmentation).
4. **Vòng lặp Huấn luyện (Forward & Backward Loop):**
   - Batch $(B, 1, 28, 28) \to$ Model $\to$ Logits $(B, 10)$.
   - $\text{Loss} = \text{CrossEntropyLoss}(\text{Logits}, \text{Labels})$.
   - `optimizer.zero_grad()` $\to$ `loss.backward()` $\to$ `optimizer.step()`.
5. **Kiểm định & Lưu Checkpoint:**
   - Đánh giá Validation Accuracy cuối mỗi epoch trong ngữ cảnh `torch.no_grad()`.
   - Nếu $\text{val\_acc} > \text{best\_val\_acc}$, lưu `best_model.pt` qua `checkpoint.py`.
6. **Đánh giá Cuối & Trực quan:** Nạp `best_model.pt` $\to$ Đánh giá 1 lần duy nhất trên Test set $\to$ Xuất biểu đồ và báo cáo.

---

## 2. Cấu Trúc Thư Mục Dự Án (Directory Layout)

```text
fashionmnist-deep-learning/
├── .gitignore                          # Loại trừ file tạm, cache, checkpoint dung lượng lớn
├── pyproject.toml                      # Cấu hình project, linter, pytest
├── requirements.txt                    # Danh sách thư viện và phiên bản tối thiểu
├── README.md                           # Tài liệu tổng quan 10 mục theo quy chuẩn
├── AI_Usage.md                         # Báo cáo sử dụng AI 7 mục (giữ placeholder sinh viên điền)
├── RULES.md                            # Quy tắc chuẩn khoa học, code style và kiểm thử
├── configs/
│   ├── default.yaml                    # Siêu tham số mặc định (seed=42, split=0.8/0.2, lr=1e-3)
│   └── experiments.yaml                # Cấu hình 5 thực nghiệm đối chứng
├── docs/
│   ├── project_description.md          # Mô tả bài toán, mục tiêu, input/output, phạm vi
│   ├── project_design.md               # Kiến trúc, data flow, M0-M11, interfaces, rủi ro
│   └── progress_status.md              # Bảng theo dõi tiến độ, DoD, decision log
├── report/
│   └── report.md                       # Báo cáo học thuật chi tiết dựa trên số liệu thực nghiệm
├── notebooks/
│   └── practice1.ipynb                 # Notebook thực hành hoàn chỉnh, có giải thích lý thuyết
├── scripts/
│   ├── check_rules.py                  # Script kiểm tra quy tắc trước commit
│   ├── train.py                        # CLI huấn luyện mô hình theo config
│   ├── evaluate.py                     # CLI đánh giá mô hình trên test set
│   └── predict.py                      # CLI dự đoán ảnh thời trang đầu vào
├── src/
│   └── fashionmnist/
│       ├── __init__.py
│       ├── config.py                   # Dataclass load và parse YAML config
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py              # Download và Dataset wrapper, Stratified Split
│       │   └── transforms.py           # Tiền xử lý, chuẩn hóa không rò rỉ dữ liệu
│       ├── models/
│       │   ├── __init__.py
│       │   └── network.py              # BaselineMLP và SimpleCNN, đếm tham số
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py              # Huấn luyện, valid loop, early stopping
│       │   └── checkpoint.py           # Lưu và tải an toàn với weights_only=True
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py              # Tính accuracy, precision, recall, macro F1
│       │   └── error_analysis.py       # Trích xuất top mẫu phân loại sai
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── plots.py                # Biểu đồ loss/acc, confusion matrix, ảnh mẫu
│       └── utils/
│           ├── __init__.py
│           ├── device.py               # Tự động chọn CPU/CUDA/MPS
│           └── seed.py                 # Cố định seed cho torch, numpy, random
├── tests/
│   ├── test_checkpoint.py              # Kiểm thử lưu và nạp checkpoint
│   ├── test_dataset.py                 # Kiểm thử kích thước tensor, split không chồng lấn
│   ├── test_model.py                   # Kiểm thử shape output của MLP và CNN
│   └── test_training.py                # Kiểm thử quá trình tính loss và cập nhật trọng số
└── artifacts/
    ├── checkpoints/                    # Lưu best_model.pt
    ├── figures/                        # Lưu biểu đồ phân tích PNG
    ├── metrics/                        # Lưu báo cáo JSON và CSV
    └── predictions/                    # Lưu ảnh dự đoán từ CLI
```

---

## 3. Danh Sách Chi Tiết 12 Module (M0 đến M11)

### Module M0: Learn & Environment Setup
- **Mục tiêu:** Nắm vững kiến thức nền tảng (Tensor, Autograd, Dataset, DataLoader), kiểm tra môi trường Python $\ge 3.10$, tạo `.gitignore`, `RULES.md`, `scripts/check_rules.py`, `AI_Usage.md` và `requirements.txt`.
- **Input:** Đặc tả yêu cầu bài toán từ giảng viên.
- **Output:** Khung cấu trúc thư mục, tệp quy tắc, môi trường thực thi chuẩn.
- **Interface:** Terminal command, script `python scripts/check_rules.py`.
- **Ràng buộc:** Chưa viết mã nghiệp vụ Deep Learning ở module này.

### Module M1: Config & Utilities
- **Mục tiêu:** Quản lý tập trung toàn bộ siêu tham số thực nghiệm qua YAML, xây dựng các tiện ích kiểm soát hạt giống ngẫu nhiên và cấp phát thiết bị phần cứng.
- **Input:** Đường dẫn `configs/default.yaml` và `configs/experiments.yaml`.
- **Output:** Đối tượng cấu hình `Config`, hàm `set_seed()`, hàm `get_device()`.
- **Interface:**
  ```python
  def load_config(path: str) -> Config: ...
  def set_seed(seed: int = 42) -> None: ...
  def get_device() -> torch.device: ...
  ```
- **Ràng buộc:** Seed phải áp dụng đồng bộ cho `random`, `numpy`, `torch.manual_seed` và `torch.cuda.manual_seed_all`.

### Module M2: Data Pipeline
- **Mục tiêu:** Tải dataset FashionMNIST, bọc trong PyTorch `Dataset`, áp dụng transforms chống rò rỉ dữ liệu, chia tập phân tầng 80/20 (Train/Val) và đóng gói vào `DataLoader`.
- **Input:** Tệp dữ liệu gốc FashionMNIST.
- **Output:** `train_loader`, `val_loader`, `test_loader`.
- **Interface:**
  ```python
  def get_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]: ...
  def get_transforms(augment: bool = False, mean: float = 0.2860, std: float = 0.3530) -> transforms.Compose: ...
  ```
- **Ràng buộc:** Tham số chuẩn hóa chỉ tính trên tập Train. Tuyệt đối không áp dụng Data Augmentation cho tập Validation và Test.

### Module M3: Models Architecture
- **Mục tiêu:** Hiện thực hai lớp mạng nơ-ron kế thừa `nn.Module`:
  1. `BaselineMLP`: Flatten $\to$ Linear(784, 256) $\to$ ReLU $\to$ Linear(256, 10). Tổng ~203.530 tham số.
  2. `SimpleCNN`: 2 khối ConvBlock (Conv2d, BatchNorm, ReLU, MaxPool) $\to$ Global Average Pooling $\to$ Linear(64, 256) $\to$ Dropout(0.3) $\to$ Linear(256, 10). Tổng ~38.122 tham số.
- **Input:** Tensor batch ảnh $(B, 1, 28, 28)$.
- **Output:** Tensor logits $(B, 10)$.
- **Interface:**
  ```python
  class BaselineMLP(nn.Module):
      def forward(self, x: torch.Tensor) -> torch.Tensor: ...
  class SimpleCNN(nn.Module):
      def forward(self, x: torch.Tensor) -> torch.Tensor: ...
  def build_model(cfg: ModelConfig) -> nn.Module: ...
  def count_parameters(model: nn.Module) -> int: ...
  ```
- **Ràng buộc:** Kiểm tra shape đầu vào nghiêm ngặt, có type hints và docstrings.

### Module M4: Train Engine
- **Mục tiêu:** Xây dựng vòng lặp huấn luyện chuẩn xác:
  - `model.train()`: `optimizer.zero_grad()` $\to$ `outputs = model(inputs)` $\to$ `loss = criterion(outputs, labels)` $\to$ `loss.backward()` $\to$ `optimizer.step()`.
  - `model.eval()` + `with torch.no_grad()`: Tính validation loss và validation accuracy.
  - Tích hợp cơ chế Early Stopping theo dõi validation accuracy.
- **Input:** Model, `train_loader`, `val_loader`, Optimizer, Criterion, Scheduler, TrainingConfig.
- **Output:** Lịch sử huấn luyện `history: Dict[str, List[float]]`, mô hình đã huấn luyện.
- **Interface:**
  ```python
  class Trainer:
      def fit(self) -> Dict[str, List[float]]: ...
      def train_epoch(self) -> Tuple[float, float]: ...
      def validate(self) -> Tuple[float, float]: ...
  ```
- **Ràng buộc:** Bắt buộc có chú thích tiếng Việt giải thích lý do phải gọi `zero_grad()`, `backward()`, `step()`, `eval()`, và `no_grad()`.

### Module M5: Checkpoint Management
- **Mục tiêu:** Cung cấp cơ chế lưu trữ và phục hồi trạng thái mô hình an toàn, tránh lỗ hổng thực thi mã độc.
- **Input:** Mô hình, optimizer, epoch, metrics, đường dẫn file `.pt`.
- **Output:** File checkpoint trên đĩa; khôi phục trọng số vào mô hình.
- **Interface:**
  ```python
  def save_checkpoint(path: Path, model: nn.Module, optimizer: optim.Optimizer, epoch: int, val_acc: float) -> None: ...
  def load_checkpoint(path: Path, model: nn.Module, map_location: torch.device) -> Dict[str, Any]: ...
  ```
- **Ràng buộc:** Sử dụng cờ `weights_only=True` khi gọi `torch.load()`.

### Module M6: Sanity Checks
- **Mục tiêu:** Xác nhận tính đúng đắn của pipeline trước khi huấn luyện dài:
  1. Overfit trên 1 batch cực nhỏ (16 mẫu): Mô hình phải đạt loss $\approx 0$ và accuracy 100%.
  2. Kiểm tra dòng gradient (Gradient Flow): Đảm bảo gradient được truyền đều qua các tầng, không bị triệt tiêu (*vanishing*) hay bùng nổ (*exploding*).
- **Input:** Mô hình chưa huấn luyện và 1 micro-batch dữ liệu.
- **Output:** Trạng thái PASS / FAIL kèm biểu đồ gradient norm.
- **Interface:** `def run_sanity_checks(model: nn.Module, sample_batch: Tuple[torch.Tensor, torch.Tensor]) -> bool: ...`
- **Ràng buộc:** Không chạy trên toàn bộ dữ liệu ở bước này.

### Module M7: Controlled Experiments
- **Mục tiêu:** Thực hiện 5 thực nghiệm có đối chứng khoa học với biến số độc lập:
  - **E1 (`baseline_mlp`):** MLP + SGD (lr=0.01, epochs=10, bs=128)
  - **E2 (`mlp_adam`):** MLP + Adam (lr=0.001, epochs=15, bs=128)
  - **E3 (`exp_cnn_small`):** CNN + Adam (lr=0.001, epochs=15, bs=128)
  - **E4 (`exp_lr_01`):** CNN + Adam (lr=0.01, epochs=15, bs=128)
  - **E5 (`exp_batch_256`):** CNN + Adam (lr=0.001, epochs=15, bs=256)
- **Input:** Cấu hình trong `configs/experiments.yaml`.
- **Output:** Các file nhật ký kết quả `artifacts/experiments/*.json`.
- **Interface:** `def run_experiments(config_path: str) -> pd.DataFrame: ...`
- **Ràng buộc:** Tuyệt đối không dùng Test set trong quá trình thực nghiệm so sánh.

### Module M8: Final Evaluation
- **Mục tiêu:** Nạp mô hình đạt Validation Accuracy cao nhất từ M7 và đánh giá DUY NHẤT một lần trên Test set (10.000 mẫu).
- **Input:** `best_model.pt`, `test_loader`.
- **Output:** File `artifacts/metrics/final_test.json`, `classification_report.csv` (Accuracy, Precision, Recall, Macro F1, Per-class F1).
- **Interface:** `def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict[str, Any]: ...`
- **Ràng buộc:** Không được điều chỉnh siêu tham số sau khi đã nhìn thấy kết quả Test.

### Module M9: Visualization & Analysis
- **Mục tiêu:** Tự động sinh và lưu trữ toàn bộ đồ thị phân tích:
  - Đồ thị đường cong Training & Validation Loss/Accuracy.
  - Ma trận nhầm lẫn (Confusion Matrix): Dạng số đếm và dạng chuẩn hóa tỉ lệ.
  - Biểu đồ cột so sánh F1-Score của 10 lớp thời trang.
  - Lưới ảnh phân tích sai số (Error Analysis): Các mẫu phân loại sai tự tin nhất.
- **Input:** Lịch sử huấn luyện, mảng dự đoán và nhãn thực tế trên Test set.
- **Output:** Các file ảnh PNG trong `artifacts/figures/`.
- **Interface:**
  ```python
  def plot_learning_curves(history: Dict, save_path: Path) -> None: ...
  def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path) -> None: ...
  def plot_error_analysis(images, y_true, y_pred, probs, save_path: Path) -> None: ...
  ```
- **Ràng buộc:** Matplotlib sử dụng backend `Agg`, gọi `plt.close(fig)` sau khi lưu.

### Module M10: Prediction & CLI Tool
- **Mục tiêu:** Xây dựng công cụ dòng lệnh hoàn chỉnh cho phép dự đoán nhãn lớp của ảnh thời trang đầu vào.
- **Input:** File ảnh tùy ý (`--image path/to/img.png`) hoặc mẫu ngẫu nhiên (`--random-test`).
- **Output:** In bảng Top-3 xác suất ra console và lưu ảnh dự đoán vào `artifacts/predictions/`.
- **Interface:**
  `python scripts/predict.py --image <path> --checkpoint artifacts/checkpoints/best_model.pt`
- **Ràng buộc:** Xử lý ngoại lệ file hỏng hoặc sai định dạng bằng thông báo lỗi tiếng Việt thân thiện.

### Module M11: Documentation & Final Deliverables
- **Mục tiêu:** Hoàn thiện và đóng gói toàn bộ sản phẩm bàn giao học thuật:
  - `README.md`: Đầy đủ 10 mục tiêu chuẩn, kèm ví dụ input/output từ lần chạy thật.
  - `AI_Usage.md`: Đầy đủ 7 mục theo chuẩn đạo đức học thuật (giữ nguyên placeholder sinh viên điền).
  - `report/report.md`: Báo cáo khoa học dựa trên số liệu thực nghiệm thực tế.
  - `notebooks/practice1.ipynb`: Notebook hoàn chỉnh, có markdown giải thích lý thuyết.
  - Toàn bộ test suite vượt qua (`pytest tests/`).
- **Input:** Toàn bộ mã nguồn, tài liệu và artifacts đã hoàn thiện.
- **Output:** Bộ tài liệu hoàn chỉnh sẵn sàng nộp và bảo vệ.

---

## 4. Quản Lý Rủi Ro và Biện Pháp Giảm Thiểu (Risks & Mitigations)

| Rủi ro Kỹ thuật / Học thuật | Tác động | Biện pháp Giảm thiểu Chặt chẽ |
|---|---|---|
| **Rò rỉ dữ liệu (Data Leakage)** | Điểm số ảo, mô hình học vẹt, mất tính khoa học | Tính $(\mu, \sigma)$ duy nhất trên tập Train; cấm Augmentation trên Val/Test; cô lập hoàn toàn tập Test đến Module M8. |
| **Tính không xác định (Non-determinism)** | Kết quả thực nghiệm thay đổi giữa các lần chạy | Cố định `seed=42` cho `random`, `numpy`, `torch`; bật cờ `torch.backends.cudnn.deterministic = True`. |
| **Lỗ hổng bảo mật Checkpoint** | Nguy cơ thực thi mã độc qua pickle | Luôn sử dụng cờ `weights_only=True` khi nạp checkpoint với `torch.load()`. |
| **Treo tiến trình do GUI Matplotlib** | Script bị dừng chờ hiển thị cửa sổ | Bắt buộc khai báo `matplotlib.use('Agg')` ở đầu tất cả các script và module vẽ hình. |
| **Quá tải tài nguyên khi chạy trên CPU** | Thời gian huấn luyện kéo dài | Thiết kế `SimpleCNN` siêu gọn (~38k tham số), chọn batch size 128 tối ưu cho luồng dữ liệu trên CPU/RAM. |
| **Bịa đặt số liệu (Hallucination)** | Vi phạm nghiêm trọng liêm chính học thuật | Toàn bộ số liệu trong báo cáo và bảng tổng hợp bắt buộc phải được đọc và trích xuất tự động từ các tệp nhật ký thực tế (`results.csv`, `*.json`). |

---

## 5. Kế Hoạch Các Cột Mốc (Milestones)

- **Milestone 1 (Nền tảng & Dữ liệu - M0 đến M2):** Thiết lập môi trường, hoàn thiện config và pipeline dữ liệu chuẩn không rò rỉ.
- **Milestone 2 (Mô hình & Huấn luyện cốt lõi - M3 đến M6):** Hiện thực BaselineMLP và SimpleCNN, xây dựng Trainer, cơ chế Checkpoint, và vượt qua Sanity Checks.
- **Milestone 3 (Thực nghiệm & Đánh giá - M7 đến M9):** Hoàn thành 5 thực nghiệm đối chứng, đánh giá Test set mô hình tốt nhất, xuất toàn bộ đồ thị và phân tích sai số.
- **Milestone 4 (Sản phẩm Bàn giao - M10 đến M11):** Hoàn thiện CLI predict tool, đồng bộ notebook `practice1.ipynb`, hoàn thiện `README.md`, `AI_Usage.md` và `report.md`.
