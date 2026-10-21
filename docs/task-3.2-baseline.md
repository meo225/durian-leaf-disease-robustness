# Task 3.2. Huấn luyện và Đánh giá Baseline Models

Báo cáo nhanh cho Task 3.2 theo kế hoạch đồ án CS406. File này theo dõi tiến độ và tổng hợp kết quả của 3 kiến trúc baseline được khóa từ [Phase 3 Baseline Candidates](phase-3-baseline-candidates.md).

Mục tiêu: Thiết lập mức hiệu năng chuẩn (Clean Reference) trên ảnh sạch trước khi chuyển sang đánh giá độ bền vững dưới ảnh suy giảm chất lượng ở Phase 4.

---

## 1. Thiết lập thực nghiệm thống nhất

* **Dataset:** DurianLDD (4.437 ảnh, 5 lớp).
* **Phân chia dữ liệu:** `manifests/split_manifest.csv` (Train: 3.105 ảnh, Validation: 444 ảnh, Test: 888 ảnh).
* **Quy tắc bảo mật tập dữ liệu:** Tập Test được **khóa hoàn toàn**, không mở trong quá trình train, tinh chỉnh siêu tham số hay lựa chọn checkpoint.
* **Tiền xử lý & Augmentation:** Resize 224×224, chuẩn hóa ImageNet, Random Horizontal/Vertical Flip, Random Rotation ±15°.
* **Siêu tham số huấn luyện:**
  * Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-2`)
  * Batch size: 32
  * Epochs tối đa: 20 (Early stopping patience: 5)
  * Metric chọn checkpoint: **Validation Macro F1**

---

## 2. Bảng tổng hợp so sánh 3 mô hình Baseline

| Thứ tự | Mô hình | Họ kiến trúc | `timm` backbone | Tham số | Val Macro F1 | Val Balanced Acc | Val Loss | Thời gian (s) | GPU | Trạng thái |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **ResNet-50** | CNN đối chứng | `resnet50` | 23.5M | **0.9679** | 0.9676 | 0.1377 | 1262.1 (~21m) | Tesla T4 | **Hoàn thành** |
| **2** | **ConvNeXt-Tiny** | Modern Vision | `convnext_tiny` | ~28.6M | — | — | — | — | — | *Chờ thực nghiệm* |
| **3** | **MobileNetV3-Small** | Lightweight | `mobilenet_v3_small` (torchvision) | 1.52M | **0.9485** | 0.9484 | 0.2841 | 751.1 (~13m) | Tesla T4 | **Hoàn thành** |

---

## 3. Chi tiết kết quả Baseline 1: ResNet-50 (Seed 42)

* **Thiết bị:** NVIDIA Tesla T4 (Kaggle GPU)
* **Thời gian huấn luyện:** 1.262,1 giây (20 epochs)
* **Epoch được chọn làm checkpoint tốt nhất:** **Epoch 19**
* **Validation Macro F1:** **0.9679** (96.79%)
* **Validation Balanced Accuracy:** **0.9676** (96.76%)
* **Validation Loss:** **0.1377**

### Recall theo từng lớp bệnh trên tập Validation:
| Lớp bệnh | Số ảnh val | Recall | Đánh giá sơ bộ |
|:---|:---:|:---:|:---|
| `ALLOCARIDARA_ATTACK` (Rầy nhảy) | 92 | **98.91%** | Nhận diện tốt nhất, tổn thương đốm vàng đặc trưng rõ nét |
| `HEALTHY_LEAF` (Lá khỏe mạnh) | 97 | **97.94%** | Phân biệt chính xác lá bình thường với lá bệnh |
| `PHOMOPSIS_LEAF_SPOT` (Đốm lá Phomopsis) | 88 | **97.73%** | Độ nhạy cao |
| `LEAF_BLIGHT` (Cháy lá) | 93 | **94.62%** | Đôi khi nhầm với đốm rong do rìa hoại tử tương đồng |
| `ALGAL_LEAF_SPOT` (Đốm rong) | 74 | **94.59%** | Lớp có số mẫu ít nhất, dễ nhầm khi vết đốm giai đoạn đầu nhỏ |

### Vị trí lưu trữ sản phẩm:
* **Checkpoint weights:** `outputs/checkpoints/resnet50_seed42.pt` (lưu local / private Kaggle dataset, không commit Git)
* **Số liệu chi tiết:** `reports/baselines/resnet50/resnet50_seed42.json`
* **Hình ảnh Confusion Matrix:** `reports/baselines/resnet50/resnet50_seed42_confusion_matrix.png`
* **Hình ảnh Lịch sử train (Loss/F1):** `reports/baselines/resnet50/resnet50_seed42_history.png`

---

## 4. Chi tiết kết quả Baseline 3: MobileNetV3-Small (Seed 42)

* **Thiết bị:** NVIDIA Tesla T4 (Kaggle GPU)
* **Thời gian huấn luyện:** 751,1 giây (29 epochs, dừng sớm). Epoch 1 mất 327,3 giây do khởi động lạnh, các epoch sau trung bình 15,1 giây.
* **Epoch được chọn làm checkpoint tốt nhất:** **Epoch 22**
* **Validation Macro F1:** **0.9485** (94.85%)
* **Validation Balanced Accuracy:** **0.9484** (94.84%)
* **Validation Loss:** **0.2841** (tại epoch 22)
* **Siêu tham số:** AdamW (`lr=5e-4`, `weight_decay=1e-2`), cosine annealing, dropout 0.2, batch size 32, tối đa 30 epochs, patience 7. Khác ResNet-50 ở learning rate, số epoch, patience, scheduler và nguồn trọng số (torchvision thay vì timm).

### Recall theo từng lớp bệnh trên tập Validation:
| Lớp bệnh | Số ảnh val | Recall | Nhầm chính |
|:---|:---:|:---:|:---|
| `HEALTHY_LEAF` (Lá khỏe mạnh) | 97 | **100.00%** | Không nhầm ảnh nào |
| `ALLOCARIDARA_ATTACK` (Rầy nhảy) | 92 | **97.83%** | 1 ảnh sang `ALGAL_LEAF_SPOT`, 1 ảnh sang `PHOMOPSIS_LEAF_SPOT` |
| `LEAF_BLIGHT` (Cháy lá) | 93 | **93.55%** | 3 ảnh sang `ALGAL_LEAF_SPOT`, 2 sang `PHOMOPSIS_LEAF_SPOT`, 1 sang `HEALTHY_LEAF` |
| `ALGAL_LEAF_SPOT` (Đốm rong) | 74 | **91.89%** | 4 ảnh sang `PHOMOPSIS_LEAF_SPOT`, 2 sang `ALLOCARIDARA_ATTACK` |
| `PHOMOPSIS_LEAF_SPOT` (Đốm lá Phomopsis) | 88 | **90.91%** | 4 ảnh sang `HEALTHY_LEAF`, 2 sang `ALGAL_LEAF_SPOT`, 1 sang `ALLOCARIDARA_ATTACK`, 1 sang `LEAF_BLIGHT` |

### Vị trí lưu trữ sản phẩm:
* **Checkpoint weights:** `outputs/checkpoints/mobilenetv3_small_seed42.pt` (lưu private Kaggle dataset, không commit Git)
* **Số liệu chi tiết:** `reports/baselines/mobilenetv3_small/mobilenetv3_small_seed42.json`
* **Hình ảnh Confusion Matrix:** `reports/baselines/mobilenetv3_small/mobilenetv3_small_seed42_confusion_matrix.png`
* **Hình ảnh Lịch sử train (Loss/F1):** `reports/baselines/mobilenetv3_small/mobilenetv3_small_seed42_history.png`
* **Nhật ký từng epoch:** `reports/baselines/mobilenetv3_small/mobilenetv3_small_seed42_history.csv`
* **Notebook đã chạy:** `notebooks/03_mobilenetv3_small.ipynb`

---

## 5. Nhận xét và Bước tiếp theo

1. **Về kết quả ResNet-50:**
   * Điểm số **96.79% Macro F1** trên ảnh sạch khẳng định mô hình và pipeline dữ liệu hoạt động ổn định, không có lỗi rò rỉ dữ liệu (data leakage) giữa các tập.
   * Đây là mốc đối chứng lý tưởng để làm rõ mức độ suy giảm hiệu năng khi bước vào Phase 4 (Corruption benchmark: ảnh rung mờ, nhiễu camera, thiếu sáng, chói sáng).

2. **Về kết quả MobileNetV3-Small:**
   * Đạt **94.85% Macro F1**, thấp hơn ResNet-50 khoảng 1,9 điểm phần trăm nhưng chỉ có 1,52M tham số, nhỏ hơn khoảng 15 lần.
   * Có 5 ảnh bệnh bị dự đoán thành lá khỏe (4 `PHOMOPSIS_LEAF_SPOT`, 1 `LEAF_BLIGHT`) và không có ảnh lá khỏe nào bị dự đoán thành bệnh.
   * Train accuracy 99.7% trong khi val loss dừng quanh 0.26 đến 0.29 từ epoch 20 trở đi, cho thấy mô hình khớp tốt tập train và val không còn cải thiện.
   * Siêu tham số không trùng ResNet-50, nên chênh lệch 1,9 điểm chưa thể quy hoàn toàn cho kiến trúc. Cần chạy lại với cùng siêu tham số nếu muốn so sánh chặt chẽ.

3. **Kế hoạch tiếp theo:**
   * Huấn luyện mô hình còn lại: `ConvNeXt-Tiny` (kiến trúc hiện đại) để hoàn thiện bảng so sánh.
   * Lựa chọn reference model chính thức phục vụ cho các nghiên cứu xử lý ảnh ở Phase 4 & Phase 5.
