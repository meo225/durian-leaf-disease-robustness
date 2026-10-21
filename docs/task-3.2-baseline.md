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
| **3** | **MobileNetV3-Small** | Lightweight | `mobilenetv3_small_100` | ~2.5M | — | — | — | — | — | *Chờ thực nghiệm* |

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

## 4. Nhận xét và Bước tiếp theo

1. **Về kết quả ResNet-50:**
   * Điểm số **96.79% Macro F1** trên ảnh sạch khẳng định mô hình và pipeline dữ liệu hoạt động ổn định, không có lỗi rò rỉ dữ liệu (data leakage) giữa các tập.
   * Đây là mốc đối chứng lý tưởng để làm rõ mức độ suy giảm hiệu năng khi bước vào Phase 4 (Corruption benchmark: ảnh rung mờ, nhiễu camera, thiếu sáng, chói sáng).

2. **Kế hoạch tiếp theo:**
   * Huấn luyện 2 mô hình còn lại: `ConvNeXt-Tiny` (kiến trúc hiện đại) và `MobileNetV3-Small` (mô hình nhẹ) theo cùng protocol để hoàn thiện bảng so sánh.
   * Lựa chọn reference model chính thức phục vụ cho các nghiên cứu xử lý ảnh ở Phase 4 & Phase 5.
