# Phase 3: Baseline Models Report

Tài liệu tổng hợp kết quả của 3 mô hình Baseline trong Phase 3 (theo danh mục đã khóa trong `configs/baselines.yaml` và kế hoạch đồ án CS406).

Các mô hình được fine-tune trên cùng một giao thức phân chia dữ liệu (`manifests/split_manifest.csv`), kích thước ảnh 224×224, 5 nhãn bệnh lá sầu riêng. Mọi checkpoint và siêu tham số được chọn dựa trên **Validation Macro F1**. Tập Test được khóa hoàn toàn.

---

## 1. Bảng so sánh tổng hợp (Baseline Leaderboard)

| Thứ tự | Mô hình | Họ kiến trúc (Family) | `timm` backbone | Số tham số | Val Macro F1 | Val Balanced Acc | Val Loss | Thời gian train (s) | Trạng thái |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **ResNet-50** | Classical CNN | `resnet50` | 23,518,277 | **0.9679** | 0.9676 | 0.1377 | 1262.1 (T4) | **Đã hoàn thành** |
| **2** | **ConvNeXt-Tiny** | Modern Vision | `convnext_tiny` | - | - | - | - | - | *Chờ thực nghiệm* |
| **3** | **MobileNetV3-Small** | Lightweight | `mobilenetv3_small_100` | - | - | - | - | - | *Chờ thực nghiệm* |

---

## 2. Chi tiết Baseline 1: ResNet-50 (Seed 42)

* **Backbone:** ResNet-50 (pretrained ImageNet-1k)
* **Số epoch tối đa:** 20 (Early stopping patience: 5)
* **Epoch được chọn:** **Epoch 19**
* **Validation Macro F1:** **0.9679** (96.79%)
* **Validation Balanced Accuracy:** **0.9676** (96.76%)
* **Validation Loss:** **0.1377**

### Recall theo từng lớp bệnh:
| Lớp bệnh | Recall |
|:---|:---:|
| `ALGAL_LEAF_SPOT` | 94.59% |
| `ALLOCARIDARA_ATTACK` | 98.91% |
| `HEALTHY_LEAF` | 97.94% |
| `LEAF_BLIGHT` | 94.62% |
| `PHOMOPSIS_LEAF_SPOT` | 97.73% |

### Tài liệu trực quan:
* **Confusion Matrix:** `resnet50/resnet50_seed42_confusion_matrix.png`
* **Lịch sử huấn luyện (Loss & F1):** `resnet50/resnet50_seed42_history.png`
* **Chi tiết dạng JSON:** `resnet50/resnet50_seed42.json`

---

## 3. Cấu trúc thư mục

```text
reports/baselines/
├── README.md                      # Báo cáo và bảng so sánh này
├── resnet50/                      # Baseline 1: CNN đối chứng
│   ├── resnet50_seed42.json
│   ├── resnet50_seed42_confusion_matrix.png
│   └── resnet50_seed42_history.png
├── convnext_tiny/                 # Baseline 2: Kiến trúc hiện đại (sắp chạy)
│   └── .gitkeep
└── mobilenetv3_small/             # Baseline 3: Mô hình gọn nhẹ (sắp chạy)
    └── .gitkeep
```
