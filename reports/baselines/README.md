# Kết quả baseline

Fine-tune trên split khóa `manifests/split_manifest.csv`, ảnh 224×224, 5 lớp, seed 42. Checkpoint chọn bằng macro F1 trên validation. Tập test chưa mở.

## Bảng so sánh

| Mô hình | Họ kiến trúc | Số tham số | Val macro F1 | Val balanced acc | Val loss | Thời gian train | Epoch chọn |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | CNN cổ điển | 23,518,277 | 0.9679 | 0.9676 | 0.1377 | 1162.8 giây | 19 |
| ConvNeXt-Tiny | Vision hiện đại | | | | | | chưa chạy |
| MobileNetV3-Small | CNN nhẹ | 1,522,981 | 0.9485 | 0.9484 | 0.2841 | 751.1 giây | 22 |

Cả hai chạy trên Tesla T4. Thời gian MobileNetV3-Small gồm epoch 1 mất 327.3 giây do khởi động lạnh, các epoch sau khoảng 15 giây.

## Recall từng lớp

| Lớp | ResNet-50 | MobileNetV3-Small |
|---|---:|---:|
| ALGAL_LEAF_SPOT | 0.9459 | 0.9189 |
| ALLOCARIDARA_ATTACK | 0.9891 | 0.9783 |
| HEALTHY_LEAF | 0.9794 | 1.0000 |
| LEAF_BLIGHT | 0.9462 | 0.9355 |
| PHOMOPSIS_LEAF_SPOT | 0.9773 | 0.9091 |

## Cấu hình khác nhau giữa hai lần chạy

| | ResNet-50 | MobileNetV3-Small |
|---|---|---|
| Nguồn model | timm `resnet50` | torchvision `mobilenet_v3_small` |
| Learning rate | 1e-4 | 5e-4 |
| Weight decay | 1e-4 | 1e-2 |
| Scheduler | không | cosine |
| Dropout | mặc định | 0.2 |
| Epoch tối đa, patience | 20, 5 | 30, 7 |

Hai model chưa chung hyperparameter, nên chênh lệch điểm chưa chỉ do kiến trúc. `configs/baselines.yaml` ghi MobileNetV3-Small là timm `mobilenetv3_small_100`, còn lần chạy này dùng bản torchvision.

Hai lần chạy này train trước khi gộp code vào `src/training/run.py`. Số liệu giữ nguyên, chỉ đổi JSON sang cùng một định dạng. Lịch sử của ResNet-50 chưa có cột lr, train_accuracy và seconds. Chạy lại notebook trên Kaggle sẽ tạo lại kết quả bằng code hiện tại.

## File

Mỗi model có thư mục riêng, tên file có hậu tố `_seed42`:

- `<model>_seed42.json`: toàn bộ số liệu, hyperparameter và lịch sử từng epoch
- `<model>_seed42_history.png`: loss và val macro F1 theo epoch
- `<model>_seed42_confusion_matrix.png`: confusion matrix của checkpoint đã chọn

Code: `src/training/run.py`, config: `configs/train_<model>.yaml`, notebook: `notebooks/03_train_<model>.ipynb`.
