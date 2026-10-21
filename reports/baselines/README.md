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

## Giao thức chung

Từ nay mọi baseline dùng cùng một cấu hình train, chỉ khác model: timm, AdamW, learning rate 2e-4, weight decay 1e-2, cosine, tối đa 30 epoch, patience 7, không dropout thêm. Xem `configs/train_<model>.yaml`.

Bảng trên là hai lần chạy cũ, khi hai model còn dùng cấu hình khác nhau (ResNet-50: lr 1e-4, weight decay 1e-4, không scheduler, 20 epoch, patience 5. MobileNetV3-Small: torchvision, lr 5e-4, weight decay 1e-2, cosine, dropout 0.2, 30 epoch, patience 7). Chưa so sánh trực tiếp được. Cần chạy lại cả hai notebook theo giao thức chung, rồi thay số vào bảng.

## File

Mỗi model có thư mục riêng, tên file có hậu tố `_seed42`:

- `<model>_seed42.json`: toàn bộ số liệu, hyperparameter và lịch sử từng epoch
- `<model>_seed42_history.png`: loss và val macro F1 theo epoch
- `<model>_seed42_confusion_matrix.png`: confusion matrix của checkpoint đã chọn

Code: `src/training/run.py`, config: `configs/train_<model>.yaml`, notebook: `notebooks/03_train_<model>.ipynb`.
