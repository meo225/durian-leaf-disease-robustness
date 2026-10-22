# Kết quả baseline

Fine-tune trên split khóa `manifests/split_manifest.csv`, ảnh 224×224, 5 lớp, seed 42, một lần chạy mỗi model. Checkpoint chọn bằng macro F1 trên validation. Tập test chưa mở.

## Bảng so sánh

| Mô hình | Họ kiến trúc | Số tham số | Val macro F1 | Val balanced acc | Val loss | Thời gian train | Epoch chọn / đã chạy |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | CNN cổ điển | 23,518,277 | 0.9720 | 0.9719 | 0.1112 | 1740.7 giây | 23 / 30 |
| ConvNeXt-Tiny | Vision hiện đại | | | | | | chưa chạy |
| MobileNetV3-Small | CNN nhẹ | 1,522,981 | 0.9292 | 0.9297 | 0.2635 | 551.7 giây | 15 / 22 |

Chạy trên Tesla T4. Epoch 1 luôn chậm hơn nhiều do khởi động lạnh (MobileNetV3-Small 243 giây, ResNet-50 305 giây), nên thời gian train gồm cả phần này.

## Recall từng lớp

| Lớp | ResNet-50 | MobileNetV3-Small |
|---|---:|---:|
| ALGAL_LEAF_SPOT | 0.9595 | 0.9459 |
| ALLOCARIDARA_ATTACK | 0.9891 | 0.9348 |
| HEALTHY_LEAF | 1.0000 | 0.9897 |
| LEAF_BLIGHT | 0.9677 | 0.9032 |
| PHOMOPSIS_LEAF_SPOT | 0.9432 | 0.8750 |

## Giao thức chung

Mọi baseline dùng cùng một cấu hình, chỉ khác model: timm, AdamW, learning rate 2e-4, weight decay 1e-2, cosine, tối đa 30 epoch, patience 7, không dropout thêm. Xem `configs/train_<model>.yaml`.

## Đọc kết quả

- Tập validation có 444 ảnh. Một điểm F1 tương đương vài ảnh, và đường val F1 dao động theo epoch. Chênh lệch dưới khoảng 0.01 chưa đủ để xếp hạng.
- Mỗi model mới chạy một seed. Phase 3 ghi rõ cần chạy lặp khi điểm gần nhau.
- MobileNetV3-Small dừng sớm ở epoch 22 nên cosine chưa giảm learning rate về mức thấp.

## File

Mỗi model có thư mục riêng, tên file có hậu tố `_seed42`:

- `<model>_seed42.json`: toàn bộ số liệu, hyperparameter và lịch sử từng epoch
- `<model>_seed42_history.png`: loss và val macro F1 theo epoch
- `<model>_seed42_confusion_matrix.png`: confusion matrix của checkpoint đã chọn

Code: `src/training/run.py`, config: `configs/train_<model>.yaml`, notebook đã chạy: `notebooks/03_train_<model>.ipynb`.
