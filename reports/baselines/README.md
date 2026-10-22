# Kết quả baseline

Fine-tune trên split khóa `manifests/split_manifest.csv`, ảnh 224×224, 5 lớp, seed 42, một lần chạy mỗi model. Checkpoint chọn bằng macro F1 trên validation. Tập test chưa mở.

## Bảng so sánh

| Mô hình | Họ kiến trúc | Số tham số | Val macro F1 | Val balanced acc | Val loss | Thời gian train | Epoch chọn / đã chạy |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | CNN cổ điển | 23,518,277 | 0.9720 | 0.9719 | 0.1112 | 1740.7 giây | 23 / 30 |
| ConvNeXt-Tiny | Vision hiện đại | 27,823,973 | 0.9775 | 0.9775 | 0.1173 | 3109.2 giây | 24 / 30 |
| MobileNetV3-Small | CNN nhẹ | 1,522,981 | 0.9292 | 0.9297 | 0.2635 | 551.7 giây | 15 / 22 |

Chạy trên Tesla T4. Epoch 1 luôn chậm hơn nhiều do khởi động lạnh (ConvNeXt-Tiny 515 giây, ResNet-50 305 giây, MobileNetV3-Small 243 giây), nên thời gian train gồm cả phần này.

## Recall từng lớp

| Lớp | ResNet-50 | ConvNeXt-Tiny | MobileNetV3-Small |
|---|---:|---:|---:|
| ALGAL_LEAF_SPOT | 0.9595 | 0.9730 | 0.9459 |
| ALLOCARIDARA_ATTACK | 0.9891 | 0.9891 | 0.9348 |
| HEALTHY_LEAF | 1.0000 | 0.9794 | 0.9897 |
| LEAF_BLIGHT | 0.9677 | 0.9462 | 0.9032 |
| PHOMOPSIS_LEAF_SPOT | 0.9432 | 1.0000 | 0.8750 |

## Giao thức chung

Mọi baseline dùng cùng một cấu hình, chỉ khác model: timm, AdamW, learning rate 2e-4, weight decay 1e-2, cosine, tối đa 30 epoch, patience 7, không dropout thêm. Xem `configs/train_<model>.yaml`.

## Đọc kết quả

- Tập validation có 444 ảnh. Một điểm F1 tương đương vài ảnh, và đường val F1 dao động theo epoch. Chênh lệch dưới khoảng 0.01 chưa đủ để xếp hạng, nên ConvNeXt-Tiny và ResNet-50 (chênh 0.0055) chưa phân biệt được.
- Mỗi model mới chạy một seed. Phase 3 ghi rõ cần chạy lặp khi điểm gần nhau.
- MobileNetV3-Small dừng sớm ở epoch 22 nên cosine chưa giảm learning rate về mức thấp.

## Reference model

**ConvNeXt-Tiny** là reference model. Đây cũng là control baseline cho task 4: cùng checkpoint, ảnh không xử lý, pipeline chuẩn.

Quy tắc chọn khóa trước khi chạy (`configs/reference_selection.yaml`): các model có macro F1 sạch trong khoảng 0.01 của model cao nhất được coi là ngang nhau. Giữa chúng chọn model có mức giảm F1 thấp nhất dưới năm suy giảm cố định trên validation, rồi mới xét độ trễ. ResNet-50 và ConvNeXt-Tiny ngang nhau về F1 sạch (chênh 0.0055). ConvNeXt-Tiny giảm ít hơn (0.207 so với 0.265), nên được chọn. MobileNetV3-Small thấp hơn rõ ở cả F1 sạch và độ bền.

Kích thước và tốc độ (Tesla T4, fp32):

| Mô hình | Dung lượng MB | ms mỗi ảnh, batch 1 | Ảnh mỗi giây, batch 32 |
|---|---:|---:|---:|
| ResNet-50 | 89.7 | 5.5 | 383 |
| ConvNeXt-Tiny | 106.1 | 6.0 | 270 |
| MobileNetV3-Small | 5.8 | 5.3 | 4318 |

Macro F1 trên validation dưới suy giảm:

| Mô hình | Sạch | Tối 0.5 | Sáng 1.5 | Tương phản 0.5 | Mờ σ 2 | Nhiễu 0.1 | Trung bình | Giảm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 0.9720 | 0.909 | 0.920 | 0.735 | 0.405 | 0.568 | 0.7073 | 0.2647 |
| ConvNeXt-Tiny | 0.9775 | 0.892 | 0.851 | 0.876 | 0.577 | 0.658 | 0.7709 | 0.2066 |
| MobileNetV3-Small | 0.9292 | 0.778 | 0.861 | 0.781 | 0.398 | 0.325 | 0.6287 | 0.3005 |

Lưu ý: ba checkpoint tái hiện đúng macro F1 đã ghi lúc train. Năm mức suy giảm chỉ để sanity check trên validation, không phải severity của task 4.1. Mờ và nhiễu làm cả ba model giảm rất mạnh. Mới một seed và validation 444 ảnh, nên F1 sạch của ConvNeXt-Tiny và ResNet-50 chưa phân biệt được. Quyết định dựa vào độ bền (chênh 0.058 mức giảm), và độ bền cũng mới đo bằng một mức cho mỗi kiểu suy giảm. Chi tiết nằm trong `reference_selection.json`, notebook `03_select_reference_model.ipynb`.

## File

Mỗi model có thư mục riêng, tên file có hậu tố `_seed42`:

- `<model>_seed42.json`: toàn bộ số liệu, hyperparameter và lịch sử từng epoch
- `<model>_seed42_history.png`: loss và val macro F1 theo epoch
- `<model>_seed42_confusion_matrix.png`: confusion matrix của checkpoint đã chọn

File `reference_selection.json` nằm cạnh thư mục các model.

Code: `src/training/run.py`, config: `configs/train_<model>.yaml`, notebook đã chạy: `notebooks/03_train_<model>.ipynb`.
