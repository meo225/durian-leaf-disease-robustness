# Kết quả baseline

Fine-tune trên split khóa `manifests/split_manifest.csv`, ảnh 224×224, 5 lớp, seed 42, một lần chạy mỗi model. Checkpoint chọn bằng macro F1 trên validation. Tập test chưa mở.

## Bảng so sánh

| Mô hình | Họ kiến trúc | Số tham số | Val macro F1 | Val balanced acc | Val loss | Thời gian train | Epoch chọn / đã chạy |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | CNN cổ điển | 23,518,277 | 0.9720 | 0.9719 | 0.1112 | 1740.7 giây | 23 / 30 |
| ConvNeXt-Tiny | Vision hiện đại | 27,823,973 | 0.9775 | 0.9775 | 0.1173 | 3109.2 giây | 24 / 30 |
| MobileNetV3-Small | CNN nhẹ | 1,522,981 | 0.9292 | 0.9297 | 0.2635 | 551.7 giây | 15 / 22 |
| Swin-Tiny (thêm sau khi khóa) | Transformer cửa sổ trượt | 27,523,199 | 0.9664 | 0.9654 | 0.1243 | 824.4 giây | 5 / 12 |
| HOG + SVM | Truyền thống | không áp dụng | 0.4003 | 0.4074 | không áp dụng | 240.7 giây (CPU) | không áp dụng |

Chạy trên Tesla T4. Epoch 1 luôn chậm hơn nhiều do khởi động lạnh (ConvNeXt-Tiny 515 giây, ResNet-50 305 giây, MobileNetV3-Small 243 giây, Swin-Tiny 190 giây), nên thời gian train gồm cả phần này.

## Recall từng lớp

| Lớp | ResNet-50 | ConvNeXt-Tiny | MobileNetV3-Small | Swin-Tiny |
|---|---:|---:|---:|---:|
| ALGAL_LEAF_SPOT | 0.9595 | 0.9730 | 0.9459 | 0.9459 |
| ALLOCARIDARA_ATTACK | 0.9891 | 0.9891 | 0.9348 | 0.9783 |
| HEALTHY_LEAF | 1.0000 | 0.9794 | 0.9897 | 0.9794 |
| LEAF_BLIGHT | 0.9677 | 0.9462 | 0.9032 | 0.9462 |
| PHOMOPSIS_LEAF_SPOT | 0.9432 | 1.0000 | 0.8750 | 0.9773 |

Recall từng lớp của HOG + SVM: ALGAL_LEAF_SPOT 0.2027, ALLOCARIDARA_ATTACK 0.5652, HEALTHY_LEAF 0.5464, LEAF_BLIGHT 0.4839, PHOMOPSIS_LEAF_SPOT 0.2386.

## Giao thức chung

Mọi baseline dùng cùng một cấu hình, chỉ khác model: timm, AdamW, learning rate 2e-4, weight decay 1e-2, cosine, tối đa 30 epoch, patience 7, không dropout thêm. Xem `configs/train_<model>.yaml`.

## Đọc kết quả

- Tập validation có 444 ảnh. Một điểm F1 tương đương vài ảnh, và đường val F1 dao động theo epoch. Chênh lệch dưới khoảng 0.01 chưa đủ để xếp hạng, nên ConvNeXt-Tiny và ResNet-50 (chênh 0.0055) chưa phân biệt được.
- Mỗi model mới chạy một seed. Phase 3 ghi rõ cần chạy lặp khi điểm gần nhau.
- MobileNetV3-Small dừng sớm ở epoch 22 nên cosine chưa giảm learning rate về mức thấp.
- Swin-Tiny dừng sớm ở epoch 12, chọn epoch 5 khi learning rate còn khoảng 1.9e-4, và val F1 dao động từ 0.93 đến 0.97 giữa các epoch. Learning rate 2e-4 chưa dò riêng cho Swin. Xem mục "Swin-Tiny: model đối chiếu thêm".

## Mốc truyền thống: HOG + SVM

Mốc rẻ theo yêu cầu môn học, để thấy khoảng cách với học sâu. Chạy trên CPU, cùng split khóa, cùng ảnh 224×224, seed 42. HOG tính trên ảnh xám (ô 16 px, khối 2×2 ô, 9 hướng, 6.084 chiều), chuẩn hóa từng chiều rồi SVM nhân RBF với tham số mặc định (C = 1, gamma = scale). Không augmentation, không dò siêu tham số. Cấu hình ở `configs/train_hog_svm.yaml`, code ở `src/training/hog_svm.py`, notebook `03_train_hog_svm.ipynb`.

Macro F1 validation 0.4003, thấp hơn ConvNeXt-Tiny 0.9775 khoảng 0.58. Mức ngẫu nhiên giữa năm lớp xấp xỉ 0.2. Hai lớp ALGAL_LEAF_SPOT và PHOMOPSIS_LEAF_SPOT bị nhận sai nhiều nhất (recall khoảng 0.2). HOG chỉ giữ cạnh và kết cấu trên ảnh xám nên bỏ mất màu, vốn là dấu hiệu chính của vết bệnh. Đây là một seed, không tối ưu, nên con số chỉ là mốc. HOG + SVM không tham gia chọn reference model và chưa đo dưới suy giảm.

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

## Swin-Tiny: model đối chiếu thêm

Swin-Tiny được thêm sau khi reference model đã chọn (nhánh `baseline-Swin-Tiny`, xem `docs/phase-3-baseline-candidates.md`). Cùng giao thức train, chỉ khác backbone `swin_tiny_patch4_window7_224`. Đo lại cả bốn model trong một lần chạy bằng `configs/reference_selection_swin.yaml` và mục 5 của notebook `03_select_reference_model.ipynb`, cùng validation và cùng năm suy giảm. Kết quả ở `reference_selection_with_swin.json`, không ghi đè `reference_selection.json`. Cả bốn checkpoint tái hiện đúng macro F1 đã ghi lúc train.

Macro F1 trên validation dưới suy giảm:

| Mô hình | Sạch | Tối 0.5 | Sáng 1.5 | Tương phản 0.5 | Mờ σ 2 | Nhiễu 0.1 | Trung bình | Giảm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 0.9720 | 0.909 | 0.920 | 0.735 | 0.405 | 0.568 | 0.7073 | 0.2647 |
| ConvNeXt-Tiny | 0.9775 | 0.892 | 0.851 | 0.876 | 0.577 | 0.658 | 0.7709 | 0.2066 |
| MobileNetV3-Small | 0.9292 | 0.778 | 0.861 | 0.781 | 0.398 | 0.325 | 0.6287 | 0.3005 |
| Swin-Tiny | 0.9664 | 0.927 | 0.877 | 0.954 | 0.653 | 0.738 | 0.8299 | 0.1365 |

Kích thước và tốc độ, đo lại trong lần chạy này (Tesla T4, fp32). Số của ba model cũ lệch nhẹ so với bảng ở mục Reference model do đo ở một phiên khác, nên so Swin với ba model này bằng bảng dưới:

| Mô hình | Dung lượng MB | ms mỗi ảnh, batch 1 | Ảnh mỗi giây, batch 32 |
|---|---:|---:|---:|
| ResNet-50 | 89.7 | 5.7 | 350 |
| ConvNeXt-Tiny | 106.1 | 6.2 | 227 |
| MobileNetV3-Small | 5.8 | 5.3 | 4315 |
| Swin-Tiny | 105.0 | 11.1 | 214 |

Đọc kết quả:

- Swin-Tiny giảm ít nhất (0.1365, so với 0.2066 của ConvNeXt-Tiny) và tốt hơn ConvNeXt-Tiny ở cả năm suy giảm. Chênh mức giảm khoảng 0.07, lớn hơn nhiều ngưỡng 0.01.
- Quy tắc đã khóa vẫn chọn ConvNeXt-Tiny. F1 sạch của Swin-Tiny kém ConvNeXt-Tiny 0.0111, vượt ngưỡng 0.01 là 0.0011, nên Swin-Tiny không nằm trong nhóm "ngang nhau" (`selection.candidates` chỉ có ResNet-50 và ConvNeXt-Tiny). Quy tắc không được đổi sau khi xem kết quả. Chênh 0.0111 cũng nằm trong vùng nhiễu của validation 444 ảnh nói ở mục Đọc kết quả.
- Mới một seed cho mỗi model, mỗi suy giảm một mức. Chưa kết luận Swin-Tiny bền hơn.
- Giả thuyết chưa kiểm tra: checkpoint Swin-Tiny ở epoch 5 (learning rate còn cao) so với epoch 24 của ConvNeXt-Tiny, và một model chưa khớp kỹ vào ảnh sạch có thể ổn định hơn dưới suy giảm. Chưa có bằng chứng.
- Độ trễ batch 1 của Swin-Tiny gấp khoảng 1.8 lần ConvNeXt-Tiny, throughput batch 32 gần nhau.
- Tập test chưa mở. Reference model vẫn là ConvNeXt-Tiny. Đưa Swin-Tiny vào task 4.2 hay đổi reference là quyết định của nhóm sau khi chạy lặp seed.

## File

Mỗi model có thư mục riêng, tên file có hậu tố `_seed42`:

- `<model>_seed42.json`: toàn bộ số liệu, hyperparameter và lịch sử từng epoch (HOG + SVM không có epoch nên `history` rỗng, và không có `_history.png`)
- `<model>_seed42_history.png`: loss và val macro F1 theo epoch
- `<model>_seed42_confusion_matrix.png`: confusion matrix của checkpoint đã chọn

File `reference_selection.json` nằm cạnh thư mục các model.

Code: `src/training/run.py`, config: `configs/train_<model>.yaml`, notebook đã chạy: `notebooks/03_train_<model>.ipynb`.
