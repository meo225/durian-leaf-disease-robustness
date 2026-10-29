# Kết quả baseline

## Tóm tắt

- **Reference model: Swin-Tiny.** Nhóm đổi từ ConvNeXt-Tiny sau khi chạy lặp ba seed (42, 43, 44). Swin-Tiny cũng là control baseline cho task 4: cùng checkpoint, ảnh không xử lý, pipeline chuẩn.
- Lý do: F1 sạch của năm model sâu gần nhau, nhưng Swin-Tiny giảm ít nhất dưới suy giảm ở cả ba seed (trung bình 0.136, so với 0.219 của ConvNeXt-Tiny và 0.260 của ResNet-50). Chi tiết ở mục "Chạy lặp seed và chọn reference".
- Tập test chưa mở. Mọi số liệu là macro F1 trên validation 444 ảnh, split khóa `manifests/split_manifest.csv`.

Trung bình ba seed:

| Mô hình | F1 sạch | F1 trung bình dưới 5 suy giảm | Mức giảm (± độ lệch chuẩn) |
|---|---:|---:|---:|
| Swin-Tiny | 0.9734 | 0.8377 | 0.136 ± 0.006 |
| ConvNeXt-Tiny | 0.9779 | 0.7590 | 0.219 ± 0.031 |
| ResNet-50 | 0.9736 | 0.7138 | 0.260 ± 0.018 |

MobileNetV3-Small và HOG + SVM không chạy lặp (xem bên dưới), chỉ có số của seed 42.

## Giao thức chung

Fine-tune trên split khóa, ảnh 224×224, 5 lớp. Mọi baseline học sâu dùng cùng một cấu hình, chỉ khác model: timm, AdamW, learning rate 2e-4, weight decay 1e-2, cosine, tối đa 30 epoch, patience 7, không dropout thêm. Xem `configs/train_<model>.yaml`. Checkpoint chọn bằng macro F1 trên validation.

Phần "Kết quả seed 42" và "Độ bền" bên dưới là một lần chạy mỗi model với seed 42. Phần chạy lặp thêm seed 43 và 44.

## Kết quả seed 42

| Mô hình | Họ kiến trúc | Số tham số | Val macro F1 | Val balanced acc | Val loss | Thời gian train | Epoch chọn / đã chạy |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | CNN cổ điển | 23,518,277 | 0.9720 | 0.9719 | 0.1112 | 1740.7 giây | 23 / 30 |
| ConvNeXt-Tiny | Vision hiện đại | 27,823,973 | 0.9775 | 0.9775 | 0.1173 | 3109.2 giây | 24 / 30 |
| MobileNetV3-Small | CNN nhẹ | 1,522,981 | 0.9292 | 0.9297 | 0.2635 | 551.7 giây | 15 / 22 |
| Swin-Tiny (thêm sau khi khóa danh sách) | Transformer cửa sổ trượt | 27,523,199 | 0.9664 | 0.9654 | 0.1243 | 824.4 giây | 5 / 12 |
| HOG + SVM | Truyền thống | không áp dụng | 0.4003 | 0.4074 | không áp dụng | 240.7 giây (CPU) | không áp dụng |

Chạy trên Tesla T4. Epoch 1 luôn chậm hơn nhiều do khởi động lạnh (ConvNeXt-Tiny 515 giây, ResNet-50 305 giây, MobileNetV3-Small 243 giây, Swin-Tiny 190 giây), nên thời gian train gồm cả phần này.

Recall từng lớp:

| Lớp | ResNet-50 | ConvNeXt-Tiny | MobileNetV3-Small | Swin-Tiny |
|---|---:|---:|---:|---:|
| ALGAL_LEAF_SPOT | 0.9595 | 0.9730 | 0.9459 | 0.9459 |
| ALLOCARIDARA_ATTACK | 0.9891 | 0.9891 | 0.9348 | 0.9783 |
| HEALTHY_LEAF | 1.0000 | 0.9794 | 0.9897 | 0.9794 |
| LEAF_BLIGHT | 0.9677 | 0.9462 | 0.9032 | 0.9462 |
| PHOMOPSIS_LEAF_SPOT | 0.9432 | 1.0000 | 0.8750 | 0.9773 |

Recall từng lớp của HOG + SVM: ALGAL_LEAF_SPOT 0.2027, ALLOCARIDARA_ATTACK 0.5652, HEALTHY_LEAF 0.5464, LEAF_BLIGHT 0.4839, PHOMOPSIS_LEAF_SPOT 0.2386.

Đọc kết quả:

- Tập validation có 444 ảnh. Một điểm F1 tương đương vài ảnh, và đường val F1 dao động theo epoch. Chênh lệch dưới khoảng 0.01 chưa đủ để xếp hạng, nên ConvNeXt-Tiny và ResNet-50 (chênh 0.0055) chưa phân biệt được.
- MobileNetV3-Small dừng sớm ở epoch 22 nên cosine chưa giảm learning rate về mức thấp.
- Swin-Tiny dừng sớm ở epoch 12, chọn epoch 5 khi learning rate còn khoảng 1.9e-4, và val F1 dao động từ 0.93 đến 0.97 giữa các epoch. Learning rate 2e-4 chưa dò riêng cho Swin.

### Mốc truyền thống: HOG + SVM

Mốc rẻ theo yêu cầu môn học, để thấy khoảng cách với học sâu. Chạy trên CPU, cùng split khóa, cùng ảnh 224×224, seed 42. HOG tính trên ảnh xám (ô 16 px, khối 2×2 ô, 9 hướng, 6.084 chiều), chuẩn hóa từng chiều rồi SVM nhân RBF với tham số mặc định (C = 1, gamma = scale). Không augmentation, không dò siêu tham số. Cấu hình ở `configs/train_hog_svm.yaml`, code ở `src/training/hog_svm.py`, notebook `03_train_hog_svm.ipynb`.

Macro F1 validation 0.4003, thấp hơn ConvNeXt-Tiny 0.9775 khoảng 0.58. Mức ngẫu nhiên giữa năm lớp xấp xỉ 0.2. Hai lớp ALGAL_LEAF_SPOT và PHOMOPSIS_LEAF_SPOT bị nhận sai nhiều nhất (recall khoảng 0.2). HOG chỉ giữ cạnh và kết cấu trên ảnh xám nên bỏ mất màu, vốn là dấu hiệu chính của vết bệnh. Đây là một seed, không tối ưu, nên con số chỉ là mốc. HOG + SVM không tham gia chọn reference model và chưa đo dưới suy giảm.

## Độ bền dưới suy giảm (seed 42)

Quy tắc chọn khóa trước khi chạy (`configs/reference_selection.yaml`): các model có macro F1 sạch trong khoảng 0.01 của model cao nhất được coi là ngang nhau. Giữa chúng chọn model có mức giảm F1 thấp nhất dưới năm suy giảm cố định trên validation, rồi mới xét độ trễ.

Lần đầu (ba model, `reference_selection.json`), ResNet-50 và ConvNeXt-Tiny ngang nhau về F1 sạch (chênh 0.0055). ConvNeXt-Tiny giảm ít hơn (0.207 so với 0.265), nên được chọn làm reference. MobileNetV3-Small thấp hơn rõ ở cả F1 sạch và độ bền.

Swin-Tiny được thêm sau khi reference đã chọn (nhánh `baseline-Swin-Tiny`, xem `docs/phase-3-baseline-candidates.md`). Cùng giao thức train, chỉ khác backbone `swin_tiny_patch4_window7_224`. Cả bốn model được đo lại trong một lần chạy bằng `configs/reference_selection_swin.yaml` và mục 5 của notebook `03_select_reference_model.ipynb`, cùng validation và năm suy giảm. Kết quả ở `reference_selection_with_swin.json`, không ghi đè `reference_selection.json`. Các checkpoint tái hiện đúng macro F1 đã ghi lúc train. Năm mức suy giảm chỉ để sanity check trên validation, không phải severity của task 4.1.

Macro F1 trên validation dưới suy giảm:

| Mô hình | Sạch | Tối 0.5 | Sáng 1.5 | Tương phản 0.5 | Mờ σ 2 | Nhiễu 0.1 | Trung bình | Giảm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 0.9720 | 0.909 | 0.920 | 0.735 | 0.405 | 0.568 | 0.7073 | 0.2647 |
| ConvNeXt-Tiny | 0.9775 | 0.892 | 0.851 | 0.876 | 0.577 | 0.658 | 0.7709 | 0.2066 |
| MobileNetV3-Small | 0.9292 | 0.778 | 0.861 | 0.781 | 0.398 | 0.325 | 0.6287 | 0.3005 |
| Swin-Tiny | 0.9664 | 0.927 | 0.877 | 0.954 | 0.653 | 0.738 | 0.8299 | 0.1365 |

Kích thước và tốc độ, đo lại trong lần chạy có Swin-Tiny (Tesla T4, fp32). Số của ba model cũ lệch nhẹ so với lần đo đầu trong `reference_selection.json` do đo ở một phiên khác, nên so Swin với ba model này bằng bảng dưới:

| Mô hình | Dung lượng MB | ms mỗi ảnh, batch 1 | Ảnh mỗi giây, batch 32 |
|---|---:|---:|---:|
| ResNet-50 | 89.7 | 5.7 | 350 |
| ConvNeXt-Tiny | 106.1 | 6.2 | 227 |
| MobileNetV3-Small | 5.8 | 5.3 | 4315 |
| Swin-Tiny | 105.0 | 11.1 | 214 |

Đọc kết quả:

- Swin-Tiny giảm ít nhất (0.1365, so với 0.2066 của ConvNeXt-Tiny) và tốt hơn ConvNeXt-Tiny ở cả năm suy giảm. Chênh mức giảm khoảng 0.07, lớn hơn nhiều ngưỡng 0.01.
- Quy tắc đã khóa ở seed 42 vẫn chọn ConvNeXt-Tiny. F1 sạch của Swin-Tiny kém ConvNeXt-Tiny 0.0111, vượt ngưỡng 0.01 là 0.0011, nên Swin-Tiny không nằm trong nhóm "ngang nhau" (`selection.candidates` chỉ có ResNet-50 và ConvNeXt-Tiny). Chênh 0.0111 nằm trong vùng nhiễu của validation 444 ảnh.
- Ở seed 42 mới một seed cho mỗi model, mỗi suy giảm một mức, nên chưa kết luận Swin-Tiny bền hơn. Phần chạy lặp bên dưới trả lời câu này.
- Giả thuyết chưa kiểm tra: checkpoint Swin-Tiny ở epoch 5 (learning rate còn cao) so với epoch 24 của ConvNeXt-Tiny, và một model chưa khớp kỹ vào ảnh sạch có thể ổn định hơn dưới suy giảm. Chưa có bằng chứng.
- Độ trễ batch 1 của Swin-Tiny gấp khoảng 1.8 lần ConvNeXt-Tiny, throughput batch 32 gần nhau.
- Mờ và nhiễu làm mọi model giảm rất mạnh.

## Chạy lặp seed và chọn reference

Để biết các chênh lệch có ổn định không, ba model được chạy lặp với seed 43 và 44: Swin-Tiny, ResNet-50 và ConvNeXt-Tiny (danh sách seed 42, 43, 44 chốt trước khi chạy, không bỏ seed nào theo kết quả). Giao thức train và phép đo độ bền giữ nguyên. Chỉ khóa `seed` trong `configs/train_<model>_seed<N>.yaml` đổi, ghi đè seed trong `configs/pipeline.yaml`. Split và seed của suy giảm không đổi, nên mọi lần chạy được chấm trên đúng cùng ảnh. Notebook `03_repeat_seeds.ipynb`, cấu hình đo `configs/reference_selection_seed<N>.yaml`.

Vì sao chọn ba model này (ConvNeXt-Tiny khi đó vẫn là reference):

- **ConvNeXt-Tiny**: reference lúc đó, cần biết điểm của nó ổn định đến đâu.
- **Swin-Tiny**: model đối chiếu thêm. Câu hỏi là nó có thật sự bền hơn ConvNeXt-Tiny không.
- **ResNet-50**: ứng viên sát nút của ConvNeXt-Tiny khi chọn reference (F1 sạch chênh 0.0055, quyết định dựa vào mức giảm 0.207 so với 0.265). Phase 3 ghi cần chạy lặp khi điểm gần nhau.

Vì sao không chạy lặp MobileNetV3-Small:

- Nó không nằm trong câu hỏi nào ở trên và đã bị loại khỏi nhóm ứng viên reference bằng quy tắc khóa ở bước F1 sạch: 0.9292, kém ConvNeXt-Tiny 0.0483, gấp khoảng năm lần ngưỡng 0.01.
- Mức giảm dưới suy giảm của nó cũng lớn nhất trong bốn model (0.3005, so với 0.2066 của ConvNeXt-Tiny, 0.2648 của ResNet-50 và 0.1365 của Swin-Tiny).
- Chênh lệch lớn hơn nhiều so với độ nhiễu của validation 444 ảnh (khoảng 0.01), nên một seed đủ để thấy nó kém hơn. Đây là đánh giá của nhóm dựa trên một seed, không phải kết quả đã đo lặp.
- Tiết kiệm GPU: khoảng 9 phút mỗi lần chạy trên Tesla T4 (552 giây ở seed 42).

### Kết quả

Mức giảm F1 dưới suy giảm (càng thấp càng bền), validation 444 ảnh:

| Seed | Swin-Tiny | ConvNeXt-Tiny | ResNet-50 |
|---|---:|---:|---:|
| 42 | 0.1365 | 0.2066 | 0.2648 |
| 43 | 0.1294 | 0.2546 | 0.2398 |
| 44 | 0.1411 | 0.1954 | 0.2748 |
| Trung bình ± độ lệch chuẩn | 0.1357 ± 0.0059 | 0.2189 ± 0.0314 | 0.2598 ± 0.0180 |
| Khoảng (nhỏ nhất đến lớn nhất) | 0.129 đến 0.141 | 0.195 đến 0.255 | 0.240 đến 0.275 |

Macro F1 sạch trung bình: ConvNeXt-Tiny 0.9779, ResNet-50 0.9736, Swin-Tiny 0.9734. F1 trung bình dưới năm suy giảm: Swin-Tiny 0.8377, ConvNeXt-Tiny 0.7590, ResNet-50 0.7138. Bảng đầy đủ ở `seed_repeat_summary.csv`.

### Quyết định: đổi reference sang Swin-Tiny

Áp quy tắc khóa (giữ model có F1 sạch trong 0.01 so với model tốt nhất, rồi lấy model có mức giảm thấp nhất) lên trung bình ba seed:

- F1 sạch của Swin-Tiny kém ConvNeXt-Tiny 0.0045, trong ngưỡng 0.01, nên Swin-Tiny là ứng viên. Ở seed 42 riêng lẻ nó kém 0.0111 và bị loại, đó là dao động của một seed.
- Trong ba ứng viên, Swin-Tiny có mức giảm thấp nhất, 0.136 so với 0.219 và 0.260.
- Nhóm quyết định đổi reference sang Swin-Tiny. Đây là quyết định sau khi xem kết quả, dựa trên trung bình ba seed chứ không dựa riêng seed 42, nơi quy tắc đã khóa loại Swin-Tiny. Việc đổi kéo theo chạy lại task 4.2 trở đi với Swin-Tiny.

### Vì sao dừng ở ba seed

- Thứ hạng mức giảm không đổi ở cả ba seed: Swin-Tiny thấp nhất, và khoảng của nó không chồng lên khoảng của hai model còn lại. Giá trị tệ nhất của Swin-Tiny (0.141) vẫn tốt hơn giá trị tốt nhất của ConvNeXt-Tiny (0.195) và ResNet-50 (0.240).
- Khoảng cách giữa Swin-Tiny và ConvNeXt-Tiny là 0.083, bằng khoảng 14 lần độ lệch chuẩn của Swin-Tiny (0.0059). Để đổi thứ hạng, một seed mới phải lệch khoảng 0.05 so với mọi giá trị đã thấy.
- Mục đích chỉ là chọn reference model, không phải ước lượng chính xác độ bền hay chứng minh bằng kiểm định thống kê. Với ba điểm, đây là lập luận dựa trên khoảng cách, không phải p-value.

## Giới hạn

- Mọi lần chạy dùng cùng split và cùng ảnh hỏng. Độ lệch chuẩn chỉ phản ánh độ biến thiên do khởi tạo và thứ tự dữ liệu, không phản ánh split khác. Kết luận chỉ áp dụng cho split khóa này.
- Độ bền đo trên validation, mỗi kiểu suy giảm một mức. Tập test chưa mở.
- Độ trễ của Swin-Tiny ở batch 1 cao hơn (khoảng 11.9 ms mỗi ảnh so với 6.3 ms của ConvNeXt-Tiny, đo ở seed 44).
- MobileNetV3-Small và HOG + SVM chỉ có một seed.

## File

Mỗi model có thư mục riêng, tên file có hậu tố `_seed<N>` (N = 42, 43, 44; HOG + SVM và MobileNetV3-Small chỉ có `_seed42`):

- `<model>_seed<N>.json`: toàn bộ số liệu, hyperparameter và lịch sử từng epoch (HOG + SVM không có epoch nên `history` rỗng, và không có `_history.png`)
- `<model>_seed<N>_history.png`: loss và val macro F1 theo epoch
- `<model>_seed<N>_confusion_matrix.png`: confusion matrix của checkpoint đã chọn

Cạnh thư mục các model:

- `reference_selection.json`: đo độ bền ba model, seed 42
- `reference_selection_with_swin.json`: đo độ bền bốn model, seed 42
- `reference_selection_seed43.json`, `reference_selection_seed44.json`: đo độ bền ba model ở seed 43 và 44
- `seed_repeat_summary.csv`: bảng tổng hợp qua ba seed

Code: `src/training/run.py`, config: `configs/train_<model>.yaml` và `configs/train_<model>_seed<N>.yaml`, notebook đã chạy: `notebooks/03_train_<model>.ipynb`, `03_repeat_seeds.ipynb`.
