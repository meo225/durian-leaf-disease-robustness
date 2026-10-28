# Phase 3. Baseline candidates

Task 3.1 của plan. Danh sách này khóa trước khi huấn luyện. Task 3.2, 3.3 và 3.4 chưa chạy.

## Tiêu chí

Cùng split DurianLDD đã khóa, năm lớp, ảnh 224×224, pipeline trong `configs/pipeline.yaml`. Checkpoint và hyperparameter chỉ chọn trên validation. Tập test không dùng ở bước này.

Mỗi candidate phải có trọng số pretrained trong `timm`, chạy được với batch 32 trên GPU Kaggle, và vừa với khoảng 3.105 ảnh train. Task 3.2 chạy ba model: CNN đối chứng, kiến trúc hiện đại, và model nhẹ.

## Không dùng làm lý do chọn

Bảng xếp hạng của Richter và Kim 2025 không chuyển sang dataset này. 93,82% của DurianLSNet và 98% của ConvNeXt trong AI-Driven 2026 không phải điểm của protocol đồ án. 99,61% của ETASR không so được.

## Không đưa vào danh sách train

DurianLSNet đúng vai model nhẹ trên DurianLDD, nhưng bài công bố dataset Kaggle và không công bố code. Không tự dựng lại mạng rồi gọi đó là baseline của bài.

Mask R-CNN thuộc bài toán localization. Xception và các model lớn hơn ConvNeXt-Tiny hoặc Swin-Tiny không đưa vào vì tập train nhỏ và hạn mức GPU.

## Danh sách khóa

Cấu hình máy đọc: `configs/baselines.yaml`. Cả ba fine-tune từ ImageNet. Thứ tự chạy là thứ tự trong bảng. ResNet-50 chạy trước.

| Thứ tự | Tên | `timm` | Vai trò |
|---|---|---|---|
| 1 | ResNet-50 | `resnet50` | CNN đối chứng. Lần train đầu để kiểm tra protocol |
| 2 | ConvNeXt-Tiny | `convnext_tiny` | Kiến trúc hiện đại, cùng cỡ tham số với ResNet-50 |
| 3 | MobileNetV3-Small | `mobilenetv3_small_100` | Model nhẹ, thay DurianLSNet khi chưa có code |

Swin-Tiny và DINOv2 không nằm trong đợt train này. Swin-Tiny chỉ xét lại khi ba điểm validation quá gần và cần thêm một họ kiến trúc. DINOv2 để frozen là phép kiểm tra sau, khi cần xem kết luận về xử lý ảnh có phụ thuộc fine-tune hay không.

## Quy tắc cho task 3.2 trở đi

Mỗi baseline có một file `configs/train_<model>.yaml` và một notebook `notebooks/03_train_<model>.ipynb`, cùng chạy qua `src/training/run.py`. Kết quả đã chốt nằm ở `reports/baselines/README.md`. Checkpoint chọn bằng macro F1 trên validation. Cùng file kết quả có balanced accuracy, recall từng lớp, confusion matrix, số tham số, thời gian train và loại GPU.

Seed đầu là 42. Chạy lặp khi điểm validation gần nhau hoặc khi chốt reference model. Richter và Kim là căn cứ để lặp, không thay cách chọn model bằng validation. Nếu điểm sạch gần nhau, sanity check độ bền trên validation trước khi chốt reference model. Tập test vẫn khóa đến lần đánh giá cuối.

## Swin-Tiny: model đối chiếu thêm sau khi khóa

Danh sách khóa ở trên không đổi. Swin-Tiny được thêm sau khi reference model đã chọn, ở nhánh `baseline-Swin-Tiny`, để xem một họ transformer cửa sổ trượt có khác ConvNeXt-Tiny về F1 sạch và độ bền hay không. Điều kiện "ba điểm validation quá gần" ở trên đã xảy ra giữa ResNet-50 và ConvNeXt-Tiny (chênh 0,0055).

Giao thức train giữ nguyên: `configs/train_swin_tiny.yaml`, notebook `03_train_swin_tiny.ipynb`, cùng `src/training/run.py`. Chỉ khác `backbone: swin_tiny_patch4_window7_224`, nhận ảnh 224×224 như pipeline. Learning rate 2e-4 chưa dò riêng cho Swin, nên điểm của Swin-Tiny có thể thấp hơn mức đạt được khi dò.

Đo độ bền bằng `configs/reference_selection_swin.yaml` và mục 5 của notebook `03_select_reference_model.ipynb`: cùng code `src/evaluation/reference.py`, cùng validation và năm suy giảm của task 3.3, chạy lại cả bốn model. Kết quả ghi vào `reports/baselines/reference_selection_with_swin.json`, không ghi đè `reference_selection.json`.

Reference model vẫn là ConvNeXt-Tiny, và tập test không mở ở bước này. Dòng "selection" trong file kết quả chỉ cho biết quy tắc đã khóa sẽ chọn gì nếu có thêm Swin-Tiny. Nếu Swin-Tiny tốt hơn, việc đổi reference model sẽ kéo theo chạy lại task 4.2 trở đi, nên cần nhóm quyết định riêng.
