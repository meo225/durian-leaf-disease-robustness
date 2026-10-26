# Đối chứng huấn luyện lại bằng augmentation suy giảm (task 4.5)

Huấn luyện lại reference model, ConvNeXt-Tiny, bằng augmentation suy giảm ngẫu nhiên, để biết xử lý ảnh đầu vào hơn, kém hay bổ sung so với việc để model tự học bền hơn. Model này được lưu để dùng lại ở P6.1.

Cùng giao thức với baseline (`configs/train_convnext_tiny.yaml`): timm, trọng số ImageNet, AdamW, learning rate 2e-4, weight decay 1e-2, cosine, tối đa 30 epoch, patience 7, seed 42, một lần chạy. Cấu hình ở `configs/train_convnext_tiny_aug.yaml`, code ở `src/training/run_aug.py` và `src/corruption/augment.py`, notebook `notebooks/07_augmentation_control.ipynb`.

## Khác baseline ở hai điểm

- **Augmentation suy giảm:** ngoài lật và xoay như baseline, mỗi ảnh train với xác suất 0,5 bị áp một điều kiện chọn đều trong 42 điều kiện của benchmark 4.1 (14 họ × 3 mức). Giá trị của điều kiện nhân với hệ số ngẫu nhiên trong 1 ± 0,1, và phần ngẫu nhiên (nhiễu, hướng mờ, vị trí chói, hình bóng) mới ở mỗi lần gọi. Cả 14 họ đều được dùng.
- **Chọn checkpoint bằng validation hỗn hợp:** 444 ảnh sạch cộng mỗi họ ở mức 2 (suy giảm cố định theo benchmark 4.1), tổng 6.660 ảnh. Macro F1 trên tập này chọn epoch. Quy tắc khóa trước khi chạy. Tập test không được mở khi train.

## Quá trình train

Chạy trên Tesla T4, 30 epoch (không dừng sớm), 3.362 giây, epoch được chọn là 26, 27.823.973 tham số. Lịch sử từng epoch trong `convnext_tiny_aug_seed42.json` và `convnext_tiny_aug_seed42_history.png`.

| Điểm | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Validation hỗn hợp (dùng để chọn epoch) | 0.9693 | 0.9684 |
| Validation sạch | 0.9768 | 0.9756 |

Validation sạch của baseline ConvNeXt-Tiny là 0.9775, nên augmentation không làm giảm điểm ảnh sạch trên validation. Điểm validation hỗn hợp không so được với validation sạch vì tập khác nhau.

## Kết quả trên test

Đánh giá một lần trên 888 ảnh test, ảnh sạch và 42 điều kiện, cùng mẫu, nhãn và suy giảm với task 4.2. Reference model là ConvNeXt-Tiny baseline (kết quả ở `reports/sensitivity/`). Hai bên dùng đúng cùng thứ tự mẫu nên so sánh ghép cặp được.

| | Reference | Huấn luyện lại | Chênh |
|---|---:|---:|---:|
| Ảnh sạch | 0.9645 | **0.9804** | +0.016 |
| Trung bình 42 điều kiện | 0.797 | **0.960** | +0.163 |

Macro F1 theo mức suy giảm (trung bình các họ):

| Mức | Reference | Huấn luyện lại |
|---|---:|---:|
| 1 | 0.920 | 0.975 |
| 2 | 0.818 | 0.964 |
| 3 | 0.654 | 0.941 |

Theo họ suy giảm (trung bình ba mức):

| Họ | Reference | Huấn luyện lại | Chênh |
|---|---:|---:|---:|
| `lowres` | 0.708 | 0.968 | +0.260 |
| `contrast_high` | 0.715 | 0.953 | +0.238 |
| `blur` | 0.746 | 0.965 | +0.218 |
| `contrast_low` | 0.753 | 0.970 | +0.217 |
| `lowlight` | 0.771 | 0.963 | +0.192 |
| `motion_blur` | 0.774 | 0.962 | +0.188 |
| `brightness_bright` | 0.763 | 0.949 | +0.186 |
| `noise` | 0.793 | 0.956 | +0.163 |
| `brightness_dark` | 0.815 | 0.975 | +0.160 |
| `sensor_noise` | 0.818 | 0.962 | +0.144 |
| `color_cast` | 0.860 | 0.970 | +0.111 |
| `jpeg` | 0.828 | 0.917 | +0.089 |
| `shadow` | 0.869 | 0.956 | +0.087 |
| `glare` | 0.946 | 0.973 | +0.028 |

Năm điều kiện yếu nhất của model huấn luyện lại: `jpeg_s3` 0.865, `brightness_bright_s3` 0.909, `shadow_s3` 0.920, `contrast_high_s3` 0.920, `jpeg_s2` 0.936.

Recall từng lớp trên ảnh sạch (test), reference rồi huấn luyện lại: ALGAL_LEAF_SPOT 0.959 → 0.966, ALLOCARIDARA_ATTACK 0.956 → 0.984, HEALTHY_LEAF 0.974 → 0.995, LEAF_BLIGHT 0.952 → 0.973, PHOMOPSIS_LEAF_SPOT 0.977 → 0.983. Không lớp nào giảm.

Kiểm tra ghép cặp sơ bộ (McNemar chính xác trên từng ảnh, hiệu chỉnh Holm trên 42 điều kiện): model huấn luyện lại tốt hơn **có ý nghĩa ở cả 42 điều kiện**, không điều kiện nào kém hơn. Với ảnh sạch, chênh +0,016 có p = 0,004 (khoảng tin cậy bootstrap 95% từ +0,005 đến +0,026). Đây là phép kiểm tra nhanh để soát kết quả. Phân tích thống kê chính thức nằm ở task 4.5 phần bản đồ phục hồi.

## Đọc kết quả

- Với suy giảm cùng loại với lúc train, huấn luyện lại bằng augmentation làm tăng độ bền rất rõ ở mọi họ và mọi mức, không mất điểm ảnh sạch. Mức nặng (mức 3) cải thiện nhiều nhất, 0.654 lên 0.941.
- So với kỹ thuật xử lý ảnh: các kỹ thuật ở task 4.3 và 4.4 đo trên validation và chỉ phục hồi một phần ở một số họ, còn huấn luyện lại hơn reference ở mọi họ. Không so trực tiếp con số giữa validation và test ở đây. So sánh chính thức trên test, cùng mẫu, nằm ở bản đồ phục hồi của task 4.5.
- Điều kiện "huấn luyện lại cộng xử lý ảnh phía trước" thuộc P6.1, chưa đo ở đây.

## Giới hạn

- **Độ bền trong phân phối.** Augmentation dùng đúng 14 họ và các mức của benchmark, nên model được luyện đúng loại suy giảm bị kiểm tra. Kết quả chưa nói gì về suy giảm lạ chưa từng thấy. Cấu hình có sẵn `exclude_families` để giữ lại một số họ, lần chạy này chưa dùng.
- **Một seed.** Chưa biết dao động do huấn luyện. Điểm ảnh sạch trên test tăng +0,016 nhưng trên validation sạch thì bằng baseline (0.9768 so với 0.9775), nên phần tăng này có thể một phần do ngẫu nhiên của lần huấn luyện hoặc của tập test. Không nên kết luận augmentation làm tăng độ chính xác ảnh sạch nếu chưa có thêm seed.
- Checkpoint chọn trên validation hỗn hợp với mức 2, nên các điểm validation mức 2 hơi lạc quan. Các số ở bảng trên là test, không bị ảnh hưởng.
- Reference model cũng chỉ có một seed.

## File

| File | Nội dung |
|---|---|
| `convnext_tiny_aug_seed42.json` | cấu hình, augmentation, lịch sử từng epoch, điểm validation hỗn hợp và validation sạch |
| `convnext_tiny_aug_seed42_history.png` | loss và macro F1 theo epoch |
| `convnext_tiny_aug_seed42_confusion_matrix.png` | confusion matrix của checkpoint đã chọn (validation hỗn hợp) |
| `augmentation_val_conditions.csv`, `augmentation_test_conditions.csv` | macro F1, balanced accuracy, accuracy, macro precision, recall từng lớp, degradation cho ảnh sạch và 42 điều kiện |
| `augmentation_val_predictions.npz`, `augmentation_test_predictions.npz` | nhãn, đường dẫn ảnh, dự đoán và xác suất từng ảnh (khóa `retrained__convnext_tiny_aug__<điều kiện>` và hậu tố `__probs`), để tính khoảng tin cậy, McNemar và PR-AUC |

Checkpoint `convnext_tiny_aug_seed42.pt` (khoảng 106 MB) không vào Git (`.gitignore` chặn `*.pt`). Cần lưu ở một dataset Kaggle riêng để dùng lại ở P6.1.
