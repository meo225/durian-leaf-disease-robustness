# Đối chứng huấn luyện lại bằng augmentation suy giảm (task 4.5)

Huấn luyện lại reference model, Swin-Tiny, bằng augmentation suy giảm ngẫu nhiên, để biết xử lý ảnh đầu vào hơn, kém hay bổ sung so với việc để model tự học bền hơn. Model này được lưu để dùng lại ở P6.1.

Cùng giao thức với baseline (`configs/train_swin_tiny.yaml`): timm, trọng số ImageNet, AdamW, learning rate 2e-4, weight decay 1e-2, cosine, tối đa 30 epoch, patience 7, seed 42, một lần chạy. Cấu hình ở `configs/train_swin_tiny_aug.yaml`, code ở `src/training/run_aug.py` và `src/corruption/augment.py`, notebook `notebooks/07_augmentation_control.ipynb`.

Reference model đã đổi từ ConvNeXt-Tiny sang Swin-Tiny sau khi chạy lặp seed (xem `reports/baselines/README.md`). Lần chạy trước của task này dùng ConvNeXt-Tiny và đã được thay bằng kết quả dưới đây.

## Khác baseline ở hai điểm

- **Augmentation suy giảm:** ngoài lật và xoay như baseline, mỗi ảnh train với xác suất 0,5 bị áp một điều kiện chọn đều trong 42 điều kiện của benchmark 4.1 (14 họ × 3 mức). Giá trị của điều kiện nhân với hệ số ngẫu nhiên trong 1 ± 0,1, và phần ngẫu nhiên (nhiễu, hướng mờ, vị trí chói, hình bóng) mới ở mỗi lần gọi. Cả 14 họ đều được dùng.
- **Chọn checkpoint bằng validation hỗn hợp:** 444 ảnh sạch cộng mỗi họ ở mức 2 (suy giảm cố định theo benchmark 4.1), tổng 6.660 ảnh. Macro F1 trên tập này chọn epoch. Quy tắc khóa trước khi chạy. Tập test không được mở khi train.

## Quá trình train

Chạy trên Tesla T4, 30 epoch (không dừng sớm), 2.243 giây, epoch được chọn là 24, 27.523.199 tham số. Lịch sử từng epoch trong `swin_tiny_aug_seed42.json` và `swin_tiny_aug_seed42_history.png`.

| Điểm | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Validation hỗn hợp (dùng để chọn epoch) | 0.9649 | 0.9648 |
| Validation sạch | 0.9715 | 0.9714 |

Validation sạch của baseline Swin-Tiny (seed 42) là 0.9664, nên augmentation không làm giảm điểm ảnh sạch trên validation (+0.005, trong vùng nhiễu của 444 ảnh). Điểm validation hỗn hợp không so được với validation sạch vì tập khác nhau.

## Kết quả trên test

Đánh giá một lần trên 888 ảnh test, ảnh sạch và 42 điều kiện, cùng mẫu, nhãn và suy giảm với task 4.2. Reference model là Swin-Tiny baseline (kết quả ở `reports/sensitivity/`). Hai bên dùng đúng cùng thứ tự mẫu nên so sánh ghép cặp được.

| | Reference | Huấn luyện lại | Chênh |
|---|---:|---:|---:|
| Ảnh sạch | 0.9622 | **0.9871** | +0.025 |
| Trung bình 42 điều kiện | 0.869 | **0.965** | +0.096 |

Macro F1 theo mức suy giảm (trung bình các họ):

| Mức | Reference | Huấn luyện lại |
|---|---:|---:|
| 1 | 0.936 | 0.980 |
| 2 | 0.889 | 0.970 |
| 3 | 0.782 | 0.947 |

Theo họ suy giảm (trung bình ba mức):

| Họ | Reference | Huấn luyện lại | Chênh |
|---|---:|---:|---:|
| `lowres` | 0.827 | 0.981 | +0.154 |
| `blur` | 0.820 | 0.972 | +0.151 |
| `jpeg` | 0.794 | 0.930 | +0.135 |
| `brightness_bright` | 0.828 | 0.960 | +0.133 |
| `contrast_high` | 0.849 | 0.964 | +0.115 |
| `motion_blur` | 0.857 | 0.967 | +0.110 |
| `sensor_noise` | 0.859 | 0.963 | +0.104 |
| `noise` | 0.861 | 0.957 | +0.095 |
| `lowlight` | 0.870 | 0.962 | +0.092 |
| `color_cast` | 0.892 | 0.970 | +0.078 |
| `shadow` | 0.888 | 0.957 | +0.069 |
| `brightness_dark` | 0.926 | 0.977 | +0.050 |
| `glare` | 0.934 | 0.977 | +0.043 |
| `contrast_low` | 0.960 | 0.980 | +0.020 |

Năm điều kiện yếu nhất của model huấn luyện lại: `jpeg_s3` 0.879, `shadow_s3` 0.920, `brightness_bright_s3` 0.927, `noise_s3` 0.931, `jpeg_s2` 0.942.

Recall từng lớp trên ảnh sạch (test), reference rồi huấn luyện lại: ALGAL_LEAF_SPOT 0.925 → 0.979, ALLOCARIDARA_ATTACK 0.973 → 0.989, HEALTHY_LEAF 0.985 → 1.000, LEAF_BLIGHT 0.963 → 0.984, PHOMOPSIS_LEAF_SPOT 0.960 → 0.983. Không lớp nào giảm.

Kiểm tra ghép cặp sơ bộ (McNemar chính xác trên từng ảnh, hiệu chỉnh Holm trên 42 điều kiện): model huấn luyện lại tốt hơn **có ý nghĩa ở 40 trong 42 điều kiện**, không điều kiện nào kém hơn. Hai điều kiện còn lại là `brightness_dark_s1` và `brightness_dark_s2` (cùng p_holm = 0,094): model huấn luyện lại vẫn đúng nhiều ảnh hơn (19 so với 9 và 25 so với 12) nhưng chưa đủ để loại trừ ngẫu nhiên. Với ảnh sạch, chênh +0,025 có p < 0,001 (4 ảnh reference đúng mà huấn luyện lại sai, 26 ảnh ngược lại; khoảng tin cậy bootstrap 95% từ +0,013 đến +0,037). Đây là phép kiểm tra nhanh để soát kết quả. Phân tích thống kê chính thức nằm ở task 4.5 phần bản đồ phục hồi.

## Đọc kết quả

- Với suy giảm cùng loại với lúc train, huấn luyện lại bằng augmentation làm tăng độ bền ở mọi họ và mọi mức, không mất điểm ảnh sạch. Mức nặng (mức 3) cải thiện nhiều nhất, 0.782 lên 0.947. Mức tăng nhỏ hơn so với lần chạy ConvNeXt-Tiny trước đó vì reference Swin-Tiny vốn đã bền hơn (trung bình 42 điều kiện 0.869, so với 0.797 của ConvNeXt-Tiny).
- Họ hưởng lợi nhiều nhất là `lowres`, `blur` và `jpeg`, đều là mất chi tiết. Hai họ ít được lợi nhất là `contrast_low` và `glare`, vì reference đã gần trần ở đó (0.960 và 0.934).
- So với kỹ thuật xử lý ảnh: các kỹ thuật ở task 4.3 và 4.4 đo trên validation và chỉ phục hồi một phần ở một số họ, còn huấn luyện lại hơn reference ở mọi họ. Không so trực tiếp con số giữa validation và test ở đây. So sánh chính thức trên test, cùng mẫu, nằm ở bản đồ phục hồi của task 4.5.
- Điều kiện "huấn luyện lại cộng xử lý ảnh phía trước" thuộc P6.1, chưa đo ở đây.

## Giới hạn

- **Độ bền trong phân phối.** Augmentation dùng đúng 14 họ và các mức của benchmark, nên model được luyện đúng loại suy giảm bị kiểm tra. Kết quả chưa nói gì về suy giảm lạ chưa từng thấy. Cấu hình có sẵn `exclude_families` để giữ lại một số họ, lần chạy này chưa dùng.
- **Một seed.** Chưa biết dao động do huấn luyện. Điểm ảnh sạch trên test tăng +0,025 nhưng trên validation sạch chỉ tăng +0,005 (0.9715 so với 0.9664), nên phần tăng ở test có thể một phần do ngẫu nhiên của lần huấn luyện hoặc của tập test. Chạy lặp baseline cho thấy F1 sạch validation của Swin-Tiny dao động khoảng 0,006 giữa các seed. Không nên kết luận augmentation làm tăng độ chính xác ảnh sạch nếu chưa có thêm seed.
- Checkpoint chọn trên validation hỗn hợp với mức 2, nên các điểm validation mức 2 hơi lạc quan. Các số ở bảng trên là test, không bị ảnh hưởng.
- Reference model cũng chỉ có một seed (seed 42, cùng checkpoint dùng ở task 4.2).

## File

| File | Nội dung |
|---|---|
| `swin_tiny_aug_seed42.json` | cấu hình, augmentation, lịch sử từng epoch, điểm validation hỗn hợp và validation sạch |
| `swin_tiny_aug_seed42_history.png` | loss và macro F1 theo epoch |
| `swin_tiny_aug_seed42_confusion_matrix.png` | confusion matrix của checkpoint đã chọn (validation hỗn hợp) |
| `augmentation_val_conditions.csv`, `augmentation_test_conditions.csv` | macro F1, balanced accuracy, accuracy, macro precision, recall từng lớp, degradation cho ảnh sạch và 42 điều kiện |
| `augmentation_val_predictions.npz`, `augmentation_test_predictions.npz` | nhãn, đường dẫn ảnh, dự đoán và xác suất từng ảnh (khóa `retrained__swin_tiny_aug__<điều kiện>` và hậu tố `__probs`), để tính khoảng tin cậy, McNemar và PR-AUC |
| `augmentation_examples.png` | ảnh mẫu augmentation |

Checkpoint `swin_tiny_aug_seed42.pt` (khoảng 105 MB) không vào Git (`.gitignore` chặn `*.pt`). Cần lưu ở một dataset Kaggle riêng để dùng lại ở P6.1.
