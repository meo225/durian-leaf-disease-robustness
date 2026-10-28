# Benchmark suy giảm ảnh (task 4.1)

Mười bốn họ suy giảm, mỗi họ ba mức, tổng 42 điều kiện. Cấu hình ở `configs/corruption.yaml`, code ở `src/corruption/`. Năm họ đầu thuộc bốn nhóm của plan. Chín họ mở rộng mô phỏng điều kiện chụp ngoài vườn.

Suy giảm áp lên đúng ảnh test đã khóa, giữ nguyên mẫu và nhãn giữa ảnh sạch và ảnh suy giảm. Phần ngẫu nhiên (nhiễu, hướng mờ, vị trí chói, hình bóng) lấy seed từ nội dung ảnh, tên điều kiện và seed 42, nên một ảnh luôn nhận cùng một suy giảm. Ảnh được làm tròn về 8 bit sau khi suy giảm.

## Hiệu chỉnh trên validation

Các mức chọn bằng cách xem ảnh mẫu (`severity_examples.png`) và đo ConvNeXt-Tiny, reference model lúc đó, trên tập validation qua hai vòng. Sau khi nhóm đổi reference sang Swin-Tiny, các mức được giữ nguyên (chúng là thuộc tính của benchmark) và bảng dưới ghi macro F1 của Swin-Tiny trên validation, đo lại bằng `reports/corruptions/validation_calibration.json`. Macro F1 sạch là 0.966. Bảng ghi giá trị mức và macro F1 trong ngoặc.

| Họ | Nhóm | Mô tả | Mức 1 | Mức 2 | Mức 3 |
|---|---|---|---|---|---|
| `brightness_dark` | plan | Tối đi (nhân độ sáng) | 0.7 (0.943) | 0.5 (0.925) | 0.3 (0.811) |
| `brightness_bright` | plan | Sáng lên, cháy sáng | 1.3 (0.936) | 1.6 (0.833) | 2.0 (0.691) |
| `contrast_low` | plan | Tương phản thấp | 0.7 (0.962) | 0.5 (0.954) | 0.3 (0.927) |
| `blur` | plan | Mờ Gauss (mất nét), sigma px | 0.5 (0.958) | 1.0 (0.828) | 2.0 (0.653) |
| `noise` | plan | Nhiễu Gauss, độ lệch chuẩn | 0.03 (0.936) | 0.06 (0.866) | 0.1 (0.759) |
| `contrast_high` | mở rộng | Tương phản cao (nắng gắt) | 1.5 (0.927) | 2.2 (0.864) | 3.0 (0.761) |
| `glare` | mở rộng | Vệt chói nắng cục bộ | 0.4 (0.960) | 0.7 (0.938) | 1.0 (0.901) |
| `shadow` | mở rộng | Bóng tán lá loang lổ | 0.4 (0.941) | 0.7 (0.895) | 1.0 (0.832) |
| `lowlight` | mở rộng | Thiếu sáng kèm nhiễu hạt (hệ số sáng) | 0.7 (0.897) | 0.5 (0.843) | 0.35 (0.754) |
| `motion_blur` | mở rộng | Mờ chuyển động (rung tay), độ dài px | 3 (0.921) | 5 (0.864) | 9 (0.739) |
| `sensor_noise` | mở rộng | Nhiễu cảm biến, peak (thấp là nhiễu hơn) | 250 (0.921) | 120 (0.864) | 60 (0.812) |
| `jpeg` | mở rộng | Nén JPEG, chất lượng (thấp là nặng hơn) | 50 (0.915) | 25 (0.863) | 10 (0.568) |
| `lowres` | mở rộng | Độ phân giải thấp, tỉ lệ giữ lại | 0.75 (0.859) | 0.6 (0.822) | 0.45 (0.779) |
| `color_cast` | mở rộng | Ám vàng đỏ (nắng chiều) | 0.15 (0.948) | 0.3 (0.925) | 0.45 (0.851) |

Các nhận xét sau là của hai vòng hiệu chỉnh bằng ConvNeXt-Tiny, không viết lại cho Swin-Tiny. Vòng đầu cho thấy `lowlight`, `motion_blur`, `sensor_noise`, `jpeg`, `lowres` và `blur` quá mạnh ngay từ mức 1 nên đã hạ. Vòng hai đạt: điểm giảm dần theo mức và không họ nào về mức đoán bừa ở mức 3. Hai họ còn lệch nhẹ nhưng giữ nguyên: `glare` ít ảnh hưởng (mức 3 vẫn 0.918) và `lowres` mức 1 đã 0.768. Hai điểm này là kết quả của model, không chỉnh thêm. Kết quả chi tiết nằm trong `notebooks/04_corruption_benchmark.ipynb`.

## Khóa

`locked: true` từ commit này. Mức không được chỉnh theo kết quả test. Test chỉ mở ở task 4.2 với `FINAL_EVAL=1`.
