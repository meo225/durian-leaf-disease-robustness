# Kết quả hiệu chỉnh bộ chọn xử lý ảnh, task 5.3

Bộ chọn nhìn một ảnh rồi quyết định giữ nguyên hay xử lý bằng unsharp hoặc bilateral. Thiết kế nằm ở `docs/task-5.1-selector-design.md`. Thư mục này ghi kết quả hiệu chỉnh trên validation và quyết định khóa bộ chọn.

**Kết luận ngắn.** Bộ chọn được khóa là bộ của lần chạy 1 gồm hai luật. Lần chạy 2 thử sửa một lỗi đã biết của luật mờ nhưng không đạt ngưỡng đã đặt trước, nên bộ chọn không đổi. File khóa là `selector_calibrated.json` ở thư mục này.

Mọi số liệu đo trên **tập validation**, 444 ảnh, với Swin-Tiny đóng băng. Tập test chưa mở.

## Hai lần chạy

| | Notebook trên Kaggle | Commit | Thư mục |
|---|---|---|---|
| Lần 1 | https://www.kaggle.com/code/vtnmai/09-selector-calibration?scriptVersionId=364940521 | `bb7cd5e` | `run1/` và `notebooks/09_selector_calibration.ipynb` |
| Lần 2 | https://www.kaggle.com/code/vtnmai/09-selector-calibration/output?scriptVersionId=364969899 | `368c71e` | `run2/` và `notebooks/09_selector_calibration_run2.ipynb` |

Notebook đã chạy của lần 1 là `notebooks/09_selector_calibration.ipynb` và của lần 2 là `notebooks/09_selector_calibration_run2.ipynb`.

## Kết quả lần 1

| Chỉ số | Không xử lý | Bộ chọn |
|---|---|---|
| Macro F1 trung bình 42 điều kiện | 0.8607 | 0.8802, tăng 0.0195 |
| Macro F1 ảnh sạch | 0.9664 | 0.9664, không đổi |
| Tỉ lệ ảnh sạch bị xử lý | 0 | 1.1 phần trăm |
| Số điều kiện tốt hơn, bằng, kém hơn | | 25, 10, 7 |

Cận trên theo từng ảnh, tức nếu biết nhãn để chọn hành động tốt nhất cho mỗi ảnh, là 0.9209. Dư địa là 0.060 và bộ chọn lấy được khoảng 32 phần trăm.

Cố định một hành động cho mọi ảnh đều tệ hơn không xử lý: unsharp_strong 0.747, unsharp_mild 0.839, bilateral_wide 0.828, bilateral_narrow 0.840. Giá trị của bộ chọn nằm ở việc biết khi nào nên xử lý, không phải ở một kỹ thuật tốt hơn.

### Hai luật tìm được

1. **Luật nhiễu.** Nếu `noise_sigma` từ 0.0106 trở lên và `luma_mean` từ 0.31 trở lên thì dùng bilateral_wide. Đóng góp 0.0052.
2. **Luật mờ.** Nếu `mscn_var` từ 0.18 trở xuống và `noise_sigma` từ 0.0034 trở xuống thì dùng unsharp_strong. Đóng góp 0.0143, tức khoảng ba phần tư tổng mức tăng.

### Tăng ở đâu

| Họ | Mức tăng F1 ở các mức 1, 2, 3 |
|---|---|
| lowres | 0.057, 0.115, 0.095 |
| blur | giảm 0.003, tăng 0.120, tăng 0.085 |
| motion_blur | 0.006, 0.058, 0.083 |
| noise | 0.000, 0.043, 0.031 |
| sensor_noise | 0.010, 0.039, 0.039 |

Hai luật tìm lại đúng bản đồ của task 4.5: unsharp cho mờ và độ phân giải thấp, bilateral cho nhiễu. Đây chưa phải bằng chứng độc lập, vì chính 4.5 đã chọn tập hành động.

Bộ chọn đứng yên ở glare, color_cast và phần lớn shadow, jpeg, lowlight. Với lowlight là do luật nhiễu bị chặn khi ảnh quá tối.

### Độ ổn định

Hiệu chỉnh trên bốn phần ảnh và chấm trên một phần giữ lại, lặp 5 lần. Mức tăng giữ lại là 0.0181 với độ lệch 0.0044 và dương ở cả 5 phần. Ngưỡng gần như không đổi giữa các phần. Hai điểm cần lưu ý:

- Ở phần 2 điều kiện phụ của luật mờ đổi từ `noise_sigma` sang `luma_mean`, nên cấu trúc luật chưa hoàn toàn cố định.
- Tỉ lệ ảnh sạch bị xử lý ở phần giữ lại là 0.022 ở phần 0 và phần 2, hơi vượt trần 0.02. Mỗi phần chỉ có khoảng 89 ảnh sạch nên số này dao động mạnh.

## Vấn đề gặp phải ở lần 1

Có 7 điều kiện kém hơn không xử lý:

| Điều kiện | Mức đổi | Tỉ lệ ảnh bị xử lý |
|---|---|---|
| shadow_s3 | giảm 0.048 | 48 phần trăm |
| contrast_low_s3 | giảm 0.009 | 68 phần trăm |
| jpeg_s3 | giảm 0.007 | 5 phần trăm |
| contrast_low_s2 | giảm 0.005 | 9 phần trăm |
| brightness_bright_s3 | giảm 0.004 | 30 phần trăm |
| blur_s1 | giảm 0.003 | 4 phần trăm |
| noise_s1 | giảm 0.0002 | 98 phần trăm |

Lỗi thật chỉ là shadow_s3 và contrast_low. Ảnh có bóng đổ hoặc tương phản thấp cũng có `mscn_var` thấp nên luật mờ tưởng là ảnh mờ rồi làm nét nhầm.

Điều này cũng giải thích vì sao brightness_dark_s3 tăng 0.050. Unsharp vô tình kéo giãn tương phản ảnh tối, chứ ảnh đó không mờ. Mức tăng này là may mắn, không nên tính là bằng chứng bộ chọn nhận ra ảnh tối.

## Thử nghiệm lần 2

Nhóm thử thêm một mệnh đề chặn thứ hai cho luật mờ, chọn một trong hai ứng viên là `luma_std` từ ngưỡng trở lên hoặc `dark_fraction` từ ngưỡng trở xuống. Quy tắc đặt trước: mệnh đề chỉ được giữ nếu mục tiêu tăng thêm ít nhất `min_gain` là 0.002. Đây là chỉnh sau khi đã thấy kết quả validation, và tập test chưa mở.

Kết quả: mệnh đề tốt nhất chỉ làm mục tiêu tăng thêm **0.0006**, dưới 0.002, nên bị bỏ. Hai luật, mục tiêu 0.8802, 25 điều kiện tốt hơn và 7 kém hơn, và kết quả kiểm tra chéo đều giống hệt lần 1. Chi tiết nằm ở `run2/calibration_log.json`, mục `extra_guard` của luật `blur`.

Lý do mệnh đề không giúp được: chặn ảnh bóng đổ và tương phản thấp có cứu được shadow_s3 và contrast_low, nhưng cùng mệnh đề cũng chặn luật mờ trên ảnh tối, nơi luật đang thắng nhờ may. Hai bên gần triệt tiêu nhau.

## Vì sao giữ bộ chọn của lần 1

1. Mệnh đề thêm không đạt ngưỡng đã đặt trước. Hạ `min_gain` để cho nó lọt là chỉnh ngưỡng sau khi thấy kết quả.
2. Hai lần chạy cho cùng một bộ chọn, nên giữ bản đơn giản hơn với hai mệnh đề mỗi luật.
3. Lần 1 là bản đã được kiểm tra chéo và đã có đầy đủ kết quả theo từng điều kiện.

Lỗi shadow_s3 và contrast_low được ghi như một giới hạn đã biết, không giấu đi.

## Kết luận

- Bộ chọn hai luật tăng macro F1 trung bình 0.0195 trên validation, không làm hại ảnh sạch, và mọi kỹ thuật cố định đều thua không xử lý. Giá trị nằm ở việc chọn đúng lúc.
- Mức tăng ổn định qua kiểm tra chéo, nhưng chỉ lấy được khoảng một phần ba dư địa theo cận trên.
- Luật mờ có lỗi đã biết với ảnh bóng đổ và tương phản thấp. Một mệnh đề chặn đơn giản không sửa được mà không mất chỗ khác. Muốn sửa cần đặc trưng tách được ảnh tối và tương phản thấp khỏi ảnh mờ, nên để làm hướng phát triển.
- Chưa có bằng chứng ngoài mẫu. Bước kiểm chứng thật nằm ở task 5.5 với 6 họ chưa thấy và task 6.1 trên test.

## Đối chiếu giả thuyết đã đăng ký

| Giả thuyết | Trạng thái |
|---|---|
| H1 bộ chọn tốt hơn không xử lý | Hướng đúng trên validation. Khoảng tin cậy sẽ tính ở 6.1 trên test |
| H2 an toàn trên ảnh sạch | Đạt. F1 sạch không đổi, tỉ lệ kích hoạt 1.1 phần trăm |
| H3 không kém kỹ thuật cố định tốt nhất | Hướng đúng, vì mọi kỹ thuật cố định đều thấp hơn bộ chọn. So ghép cặp theo ảnh làm ở 6.1 |
| H4 kém huấn luyện lại | Chưa đo. Dự kiến kém vì trần oracle thấp hơn mức tăng của huấn luyện lại |

## Giới hạn

- Số liệu trên validation, nơi chính ngưỡng được tìm ra.
- Mười bốn họ suy giảm đều là họ đã thấy lúc hiệu chỉnh.
- Tập hành động đã được chọn từ kết quả test ở 4.5 và cấu hình xử lý được chọn trên validation. Doc 5.1 đã ghi rõ.
- Mức 3 của một số họ trong 14 họ khá nặng, nên số liệu mức 3 là kết quả với ảnh xấu thật.

## Khóa

`selector_calibrated.json` ở thư mục này có `locked: true`. Không đổi luật hay ngưỡng sau khi mở test ở task 6.1.

## File trong thư mục

- `selector_calibrated.json`: bộ chọn đã khóa, lấy từ lần 1.
- `run1/`: kết quả đầy đủ lần 1 gồm bộ chọn, nhật ký tìm ngưỡng, điểm từng điều kiện và biểu đồ theo họ.
- `run2/calibration_log.json`: nhật ký lần 2, có mục thử mệnh đề chặn thêm.
