# Phân tích độ nhạy của reference model với suy giảm ảnh (task 4.2)

ConvNeXt-Tiny (reference model, seed 42) đo trên ảnh sạch và 42 điều kiện của benchmark 4.1, ở validation và test. Notebook: `notebooks/06_degradation_sensitivity.ipynb`. Code: `src/evaluation/sensitivity.py`. Ngưỡng phân loại đặt trước trong `configs/sensitivity.yaml`.

## Thiết lập

- Test mở một lần với `FINAL_EVAL=1`, mức suy giảm đã khóa ở 4.1 và không chỉnh theo kết quả.
- Mọi so sánh là ghép cặp trên cùng mẫu. Khoảng tin cậy 95% bằng bootstrap 2000 lần lấy lại mẫu theo cặp. Kiểm định McNemar chính xác hai phía trên các ảnh đổi từ đúng sang sai và ngược lại, hiệu chỉnh Holm trên 42 phép thử.
- Chạy lại validation khớp bảng hiệu chỉnh 4.1: lệch lớn nhất 0,0005 (`shadow_s1`).
- Degradation là macro F1 sạch trừ macro F1 suy giảm.

## Điểm trên ảnh sạch

| Split | Ảnh | Macro F1 (95%) | Accuracy | Ảnh sai |
|---|---|---|---|---|
| validation | 444 | 0,9775 (0,963 đến 0,990) | 0,9775 | 10 |
| test | 888 | 0,9645 (0,952 đến 0,976) | 0,9640 | 32 |

Validation cao hơn test, một phần vì checkpoint được chọn trên validation.

## Degradation theo họ và mức (test)

Dấu `*` là khác biệt có ý nghĩa sau Holm (alpha 0,05). Loại: severe từ 0,15, moderate từ 0,05, mild từ 0,02 và có ý nghĩa, theo giá trị trung bình qua ba mức. `sớm` là đã giảm từ 0,05 ngay ở mức 1.

| Họ | Nhóm | Mức 1 | Mức 2 | Mức 3 | Trung bình | Loại | Task xử lý |
|---|---|---|---|---|---|---|---|
| `lowres` | extension | 0.199* | 0.256* | 0.314* | 0.256 | severe, sớm | 4.4 |
| `contrast_high` | extension | 0.064* | 0.231* | 0.454* | 0.250 | severe, sớm | 4.3 |
| `blur` | plan | 0.012 | 0.258* | 0.384* | 0.218 | severe | 4.4 |
| `contrast_low` | plan | 0.013 | 0.096* | 0.525* | 0.211 | severe | 4.3 |
| `brightness_bright` | plan | 0.031* | 0.173* | 0.401* | 0.202 | severe | 4.3 |
| `lowlight` | extension | 0.078* | 0.161* | 0.341* | 0.193 | severe, sớm | 4.3+4.4 |
| `motion_blur` | extension | 0.066* | 0.192* | 0.313* | 0.190 | severe, sớm | 4.4 |
| `noise` | plan | 0.014 | 0.167* | 0.334* | 0.171 | severe | 4.4 |
| `brightness_dark` | plan | 0.013 | 0.089* | 0.347* | 0.150 | moderate | 4.3 |
| `sensor_noise` | extension | 0.055* | 0.152* | 0.231* | 0.146 | moderate, sớm | 4.4 |
| `jpeg` | extension | 0.049* | 0.072* | 0.288* | 0.136 | moderate | không có |
| `color_cast` | extension | 0.016 | 0.102* | 0.197* | 0.105 | moderate | 4.3 |
| `shadow` | extension | 0.017 | 0.082* | 0.188* | 0.096 | moderate | 4.3 |
| `glare` | extension | -0.003 | 0.022* | 0.037* | 0.019 | negligible | 4.3 |

- 35 trên 42 điều kiện khác ảnh sạch có ý nghĩa. Bảy điều kiện không có ý nghĩa đều ở mức 1: `brightness_dark_s1`, `contrast_low_s1`, `blur_s1`, `noise_s1`, `glare_s1`, `shadow_s1`, `color_cast_s1`. Suy giảm nhẹ trong các họ này chưa phân biệt được với dao động chọn mẫu.
- `glare` giảm đến 0,037 ở mức 3, có ý nghĩa thống kê ở mức 2 và 3 nhưng quá nhỏ để ảnh hưởng thực tế. `brightness_dark` nằm sát ngưỡng severe (0,1497 so với 0,15).
- Năm họ giảm từ 0,05 ngay ở mức 1: `lowres` (0,199), `lowlight`, `contrast_high`, `motion_blur`, `sensor_noise`. `lowres` hỏng nặng nhất ngay cả ở mức nhẹ nhất.
- Mức 3 nặng nhất: `contrast_low_s3` (F1 0,439), `contrast_high_s3` (0,511), `brightness_bright_s3` (0,564), `blur_s3` (0,580).

Hình: `figures/test_degradation_heatmap.png`, `figures/test_severity_curves.png`, `figures/test_priority.png`.

## Phân tích theo lớp

Recall từng lớp thay đổi không đều. Hình `figures/test_class_recall_change.png`, `figures/test_confusion_worst.png`.

- `ALLOCARIDARA_ATTACK` mất nhiều nhất khi ảnh mờ hoặc độ phân giải thấp: recall còn 0,21 ở `lowres_s1`, 0,08 ở `blur_s2`, 0,00 ở `blur_s3` và `lowres_s3`, 0,07 ở `motion_blur_s3`. Ở `lowres_s1`, 136 trên 148 ảnh bị đổi sang sai thuộc lớp này. Lớp dựa vào chi tiết nhỏ nên mất ngay khi chi tiết bị làm mờ.
- Khi mờ, ảnh đổi sang sai chủ yếu thành `ALGAL_LEAF_SPOT` (khoảng 0,6) và `PHOMOPSIS_LEAF_SPOT`.
- Với `noise`, `sensor_noise` và `jpeg_s1`, dự đoán dồn về `ALLOCARIDARA_ATTACK`: 0,60 đến 0,79 số ảnh đổi sang sai ở `noise_s2`, `noise_s3` và `sensor_noise_s1` đến `s3`, 0,56 ở `jpeg_s1`. Lớp này trở thành lớp mặc định khi có nhiễu.
- `contrast_high` làm mất `ALGAL_LEAF_SPOT` (recall giảm 0,85 ở mức 3), `brightness_bright` làm mất `ALGAL_LEAF_SPOT` và `HEALTHY_LEAF`. Ảnh đổi sang sai dồn về `LEAF_BLIGHT` (0,52 và 0,69 ở mức 3).
- `contrast_low_s3` làm sụt recall của ba lớp: `ALLOCARIDARA_ATTACK` (−0,78), `LEAF_BLIGHT` (−0,70), `PHOMOPSIS_LEAF_SPOT` (−0,69). `HEALTHY_LEAF` hầu như giữ (−0,04), `ALGAL_LEAF_SPOT` giảm 0,18.

Hình `figures/test_flip_destinations.png`. Bảng đầy đủ: `flip_destinations_test.csv`.

## Prediction flip theo từng mẫu

Trên 856 ảnh đúng ở ảnh sạch, 86 ảnh không bị đổi ở điều kiện nào và 30 ảnh bị đổi ở ít nhất một nửa trong 42 điều kiện. Theo lớp, `ALLOCARIDARA_ATTACK` dễ đổi nhất (cả 175 ảnh đều bị đổi ở ít nhất một điều kiện, 17 ảnh bị đổi ở ít nhất một nửa số điều kiện, nhiều nhất là 37 trên 42), `LEAF_BLIGHT` bền nhất (40 trên 178 ảnh không đổi). Danh sách ảnh: `sample_flips_test.csv`. Hình `figures/test_sample_flips.png`, `figures/test_flip_counts.png`.

## Validation so với test

Spearman giữa degradation của 42 điều kiện trên hai split là 0,984. Chênh lệch macro F1 trung bình 0,016 và lớn nhất 0,052. Các mức hiệu chỉnh trên validation cho mức giảm tương tự ở test. Hình `figures/val_vs_test_degradation.png`.

## Thống kê chất lượng ảnh

Đo trên 888 ảnh test ở kích thước 224. Ảnh sạch có độ sáng 0,458, nhiễu (sigma Immerkaer) 1,73 với khoảng 10 đến 90 phần trăm là 1,21 đến 2,28, độ nét (phương sai Laplacian) 548 với khoảng 288 đến 840.

- `noise_s1` đã có nhiễu 5,6 và `sensor_noise_s1` 7,9, gấp 3 đến 4,5 lần ảnh sạch và vượt xa khoảng 10 đến 90 phần trăm.
- `blur_s1` có độ nét 294, sát cận dưới của ảnh sạch. `lowres_s1` có độ nét 171, thấp hơn cận dưới. `jpeg` hầu như không đổi độ nét (1008 ở mức 1 và 529 ở mức 3) nên mức giảm của họ này đến từ các vết khối, không phải mất nét.
- Chưa đối chiếu với bộ Vietnamese durian leaf vì chưa có dữ liệu. Chưa đánh giá được mức nào là thực tế với ảnh chụp ngoài vườn.

Bảng: `quality_by_condition_test.csv`. Hình `figures/test_quality_curves.png`.

## Đầu vào cho task 4.5

`priority_for_4_5.json` ghi thứ hạng 14 họ, 31 điều kiện giảm từ 0,05 và có ý nghĩa, họ chưa có kỹ thuật xử lý (`jpeg`) và họ có thể bỏ qua (`glare`).

- Ưu tiên theo thứ tự: `lowres`, `contrast_high`, `blur`, `contrast_low`, `brightness_bright`, `lowlight`, `motion_blur`, `noise`.
- `brightness_bright` và `contrast_high` thuộc nhóm nặng, nhưng 4.3 trên validation cho thấy xử lý gần như không phục hồi được hai họ này. Đây là hai họ cần đối chiếu với model huấn luyện lại bằng augmentation ở 4.5.
- `jpeg` giảm trung bình 0,136 mà chưa có kỹ thuật tương ứng, nên đối chứng huấn luyện lại cũng là hướng duy nhất cho họ này.

## Hạn chế

- Một model, một seed huấn luyện. Khoảng tin cậy chỉ phản ánh độ ngẫu nhiên của việc chọn mẫu test, không phản ánh biến động giữa các lần huấn luyện.
- Mức suy giảm là tổng hợp, chưa so với ảnh thực địa. Mức giảm tuyệt đối không nên được hiểu là mức giảm khi dùng ngoài vườn.
- Ngưỡng severe, moderate, mild chọn trước nhưng vẫn là quy ước. Họ sát ngưỡng (`brightness_dark`) có thể đổi loại với ngưỡng khác.

## File

| File | Nội dung |
|---|---|
| `sensitivity_conditions_test.csv`, `sensitivity_conditions_val.csv` | Mỗi điều kiện: macro F1 và khoảng tin cậy, degradation và khoảng tin cậy, số ảnh đổi, p McNemar và p Holm, recall từng lớp |
| `sensitivity_families_test.csv` | Mỗi họ: degradation trung bình, lớn nhất, ở mức 1, loại, task xử lý |
| `priority_for_4_5.json` | Đầu vào cho task 4.5 |
| `sensitivity_summary.json` | Điểm sạch, đồng thuận giữa hai split, tham số bootstrap |
| `flip_destinations_test.csv`, `sample_flips_test.csv` | Lớp nhận ảnh bị đổi, số lần đổi từng ảnh |
| `quality_by_condition_test.csv` | Thống kê chất lượng ảnh theo điều kiện |
| `sensitivity_predictions.npz` | Xác suất từng ảnh, 43 bộ ảnh, validation và test, để 4.5 so ghép cặp |
| `figures/` | Mười hình |
