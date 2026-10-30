# Phân tích độ nhạy của reference model với suy giảm ảnh (task 4.2)

Swin-Tiny (reference model, seed 42) đo trên ảnh sạch và 42 điều kiện của benchmark 4.1, ở validation và test. Notebook: `notebooks/06_degradation_sensitivity.ipynb`. Code: `src/evaluation/sensitivity.py`. Ngưỡng phân loại đặt trước trong `configs/sensitivity.yaml`.

Reference model đã đổi từ ConvNeXt-Tiny sang Swin-Tiny sau khi chạy lặp seed (xem `reports/baselines/README.md`). Kết quả dưới đây thay cho lần đo ConvNeXt-Tiny trước đó, giữ nguyên mức suy giảm đã khóa ở 4.1.

## Thiết lập

- Test mở lại một lần với `FINAL_EVAL=1` cho Swin-Tiny, mức suy giảm đã khóa ở 4.1 và không chỉnh theo kết quả.
- Mọi so sánh là ghép cặp trên cùng mẫu. Khoảng tin cậy 95% bằng bootstrap 2000 lần lấy lại mẫu theo cặp. Kiểm định McNemar chính xác hai phía trên các ảnh đổi từ đúng sang sai và ngược lại, hiệu chỉnh Holm trên 42 phép thử.
- Degradation là macro F1 sạch trừ macro F1 suy giảm.

## Điểm trên ảnh sạch

| Split | Ảnh | Macro F1 (95%) | Accuracy | Ảnh sai |
|---|---|---|---|---|
| validation | 444 | 0,9664 (0,949 đến 0,981) | 0,9662 | 15 |
| test | 888 | 0,9622 (0,949 đến 0,974) | 0,9628 | 33 |

Hai split gần nhau (chênh 0,004), trong vùng nhiễu của khoảng tin cậy.

## Degradation theo họ và mức (test)

Dấu `*` là khác biệt có ý nghĩa sau Holm (alpha 0,05). Loại: severe từ 0,15, moderate từ 0,05, mild từ 0,02 và có ý nghĩa, theo giá trị trung bình qua ba mức. `sớm` là đã giảm từ 0,05 ngay ở mức 1.

| Họ | Nhóm | Mức 1 | Mức 2 | Mức 3 | Trung bình | Loại | Task xử lý |
|---|---|---|---|---|---|---|---|
| `jpeg` | extension | 0.033* | 0.096* | 0.375* | 0.168 | severe | không có |
| `blur` | plan | 0.015* | 0.107* | 0.304* | 0.142 | moderate | 4.4 |
| `lowres` | extension | 0.100* | 0.118* | 0.189* | 0.136 | moderate, sớm | 4.4 |
| `brightness_bright` | plan | 0.031* | 0.115* | 0.259* | 0.135 | moderate | 4.3 |
| `contrast_high` | extension | 0.028* | 0.121* | 0.189* | 0.113 | moderate | 4.3 |
| `motion_blur` | extension | 0.032* | 0.080* | 0.204* | 0.105 | moderate | 4.4 |
| `sensor_noise` | extension | 0.053* | 0.090* | 0.166* | 0.103 | moderate, sớm | 4.4 |
| `noise` | plan | 0.022* | 0.084* | 0.196* | 0.101 | moderate | 4.4 |
| `lowlight` | extension | 0.044* | 0.077* | 0.156* | 0.092 | moderate | 4.3+4.4 |
| `shadow` | extension | 0.003 | 0.052* | 0.169* | 0.075 | moderate | 4.3 |
| `color_cast` | extension | 0.016 | 0.060* | 0.133* | 0.070 | moderate | 4.3 |
| `brightness_dark` | plan | -0.007 | 0.001 | 0.113* | 0.036 | mild | 4.3 |
| `glare` | extension | 0.005 | 0.027* | 0.052* | 0.028 | mild | 4.3 |
| `contrast_low` | plan | -0.004 | 0.000 | 0.011 | 0.002 | negligible | 4.3 |

- 34 trên 42 điều kiện khác ảnh sạch có ý nghĩa. Tám điều kiện không có ý nghĩa: `brightness_dark_s1`, `brightness_dark_s2`, `contrast_low_s1`, `contrast_low_s2`, `contrast_low_s3`, `glare_s1`, `shadow_s1`, `color_cast_s1`. Suy giảm nhẹ trong các họ này chưa phân biệt được với dao động chọn mẫu.
- Chỉ `jpeg` đạt mức severe, và chủ yếu do mức 3 (`jpeg_s3` giảm 0,375, F1 còn 0,587, điều kiện tệ nhất). Các họ còn lại đều moderate hoặc nhẹ hơn: Swin-Tiny bền hơn ConvNeXt-Tiny ở hầu hết họ (task 4.2 cũ có tám họ severe).
- `contrast_low` hầu như không ảnh hưởng (giảm 0,011 ở mức 3, không có ý nghĩa), `brightness_dark` chỉ giảm có ý nghĩa ở mức 3. `glare` giảm tối đa 0,052, có ý nghĩa ở mức 2 và 3 nhưng quá nhỏ để ảnh hưởng thực tế.
- Hai họ giảm từ 0,05 ngay ở mức 1: `lowres` (0,100, hỏng nặng nhất ở mức nhẹ nhất) và `sensor_noise` (0,053). `lowlight` sát ngưỡng (0,044).
- Mức 3 nặng nhất: `jpeg_s3` (F1 0,587), `blur_s3` (0,658), `brightness_bright_s3` (0,704), `motion_blur_s3` (0,758), `noise_s3` (0,766).

Hình: `figures/test_degradation_heatmap.png`, `figures/test_severity_curves.png`, `figures/test_priority.png`.

## Phân tích theo lớp

Recall từng lớp thay đổi không đều. Recall ảnh sạch (test): `ALGAL_LEAF_SPOT` 0,925, `ALLOCARIDARA_ATTACK` 0,973, `HEALTHY_LEAF` 0,985, `LEAF_BLIGHT` 0,963, `PHOMOPSIS_LEAF_SPOT` 0,960. Hình `figures/test_class_recall_change.png`, `figures/test_confusion_worst.png`.

- `ALLOCARIDARA_ATTACK` mất nhiều nhất khi ảnh mờ hoặc độ phân giải thấp: recall còn 0,58 ở `lowres_s1`, 0,54 ở `blur_s2`, 0,09 ở `blur_s3`, 0,30 ở `lowres_s3`, 0,39 ở `motion_blur_s3`. Ở `blur_s3`, 162 trên 230 ảnh bị đổi sang sai thuộc lớp này, ở `lowres_s1` là 72 trên 84. Lớp dựa vào chi tiết nhỏ nên mất ngay khi chi tiết bị làm mờ.
- Khi mờ, ảnh đổi sang sai chủ yếu thành `PHOMOPSIS_LEAF_SPOT` (0,40 đến 0,87) và `ALGAL_LEAF_SPOT` (0,28 đến 0,35 từ mức 2).
- Với `noise` và `sensor_noise`, ảnh đổi sang sai dồn về `HEALTHY_LEAF` (0,48 và 0,49 ở mức 3) và `PHOMOPSIS_LEAF_SPOT` (khoảng 0,38). Ở mức 3, recall `ALGAL_LEAF_SPOT` còn 0,50 (`noise_s3`) và 0,62 (`sensor_noise_s3`). Khác với ConvNeXt-Tiny, nhiễu không còn dồn về `ALLOCARIDARA_ATTACK`.
- `jpeg_s3` làm sụt `PHOMOPSIS_LEAF_SPOT` (recall 0,34) và `ALLOCARIDARA_ATTACK` (0,25), ảnh đổi sang sai dồn về `HEALTHY_LEAF` (0,73). Các vết khối nén làm vết bệnh nhỏ bị đọc thành lá khỏe.
- `contrast_high_s3` làm sụt `ALGAL_LEAF_SPOT` (recall 0,45) và `LEAF_BLIGHT` (0,68). `brightness_bright_s3` làm sụt `ALGAL_LEAF_SPOT` (0,36) và `LEAF_BLIGHT` (0,51). Cả hai đổi sang sai chủ yếu thành `PHOMOPSIS_LEAF_SPOT` (0,56 đến 0,61 ở các mức).
- `brightness_dark_s3` và `lowlight_s3` làm sụt `PHOMOPSIS_LEAF_SPOT` (0,77 và 0,68) và `ALLOCARIDARA_ATTACK` (0,75 và 0,64). `HEALTHY_LEAF` hầu như giữ ở mọi điều kiện (từ 0,90 trở lên).

Hình `figures/test_flip_destinations.png`. Bảng đầy đủ: `flip_destinations_test.csv`.

## Prediction flip theo từng mẫu

Trên 855 ảnh đúng ở ảnh sạch, 263 ảnh không bị đổi ở điều kiện nào và 23 ảnh bị đổi ở ít nhất 21 trong 42 điều kiện (nhiều nhất 35). Theo lớp, `ALLOCARIDARA_ATTACK` dễ đổi nhất (176 trên 178 ảnh bị đổi ở ít nhất một điều kiện, 10 ảnh bị đổi ở ít nhất 21 điều kiện), `HEALTHY_LEAF` bền nhất (129 trên 193 ảnh không đổi, và không ảnh nào bị đổi ở từ 21 điều kiện trở lên). Danh sách ảnh: `sample_flips_test.csv`. Hình `figures/test_sample_flips.png`, `figures/test_flip_counts.png`.

## Validation so với test

Spearman giữa degradation của 42 điều kiện trên hai split là 0,955. Chênh lệch macro F1 trung bình 0,016 và lớn nhất 0,052. Các mức hiệu chỉnh trên validation cho mức giảm tương tự ở test. Hình `figures/val_vs_test_degradation.png`.

## Thống kê chất lượng ảnh

Số liệu này chỉ phụ thuộc ảnh, không phụ thuộc model, nên giữ nguyên so với lần đo trước. Đo trên 888 ảnh test ở kích thước 224. Ảnh sạch có độ sáng 0,458, nhiễu (sigma Immerkaer) 1,73 với khoảng 10 đến 90 phần trăm là 1,21 đến 2,28, độ nét (phương sai Laplacian) 548 với khoảng 288 đến 840.

- `noise_s1` đã có nhiễu 5,6 và `sensor_noise_s1` 7,9, gấp 3 đến 4,5 lần ảnh sạch và vượt xa khoảng 10 đến 90 phần trăm.
- `blur_s1` có độ nét 294, sát cận dưới của ảnh sạch. `lowres_s1` có độ nét 171, thấp hơn cận dưới. `jpeg` hầu như không đổi độ nét (1008 ở mức 1 và 529 ở mức 3) nên mức giảm của họ này đến từ các vết khối, không phải mất nét.
- Chưa đối chiếu với bộ Vietnamese durian leaf vì chưa có dữ liệu. Chưa đánh giá được mức nào là thực tế với ảnh chụp ngoài vườn.

Bảng: `quality_by_condition_test.csv`. Hình `figures/test_quality_curves.png`.

## Đầu vào cho task 4.5

`priority_for_4_5.json` ghi thứ hạng 14 họ, 26 điều kiện giảm từ 0,05 và có ý nghĩa, họ chưa có kỹ thuật xử lý (`jpeg`) và họ có thể bỏ qua (`contrast_low`).

- Ưu tiên theo thứ tự: `jpeg`, `blur`, `lowres`, `brightness_bright`, `contrast_high`, `motion_blur`, `sensor_noise`, `noise`, `lowlight`.
- `jpeg` là họ nặng nhất mà chưa có kỹ thuật tương ứng, nên đối chứng huấn luyện lại là hướng chính cho họ này.
- `brightness_bright` và `contrast_high` thuộc nhóm đáng kể, nhưng 4.3 trên validation cho thấy xử lý gần như không phục hồi được hai họ này. Đây là hai họ cần đối chiếu với model huấn luyện lại bằng augmentation ở 4.5.

## Hạn chế

- Một model, một seed huấn luyện cho phép đo này. Khoảng tin cậy chỉ phản ánh độ ngẫu nhiên của việc chọn mẫu test, không phản ánh biến động giữa các lần huấn luyện. Chạy lặp baseline (`reports/baselines/README.md`) cho thấy mức giảm F1 của Swin-Tiny dưới năm suy giảm dao động khoảng 0,006 giữa các seed, nhưng chưa chạy lặp trên toàn bộ 42 điều kiện.
- Mức suy giảm là tổng hợp, chưa so với ảnh thực địa. Mức giảm tuyệt đối không nên được hiểu là mức giảm khi dùng ngoài vườn.
- Ngưỡng severe, moderate, mild chọn trước nhưng vẫn là quy ước. Họ sát ngưỡng `sớm` (`sensor_noise` 0,053 và `lowlight` 0,044 so với 0,05) có thể đổi loại với ngưỡng khác.

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
