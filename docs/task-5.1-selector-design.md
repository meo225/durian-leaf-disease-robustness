# Task 5.1. Thiết kế bộ chọn xử lý ảnh thích ứng

Đặc tả của phương pháp đề xuất ở Phase 5, viết **trước khi** hiệu chỉnh (task 5.3) và trước khi mở tập test ở task 6.1. Cài đặt ở `src/selector/`, cấu hình ở `configs/selector.yaml`, notebook hiệu chỉnh ở `notebooks/09_selector_calibration.ipynb`.

## 1. Câu hỏi nghiên cứu

Khi nào xử lý ảnh thích ứng đáng dùng so với huấn luyện lại model? Bộ chọn xử lý ảnh nhìn ảnh đầu vào, ước lượng chất lượng bằng đặc trưng nhẹ, rồi quyết định giữ nguyên hoặc áp một kỹ thuật xử lý, và không động đến reference model. Nếu bộ chọn không thắng thì báo kết quả âm. Kết quả âm là kết quả hợp lệ của đề tài.

## 2. Bằng chứng đầu vào từ Phase 4

Số liệu trên test của reference model Swin-Tiny (`reports/recovery_map/README.md`).

| Phát hiện | Số liệu | Hệ quả cho thiết kế |
|---|---|---|
| Chỉ 5 trên 33 cặp (kỹ thuật, họ) có bằng chứng phục hồi | unsharp cho `blur`, `motion_blur`, `lowres`; bilateral cho `noise`, `sensor_noise` | Tập hành động chỉ gồm các cấu hình của 5 cặp này |
| Không kỹ thuật cố định nào hơn không xử lý | Gamma +0,001, các kỹ thuật khác −0,003 đến −0,120 | Mặc định phải là giữ nguyên |
| Làm nét hại nặng trên nhiễu | unsharp trên `noise`: −0,38 F1 | Luật nhiễu đứng trước luật mờ, luật mờ có mệnh đề chặn nhiễu |
| Chi phí ảnh sạch | unsharp mạnh −0,075 F1, bilateral −0,009 đến −0,012 | Ràng buộc kích hoạt trên ảnh sạch khi hiệu chỉnh |
| Trần của bộ chọn trên 10 kỹ thuật | oracle theo họ +0,022 so với không xử lý | Mức tăng kỳ vọng nhỏ, tiêu chí đánh giá phải phản ánh điều đó |
| Huấn luyện lại bằng augmentation | +0,096 (trong phân phối augmentation, một seed) | Bộ chọn không được trình bày như thay thế huấn luyện lại |

## 3. Kiến trúc

Bốn thành phần tách rời, bật hoặc tắt riêng khi ablation (task 5.4), dùng chung code cho đánh giá và inference (`src/selector/pipeline.py`).

```text
ảnh ──► ước lượng chất lượng ──► bộ chọn (luật) ──► kỹ thuật xử lý ──► reference model đóng băng ──► dự đoán
        features.py              rules.py            src/processing    Swin-Tiny, eval, không gradient
```

Ảnh đi vào ở kích thước đã chuẩn hóa của pipeline (224), trước bước chuẩn hóa trung bình và độ lệch chuẩn. Đặc trưng tính trên ảnh chưa xử lý.

## 4. Hành động

Giữ nguyên là mặc định. Bốn cấu hình còn lại là các cấu hình đã khóa trên validation ở task 4.3 và 4.4, không đổi tham số.

| Hành động | Kỹ thuật, cấu hình | Họ có bằng chứng ở 4.5 |
|---|---|---|
| `keep` | không xử lý | |
| `unsharp_strong` | unsharp, σ 1,0, amount 3,0 | `blur`, `lowres` |
| `unsharp_mild` | unsharp, σ 2,0, amount 1,0 | `motion_blur` |
| `bilateral_wide` | bilateral, d 7, σ màu 30, σ không gian 5 | `noise` |
| `bilateral_narrow` | bilateral, d 5, σ màu 30, σ không gian 3 | `sensor_noise` |

`tests/test_selector.py` kiểm tra mỗi hành động khớp cấu hình đã khóa và có bằng chứng trong `reports/recovery_map/recovery_summary.json`.

## 5. Đặc trưng chất lượng

Tính không cần ảnh sạch tương ứng, trên độ sáng Y = 0,299 R + 0,587 G + 0,114 B trong khoảng 0 đến 1. Sáu nhóm, để ablation bỏ từng nhóm.

| Nhóm | Đặc trưng | Định nghĩa |
|---|---|---|
| exposure | `luma_mean`, `dark_fraction`, `bright_fraction` | Độ sáng trung bình; tỉ lệ điểm ảnh dưới 0,1 và trên 0,9 |
| contrast | `luma_std` | Độ lệch chuẩn của độ sáng |
| sharpness | `sharpness` | Phương sai của Laplacian chia phương sai độ sáng. Chuẩn hóa nên bất biến khi nhân độ sáng hoặc độ tương phản, ảnh tối không bị nhận nhầm là mờ |
| noise | `noise_sigma`, `noise_ratio` | Ước lượng nhiễu của Immerkaer (1996); nhiễu chia độ lệch chuẩn độ sáng |
| color | `cast` | log của tỉ số kênh đỏ trên kênh xanh dương |
| naturalness | `mscn_var` | Phương sai của hệ số MSCN, đặc trưng nền của BRISQUE (Mittal và cộng sự, 2012) |

`mscn_var` là thống kê tự nhiên của ảnh, **không phải điểm BRISQUE đầy đủ** vì không dùng bộ hồi quy đã huấn luyện của BRISQUE. Nếu cần điểm BRISQUE đầy đủ thì thêm sau, như một đặc trưng mới.

## 6. Luật và quy trình hiệu chỉnh

Luật xét theo thứ tự, luật đầu tiên khớp thì thắng, không luật nào khớp thì `keep`.

1. `noise`: nếu một đặc trưng nhiễu vượt ngưỡng (và độ sáng trên một ngưỡng, nếu mệnh đề chặn được chọn) thì dùng `bilateral_wide` hoặc `bilateral_narrow`.
2. `blur`: nếu độ nét dưới ngưỡng (và độ sáng, nhiễu đủ thấp, nếu mệnh đề chặn được chọn) thì dùng `unsharp_strong` hoặc `unsharp_mild`.

Task 5.3 chọn hành động, đặc trưng và ngưỡng của từng luật (`src/selector/calibrate.py`):

- **Dữ liệu:** chỉ validation, 444 ảnh, ảnh sạch và 42 điều kiện của benchmark 4.1. Không mở tập test.
- **Cách chấm:** dự đoán của reference với mọi hành động được tính trước một lần, mỗi bộ ngưỡng được chấm bằng cách tra bảng. Lưới ngưỡng theo phân vị của đặc trưng, thêm phân vị cao của ảnh sạch.
- **Mục tiêu:** macro F1 trung bình 42 điều kiện suy giảm.
- **Ràng buộc trên ảnh sạch của validation:** macro F1 giảm tối đa 0,005 so với không xử lý, và tối đa 2% ảnh sạch bị kích hoạt xử lý.
- **Giữ luật:** một luật chỉ được giữ nếu tăng mục tiêu ít nhất 0,002. Không thì luật bị bỏ và bộ chọn có thể rỗng.
- **Hòa:** khi hai bộ ngưỡng cho cùng mục tiêu, chọn bộ kích hoạt ít ảnh hơn.
- **Ổn định:** cross-validation 5 phần theo ảnh, chia theo lớp, hiệu chỉnh trên 4 phần và chấm trên phần còn lại.

## 7. Từ chối và đề nghị chụp lại

Bộ chọn không từ chối ảnh. Quyết định từ chối dựa trên độ tin cậy của reference model (xác suất lớn nhất), đo bằng đường risk-coverage ở task 6.1. Đặc trưng ảnh một mình chưa có bằng chứng dự đoán lỗi phân loại.

## 8. Giả thuyết khóa trước

Đánh giá trên test ở task 6.1, với bộ chọn đã khóa (`locked: true`) và bootstrap 95% theo ảnh như bản đồ phục hồi.

| | Giả thuyết | Cách kiểm |
|---|---|---|
| H1 | Bộ chọn tốt hơn không xử lý | Chênh lệch macro F1 trung bình 42 điều kiện, cận dưới khoảng tin cậy lớn hơn 0 |
| H2 | Bộ chọn an toàn trên ảnh sạch | Macro F1 ảnh sạch giảm không quá 0,01, tỉ lệ kích hoạt trên ảnh sạch không quá 0,02 |
| H3 | Bộ chọn không kém kỹ thuật cố định tốt nhất | Kỹ thuật cố định chọn trên validation, so ghép cặp theo ảnh |
| H4 | Bộ chọn kém huấn luyện lại | Dự kiến, vì trần oracle (+0,022) thấp hơn huấn luyện lại (+0,096) |

Nếu H1 hoặc H2 không đạt thì báo bộ chọn không có giá trị trên cấu hình này, kèm số liệu. Không đổi ngưỡng hay luật sau khi xem test.

## 9. Hạn chế đã biết

- **Tập hành động chịu ảnh hưởng của test.** Danh sách 5 cặp có bằng chứng lấy từ kết quả test ở task 4.5 (quy tắc "có bằng chứng" đã khóa trước khi mở test, nhưng danh sách là kết quả đo trên test). Ngưỡng thì chỉ hiệu chỉnh trên validation. Cấu hình của các cặp này cũng được chọn trên validation ở 4.3 và 4.4, nên lựa chọn hành động có cơ sở độc lập trên validation, nhưng không loại trừ được hoàn toàn ảnh hưởng gián tiếp. Cần nêu ở báo cáo.
- **Validation dùng hai lần.** Cấu hình xử lý ảnh đã được chọn trên validation, rồi ngưỡng cũng hiệu chỉnh trên đó. Mức tăng trên validation có thể lạc quan; cross-validation chỉ kiểm tra sang ảnh khác, không kiểm tra sang bước chọn cấu hình. Đánh giá trung thực là task 6.1.
- **Họ đã thấy.** Ngưỡng hiệu chỉnh trên các họ của benchmark 4.1. Tổng quát hóa sang họ chưa thấy đo ở task 5.5 và 6.1.
- **Đặc trưng phụ thuộc nội dung.** Ước lượng nhiễu của Immerkaer cũng tính kết cấu mịn của lá nên là cận trên của nhiễu. Mệnh đề chặn độ sáng giảm nhầm lẫn giữa nhiễu và thiếu sáng, nhưng không loại hẳn.
- **Mô phỏng.** Suy giảm là tổng hợp, không phải ảnh hỏng thật. Ảnh dữ liệu ngoài ở task 6.2 mới là bằng chứng gần thực tế.
- **Một reference, một seed.** Kết luận về từng kỹ thuật phụ thuộc Swin-Tiny, xem `reports/recovery_map/README.md`.

## 10. Dùng

```bash
pytest tests/test_selector.py
```

Hiệu chỉnh chạy trên Kaggle bằng `notebooks/09_selector_calibration.ipynb`, ghi `reports/selector/selector_calibrated.json` (`locked: false`), `calibration_log.json`, `calibration_conditions_val.csv` và `selector_by_family.png`.

```python
from src.selector.pipeline import AdaptivePipeline

pipeline = AdaptivePipeline.from_files(reference=model, pipeline_cfg=pipeline_cfg, device="cuda")
probabilities, decisions = pipeline.predict_proba(images_uint8)  # decisions[i].action, decisions[i].features
```

Ablation theo nhóm đặc trưng (task 5.4) dùng `fit(..., dropped_groups=[...])`.

## Tài liệu tham khảo

- Pei và cộng sự (2018), *Effects of Image Degradations to CNN-based Image Classification*, arXiv:1810.05552. Xử lý ảnh không bằng huấn luyện trên ảnh suy giảm.
- Pei và cộng sự (2018), *Does Haze Removal Help CNN-based Image Classification?*, arXiv:1810.05716.
- Ono và cộng sự (2025), *Recognition-Oriented Low-Light Image Enhancement*, arXiv:2501.04210. Bộ lọc học theo loss của model nhận diện, không cần huấn luyện lại model.
- Gao và cộng sự (2025), *QuARF*, AAAI-25. Ý tưởng thích ứng theo chất lượng ảnh.
- Mittal, Moorthy, Bovik (2012), *No-Reference Image Quality Assessment in the Spatial Domain*, IEEE TIP. MSCN và BRISQUE.
- Immerkaer (1996), *Fast Noise Variance Estimation*, Computer Vision and Image Understanding.

## Kết quả

Kết quả hiệu chỉnh lần chạy đầu trên validation nằm ở `reports/selector/README.md`.
