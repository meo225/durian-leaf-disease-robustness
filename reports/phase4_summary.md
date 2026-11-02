# Tổng hợp Phase 4: benchmark suy giảm và xử lý ảnh

Reference model: Swin-Tiny (seed 42), macro F1 sạch trên test 0,9622. Test mở ở bước 4.2 và 4.5, mỗi bước hai lần: ConvNeXt-Tiny rồi Swin-Tiny (nhật ký ở `manifests/SPLIT_PROTOCOL.md`, mục 3.4). Báo cáo chi tiết từng task nằm trong thư mục tương ứng.

| Task | Nội dung | Tập đo | Báo cáo |
|---|---|---|---|
| 4.1 | Benchmark 14 họ × 3 mức = 42 điều kiện | validation (khóa), test chưa mở | [corruptions](corruptions/README.md) |
| 4.2 | Phân tích độ nhạy | test | [sensitivity](sensitivity/README.md) |
| 4.3 | Xử lý ánh sáng và màu | validation | [processing](processing/README.md) |
| 4.4 | Khử nhiễu và làm nét | validation | [processing](processing/README.md) |
| 4.5 | Bản đồ phục hồi, oracle, huấn luyện lại | test | [recovery_map](recovery_map/README.md), [augmentation](augmentation/README.md) |

## 4.1 Benchmark suy giảm

- 14 họ: brightness (tối, sáng), contrast (thấp, cao), blur, noise, glare, shadow, lowlight, motion blur, sensor noise, jpeg, lowres, color cast. Mỗi họ 3 mức, tổng 42 điều kiện.
- Hiệu chỉnh trên validation, điểm giảm dần theo mức, test chưa mở khi khóa.

## 4.2 Độ nhạy của reference (test 888 ảnh)

- F1 sạch 0,9622, khoảng tin cậy 95% [0,949; 0,974].
- 34/42 điều kiện giảm có ý nghĩa (McNemar chính xác, Holm). 8 điều kiện không có ý nghĩa: brightness_dark mức 1-2, contrast_low mức 1-3, glare, shadow và color_cast mức 1.
- Mức giảm trung bình theo họ:
  - Nặng: jpeg 0,168 (jpeg_s3 giảm 0,375, F1 còn 0,587).
  - Vừa (từ 0,05): blur 0,142, lowres 0,136, brightness_bright 0,135, contrast_high 0,113, motion_blur 0,105, sensor_noise 0,103, noise 0,101, lowlight 0,092, shadow 0,075, color_cast 0,070.
  - Nhẹ: brightness_dark 0,036, glare 0,028. Không đáng kể: contrast_low 0,002.
- Lớp ALLOCARIDARA mất nhiều nhất khi mờ hoặc độ phân giải thấp: recall 0,09 ở blur_s3, 0,30 ở lowres_s3, 0,39 ở motion_blur_s3 (162/230 ảnh đổi sang sai ở blur_s3 thuộc lớp này).
- Nhiễu và jpeg làm dự đoán dồn về HEALTHY_LEAF (0,48 đến 0,49 ở noise_s3 và sensor_noise_s3, 0,73 ở jpeg_s3). brightness_bright và contrast_high dồn về PHOMOPSIS (khoảng 0,6).
- Mức hiệu chỉnh trên validation vẫn đúng ở test (Spearman 0,955). Chưa đối chiếu với bộ Vietnamese durian leaf.

## 4.3 Xử lý ánh sáng và màu (validation, 1 seed, F1 sạch 0,9664)

- Rõ nhất: brightness_dark mức 3 0,81 → 0,94 (Retinex), trung bình ba mức +0,055. lowlight mức 3 0,75 → 0,81 (gamma tự động), trung bình +0,035. color_cast mức 3 0,85 → 0,92 (gray-world), trung bình ba mức chỉ +0,003.
- Nhỏ (từ +0,01 trở xuống): shadow, glare, contrast_low (CLAHE mức 3: 0,93 → 0,94). Gần như không phục hồi: brightness_bright, contrast_high.
- Chi phí ảnh sạch: mọi kỹ thuật giảm F1 (−0,005 đến −0,037), gray-world −0,037, Retinex −0,026.
- Kỹ thuật tốt nhất mỗi điều kiện chọn trên chính validation nên chỉ để quan sát. Cấu hình: `processing/illumination_selected.json`.

## 4.4 Khử nhiễu và làm nét (validation, 1 seed, lưới v3 gồm 29 cấu hình)

- Làm nét giúp ở mức vừa: unsharp cho lowres 0,82 → 0,91, blur_s2 0,83 → 0,95, motion_blur_s3 0,74 → 0,83. blur mức 3 chỉ phục hồi một phần (0,65 → 0,74), blur mức 1 còn giảm nhẹ.
- Khử nhiễu giúp nhỏ: bilateral +0,017 đến +0,031 (noise, sensor_noise, lowlight), Gaussian σ 0,5 tương đương. NLM gần như không lọc, median luôn hại.
- Chi phí ảnh sạch: unsharp −0,054, bilateral −0,019, Gaussian −0,009, NLM −0,007, median −0,065 (ALLOCARIDARA −0,27).
- Cấu hình: `processing/restoration_selected.json`.

## 4.5 Bản đồ phục hồi, oracle và huấn luyện lại (test 888 ảnh)

| Chiến lược | F1 sạch | TB 42 điều kiện | Chênh so với không xử lý |
|---|---:|---:|---|
| Không xử lý | 0,962 | 0,869 | |
| Oracle chọn theo họ trên validation | 0,962 | 0,891 | +0,022 (tốt hơn 15, kém hơn 1 điều kiện) |
| Kỹ thuật cố định tốt nhất (gamma) | 0,959 | 0,870 | +0,001 (khoảng tin cậy gồm 0) |
| Huấn luyện lại bằng augmentation | 0,987 | 0,965 | +0,096 (tốt hơn 40/42, không điều kiện nào kém hơn) |

- Bản đồ phục hồi: 5/33 cặp (kỹ thuật, họ) có bằng chứng: unsharp cho lowres (+0,089), motion_blur (+0,056), blur (+0,038); bilateral cho sensor_noise (+0,038), noise (+0,030). Retinex, CLAHE, gamma không còn đạt quy tắc vì Swin-Tiny đã bền với các họ ánh sáng.
- Không kỹ thuật cố định nào hơn không xử lý. Unsharp mạnh nhất tốn 0,075 F1 trên ảnh sạch.
- Huấn luyện lại (augmentation 14 họ, 42 điều kiện, xác suất 0,5, dao động ±10%, chọn epoch 24/30 theo validation hỗn hợp): trung bình theo mức 0,980 / 0,970 / 0,947. Hơn oracle ở 41/42 điều kiện. Yếu nhất: jpeg_s3 0,879. Tăng nhiều nhất: lowres, blur +0,15, jpeg +0,14, brightness_bright +0,13.

## Giới hạn

- Một seed cho reference và cho model huấn luyện lại (seed 43, 44 chưa chạy). Khoảng tin cậy bootstrap không phản ánh dao động do huấn luyện.
- Độ bền sau huấn luyện lại chỉ đo trong phân phối augmentation. F1 sạch test tăng +0,025 nhưng validation sạch chỉ tăng +0,005, nên chưa kết luận augmentation tăng độ chính xác ảnh sạch.
- Kết luận về từng kỹ thuật phụ thuộc vào reference model, không chuyển nguyên từ ConvNeXt-Tiny sang Swin-Tiny.
- Test đã mở ở 4.2 và 4.5, mỗi bước hai lần (ConvNeXt-Tiny rồi Swin-Tiny), tham số giữ nguyên từ validation. Việc đổi reference sang Swin-Tiny được quyết định sau khi đã thấy kết quả test của ConvNeXt-Tiny ở 4.2. Tiêu chí đổi chỉ dùng validation, nhưng không loại trừ được hoàn toàn ảnh hưởng của việc đã biết kết quả test.
