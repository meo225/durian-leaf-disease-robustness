# Bản đồ phục hồi (task 4.5)

Đo từng kỹ thuật xử lý ảnh của task 4.3 và 4.4 trên tập **test** bằng cùng một reference model, Swin-Tiny đóng băng và train trên ảnh sạch (seed 42), trên cùng 888 ảnh và nhãn, theo kỹ thuật × họ suy giảm × mức. Trả lời ba câu: kỹ thuật nào phục hồi nhận diện và ở mức nào, chi phí trên ảnh sạch và theo từng lớp là bao nhiêu, và cận trên nếu biết trước loại suy giảm (oracle) so với huấn luyện lại bằng augmentation.

Tiêu chí là macro F1 của bài toán nhận diện bệnh, không phải ảnh nhìn đẹp hơn. Code ở `src/evaluation/recovery_map.py` (chạy model, cần GPU), `recovery_stats.py` và `recovery_report.py` (thống kê, chạy trên CPU), cấu hình ở `configs/recovery_map.yaml`, notebook `notebooks/08_recovery_map.ipynb`.

Reference model đã đổi từ ConvNeXt-Tiny sang Swin-Tiny (xem `reports/baselines/README.md`). Lần chạy trước của task này dùng ConvNeXt-Tiny và đã được thay bằng kết quả dưới đây. Swin-Tiny vốn bền hơn (macro F1 trung bình 42 điều kiện 0.869, so với 0.797), nên mức suy giảm để phục hồi nhỏ hơn và các kết luận cũng khác, xem phần "Đọc kết quả".

## Cách đo

- **Kỹ thuật và cấu hình:** 10 kỹ thuật (gamma, gamma tự động, CLAHE, Retinex, gray-world từ task 4.3; Gaussian, median, bilateral, NLM, unsharp từ task 4.4), mỗi họ dùng cấu hình đã khóa trên validation (`reports/processing/*_selected.json`). Họ phù hợp với kỹ thuật dùng cấu hình tốt nhất của họ, họ không phù hợp dùng cấu hình chung. Không chỉnh tham số theo kết quả test.
- **Phạm vi:** mọi kỹ thuật chạy trên cả 42 điều kiện của benchmark 4.1 và trên ảnh sạch, tổng 459 ô (kể cả cấu hình chung chạy thêm trên các họ có cấu hình riêng): 138 ô phù hợp và 321 ô đối chứng.
- **Thống kê:** khoảng tin cậy 95% bằng bootstrap theo ảnh (1.000 lần, cùng một ma trận lấy lại mẫu cho mọi ô), McNemar chính xác ghép cặp trên từng ảnh, hiệu chỉnh Holm trên toàn bộ 459 ô, mức ý nghĩa 0,05. Normalized recovery chỉ tính khi degradation lớn hơn 0,02.
- **Quy tắc "có bằng chứng"** (khóa trước khi mở test, `configs/recovery_map.yaml`): khoảng tin cậy của recovery trung bình ba mức nằm trên 0, và ít nhất hai mức có recovery dương có ý nghĩa sau Holm. Chi phí ảnh sạch chỉ gắn nhãn `costly` (cận trên khoảng tin cậy dưới −0,02), không dùng để loại.
- **Kiểm tra tái hiện:** dự đoán "không xử lý" ở đây đúng cùng mẫu với task 4.2 (888 ảnh, cùng đường dẫn và nhãn) và chỉ lệch 2 dự đoán trên 43 bộ ảnh (`contrast_high_s1` và `jpeg_s3`, mỗi bộ 1 ảnh) do làm tròn xác suất, nên so ghép cặp với 4.2 là hợp lệ.
- **Lần mở test thứ hai.** Các bước 4.2 đến 4.5 từng được chấm trên test với ConvNeXt-Tiny. Chạy lại với Swin-Tiny là lần mở test thứ hai, giữ nguyên tham số đã chọn trên validation.

## 1. Kỹ thuật nào phục hồi, ở mức nào

Trong 33 cặp (kỹ thuật, họ) phù hợp, **5 cặp có bằng chứng**. Recovery là macro F1 sau xử lý trừ macro F1 khi suy giảm, trung bình ba mức, kèm khoảng tin cậy 95%. "Mức có ý nghĩa" là số mức có recovery dương có ý nghĩa sau Holm.

| Họ | Kỹ thuật, cấu hình | F1 suy giảm → sau xử lý | Recovery | Mức có ý nghĩa | Chi phí ảnh sạch |
|---|---|---|---|---:|---|
| `lowres` | unsharp, σ 1.0, amount 3.0 | 0.826 → 0.915 | +0.089 [+0.073, +0.106] | 3 | **−0.075** [−0.095, −0.053] |
| `motion_blur` | unsharp, σ 2.0, amount 1.0 | 0.857 → 0.913 | +0.056 [+0.045, +0.068] | 2 | −0.027 [−0.041, −0.013] |
| `blur` | unsharp, σ 1.0, amount 3.0 | 0.820 → 0.858 | +0.038 [+0.027, +0.049] | 2 | **−0.075** [−0.095, −0.053] |
| `sensor_noise` | bilateral, d 5, sigma_color 30 | 0.859 → 0.897 | +0.038 [+0.027, +0.048] | 2 | −0.009 [−0.017, −0.001] |
| `noise` | bilateral, d 7, sigma_color 30 | 0.861 → 0.891 | +0.030 [+0.021, +0.039] | 2 | −0.012 [−0.021, −0.004] |

Ví dụ theo mức: unsharp cho `lowres` 0.862 → 0.914, 0.844 → 0.932, 0.773 → 0.901 ở ba mức (normalized recovery 0,52, 0,74, 0,68); unsharp cho `motion_blur` mức 3 0.758 → 0.855; unsharp cho `blur` mức 2 0.855 → 0.938 nhưng mức 3 chỉ 0.658 → 0.712 (normalized recovery 0,18) và mức 1 hại (−0.022). Bilateral cho nhiễu cải thiện +0.03 đến +0.05 ở mức 2 và 3 và không đáng kể ở mức 1. Ở mức nhẹ, hầu hết các cặp không có lợi vì ảnh chưa hỏng nhiều.

**Không có bằng chứng:**
- **Các họ ánh sáng và tương phản không còn kỹ thuật nào đạt quy tắc.** `brightness_dark`: gamma tự động +0.035 [+0.025, +0.045], Retinex +0.032, gamma +0.030, nhưng mỗi kỹ thuật chỉ một mức có ý nghĩa; CLAHE +0.009 (khoảng tin cậy gồm 0). `lowlight`: gamma +0.021, một mức có ý nghĩa; CLAHE hại (−0.057). `contrast_low`: CLAHE −0.003, Retinex hại (−0.017). `brightness_bright`: gamma +0.002, gamma tự động hại (−0.014).
- `glare` (không kỹ thuật nào vượt +0.004), `shadow` (kỹ thuật tốt nhất, gamma, +0.008 và không phải kỹ thuật phù hợp), `contrast_high` (bilateral +0.001, họ không phù hợp), `jpeg` (không kỹ thuật phù hợp, kỹ thuật tốt nhất +0.013, một mức).
- `noise`: Gaussian +0.022 nhưng không mức nào có ý nghĩa, NLM không đổi (−0.001), median hại (−0.026).
- `color_cast`: gray-world phù hợp nhưng chỉ +0.012 [−0.007, +0.031] với một mức có ý nghĩa, và tốn 0.039 F1 ảnh sạch.

Tổng quan 459 ô: 138 ô phù hợp (25 tăng có ý nghĩa, 14 giảm có ý nghĩa), 321 ô đối chứng (2 tăng, 134 giảm), phần còn lại không khác biệt. Làm nét hại nặng ở họ không phù hợp (unsharp: −0.38 trên `noise` và `sensor_noise`, −0.34 trên `lowlight`, −0.22 trên `jpeg`, −0.14 trên `brightness_bright` và `contrast_high`). Median có recovery trung bình âm ở 13 trên 14 họ, gray-world ở 12, Retinex ở 13.

## 2. Chi phí trên ảnh sạch

Thay đổi macro F1 trên 888 ảnh sạch của các cấu hình tốn kém (cận trên khoảng tin cậy dưới −0,02), và lớp bị hại nhiều nhất:

| Cấu hình | Thay đổi F1 sạch | Lớp giảm recall nhiều nhất |
|---|---:|---|
| unsharp σ 1.0, amount 3.0 (`blur`, `lowres`) | −0.075 [−0.095, −0.053] | LEAF_BLIGHT −0.187 |
| median 3×3 | −0.054 [−0.070, −0.039] | ALLOCARIDARA_ATTACK −0.230 |
| gray-world | −0.039 [−0.054, −0.022] | HEALTHY_LEAF −0.209 |

Chi phí có ý nghĩa nhưng nhỏ hơn ở gamma 0.6 (−0.024, ALGAL_LEAF_SPOT −0.089), unsharp σ 2.0, amount 1.0 (−0.027, LEAF_BLIGHT −0.112), Gaussian (−0.015) và Retinex, CLAHE, bilateral (−0.009 đến −0.012). Gamma tự động và NLM gần như không tốn gì (−0.003 và −0.005, khoảng tin cậy gồm 0).

Kỹ thuật phục hồi nhiều nhất cũng đắt nhất: unsharp amount 3.0 phục hồi `lowres` +0.089 nhưng mất 0.075 F1 ảnh sạch và hơn 0.18 recall LEAF_BLIGHT. Bilateral có chi phí nhỏ (−0.009 và −0.012) và là kỹ thuật có tỉ lệ lợi trên hại tốt nhất trong các cặp có bằng chứng.

## 3. Oracle, kỹ thuật cố định và huấn luyện lại

Trung bình macro F1 trên 42 điều kiện (test), kèm F1 ảnh sạch:

| Chiến lược | F1 sạch | TB 42 điều kiện | Chênh so với không xử lý | Điều kiện tốt hơn / kém hơn |
|---|---:|---:|---|---:|
| Không xử lý | 0.962 | 0.869 | | |
| Oracle chọn theo họ trên validation | 0.962 | 0.891 | **+0.022** [+0.018, +0.026] | 15 / 1 |
| Oracle chọn theo từng điều kiện trên test (cận trên lạc quan) | 0.962 | 0.895 | +0.026 [+0.023, +0.030] | 16 / 0 |
| Cố định: gamma | 0.959 | 0.870 | +0.001 [−0.001, +0.003] | 2 / 0 |
| Cố định: gamma tự động | 0.960 | 0.866 | −0.003 [−0.007, −0.000] | 1 / 1 |
| Cố định: CLAHE | 0.950 | 0.858 | −0.011 | 2 / 12 |
| Cố định: Retinex | 0.953 | 0.851 | −0.018 | 1 / 15 |
| Cố định: Gaussian, NLM | 0.948, 0.957 | 0.857, 0.858 | −0.012, −0.011 | 3 / 19, 0 / 5 |
| Cố định: bilateral | 0.954 | 0.847 | −0.022 | 5 / 23 |
| Cố định: gray-world, median, unsharp | 0.924, 0.908, 0.887 | 0.840, 0.815, 0.749 | −0.029, −0.054, −0.120 | 1 / 20, 1 / 34, 8 / 33 |
| Huấn luyện lại bằng augmentation (seed 42) | 0.987 | **0.965** | **+0.096** [+0.086, +0.108] | 40 / 0 |

Oracle chọn kỹ thuật theo họ bằng validation (ứng viên là các kỹ thuật phù hợp với họ, cộng không xử lý): `brightness_dark` Retinex; `contrast_low` và `shadow` CLAHE; `brightness_bright` và `lowlight` gamma; `glare` gamma tự động; `blur`, `motion_blur` và `lowres` unsharp; `noise` Gaussian; `sensor_noise` bilateral; `color_cast` gray-world; `contrast_high` và `jpeg` không xử lý. Oracle này kém hơn không xử lý ở một điều kiện, `color_cast_s1` (−0.037), nơi gray-world được chọn trên validation. Theo mức, TB 42 điều kiện: không xử lý 0.936, 0.889, 0.782; oracle 0.937, 0.909, 0.826; huấn luyện lại 0.980, 0.970, 0.947.

Chi tiết từng điều kiện nằm ở `strategies_by_condition_test.csv`, so với oracle ở `retrained_vs_oracle_test.csv`.

## Đọc kết quả

- **Không kỹ thuật cố định nào hơn không xử lý** (với cấu hình chung đã khóa, áp lên mọi ảnh kể cả ảnh sạch). Gamma hòa (+0.001, khoảng tin cậy gồm 0), mọi kỹ thuật khác kém hơn. Đây là bằng chứng cho giả thuyết H3: xử lý cố định có thể tác dụng ngược, đặc biệt trên ảnh sạch và với tổn thương nhỏ.
- **Swin-Tiny làm lợi ích của xử lý ảnh nhỏ đi nhiều.** Với ConvNeXt-Tiny, Retinex, CLAHE và gamma phục hồi `brightness_dark`, `contrast_low`, `lowlight` đến +0.15. Swin-Tiny đã bền với các họ ánh sáng (F1 suy giảm 0.926 ở `brightness_dark`, so với 0.815), nên chỉ còn +0.03 đến +0.04 và không đạt quy tắc. Chỉ làm nét (nhóm mờ, độ phân giải thấp) và bilateral (nhiễu) còn bằng chứng. Kết luận về từng kỹ thuật phụ thuộc vào reference model, không nên chuyển nguyên từ model này sang model khác.
- **Chọn xử lý theo loại suy giảm vẫn có giá trị, nhưng trần thấp:** oracle theo validation +0,022, tốt hơn có ý nghĩa ở 15 điều kiện và kém hơn ở 1. Oracle chọn bằng test cao hơn một chút (0.895 so với 0.891), nên trần này không bị thổi phồng nhiều bởi việc chọn trên cùng tập đo. Đây là trần của một bộ chọn thích ứng trên đúng 10 kỹ thuật này.
- **Huấn luyện lại vượt xa oracle:** hơn oracle theo họ ở 41 trên 42 điều kiện có ý nghĩa (điều kiện còn lại là `brightness_dark_s2`, +0.013, p_holm = 0,080), không điều kiện nào kém hơn; trung bình +0,075. Nói cách khác, dù bộ chọn thích ứng hoàn hảo, xử lý ảnh cổ điển cũng không đạt mức huấn luyện lại. Giá trị của xử lý ảnh thích ứng nằm ở trường hợp không thể huấn luyện lại classifier đã triển khai, và với Swin-Tiny giá trị đó nhỏ (tối đa khoảng +0.02 đến +0.03 trung bình). Đây là kết quả âm hợp lệ và cần báo thẳng.
- Huấn luyện lại cũng tăng F1 ảnh sạch (0.987 so với 0.962, +0.025 [+0.013, +0.037]), nhưng với một seed và mức tăng trên validation sạch chỉ +0.005, chưa đủ để kết luận augmentation làm tăng độ chính xác ảnh sạch, xem `reports/augmentation/README.md`.

## Giới hạn

- Một seed cho reference và một seed cho model huấn luyện lại ở bảng này. Khoảng tin cậy bootstrap chỉ bắt được dao động do tập ảnh test, không bắt được dao động do huấn luyện.
- Kết luận "kỹ thuật cố định không giúp" chỉ cho **cấu hình chung đã khóa**, được chọn bằng macro F1 trung bình trên các họ phù hợp mà không xét chi phí ảnh sạch (với unsharp đó là cấu hình mạnh nhất, mất 0,075 F1 sạch). Một cấu hình nhẹ hơn chưa được đo trên test. "Kỹ thuật cố định tốt nhất" cho Phase 5 nên chọn trên validation, với cả 42 điều kiện và ảnh sạch, rồi đo trên test một lần.
- Augmentation của model huấn luyện lại dùng đúng 14 họ và các mức của benchmark, nên so sánh với oracle là độ bền trong phân phối. Chưa có kết quả giữ lại một số họ khỏi huấn luyện.
- Oracle chỉ gồm 10 kỹ thuật và cấu hình đã khóa. `lowlight` được cả task 4.3 và 4.4 xử lý nhưng việc ghép hai bước (ví dụ làm sáng rồi khử nhiễu) chưa được đo.
- Holm trên 459 ô nên các lợi ích nhỏ (dưới khoảng 0,02) khó đạt ý nghĩa. Nhiều cặp sát ngưỡng bị loại bởi quy tắc đã khóa (gamma, gamma tự động và Retinex cho `brightness_dark`, gamma cho `lowlight`, mỗi cặp chỉ một mức có ý nghĩa); quy tắc không được đổi sau khi xem test, các cặp đó vẫn có trong `recovery_family_test.csv`.
- Chỉ 2 ô đối chứng tăng có ý nghĩa (quan sát thăm dò, không nằm trong quy tắc khóa và không được đưa vào danh sách đầu vào của Phase 5).
- Không có mặt nạ vết bệnh, nên việc mất chi tiết vết bệnh chỉ đo gián tiếp qua recall từng lớp.

## File

| File | Nội dung |
|---|---|
| `recovery_summary.json` | tóm tắt: kiểm tra tái hiện, quy tắc, danh sách cặp có bằng chứng và không có bằng chứng, bảng chiến lược |
| `recovery_map_test.csv` | 459 ô: kỹ thuật × cấu hình × điều kiện, F1 trước và sau, degradation, recovery, normalized recovery và khoảng tin cậy, McNemar, Holm, recall từng lớp |
| `recovery_family_test.csv` | theo (kỹ thuật, họ): recovery trung bình ba mức kèm khoảng tin cậy, số mức có ý nghĩa, chi phí ảnh sạch, `evidence`, `costly` |
| `clean_cost_test.csv` | thay đổi F1 và recall từng lớp trên ảnh sạch của từng cấu hình, kèm khoảng tin cậy và p đã hiệu chỉnh Holm |
| `strategies_test.csv`, `strategies_by_condition_test.csv` | các chiến lược (không xử lý, oracle, cố định, huấn luyện lại): tổng và từng điều kiện |
| `oracle_choices_test.csv`, `retrained_vs_oracle_test.csv` | kỹ thuật oracle chọn cho từng điều kiện, và so sánh model huấn luyện lại với oracle |
| `recovery_heatmap.png`, `normalized_recovery_heatmap.png`, `strategies_by_family.png` | heatmap (ô khung đen là cặp phù hợp) và macro F1 theo họ của các chiến lược |

Dự đoán và xác suất từng ảnh (`recovery_predictions_test.npz`, 3,9 MB) ở lại trong output Kaggle của notebook 08, không đưa vào Git.
