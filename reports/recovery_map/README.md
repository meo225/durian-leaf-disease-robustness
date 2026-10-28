# Bản đồ phục hồi (task 4.5)

Đo từng kỹ thuật xử lý ảnh của task 4.3 và 4.4 trên tập **test** bằng cùng một reference model, ConvNeXt-Tiny đóng băng và train trên ảnh sạch, trên cùng 888 ảnh và nhãn, theo kỹ thuật × họ suy giảm × mức. Trả lời ba câu: kỹ thuật nào phục hồi nhận diện và ở mức nào, chi phí trên ảnh sạch và theo từng lớp là bao nhiêu, và cận trên nếu biết trước loại suy giảm (oracle) so với huấn luyện lại bằng augmentation.

Tiêu chí là macro F1 của bài toán nhận diện bệnh, không phải ảnh nhìn đẹp hơn. Code ở `src/evaluation/recovery_map.py` (chạy model, cần GPU), `recovery_stats.py` và `recovery_report.py` (thống kê, chạy trên CPU), cấu hình ở `configs/recovery_map.yaml`, notebook `notebooks/08_recovery_map.ipynb`.

## Cách đo

- **Kỹ thuật và cấu hình:** 10 kỹ thuật (gamma, gamma tự động, CLAHE, Retinex, gray-world từ task 4.3; Gaussian, median, bilateral, NLM, unsharp từ task 4.4), mỗi họ dùng cấu hình đã khóa trên validation (`reports/processing/*_selected.json`). Họ phù hợp với kỹ thuật dùng cấu hình tốt nhất của họ, họ không phù hợp dùng cấu hình chung. Không chỉnh tham số theo kết quả test.
- **Phạm vi:** mọi kỹ thuật chạy trên cả 42 điều kiện của benchmark 4.1 và trên ảnh sạch, tổng 468 ô (kể cả cấu hình chung chạy thêm trên các họ có cấu hình riêng).
- **Thống kê:** khoảng tin cậy 95% bằng bootstrap theo ảnh (1.000 lần, cùng một ma trận lấy lại mẫu cho mọi ô), McNemar chính xác ghép cặp trên từng ảnh, hiệu chỉnh Holm trên toàn bộ 468 ô, mức ý nghĩa 0,05. Normalized recovery chỉ tính khi degradation lớn hơn 0,02.
- **Quy tắc "có bằng chứng"** (khóa trước khi mở test, `configs/recovery_map.yaml`): khoảng tin cậy của recovery trung bình ba mức nằm trên 0, và ít nhất hai mức có recovery dương có ý nghĩa sau Holm. Chi phí ảnh sạch chỉ gắn nhãn `costly` (cận trên khoảng tin cậy dưới −0,02), không dùng để loại.
- **Kiểm tra tái hiện:** dự đoán "không xử lý" ở đây đúng cùng mẫu với task 4.2 (888 ảnh, cùng đường dẫn và nhãn) và chỉ lệch 1 dự đoán trên 43 bộ ảnh do làm tròn xác suất, nên so ghép cặp với 4.2 là hợp lệ.

## 1. Kỹ thuật nào phục hồi, ở mức nào

Trong 33 cặp (kỹ thuật, họ) phù hợp, **12 cặp có bằng chứng**. Recovery là macro F1 sau xử lý trừ macro F1 khi suy giảm, trung bình ba mức, kèm khoảng tin cậy 95%. "Mức có ý nghĩa" là số mức có recovery dương có ý nghĩa sau Holm.

| Họ | Kỹ thuật, cấu hình | F1 suy giảm → sau xử lý | Recovery | Mức có ý nghĩa | Chi phí ảnh sạch |
|---|---|---|---|---:|---|
| `brightness_dark` | Retinex, `msr_small_a0.5` | 0.815 → 0.952 | +0.137 [+0.122, +0.154] | 2 | −0.014 [−0.024, −0.004] |
| `brightness_dark` | gamma tự động, `auto_0.4` | 0.815 → 0.892 | +0.077 [+0.067, +0.088] | 2 | +0.002 [−0.004, +0.008] |
| `brightness_dark` | gamma, `gamma_0.6` | 0.815 → 0.888 | +0.073 [+0.062, +0.084] | 2 | −0.009 [−0.019, +0.001] |
| `brightness_dark` | CLAHE, `clahe_c1_t4` | 0.815 → 0.881 | +0.067 [+0.054, +0.079] | 2 | −0.020 [−0.032, −0.008] |
| `contrast_low` | Retinex, `msr_large_a0.5` | 0.753 → 0.901 | +0.147 [+0.132, +0.163] | 2 | −0.014 [−0.025, −0.004] |
| `contrast_low` | CLAHE, `clahe_c1_t4` | 0.753 → 0.870 | +0.117 [+0.104, +0.130] | 2 | −0.020 [−0.032, −0.008] |
| `lowlight` | gamma, `gamma_0.6` | 0.771 → 0.860 | +0.089 [+0.074, +0.104] | 2 | −0.009 [−0.019, +0.001] |
| `lowlight` | gamma tự động, `auto_0.4` | 0.771 → 0.859 | +0.088 [+0.073, +0.103] | 2 | +0.002 [−0.004, +0.008] |
| `lowres` | unsharp, σ 1.0, amount 3.0 | 0.708 → 0.929 | +0.221 [+0.205, +0.238] | 3 | **−0.218** [−0.249, −0.189] |
| `motion_blur` | unsharp, σ 1.0, amount 2.0 | 0.774 → 0.927 | +0.153 [+0.137, +0.170] | 3 | **−0.113** [−0.137, −0.090] |
| `blur` | unsharp, σ 2.0, amount 1.0 | 0.746 → 0.824 | +0.077 [+0.067, +0.087] | 2 | **−0.065** [−0.084, −0.045] |
| `sensor_noise` | bilateral, d 7, sigma_color 30 | 0.818 → 0.865 | +0.047 [+0.035, +0.060] | 2 | −0.014 [−0.026, −0.002] |

Ví dụ theo mức: Retinex cho `brightness_dark` 0.617 → 0.946 ở mức 3 (normalized recovery 0,95) và 0.876 → 0.954 ở mức 2, nhưng không đổi ở mức 1; Retinex cho `contrast_low` 0.439 → 0.824 ở mức 3; unsharp cho `lowres` 0.766 → 0.961, 0.708 → 0.961, 0.650 → 0.866 ở ba mức. Ở mức nhẹ, hầu hết các cặp không có lợi vì ảnh chưa hỏng nhiều. Mờ nặng (`blur` mức 3) chỉ phục hồi một phần (+0.059).

**Không có bằng chứng:**
- `brightness_bright` và `contrast_high`, hai họ nặng: kỹ thuật tốt nhất chỉ +0.031 (gamma, một mức) và +0.025 (bilateral, họ không phù hợp). Gamma tự động còn làm hại `brightness_bright` (−0.056).
- `noise`: bilateral +0.040 [+0.029, +0.053] nhưng chỉ một mức có ý nghĩa, nên không đạt quy tắc. Gaussian và NLM không đổi (−0.001, 0.000), median hại (−0.067).
- `shadow` (Retinex +0.026, một mức), `glare` (không kỹ thuật nào vượt 0.003), `jpeg` (không kỹ thuật phù hợp, kỹ thuật tốt nhất +0.002).
- `color_cast`: gray-world phù hợp nhưng **hại** ở cả ba mức (−0.103). CLAHE trên `lowlight` hại nặng (−0.291), vì khuếch đại nhiễu hạt.

Tổng quan 468 ô: 99 ô phù hợp (32 tăng có ý nghĩa, 13 giảm có ý nghĩa), 321 ô đối chứng (15 tăng, 143 giảm), phần còn lại không khác biệt. Làm nét trên họ nhiễu và `lowlight` hại nặng (unsharp: −0.50 trên `noise`, −0.58 trên `sensor_noise`, −0.55 trên `lowlight`), và hại ở hầu hết họ không phù hợp. Median hại ở cả 14 họ.

## 2. Chi phí trên ảnh sạch

Thay đổi macro F1 trên 888 ảnh sạch của các cấu hình tốn kém (cận trên khoảng tin cậy dưới −0,02), và lớp bị hại nhiều nhất:

| Cấu hình | Thay đổi F1 sạch | Lớp giảm recall nhiều nhất |
|---|---:|---|
| unsharp σ 1.0, amount 3.0 (`lowres`) | −0.218 [−0.249, −0.189] | PHOMOPSIS_LEAF_SPOT −0.534 |
| unsharp σ 1.0, amount 2.0 (`motion_blur`) | −0.113 [−0.137, −0.090] | PHOMOPSIS_LEAF_SPOT −0.301 |
| unsharp σ 2.0, amount 1.0 (`blur`) | −0.065 [−0.084, −0.045] | PHOMOPSIS_LEAF_SPOT −0.193 |
| median 3×3 | −0.116 [−0.139, −0.094] | ALLOCARIDARA_ATTACK −0.470 |
| gray-world | −0.090 [−0.113, −0.070] | HEALTHY_LEAF −0.276 |

Kỹ thuật phục hồi nhiều nhất cũng đắt nhất: ba cặp unsharp đều có bằng chứng nhưng đều `costly`. Retinex (−0.014) và CLAHE (−0.020) có chi phí có ý nghĩa nhưng nhỏ. **Gamma tự động gần như không tốn gì** (+0.002) mà vẫn phục hồi `brightness_dark` và `lowlight`, nên có tỉ lệ lợi trên hại tốt nhất trong các kỹ thuật có bằng chứng.

## 3. Oracle, kỹ thuật cố định và huấn luyện lại

Trung bình macro F1 trên 42 điều kiện (test), kèm F1 ảnh sạch:

| Chiến lược | F1 sạch | TB 42 điều kiện | Chênh so với không xử lý | Điều kiện tốt hơn / kém hơn |
|---|---:|---:|---|---:|
| Không xử lý | 0.964 | 0.797 | | |
| Oracle chọn theo họ trên validation | 0.964 | 0.867 | **+0.069** [+0.065, +0.074] | 20 / 0 |
| Oracle chọn theo từng điều kiện trên test (cận trên lạc quan) | 0.964 | 0.867 | +0.070 [+0.065, +0.075] | 20 / 0 |
| Cố định: gamma | 0.960 | 0.800 | +0.002 [−0.001, +0.006] | 7 / 11 |
| Cố định: gamma tự động | 0.968 | 0.799 | +0.002 [−0.002, +0.005] | 3 / 5 |
| Cố định: Retinex | 0.950 | 0.797 | 0.000 [−0.006, +0.006] | 6 / 19 |
| Cố định: CLAHE | 0.944 | 0.784 | −0.013 | 12 / 14 |
| Cố định: bilateral, NLM | 0.957, 0.965 | 0.778, 0.783 | −0.019, −0.014 | 5 / 18, 2 / 11 |
| Cố định: median, gray-world, unsharp | 0.848, 0.874, 0.747 | 0.694, 0.658, 0.624 | −0.103, −0.139, −0.173 | 0 / 35, 0 / 41, 9 / 32 |
| Huấn luyện lại bằng augmentation (seed 42) | 0.980 | **0.960** | **+0.163** [+0.153, +0.173] | 42 / 0 |

Oracle chọn kỹ thuật theo họ bằng validation (ứng viên là các kỹ thuật phù hợp với họ, cộng không xử lý): `brightness_dark` và `shadow` chọn Retinex, `contrast_low` Retinex, `lowlight` và `brightness_bright` gamma, `blur`, `motion_blur`, `lowres` unsharp, `noise` và `sensor_noise` bilateral, `glare` gamma tự động, `color_cast`, `contrast_high` và `jpeg` không xử lý. Theo mức, TB 42 điều kiện: không xử lý 0.920, 0.818, 0.654; oracle 0.942, 0.897, 0.761; huấn luyện lại 0.975, 0.964, 0.941.

Chi tiết từng điều kiện nằm ở `strategies_by_condition_test.csv`, so với oracle ở `retrained_vs_oracle_test.csv`.

## Đọc kết quả

- **Không kỹ thuật cố định nào hơn không xử lý** (với cấu hình chung đã khóa, áp lên mọi ảnh kể cả ảnh sạch). Đây là bằng chứng cho giả thuyết H3: xử lý cố định có thể tác dụng ngược, đặc biệt trên ảnh sạch và với tổn thương nhỏ.
- **Chọn xử lý theo loại suy giảm có giá trị thật:** oracle +0,069, tốt hơn có ý nghĩa ở 20 điều kiện và không điều kiện nào kém hơn. Oracle chọn bằng validation gần trùng oracle chọn bằng test (0.867 và 0.867), nên trần này không bị thổi phồng bởi việc chọn trên cùng tập đo. Đây là trần của một bộ chọn thích ứng trên đúng 10 kỹ thuật này.
- **Huấn luyện lại vượt cả oracle:** hơn oracle ở 37 trên 42 điều kiện có ý nghĩa, 5 điều kiện không khác biệt (`glare_s1`, `sensor_noise_s1`, `lowres_s1`, `lowres_s2`, `motion_blur_s2`, đều là điều kiện oracle đã gần mức ảnh sạch), không điều kiện nào kém hơn; trung bình +0,093. Nói cách khác, dù bộ chọn thích ứng hoàn hảo, xử lý ảnh cổ điển cũng không đạt mức huấn luyện lại. Giá trị của xử lý ảnh thích ứng nằm ở trường hợp không thể huấn luyện lại classifier đã triển khai. Đây là kết quả âm hợp lệ và cần báo thẳng.
- Điểm cộng của huấn luyện lại trên ảnh sạch (0.980 so với 0.964) chưa đủ để kết luận, xem `reports/augmentation/README.md`.

## Giới hạn

- Một seed cho reference và một seed cho model huấn luyện lại ở bảng này. Khoảng tin cậy bootstrap chỉ bắt được dao động do tập ảnh test, không bắt được dao động do huấn luyện.
- Kết luận "kỹ thuật cố định không giúp" chỉ cho **cấu hình chung đã khóa**, được chọn bằng macro F1 trung bình trên các họ phù hợp mà không xét chi phí ảnh sạch (với unsharp đó là cấu hình mạnh nhất, mất 0,218 F1 sạch). Một cấu hình nhẹ hơn chưa được đo trên test. "Kỹ thuật cố định tốt nhất" cho Phase 5 nên chọn trên validation, với cả 42 điều kiện và ảnh sạch, rồi đo trên test một lần.
- Augmentation của model huấn luyện lại dùng đúng 14 họ và các mức của benchmark, nên so sánh với oracle là độ bền trong phân phối. Các lần chạy bổ sung (seed 43 và giữ lại ba họ khỏi huấn luyện) đã chạy nhưng kết quả chưa đưa vào repo.
- Oracle chỉ gồm 10 kỹ thuật và cấu hình đã khóa. `lowlight` được cả task 4.3 và 4.4 xử lý nhưng việc ghép hai bước (ví dụ làm sáng rồi khử nhiễu) chưa được đo.
- Holm trên 468 ô nên các lợi ích nhỏ (dưới khoảng 0,02) khó đạt ý nghĩa. Một số cặp sát ngưỡng bị loại bởi quy tắc đã khóa (bilateral cho `noise`, Retinex cho `shadow`); quy tắc không được đổi sau khi xem test, các cặp đó vẫn có trong `recovery_family_test.csv`.
- 15 ô đối chứng tăng có ý nghĩa ngoài dự kiến (ví dụ CLAHE trên `blur`, `motion_blur`, `lowres` từ +0,04 đến +0,08, gamma trên `color_cast` +0,040) là quan sát thăm dò, không nằm trong quy tắc khóa và không được đưa vào danh sách đầu vào của Phase 5.
- Không có mặt nạ vết bệnh, nên việc mất chi tiết vết bệnh chỉ đo gián tiếp qua recall từng lớp.

## File

| File | Nội dung |
|---|---|
| `recovery_summary.json` | tóm tắt: kiểm tra tái hiện, quy tắc, danh sách cặp có bằng chứng và không có bằng chứng, bảng chiến lược |
| `recovery_map_test.csv` | 468 ô: kỹ thuật × cấu hình × điều kiện, F1 trước và sau, degradation, recovery, normalized recovery và khoảng tin cậy, McNemar, Holm, recall từng lớp |
| `recovery_family_test.csv` | theo (kỹ thuật, họ): recovery trung bình ba mức kèm khoảng tin cậy, số mức có ý nghĩa, chi phí ảnh sạch, `evidence`, `costly` |
| `clean_cost_test.csv` | thay đổi F1 và recall từng lớp trên ảnh sạch của từng cấu hình, kèm khoảng tin cậy và p đã hiệu chỉnh Holm |
| `strategies_test.csv`, `strategies_by_condition_test.csv` | các chiến lược (không xử lý, oracle, cố định, huấn luyện lại): tổng và từng điều kiện |
| `oracle_choices_test.csv`, `retrained_vs_oracle_test.csv` | kỹ thuật oracle chọn cho từng điều kiện, và so sánh model huấn luyện lại với oracle |
| `recovery_heatmap.png`, `normalized_recovery_heatmap.png`, `strategies_by_family.png` | heatmap (ô khung đen là cặp phù hợp) và macro F1 theo họ của các chiến lược |

Dự đoán và xác suất từng ảnh (`recovery_predictions_test.npz`, 3,9 MB) ở lại trong output Kaggle của notebook 08, không đưa vào Git.
