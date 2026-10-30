# Kết quả xử lý ảnh (Phase 4)

Thư mục này gom kết quả các kỹ thuật xử lý ảnh trên benchmark suy giảm của task 4.1. Mỗi kỹ thuật được đo bằng cùng reference model, Swin-Tiny đóng băng và train trên ảnh sạch, trên cùng mẫu và nhãn. Tiêu chí là macro F1 của bài toán nhận diện bệnh, không phải ảnh nhìn đẹp hơn.

Reference model đã đổi từ ConvNeXt-Tiny sang Swin-Tiny (xem `reports/baselines/README.md`), nên task 4.3 và 4.4 được chạy lại và chọn lại tham số trên validation. Kết quả dưới đây thay cho lần chạy ConvNeXt-Tiny trước đó.

Mục dưới đây là task 4.4 (khử nhiễu và làm nét). Cấu hình đã chọn của task 4.3 (ánh sáng, tương phản, màu) nằm ở `illumination_selected.json`, kết quả tóm tắt ở mục cuối.

## Task 4.4. Khử nhiễu và làm nét

Tất cả số liệu trên **tập validation** (444 ảnh), một seed. Tập test chưa mở. Tham số được chọn và khóa ở đây, task 4.5 đo lại trên test với đúng các cấu hình này. F1 sạch của reference model là 0.9664.

### Cách làm

| Nhóm | Kỹ thuật | Họ suy giảm phù hợp |
|---|---|---|
| Khử nhiễu | Gaussian, median, bilateral, non-local means (NLM) | `noise`, `sensor_noise`, `lowlight` |
| Làm nét | unsharp mask | `blur`, `motion_blur`, `lowres` |

- `jpeg` không có kỹ thuật phù hợp, nên làm điều kiện đối chứng cho cả năm kỹ thuật. Mỗi kỹ thuật còn chạy thêm trên các họ không phù hợp với nó để thấy nó làm hại ở đâu.
- Mỗi kỹ thuật chạy một lưới tham số (29 cấu hình tổng cộng, xem `configs/restoration.yaml`, version 3). Điểm của một cấu hình là macro F1 trung bình trên ba mức của họ. Các cấu hình trong khoảng 0,003 so với tốt nhất thì chọn cấu hình giữ F1 ảnh sạch cao hơn. Mỗi kỹ thuật có một cấu hình cho từng họ phù hợp, và một cấu hình chung dùng làm "kỹ thuật cố định".
- Code ở `src/processing/restoration.py`, notebook `notebooks/06_restoration_benchmark.ipynb`, cùng giao diện và cùng luật chọn với task 4.3.
- Lưới version 3 (mở rộng thêm mức nhẹ hơn cho khử nhiễu và mức mạnh hơn cho unsharp) được giữ nguyên cho Swin-Tiny, vẫn chỉ chọn trên validation.

### Kết quả

Recovery là macro F1 sau xử lý trừ macro F1 khi suy giảm, trung bình trên ba mức, với cấu hình đã chọn cho từng họ. Dương là xử lý giúp, âm là làm tệ hơn. Chỉ gồm cặp kỹ thuật và họ phù hợp.

| Họ | F1 không xử lý (TB 3 mức) | Gaussian | Median | Bilateral | NLM | Unsharp |
|---|---:|---:|---:|---:|---:|---:|
| `noise` | 0.854 | **+0.027** | −0.023 | +0.025 | −0.002 | |
| `sensor_noise` | 0.866 | +0.025 | −0.035 | **+0.031** | −0.001 | |
| `lowlight` | 0.831 | +0.006 | −0.048 | **+0.017** | −0.009 | |
| `blur` | 0.813 | | | | | **+0.063** |
| `motion_blur` | 0.841 | | | | | **+0.054** |
| `lowres` | 0.820 | | | | | **+0.091** |

**Làm nét giúp rõ ở mức vừa, khử nhiễu chỉ giúp nhỏ.** Swin-Tiny vốn bền hơn ConvNeXt-Tiny với mọi họ này (F1 không xử lý cao hơn ConvNeXt-Tiny từ 0,025 đến 0,12), nên khoảng trống để phục hồi nhỏ hơn.

Unsharp, F1 trước và sau xử lý theo mức:

| Họ (cấu hình) | Mức 1 | Mức 2 | Mức 3 |
|---|---|---|---|
| `blur` (σ 1.0, amount 3.0) | 0.958 → 0.941 | 0.828 → 0.949 | 0.653 → 0.738 |
| `motion_blur` (σ 2.0, amount 1.0) | 0.921 → 0.937 | 0.864 → 0.923 | 0.739 → 0.826 |
| `lowres` (σ 1.0, amount 3.0) | 0.859 → 0.921 | 0.822 → 0.937 | 0.779 → 0.874 |

- `blur` và `lowres` ở mức 2 gần về F1 sạch (0.949 và 0.937). `motion_blur` mức 3 phục hồi +0.087.
- Mờ rất nhẹ (`blur` mức 1) bị hại (−0.017) vì ảnh chưa hỏng nhiều. Mờ nặng (`blur` mức 3) chỉ phục hồi một phần (+0.085).

Khử nhiễu, F1 trước và sau xử lý theo mức:

| Họ (kỹ thuật, cấu hình) | Mức 1 | Mức 2 | Mức 3 |
|---|---|---|---|
| `noise` (Gaussian σ 0.5) | 0.936 → 0.932 | 0.866 → 0.898 | 0.759 → 0.812 |
| `noise` (bilateral d 7, sigma_color 30) | 0.936 → 0.938 | 0.866 → 0.909 | 0.759 → 0.789 |
| `sensor_noise` (bilateral d 5, sigma_color 30) | 0.921 → 0.928 | 0.864 → 0.902 | 0.812 → 0.861 |
| `lowlight` (bilateral d 7, sigma_color 10) | 0.897 → 0.903 | 0.843 → 0.872 | 0.754 → 0.771 |

- Gaussian σ 0.5 và bilateral cho recovery gần nhau ở `noise` (+0.027 và +0.025) và `sensor_noise` (+0.025 và +0.031). Chênh chỉ 0,002 đến 0,006, nên không nên xếp hạng hai kỹ thuật này trên một seed và 444 ảnh.
- Lợi ích tập trung ở mức vừa và nặng. Nhiễu nhẹ (mức 1) không đổi hoặc hơi tệ đi. `lowlight` chỉ +0.017, vì mất sáng không phải do nhiễu.
- NLM cho cấu hình tốt nhất là mức nhẹ nhất của lưới (h 1.5), tức gần như không lọc: recovery khoảng 0. Median luôn hại ở mọi họ (−0.02 đến −0.05), kể cả với cửa sổ nhỏ nhất (3×3).

### Chi phí trên ảnh sạch

Thay đổi macro F1 sạch (gốc 0.9664) và recall từng lớp, với cấu hình chung của từng kỹ thuật (`restoration_clean_cost.csv`):

| Kỹ thuật, cấu hình | F1 sạch sau xử lý | Thay đổi | Lớp giảm recall nhiều nhất |
|---|---:|---:|---|
| Gaussian σ 0.5 | 0.9577 | −0.009 | ALLOCARIDARA_ATTACK −0.04 |
| NLM h 1.5 | 0.9593 | −0.007 | ALLOCARIDARA_ATTACK −0.03 |
| Bilateral d 5, sigma_color 30 | 0.9474 | −0.019 | ALLOCARIDARA_ATTACK −0.08 |
| Median 3×3 | 0.9013 | −0.065 | ALLOCARIDARA_ATTACK −0.27 |
| Unsharp σ 1.0, amount 3.0 | 0.9120 | −0.054 | LEAF_BLIGHT −0.14 |

- Chi phí đo cho cấu hình chung. Cấu hình riêng của từng họ (ví dụ bilateral d 7 hay unsharp σ 2.0, amount 1.0) chưa có dòng chi phí riêng.
- Kỹ thuật phục hồi tốt nhất, unsharp mạnh, cũng làm hại ảnh sạch nhiều nhất (−0.054, lớp LEAF_BLIGHT và PHOMOPSIS_LEAF_SPOT giảm 0.14 và 0.09). Áp cố định lên mọi ảnh thì mất nhiều hơn được ở các họ mờ nhẹ. Các bộ lọc làm mịn hại nhiều nhất lớp ALLOCARIDARA_ATTACK, lớp dựa vào chi tiết nhỏ.
- Kết quả này ủng hộ việc chọn xử lý theo từng ảnh, nhưng chưa phải bằng chứng: bộ chọn thích ứng được đánh giá ở các task sau.

### Đối chứng

Cấu hình chung của mỗi kỹ thuật chạy trên các họ không phù hợp. Recovery trung bình:

| Kỹ thuật | Họ không phù hợp | Recovery |
|---|---|---:|
| Unsharp | `noise`, `sensor_noise`, `lowlight` | −0.40, −0.39, −0.33 |
| Unsharp | `jpeg` | −0.23 |
| Gaussian | `blur`, `motion_blur`, `lowres`, `jpeg` | −0.02, −0.02, −0.02, +0.01 |
| Median | `blur`, `motion_blur`, `lowres`, `jpeg` | −0.04, −0.04, −0.03, +0.01 |
| NLM | `blur`, `motion_blur`, `lowres`, `jpeg` | −0.01, −0.02, 0.00, −0.02 |
| Bilateral | `blur`, `motion_blur`, `lowres`, `jpeg` | −0.03, −0.04, −0.03, +0.02 |

- Làm nét trên ảnh nhiễu khuếch đại nhiễu và làm hại nặng, đúng với lý do làm nét chỉ nên áp cho họ mờ. Làm mịn trên ảnh mờ làm tệ thêm nhẹ.
- JPEG: không kỹ thuật nào cho recovery đáng kể (bilateral +0.025, Gaussian +0.014, median +0.009 trung bình ba mức). Trên từng điều kiện, median 3×3 cho +0.073 ở `jpeg_s3` (F1 0.568 → 0.641) nhưng đây là chọn trên chính tập đo, chỉ để quan sát. Giả thuyết "JPEG không có kỹ thuật phù hợp" cơ bản được ủng hộ.

### Giới hạn

- Một seed, validation 444 ảnh. Chênh lệch khoảng 0.03 trở xuống chưa đủ để kết luận. Chưa có khoảng tin cậy hay kiểm định ghép cặp, task 4.5 làm trên test.
- Cấu hình chọn trên chính tập đo nên điểm hơi lạc quan. Cột "kỹ thuật tốt nhất mỗi điều kiện" trong `restoration_best_per_condition.csv` chọn trên cùng tập nên chỉ để quan sát, không phải kết luận.
- Một số cấu hình vẫn nằm ở biên lưới: unsharp cho `blur` và `lowres` (σ 1.0, amount 3.0 là mức mạnh nhất), NLM h 1.5 (mức nhẹ nhất). Mức 2 của `blur` và `lowres` đã gần F1 sạch nên lợi ích thêm nhỏ.
- Chi phí trên ảnh sạch đo trên tập validation, theo cấu hình chung. Không có mặt nạ vết bệnh nên việc mất chi tiết vết bệnh chỉ đo gián tiếp qua recall từng lớp.
- Mỗi kỹ thuật tối ưu độc lập. Ghép hai bước (ví dụ khử nhiễu rồi làm sáng với `lowlight`, họ cả hai task cùng xử lý) chưa được đo.

## Task 4.3. Ánh sáng, tương phản và màu (tóm tắt)

Validation, một seed, F1 sạch 0.9664. Test chưa mở. Cấu hình đã chọn: `illumination_selected.json`. Recovery trung bình ba mức với cấu hình đã chọn:

| Họ | Kỹ thuật | Recovery |
|---|---|---:|
| `brightness_dark` | Retinex (msr_large_a0.5) | +0.055 |
| `brightness_dark` | gamma tự động / gamma 0.6 / CLAHE | +0.054 / +0.052 / +0.037 |
| `lowlight` | gamma 0.7 / gamma tự động | +0.035 / +0.029 |
| `brightness_bright` | gamma 1.25 | +0.008 |
| `contrast_low`, `shadow`, `glare`, `color_cast` | CLAHE, CLAHE, gamma tự động, gray-world | +0.002, +0.002, +0.004, +0.003 |
| `contrast_high` | không kỹ thuật nào | gần 0 hoặc âm |

- Rõ nhất ở `brightness_dark` mức 3 (0.811 → 0.942 với Retinex) và `lowlight` mức 3 (0.754 → 0.813 với gamma tự động). `color_cast` mức 3 phục hồi 0.851 → 0.915 với gray-world, nhưng trung bình ba mức chỉ +0.003 vì mức 1 và 2 gần như không hỏng.
- Phục hồi nhỏ ở `shadow`, `glare`, `contrast_low`. Gần như không phục hồi ở `brightness_bright` (gray-world mức 3: 0.691 → 0.752 là điều kiện duy nhất có lợi) và `contrast_high` (mức 1 và 2 còn giảm nhẹ).
- Mọi kỹ thuật đều giảm F1 ảnh sạch: gamma tự động −0.005, gamma 0.85 −0.007, CLAHE −0.007, Retinex −0.026, gray-world −0.037 (hại nhất).
- Cùng các giới hạn như 4.4: một seed, chọn trên chính validation, bảng "kỹ thuật tốt nhất mỗi điều kiện" (`illumination_best_per_condition.csv`) chỉ để quan sát.

### File

Notebook `06_restoration_benchmark.ipynb` và notebook của task 4.3 sinh các file sau. Bản chốt chép vào thư mục này. Dự đoán từng ảnh và ảnh minh họa (khoảng 3,5 MB mỗi hình, nặng nên không đưa vào Git) ở lại trong output Kaggle của notebook.

| File | Nội dung |
|---|---|
| `restoration_selected.json`, `illumination_selected.json` | cấu hình đã chọn cho từng kỹ thuật và họ, kèm cấu hình chung, chi phí ảnh sạch |
| `restoration_matrix.csv`, `illumination_matrix.csv` | bản đồ recovery: kỹ thuật × điều kiện, kèm degradation, recovery, normalized recovery |
| `restoration_best_per_condition.csv`, `illumination_best_per_condition.csv` | kỹ thuật tốt nhất mỗi điều kiện (chỉ để quan sát) |
| `restoration_clean_cost.csv`, `illumination_clean_cost.csv` | thay đổi macro F1 và recall từng lớp trên ảnh sạch |
| `restoration_*.png`, `illumination_*.png` | heatmap recovery và normalized recovery, bảng xếp hạng, kỹ thuật tốt nhất mỗi điều kiện, chi phí ảnh sạch, confusion matrix ca điển hình |
| `restoration_predictions.npz` | dự đoán từng ảnh của mọi lượt chạy (Kaggle) |
