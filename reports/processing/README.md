# Kết quả xử lý ảnh (Phase 4)

Thư mục này gom kết quả các kỹ thuật xử lý ảnh trên benchmark suy giảm của task 4.1. Mỗi kỹ thuật được đo bằng cùng reference model, ConvNeXt-Tiny đóng băng và train trên ảnh sạch, trên cùng mẫu và nhãn. Tiêu chí là macro F1 của bài toán nhận diện bệnh, không phải ảnh nhìn đẹp hơn.

Mục dưới đây là task 4.4 (khử nhiễu và làm nét). Cấu hình đã chọn của task 4.3 (ánh sáng, tương phản, màu) nằm ở `illumination_selected.json`.

## Task 4.4. Khử nhiễu và làm nét

Tất cả số liệu trên **tập validation** (444 ảnh), một seed. Tập test chưa mở. Tham số được chọn và khóa ở đây, task 4.5 đo lại trên test với đúng các cấu hình này. F1 sạch của reference model là 0.9775.

### Cách làm

| Nhóm | Kỹ thuật | Họ suy giảm phù hợp |
|---|---|---|
| Khử nhiễu | Gaussian, median, bilateral, non-local means (NLM) | `noise`, `sensor_noise`, `lowlight` |
| Làm nét | unsharp mask | `blur`, `motion_blur`, `lowres` |

- `jpeg` không có kỹ thuật phù hợp, nên làm điều kiện đối chứng cho cả năm kỹ thuật. Mỗi kỹ thuật còn chạy thêm trên các họ không phù hợp với nó để thấy nó làm hại ở đâu.
- Mỗi kỹ thuật chạy một lưới tham số (29 cấu hình tổng cộng, xem `configs/restoration.yaml`). Điểm của một cấu hình là macro F1 trung bình trên ba mức của họ. Các cấu hình trong khoảng 0,003 so với tốt nhất thì chọn cấu hình giữ F1 ảnh sạch cao hơn. Mỗi kỹ thuật có một cấu hình cho từng họ phù hợp, và một cấu hình chung dùng làm "kỹ thuật cố định".
- Code ở `src/processing/restoration.py`, notebook `notebooks/06_restoration_benchmark.ipynb`, cùng giao diện và cùng luật chọn với task 4.3.
- Lưới chạy hai lần. Lần đầu (version 1, 19 cấu hình) chọn đúng giá trị nhẹ nhất hoặc mạnh nhất của lưới (Gaussian σ 0.5, NLM h 5, bilateral sigma_color 30, unsharp amount 2.0), nên tối ưu có thể nằm ngoài lưới. Version 3 thêm mức nhẹ hơn cho khử nhiễu và mức mạnh hơn cho unsharp, vẫn chỉ chọn trên validation. Bảng dưới là version 2.

### Kết quả

Recovery là macro F1 sau xử lý trừ macro F1 khi suy giảm, trung bình trên ba mức, với cấu hình đã chọn cho từng họ. Dương là xử lý giúp, âm là làm tệ hơn. Chỉ gồm cặp kỹ thuật và họ phù hợp.

| Họ | F1 không xử lý (TB 3 mức) | Gaussian | Median | Bilateral | NLM | Unsharp |
|---|---:|---:|---:|---:|---:|---:|
| `noise` | 0.809 | +0.001 | −0.067 | **+0.027** | +0.001 | |
| `sensor_noise` | 0.841 | −0.002 | −0.088 | **+0.033** | 0.000 | |
| `lowlight` | 0.789 | +0.005 | −0.133 | **+0.022** | −0.005 | |
| `blur` | 0.741 | | | | | **+0.087** |
| `motion_blur` | 0.770 | | | | | **+0.154** |
| `lowres` | 0.699 | | | | | **+0.235** |

**Làm nét giúp rõ rệt, khử nhiễu hầu như không giúp.**

Unsharp, F1 trước và sau xử lý theo mức:

| Họ (cấu hình) | Mức 1 | Mức 2 | Mức 3 |
|---|---|---|---|
| `blur` (σ 2.0, amount 1.0) | 0.948 → 0.951 | 0.698 → 0.909 | 0.577 → 0.624 |
| `motion_blur` (σ 1.0, amount 2.0) | 0.909 → 0.945 | 0.768 → 0.955 | 0.635 → 0.873 |
| `lowres` (σ 1.0, amount 3.0) | 0.768 → 0.963 | 0.692 → 0.955 | 0.636 → 0.883 |

- `lowres` ở mức 1 và 2 gần như về F1 sạch. `motion_blur` và `blur` ở mức vừa phục hồi mạnh (normalized recovery 0,89 và 0,75 ở mức 2).
- Mờ rất nhẹ (`blur` mức 1) không đổi vì ảnh chưa hỏng nhiều, mờ nặng (`blur` mức 3) chỉ phục hồi một phần (+0.047).

Bilateral, cấu hình tốt nhất khử nhiễu:

| Họ (cấu hình) | Mức 1 | Mức 2 | Mức 3 |
|---|---|---|---|
| `noise` (d 7, sigma_color 30) | 0.957 → 0.950 | 0.849 → 0.931 | 0.622 → 0.629 |
| `sensor_noise` (d 7, sigma_color 30) | 0.926 → 0.956 | 0.848 → 0.911 | 0.750 → 0.755 |
| `lowlight` (d 7, sigma_color 10) | 0.904 → 0.913 | 0.826 → 0.861 | 0.637 → 0.660 |

- Bilateral chỉ có lợi ở nhiễu mức vừa. Nhiễu nhẹ không đổi, nhiễu nặng (mức 3) không kỹ thuật nào phục hồi được (recovery từ +0.005 đến +0.007).
- Với Gaussian và NLM, cấu hình tốt nhất phần lớn là mức nhẹ nhất của lưới (Gaussian σ 0.3 cho `noise` và `sensor_noise`, NLM h 1.5 cho `noise` và `lowlight`), tức gần như không lọc: recovery khoảng 0 và F1 sạch gần như không đổi. Nói cách khác, làm mịn nhiều hơn chỉ làm hại. Median luôn hại ở mọi họ (−0.04 đến −0.13), kể cả với cửa sổ nhỏ nhất (3×3).

### Chi phí trên ảnh sạch

Thay đổi macro F1 sạch (gốc 0.9775) và lớp bị hại nhiều nhất, với cấu hình đã chọn cho từng họ:

| Kỹ thuật, cấu hình | F1 sạch sau xử lý | Thay đổi | Lớp giảm recall nhiều nhất |
|---|---:|---:|---|
| Gaussian σ 0.3 (cho `noise`, `sensor_noise`) | 0.9775 | 0.000 | |
| Gaussian σ 0.5 (cho `lowlight`) | 0.9499 | −0.028 | ALLOCARIDARA_ATTACK −0.10 |
| NLM h 1.5 (cho `noise`, `lowlight`) | 0.9771 | −0.0003 | PHOMOPSIS_LEAF_SPOT −0.02 |
| NLM h 3 (cho `sensor_noise`) | 0.9288 | −0.049 | PHOMOPSIS_LEAF_SPOT −0.13 |
| Bilateral sigma_color 10 | 0.9747 | −0.003 | PHOMOPSIS_LEAF_SPOT −0.03 |
| Bilateral sigma_color 30 | 0.9427 | −0.035 | ALLOCARIDARA_ATTACK −0.13 |
| Median 3×3 | 0.8600 | −0.117 | ALLOCARIDARA_ATTACK −0.47 |
| Unsharp cho `blur` (σ 2.0, amount 1.0) | 0.9120 | −0.066 | ALGAL_LEAF_SPOT −0.19 |
| Unsharp cho `motion_blur` (σ 1.0, amount 2.0) | 0.8418 | −0.136 | PHOMOPSIS_LEAF_SPOT −0.33 |
| Unsharp cho `lowres` (σ 1.0, amount 3.0) | 0.7429 | −0.235 | PHOMOPSIS_LEAF_SPOT −0.51 |

- Kỹ thuật phục hồi tốt nhất, unsharp mạnh, cũng làm hại ảnh sạch nhiều nhất. Áp cố định lên mọi ảnh thì mất nhiều hơn được. Hai lớp bệnh có đốm nhỏ (PHOMOPSIS_LEAF_SPOT, ALGAL_LEAF_SPOT) bị hại nặng nhất, khớp với dự đoán rằng làm nét mạnh xóa hoặc làm lệch đặc trưng nhỏ của vết bệnh.
- Kết quả này ủng hộ việc chọn xử lý theo từng ảnh, nhưng chưa phải bằng chứng: bộ chọn thích ứng được đánh giá ở các task sau.

### Đối chứng

Cấu hình chung của mỗi kỹ thuật chạy trên các họ không phù hợp. Recovery trung bình:

| Kỹ thuật | Họ không phù hợp | Recovery |
|---|---|---:|
| Unsharp | `noise`, `sensor_noise`, `lowlight` | −0.53, −0.61, −0.59 |
| Unsharp | `jpeg` | −0.32 |
| Median | `blur`, `motion_blur`, `lowres`, `jpeg` | −0.06, −0.10, −0.04, −0.04 |
| NLM | `blur`, `motion_blur`, `lowres`, `jpeg` | −0.01, −0.02, −0.01, −0.01 |
| Bilateral | `blur`, `motion_blur`, `lowres`, `jpeg` | −0.02, −0.05, −0.03, +0.01 |

- Làm nét trên ảnh nhiễu khuếch đại nhiễu và làm hại nặng, đúng với lý do làm nét chỉ nên áp cho họ mờ.
- JPEG: không kỹ thuật nào cho recovery đáng kể (bilateral +0.007, Gaussian 0.000). Giả thuyết "JPEG không có kỹ thuật phù hợp" được ủng hộ.

### Giới hạn

- Một seed, validation 444 ảnh. Chênh lệch khoảng 0.03 trở xuống chưa đủ để kết luận. Chưa có khoảng tin cậy hay kiểm định ghép cặp, task 4.5 làm trên test.
- Cấu hình chọn trên chính tập đo nên điểm hơi lạc quan. Cột "kỹ thuật tốt nhất mỗi điều kiện" trong `restoration_best_per_condition.csv` chọn trên cùng tập nên chỉ để quan sát, không phải kết luận.
- Một số cấu hình vẫn nằm ở biên lưới: unsharp cho `lowres` (amount 3.0 là mức mạnh nhất), bilateral cho `lowlight` (sigma_color 10 là mức nhẹ nhất). Mức 1 và 2 của `lowres` đã gần F1 sạch nên lợi ích thêm nhỏ. Bilateral cho `lowlight` chỉ +0.022, nhỏ hơn nhiều so với các kỹ thuật ánh sáng của task 4.3 (Gamma khoảng +0.08).
- Chi phí trên ảnh sạch đo trên tập validation, theo cấu hình đã chọn. Không có mặt nạ vết bệnh nên việc mất chi tiết vết bệnh chỉ đo gián tiếp qua recall từng lớp.
- Mỗi kỹ thuật tối ưu độc lập. Ghép hai bước (ví dụ khử nhiễu rồi làm sáng với `lowlight`, họ cả hai task cùng xử lý) chưa được đo.

### File

Notebook `06_restoration_benchmark.ipynb` sinh các file sau. Bản chốt chép vào thư mục này. Dự đoán từng ảnh và ảnh minh họa (khoảng 3,5 MB mỗi hình, nặng nên không đưa vào Git) ở lại trong output Kaggle của notebook.

| File | Nội dung |
|---|---|
| `restoration_selected.json` | cấu hình đã chọn cho từng kỹ thuật và họ, kèm cấu hình chung, chi phí ảnh sạch |
| `restoration_matrix.csv` | bản đồ recovery: kỹ thuật × điều kiện, kèm degradation, recovery, normalized recovery |
| `restoration_best_per_condition.csv` | kỹ thuật tốt nhất mỗi điều kiện (chỉ để quan sát) |
| `restoration_clean_cost.csv` | thay đổi macro F1 và recall từng lớp trên ảnh sạch |
| `restoration_*.png` | heatmap recovery và normalized recovery, bảng xếp hạng, kỹ thuật tốt nhất mỗi điều kiện, chi phí ảnh sạch, confusion matrix ca điển hình |
| `restoration_predictions.npz` | dự đoán từng ảnh của mọi lượt chạy (Kaggle) |
