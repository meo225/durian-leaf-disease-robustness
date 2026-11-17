# Kết quả bộ họ chưa thấy v2, task 5.7 và 5.8

Bộ này thay bộ v1 làm bộ đánh giá chính. Thiết kế ở `docs/task-5.8-unseen-families-v2.md` và `docs/task-5.7-msd-family-distance.md`, cấu hình khóa ở `configs/unseen_families_v2.yaml`.

Mọi số liệu đo trên validation, không dùng test. Cấu hình đã khóa bằng sha256 `fe72d34a4c599f62`, mở test cần `FINAL_FAMILY_EVAL=1` và một dòng trong `manifests/SPLIT_PROTOCOL.md` mục 3.4. Chưa có kết quả hệ thống nào, nên chưa có kết luận về bộ chọn hay mô hình huấn luyện lại.

| Bước | Notebook | File |
|---|---|---|
| Khớp độ khó | `notebooks/13_unseen_v2_calibration.ipynb` | `calibration.json`, `candidates_surrogate.json`, `realised_difficulty.csv`, `severity_examples_*.png` |
| MSD, nhóm, vai trò, khóa | `notebooks/14_msd_family_distance.ipynb` | `msd.json`, `msd_*.csv`, `roles.csv`, `seen_pairwise_distance.*`, `msd_to_degradation_aug.png` |
| Bảng tóm tắt thêm | `python -m src.evaluation.v2_summary` | `msd_threshold_sensitivity.csv`, `difficulty_by_group.csv`, `role_counts.csv` |

File lượt hiệu chỉnh đầu nằm ở `pass1_v2.0/` để truy vết.

## Kết luận ngắn

1. Còn 9 trên 20 họ đủ điều kiện. Chín họ này có 3 mức tăng dần, mỗi mức lệch mục tiêu ΔF1 không quá 0.05 trên mô hình thay thế.
2. Nhóm theo MSD là 4 gần, 3 xa, 2 chưa rõ. Vai trò chia theo quy tắc đăng ký trước, kết quả 3 train, 2 val, 4 test.
3. Độ khó trên Swin-Tiny vẫn lệch theo nhóm. Ở mức 3 nhóm gần chỉ mất 0.179 điểm F1, nhóm xa mất 0.302, trong khi mục tiêu là 0.33. Mức được khớp bằng ConvNeXt-Tiny nên đúng quy tắc, nhưng so trực tiếp gần với xa trên Swin vẫn là so hai độ khó khác nhau.
4. Tập test chỉ có một họ xa là snow. Không thể kiểm định gần so với xa trên test.
5. Nhãn gần xa phụ thuộc vào một họ ngoại lai trong ngưỡng. Năm trên chín họ đổi nhãn nếu đổi quy tắc ngưỡng.
6. Nhãn gần xa theo tên ở bộ v1 chỉ khớp MSD ở 1 trên 6 họ, nên chênh lệch gần xa của v1 không dùng làm bằng chứng.

## Chín họ được giữ

Ba mức là giá trị tham số, các họ ImageNet-C dưới 1 là trộn alpha với mức 1 của ImageNet-C theo bản sửa v2.1. MSD là khoảng cách tới augmentation suy giảm, kèm khoảng tin cậy bootstrap 95 phần trăm.

| Họ | Nhóm | Vai trò | Ba mức | MSD | Khoảng tin cậy |
|---|---|---|---|---|---|
| inc_defocus_blur | gần | val | 0.2, 0.5, 0.7 | 0.78 | 0.64 đến 0.88 |
| inc_motion_blur | gần | test | 0.35, 0.7, 2 | 1.29 | 1.11 đến 1.49 |
| inc_impulse_noise | gần | test | 0.35, 0.5, 0.7 | 1.40 | 1.24 đến 1.51 |
| inc_zoom_blur | gần | train | 0.35, 0.7, 3 | 2.44 | 2.11 đến 2.66 |
| inc_brightness | chưa rõ | test | 2, 3, 4 | 2.67 | 2.38 đến 3.18 |
| inc_frost | chưa rõ | train | 0.7, 1, 3 | 3.07 | 2.84 đến 3.46 |
| cbar_checkerboard_cutout | xa | val | 1, 5, 9 | 3.56 | 3.38 đến 3.97 |
| inc_snow | xa | test | 0.35, 0.7, 3 | 3.77 | 3.59 đến 4.15 |
| cbar_cocentric_sine_waves | xa | train | 2, 4, 6 | 3.82 | 3.55 đến 4.20 |

Ngưỡng gần xa là 3.14, bằng MSD leave-one-out lớn nhất của 14 họ đã thấy. Biểu đồ `msd_to_degradation_aug.png` vẽ cả 14 họ đã thấy, 9 họ mới và ngưỡng.

Hai họ blur trong vai trò test, motion_blur và impulse_noise, gần với họ đã thấy cùng cơ chế nên không phải họ lạ theo nghĩa mạnh. Đây là hệ quả của việc gọi nhóm gần theo số đo.

## Họ bị loại

Có 29 họ ứng viên. Chín họ bị loại ở bước chạy thử, còn 20 họ vào khớp độ khó, trong đó 11 họ bị loại vì độ khó.

Bị loại vì độ khó. Không mức nào nằm trong 0.05 quanh mục tiêu. Chi tiết ở `calibration.json`, mục `dropped_by_difficulty`.

| Họ | Mục tiêu không với tới | Độ lệch gần nhất |
|---|---|---|
| inc_contrast | 0.330 | 0.076 |
| inc_gaussian_noise | 0.330 | 0.072 |
| inc_shot_noise | 0.330 | 0.075 |
| inc_speckle_noise | 0.330 | 0.097 |
| inc_jpeg_compression | 0.330 | 0.063 |
| inc_pixelate | 0.330 | 0.061 |
| inc_saturate | 0.330 | 0.100 |
| inc_spatter | 0.330 | 0.105 |
| inc_elastic_transform | 0.330 | 0.189 |
| cbar_blue_noise_sample | 0.330 | 0.085 |
| cbar_single_frequency_greyscale | 0.036 | 0.121 |

Mười họ bị loại vì mục tiêu nặng 0.33 nằm giữa hai mức liên tiếp của lưới. Ví dụ gaussian_noise nhảy từ 0.23 lên 0.40 và contrast từ 0.10 lên 0.41. Riêng single_frequency_greyscale quá nặng ngay từ mức 1, ΔF1 0.16 so với mục tiêu nhẹ 0.036. Ba họ nhiễu gaussian, shot và speckle cùng bị loại, nên bộ v2 chỉ còn impulse_noise làm họ nhiễu.

Bị loại ở bước chạy thử. Sáu họ ImageNet-C-bar chậm hơn 0.5 giây mỗi ảnh gồm brownish_noise, caustic_refraction, inverse_sparkles, perlin_noise, plasma_noise và sparkles. Ba họ ImageNet-C không chạy được trong môi trường Kaggle hiện tại là fog do `np.float_` bị bỏ ở numpy 2, gaussian_blur và glass_blur do `multichannel` bị bỏ ở scikit-image 0.25. Ba họ này mất vì lỗi tương thích thư viện, không phải vì độ khó. Sửa chúng sẽ là một bản sửa quy tắc nữa sau khi đã thấy kết quả, nên không làm và ghi lại ở đây.

## Độ khó thực trên Swin-Tiny theo nhóm

Mức được khớp bằng ConvNeXt-Tiny. Swin-Tiny là reference nên độ khó thực của nó được báo riêng và không dùng để chọn mức. Bảng dưới là trung bình ΔF1 theo nhóm và mức, nguồn `difficulty_by_group.csv`.

| Nhóm | Mức | Số họ | Mục tiêu | ConvNeXt-Tiny | Swin-Tiny | Sai lệch của Swin |
|---|---|---|---|---|---|---|
| gần | 1 | 4 | 0.036 | 0.036 | 0.021 | 0.016 |
| gần | 2 | 4 | 0.129 | 0.132 | 0.065 | 0.064 |
| gần | 3 | 4 | 0.330 | 0.332 | 0.179 | 0.151 |
| chưa rõ | 3 | 2 | 0.330 | 0.320 | 0.290 | 0.055 |
| xa | 1 | 3 | 0.036 | 0.036 | 0.040 | 0.007 |
| xa | 2 | 3 | 0.129 | 0.143 | 0.133 | 0.031 |
| xa | 3 | 3 | 0.330 | 0.326 | 0.302 | 0.033 |

Các họ nhóm gần đều là mờ hoặc nhiễu xung, và Swin chịu loại này tốt hơn ConvNeXt. Ví dụ impulse_noise mức 3 chỉ mất 0.076 điểm trên Swin, và mức 2 mất 0.034 thấp hơn mức 1 là 0.038. Đây là đúng cái nhiễu gộp giữa nhóm và độ khó mà bản v1 gặp, giờ chỉ còn trên reference chứ không còn trên mô hình thay thế. Cách xử lý ở phần kế hoạch phân tích.

## Độ nhạy của nhóm với ngưỡng

Ngưỡng đăng ký trước là MSD leave-one-out lớn nhất. Giá trị này là của jpeg, 3.14. Mười ba họ đã thấy còn lại đều không quá 2.69. Bảng dưới đổi ngưỡng sang giá trị thứ hai lớn nhất, 2.69, và trung vị, 1.47. Đây là phân tích thăm dò, nhãn đã khóa không đổi. Nguồn `msd_threshold_sensitivity.csv`.

| Họ | MSD | Nhãn đã khóa | Ngưỡng thứ hai | Ngưỡng trung vị |
|---|---|---|---|---|
| inc_defocus_blur | 0.78 | gần | gần | gần |
| inc_motion_blur | 1.29 | gần | gần | chưa rõ |
| inc_impulse_noise | 1.40 | gần | gần | chưa rõ |
| inc_zoom_blur | 2.44 | gần | gần | xa |
| inc_brightness | 2.67 | chưa rõ | chưa rõ | xa |
| inc_frost | 3.07 | chưa rõ | xa | xa |
| cbar_checkerboard_cutout | 3.56 | xa | xa | xa |
| inc_snow | 3.77 | xa | xa | xa |
| cbar_cocentric_sine_waves | 3.82 | xa | xa | xa |

Ba họ xa chỉ nằm trên ngưỡng từ 0.4 đến 0.7. Tài liệu 5.7 đã nêu MSD tính từ 1000 mẫu thiên về giá trị lớn hơn bản 100 000 mẫu của bài báo, nên chưa thể bảo đảm nhãn giữ nguyên khi tăng mẫu.

## Nhãn theo tên ở bộ v1 so với MSD

Nguồn `msd_v1_families.csv`.

| Họ v1 | Theo tên | Theo MSD | MSD |
|---|---|---|---|
| defocus_blur | gần | gần | 0.34 |
| chroma_noise | gần | xa | 3.61 |
| fog | xa | gần | 1.41 |
| spatter | xa | gần | 1.53 |
| rain_streak | xa | chưa rõ | 3.29 |
| saturate | xa | chưa rõ | 3.42 |

Nhóm gần 0.732 so với nhóm xa 0.847 ở bản v1 dựa trên nhãn này, nên không dùng làm bằng chứng về gần xa.

## Trùng lặp với TrivialAugment

Nguồn `msd_seen_to_trivialaugment.csv` và `ta_overlap` trong `msd.json`. Họ đã thấy gần TrivialAugment nhất là contrast_low 0.28, brightness_bright 0.35, brightness_dark 0.57, contrast_high 0.65, đều ứng với phép tương ứng của TrivialAugment. Các họ còn lại xa hơn nhiều, từ 1.5 đến 3.4. Xác suất một lần lấy mẫu TrivialAugment cho hệ số trong khoảng sai số của một mức chỉ từ 0 đến 0.012. Năm trên 15 mức của brightness, contrast và saturate nằm ngoài khoảng hệ số của TrivialAugment.

Với họ mới, khoảng tin cậy MSD tới TrivialAugment và tới augmentation suy giảm chồng nhau ở 7 trên 9 họ. Hai họ còn lại là motion_blur và impulse_noise, xa TrivialAugment hơn xa augmentation suy giảm. Chi tiết trong `msd_new_families.csv`.

## Kế hoạch phân tích, đăng ký trước khi mở test

Phần này viết sau khi thấy bảng độ khó trên validation và trước khi có bất kỳ kết quả hệ thống nào. Mọi phân tích khác sẽ gọi là thăm dò.

1. Báo từng họ riêng, kèm ΔF1 và khoảng tin cậy bootstrap theo ảnh. Không báo một con số gộp thay cho bảng từng họ.
2. Không kiểm định gộp gần so với xa trên test, vì test chỉ có một họ xa. So sánh gần xa chỉ mô tả, và nhóm chưa rõ không tham gia.
3. Biến giải thích chính là MSD liên tục chứ không phải nhãn nhị phân. Spearman giữa MSD và kết quả hệ thống báo kèm khoảng bootstrap, ghi rõ chỉ có 9 họ nên khoảng rất rộng.
4. Cạnh mọi kết quả báo ΔF1 thực của Swin-Tiny không xử lý ở cùng họ và mức. So giữa các họ dùng phần ΔF1 lấy lại được chia cho ΔF1 của Swin-Tiny, và báo cả số gốc.
5. Nhãn gần xa là nhãn đã khóa. Bảng độ nhạy ngưỡng đi kèm khi báo kết quả theo nhóm.
6. Bộ v2 mở test một lần, sau khi các nhánh 5.9, 5.10 và 5.11 đã chốt cấu hình, và ghi một dòng vào nhật ký `manifests/SPLIT_PROTOCOL.md` mục 3.4.
7. Họ vai trò train và val chỉ dùng để chọn hay hiệu chỉnh, không báo như kết quả cuối.

## Giới hạn

- Mức khớp trên 444 ảnh validation, mỗi ΔF1 có sai số lấy mẫu khoảng 0.02. Dung sai 0.05 đã tính điều này.
- MSD dùng 100 ảnh validation, 1000 mẫu và bootstrap 200 lần. Cùng tập validation còn dùng cho khớp mức và chọn checkpoint ở 5.9, nên có tái sử dụng validation.
- Mức nhẹ của ImageNet-C là trộn alpha của nhóm, không phải mức gốc. Với họ nhòe như blur, ảnh mức nhẹ là ảnh gốc chồng ảnh mờ, khác mờ thật ở cường độ thấp.
- Chỉ có mô hình thay thế và reference, một seed, ảnh lá 224 và 5 lớp. Không khẳng định điều gì về ảnh chụp thật.
- Tập test đã mở nhiều lần ở các bước trước, ghi ở `manifests/SPLIT_PROTOCOL.md`. Họ của v2 sạch với test từ lúc khóa.
- Độ khó thực trên Swin lệch theo nhóm như đã nêu, và chỉ có một họ xa ở test.

## Các bản sửa đã công khai

Cả hai bản sửa ghi trong `registration.amendments` của config, nằm trong mã băm khóa.

- v2.1, ngày 15 tháng 11. Mở lưới mức nhẹ cho ImageNet-C và seed theo họ và giá trị. Quyết định sau khi thấy bảng ΔF1 ứng viên của lượt đầu, chưa thấy kết quả hệ thống nào, test chưa mở. Lượt đầu giữ 2 trên 20 họ.
- v2.1-fix, ngày 16 tháng 11. Seed cho impulse_noise vì skimage không seed nên điều kiện không tái lập. Quy tắc chọn không đổi.
