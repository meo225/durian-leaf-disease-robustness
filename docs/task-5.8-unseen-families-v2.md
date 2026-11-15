# Task 5.8. Bộ họ chưa thấy v2 (sửa 5.5)

Nhánh `unseen-families-v2-5.5-5.7-5.8`. Cấu hình `configs/unseen_families_v2.yaml`, notebook `13_unseen_v2_calibration.ipynb` và `14_msd_family_distance.ipynb` (xong phần khóa), mã ở `src/corruption/external.py`, `family_lock.py`, `family_roles.py` và `src/evaluation/difficulty.py`, `unseen_families_v2.py`.

## Vì sao làm lại

Bộ v1 (task 5.5, `docs/task-5.5-unseen-families.md`) có năm điểm chưa khách quan:

1. Mức do nhóm chọn tay và hiệu chỉnh bằng chính Swin-Tiny, nên so hai họ là so hai độ khó khác nhau (mất 0.809 so với 0.774, nhóm gần 0.732 so với nhóm xa 0.847).
2. Sáu họ do chính nhóm cài, cùng codebase với augmentation. `fog` và `spatter` dùng trường ngẫu nhiên giả.
3. Nhóm gần/xa chia theo tên. `saturate` trùng phép `Color` của TrivialAugment, `fog` gần `contrast_low`.
4. Chưa đo bộ chọn và mô hình huấn luyện lại trên bộ này (đó là việc của 5.11).
5. Recall từng lớp dựa trên ít ảnh, không có khoảng tin cậy.

v1 **giữ nguyên** làm tham khảo, không xóa và không sửa. Bộ v2 là bộ đánh giá chính từ đây.

## Thiết kế

| Vấn đề | Cách xử lý | Chỗ trong mã |
|---|---|---|
| Họ do nhóm tự cài | Chỉ lấy từ ImageNet-C (`imagecorruptions` 1.1.2) và ImageNet-C-bar (vendored, MIT, khớp mã gốc từng byte). Họ vai trò test không bao giờ là họ nhóm cài | `external.py`, `third_party/imagenet_c_bar`, `family_roles.py` |
| Độ khó khác nhau giữa họ | Ba mức mỗi họ được khớp với các phân vị 0.2, 0.5, 0.8 của ΔF1 mô hình thay thế (ConvNeXt-Tiny) trên 42 điều kiện đã thấy, dung sai ±0.05. Không khớp thì bỏ họ, không nới dung sai | `difficulty.py` |
| Mức chọn bằng chính mô hình được đánh giá | Mô hình thay thế khác mô hình tham chiếu. Độ khó thực của Swin-Tiny báo riêng, không dùng để chọn | notebook 13, mục 7 |
| Chọn họ theo cảm tính | Quy tắc đăng ký trước ở `registration`: nguồn, lưới mức, ngưỡng thời gian, phân vị, dung sai | `unseen_families_v2.py` |
| Nhóm gần/xa chia theo tên | MSD của Swin-Tiny tới augmentation suy giảm, ngưỡng là MSD leave-one-out lớn nhất của 14 họ đã thấy. Họ có khoảng bootstrap chứa ngưỡng là `ambiguous` | `msd.py` (task 5.7) |
| Dùng họ làm cả train, chọn thiết kế và test | Chia train 0.4, val 0.2, test 0.4, phân tầng theo nhóm MSD, thứ tự theo sha256(seed:tên) | `family_roles.py` |
| Đổi cấu hình sau khi xem kết quả | `lock.sha256` phủ protocol, version, seed, registration và toàn bộ họ. Sửa bất kỳ phần nào thì `load_conditions` báo lỗi. Mở test cần thêm `FINAL_FAMILY_EVAL=1` | `family_lock.py` |
| Recall từng lớp ít ảnh | Khoảng Wilson 95% (`per_class_recall_wilson`). ALLOCARIDARA_ATTACK 46 ảnh, recall 0.26 có khoảng 0.16 đến 0.40 | `intervals.py` |

### Danh sách ứng viên

Ba quy tắc chọn ứng viên, ghi ở `registration.candidates`, quyết định theo môi trường và chi phí tính toán, không theo kết quả:

- `probe_families` chạy thử từng họ trên ảnh 224. Họ lỗi bị loại và ghi lý do. Ở môi trường thử, `fog`, `glass_blur`, `gaussian_blur` của `imagecorruptions` 1.1.2 lỗi với numpy 2 và scikit-image mới. Không vá, vì vá là đổi định nghĩa họ.
- Họ chậm hơn 0.5 giây mỗi ảnh bị loại. Đánh giá cuối chạy mọi họ trên 888 ảnh, ba mức, nhiều hệ thống. Trên máy thử, `sparkles` (2.7 giây) và `caustic_refraction` (3.1 giây) rơi vào đây, `plasma_noise` và `perlin_noise` (khoảng 1 giây) cũng vậy. Quyết định cuối lấy theo lần chạy trên Kaggle.
- Lưới mức: ImageNet-C là 1 đến 5, ImageNet-C-bar là 1 đến 10 bước 1 (bản gốc chọn năm mức rời rạc từ thang liên tục này). Bước 1 thay vì 0.5 để hiệu chỉnh không quá vài giờ.

### Điều kiện đặt tên

`<họ>_s<mức>` như các config khác, họ có tiền tố nguồn (`inc_`, `cbar_`) để không trùng `defocus_blur`, `spatter`, `saturate`, `fog` của v1. Điều kiện ứng viên khi hiệu chỉnh tên `<họ>_c<mức>`.

## Cách chạy

1. Chạy `13_unseen_v2_calibration.ipynb` trên Kaggle (GPU, Internet On, 4 CPU). Tải zip về, giải vào repo, commit `reports/unseen_v2/` (cần `calibration.json`).
2. Chạy `14_msd_family_distance.ipynb`. Notebook đo MSD, chia nhóm và vai trò rồi ghi `configs/unseen_families_v2.yaml` với `locked: true`. Tải zip về, giải vào repo, xem `git diff`, commit.
3. Từ lúc khóa, không chỉnh. Cả hai notebook từ chối chạy lại khi config đã khóa. Muốn đổi quy tắc thì đăng ký bộ mới (version 3).

Kiểm tra tự động khi chạy xong: số điều kiện bằng 3 lần số họ, mở val không cần biến môi trường, mở test bị chặn khi thiếu `FINAL_FAMILY_EVAL=1`.

## Giới hạn

- Mức khớp trên **444 ảnh validation**, mỗi ΔF1 có sai số lấy mẫu khoảng ±0.02. Dung sai 0.05 đã tính điều này, nhưng ba mức của một họ vẫn có thể chồng lấn nhẹ về độ khó.
- Cùng 444 ảnh validation dùng cho khớp mức, MSD, chọn checkpoint ở 5.9 và phân tích thăm dò. Đây là tái sử dụng validation, giảm bớt vì mức khớp bằng mô hình thay thế còn MSD dùng 100 ảnh chọn theo seed cố định, nhưng không triệt tiêu.
- ImageNet-C và ImageNet-C-bar thiết kế cho ImageNet. Với ảnh lá 224×224 và 5 lớp, một số họ có thể không giống suy giảm thật ngoài vườn. Ảnh mẫu ở notebook 13 mục 9 là chỗ kiểm tra bằng mắt. Nhận xét ghi vào báo cáo, không đổi mức.
- ImageNet-C đã gồm họ cùng cơ chế với vài họ đã thấy (nhiễu, mờ, độ sáng, tương phản, jpeg). MSD sẽ xếp chúng vào nhóm gần, và đó là mục đích: nhóm gần/xa do đo.
- Không giải quyết được ở nhánh này: tập test đã mở 6 lần trước đây (`manifests/SPLIT_PROTOCOL.md`, mục 3 điểm 4), và họ ở vai trò test của v2 chỉ sạch từ lúc khóa.
- Chưa có số liệu: nhánh này chứa mã, test và quy tắc đăng ký trước. Số liệu thật cần chạy notebook 13 và 14 trên Kaggle (GPU, dataset, checkpoint ConvNeXt-Tiny và Swin-Tiny seed 42).
