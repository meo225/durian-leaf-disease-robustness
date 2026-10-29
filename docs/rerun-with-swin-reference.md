# Chạy lại Phase 4 với Swin-Tiny làm reference model

Nhóm đã đổi reference từ ConvNeXt-Tiny sang Swin-Tiny (xem `reports/baselines/README.md`, mục "Chạy lặp seed và chọn reference"). Mọi kết quả Phase 4 hiện có dùng ConvNeXt-Tiny seed 42, nên cần chạy lại. File này ghi những gì đã chuẩn bị và thứ tự chạy.

## Đã chuẩn bị trong repo

- `reports/baselines/reference_final.json`: ghi reference là `swin_tiny`. `reference_selection.json` giữ nguyên làm hồ sơ seed 42.
- `src/evaluation/robustness.py` đọc reference từ `reference_final.json`.
- `configs/illumination.yaml`, `restoration.yaml`, `sensitivity.yaml` trỏ tới `swin_tiny_seed42.pt`.
- `configs/train_swin_tiny_aug.yaml` (mới): đối chứng augmentation cho Swin-Tiny, cùng giao thức với `train_convnext_tiny_aug.yaml` (giữ lại làm hồ sơ).
- Notebook 04, 05, 06 (độ nhạy), 06 (phục hồi) và 07 đọc `reference_final.json`, assert `swin_tiny` và nạp `configs/train_swin_tiny*.yaml`. Output cũ trong các notebook này vẫn là của ConvNeXt-Tiny cho đến khi chạy lại.

## Thứ tự chạy trên Kaggle

1. Đưa `swin_tiny_seed42.pt` vào dataset checkpoint (hiện dataset chỉ chắc chắn có checkpoint ConvNeXt-Tiny). Checkpoint không vào Git.
2. `04_corruption_benchmark`: đo lại Swin-Tiny trên validation với các mức đã khóa trong `configs/corruption.yaml`.
3. `06_degradation_sensitivity` (task 4.2).
4. `05_illumination_benchmark` (task 4.3).
5. `06_restoration_benchmark` (task 4.4).
6. `07_augmentation_control` (task 4.5): train lại Swin-Tiny với augmentation, khoảng 14 phút ở baseline và lâu hơn với 30 epoch đầy đủ.
7. Chép kết quả vào `reports/corruptions`, `sensitivity`, `processing`, `augmentation`, rồi cập nhật README từng thư mục.

## Quyết định cần nhóm chốt trước khi chạy

- **Các mức suy giảm của 4.1.** Các mức trong `configs/corruption.yaml` được chọn bằng cách đo ConvNeXt-Tiny trên validation. Đề xuất giữ nguyên mức đã khóa, vì chúng là thuộc tính của benchmark chứ không của model; chỉ ghi chú rằng chúng được hiệu chỉnh bằng ConvNeXt-Tiny. Đổi mức sau khi xem kết quả Swin-Tiny sẽ làm benchmark thay đổi theo model.
- **Tập test.** Các bước 4.2, 4.3, 4.4 và 4.5 từng được chấm trên 888 ảnh test với ConvNeXt-Tiny. Chạy lại với Swin-Tiny là lần mở test thứ hai. Cần ghi rõ trong báo cáo, và giữ nguyên tham số đã chọn trên validation, không chỉnh theo test.
- **Kết quả ConvNeXt-Tiny cũ.** Chúng sẽ bị ghi đè trong `reports/`. Nếu cần giữ để đối chiếu, chép sang thư mục riêng (ví dụ `reports/<mục>/convnext_reference/`) trước khi chạy; nếu không, vẫn lấy lại được từ lịch sử Git.

## Việc cập nhật tài liệu sau khi chạy

`README.md` gốc, `reports/corruptions/README.md`, `reports/sensitivity/README.md`, `reports/processing/README.md` và `reports/augmentation/README.md` hiện còn ghi ConvNeXt-Tiny là reference. Cập nhật sau khi có số liệu Swin-Tiny, không sửa trước.

## Notebook 08 (bản đồ phục hồi, task 4.5)

Đã chuẩn bị: `configs/recovery_map.yaml` (checkpoint, `train_config` và tiền tố `retrained__swin_tiny_aug`), `src/evaluation/recovery_report.py` (`reference_model` lấy từ `train_config` thay vì gắn cứng `convnext_tiny`), và notebook `08_recovery_map.ipynb` (đọc `reference_final.json`, assert `swin_tiny`).

Notebook này đọc đầu ra của 4.2 đến 4.5, nên chỉ chạy được sau bước 7 ở trên, và là bước chạy cuối. Sau khi chạy, thay `reports/recovery_map/` (README, CSV, ảnh) và ba ảnh `reports/processing/restoration_*recovery*.png`, vì kết quả hiện là của ConvNeXt-Tiny. Dữ liệu giả trong `tests/test_recovery_report.py` dùng tiền tố cũ nhưng độc lập với cấu hình nên không cần sửa.
