# Lượt hiệu chỉnh đầu (quy tắc v2.0), giữ để truy vết

Notebook 13, ConvNeXt-Tiny, 444 ảnh validation, lưới ImageNet-C 1 đến 5, seed theo tên điều kiện. Kết quả: **giữ 2 trên 20 họ** (`inc_brightness`, `cbar_checkerboard_cutout`), 18 họ bị bỏ vì không có mức nào trong ±0.05 của mục tiêu ΔF1 (0.036, 0.129, 0.330). Không dùng để khóa bộ họ.

Lý do sửa quy tắc và nội dung bản sửa v2.1: `docs/task-5.8-unseen-families-v2.md`, mục "Bản sửa v2.1", và `registration.amendments` trong `configs/unseen_families_v2.yaml`. Số trong `candidates_surrogate.json` không dùng lại cho v2.1 vì seed ảnh suy giảm đã đổi.
