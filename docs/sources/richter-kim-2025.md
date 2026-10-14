# Richter và Kim 2025

DOI: https://doi.org/10.1038/s41598-025-03235-w

## Vai trò trong đồ án

Follow ở Phase 3 cho cách so baseline công bằng. Không lấy bảng xếp hạng của bài làm kết luận cho dataset sầu riêng. Không dùng bài này như contribution xử lý ảnh.

## Việc tác giả làm

23 model trên 18 dataset mở. Hai setting: transfer learning, và transfer learning kèm fine-tuning. Mỗi tổ hợp chạy 5 lần. Tổng 4140 model được train.

Transfer learning dùng kiến thức đã học từ dữ liệu trước cho dataset mới. Fine-tuning tiếp tục cập nhật một phần hoặc toàn bộ pretrained model bằng dataset mới.

## Insight

Hiệu năng phụ thuộc dataset. Kiến trúc tốt trên dataset này không tự thành tốt nhất trên dataset khác.

Chạy lặp lại quan trọng vì một seed có thể may hoặc không may.

Transfer learning là lựa chọn thực tế khi dataset bệnh cây không lớn.

Cần học cách dựng benchmark công bằng, không học tên model đứng đầu bảng.

## Cách dùng ở Phase 3

Benchmark vài kiến trúc đại diện trên cùng split và cùng metric. Chạy nhiều seed cho candidate quan trọng nếu đủ tài nguyên. Chọn reference model bằng hiệu năng, độ ổn định và chi phí tính toán.
