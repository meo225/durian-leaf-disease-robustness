# DurianLSNet 2026

Nguyen-Tat, T. B. và Nong, D. T. Lightweight deep learning framework and real-world dataset for durian leaf disease detection in smart agriculture. Displays, 93, 103451.

DOI: https://doi.org/10.1016/j.displa.2026.103451

## Vai trò trong đồ án

Nguồn sát gap nhất trong nhóm bài sầu riêng. DurianLDD là ứng viên dataset chính. DurianLSNet là ứng viên baseline nhẹ nếu cài lại được. 93,82% không phải điểm độ bền.

## Bài toán

Xây hệ nhận diện đủ nhẹ cho mobile hoặc IoT, không chỉ tối ưu accuracy trên máy mạnh.

## Dữ liệu

DurianLDD, 4.437 ảnh field tại Việt Nam, chuyên gia nông nghiệp gán nhãn. Năm lớp: ALGAL_LEAF_SPOT 733, ALLOCARIDARA_ATTACK 913, HEALTHY_LEAF 976, LEAF_BLIGHT 937, PHOMOPSIS_LEAF_SPOT 878.

Split mô tả là ngẫu nhiên, 70% train, 10% validation, 20% test.

Ảnh cuối resize 224×224, lưu JPG. Thí nghiệm chính không dùng augmentation như xoay hay lật.

## Model và train

CNN nhẹ: depthwise separable convolution, bottleneck residual, squeeze-and-excitation, channel shuffle. Loss cross-entropy, Adam, learning rate 0,001, batch 32, tối đa 300 epoch. ModelCheckpoint theo validation accuracy. Không early stopping.

Metric: accuracy, precision, recall, F1, số tham số, kích thước model, thời gian inference.

## Kết quả

Accuracy 93,82%, F1 93,73%, precision 93,87%, recall 93,71%. Khoảng 1,238 triệu tham số, 4,76 MB. Inference khoảng 0,12 giây mỗi ảnh trên phần cứng tương đương mobile.

Ablation: bỏ DB block và SE block còn khoảng 87,87%. Đủ hai thành phần đạt 93,82%.

Confusion matrix và Grad-CAM. Một ca sai được mô tả là thiếu sáng làm dấu hiệu bệnh không rõ. Cũng có nhầm khi vùng tổn thương gần đặc trưng của lớp khác.

## Lấy và không lấy

Lấy dataset, cách báo accuracy, F1, precision, recall, số tham số, kích thước và latency. Lấy DurianLSNet làm baseline nhẹ nếu tái lập được.

Không coi 93,82% là điểm robust. Bài đánh giá trên split cùng dataset, chưa có benchmark brightness, contrast, blur, noise theo mức, chưa có đường degradation và chưa thử enhancement để xem nhận diện có hồi hay không. Việc không dùng augmentation không có nghĩa augmentation luôn có hại.
