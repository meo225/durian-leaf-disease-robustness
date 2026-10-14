# Classification of Durian Foliar Diseases 2025

Chatthaicharoen, P., Duangburong, S. và Phruksaphanrat, B. Engineering, Technology & Applied Science Research, 15(5), 27653–27659.

DOI: https://doi.org/10.48084/etasr.12461

## Vai trò trong đồ án

Cho thấy localization tổn thương có thể hữu ích khi nền gây nhiễu, và cho ví dụ preprocessing đã được dùng trên sầu riêng: xóa nền và bilateral sharpening. Không cô lập đóng góp của từng bước. Không so 99,61% với 98% hay 93,82%.

## Bài toán

So ResNet-9, Xception và Mask R-CNN trên bệnh lá sầu riêng field tại Thái Lan. Mask R-CNN còn localization và segmentation.

## Dữ liệu

648 ảnh gốc. 228 ảnh field tại Yala, Thái Lan. 420 ảnh mở từ Roboflow. Bốn nhóm: algal spot, blight, other diseased leaf, healthy.

Tiền xử lý: bỏ trùng, chuyên gia kiểm nhãn, chuẩn kích thước, xóa nền, bilateral sharpening, resize 400×400.

Train và validation ban đầu 540 ảnh được augmentation thành 10.800. Test ban đầu 108 ảnh được augmentation thành 2.160. Điểm này bắt buộc phải đọc cùng mọi con số của bài.

## Model

Xception dùng ImageNet transfer learning. Thử 60, 70, 80 epoch, Adam hoặc SGD, learning rate 0,01, 0,005, 0,001, batch 128.

Mask R-CNN dùng ResNet-101, FPN, RPN, RoI Align, pretrained COCO. Khác Xception ở chỗ có localization và segmentation tổn thương.

## Kết quả

ResNet-9: 83,50% accuracy. Xception: 89,42%. Mask R-CNN: 99,61% accuracy, precision 97,80%, recall 98,81%, F1 98,30%. Mask R-CNN train lâu hơn Xception.

## Lấy và không lấy

Có thể lấy Xception như baseline cổ điển nếu đủ compute. Ý tưởng localization và ảnh hưởng của nền hữu ích cho phân tích lỗi.

Không kết luận Mask R-CNN tốt nhất cho đồ án. Dữ liệu gốc nhỏ và augmentation rất mạnh, kể cả test theo mô tả bài. 99,61% không phải mục tiêu để đặt cho dataset của nhóm.
