# Quantifying the reliability gap 2026

DOI: https://doi.org/10.3389/fpls.2026.1826962

## Vai trò trong đồ án

Đây là nguồn chính cho cách đánh giá. Đồ án chuyển logic controlled-to-field của bài thành chuỗi clean, corrupted, processed trên cùng một mẫu.

## Vấn đề

Bài không nhằm thêm một model accuracy cao. Câu hỏi là khi rời dataset sạch sang ảnh field, model còn đáng tin đến mức nào. Reliability gồm cả việc biết khi nào mình không chắc, không chỉ đoán đúng.

## Thiết kế

Train trên PlantVillage, đánh giá trên PlantDoc. Ngoài accuracy và macro F1, bài đo calibration, selective prediction và phát hiện mẫu unknown.

Calibration là mức confidence có phản ánh đúng xác suất đúng hay không. Selective prediction cho phép từ chối ca khó thay vì buộc trả lời hết.

Bài đo 17 đặc trưng ảnh mức thấp về màu, độ sáng, edge, texture và foreground occupancy, tức phần vật thể chính chiếm bao nhiêu diện tích ảnh.

## Kết quả cần nhớ

Fine-tuned ResNet-50 đạt 99,73% accuracy trên PlantVillage và 32,05% trên PlantDoc. Mức giảm 67,7 điểm phần trăm.

Mean confidence trên PlantDoc vẫn khoảng 79,76%. Model sai nhiều nhưng vẫn thường nghĩ mình đúng.

Grad-CAM cho attention chuyển từ vùng tổn thương trên PlantVillage sang nền trên PlantDoc.

AdaBN và feature moment matching giúp một phần. DANN-style adaptation làm accuracy giảm. Temperature scaling cải thiện calibration nhưng không sửa domain shift.

## Insight

Accuracy sạch cao không chứng minh dùng được ngoài thực địa.

Confidence cao không có nghĩa dự đoán đáng tin khi dữ liệu bị shift.

Thống kê ảnh giúp mô tả khác biệt giữa ảnh sạch và ảnh suy giảm trước khi nhìn điểm phân loại.

Grad-CAM kiểm tra model nhìn tổn thương hay nền.

Một biện pháp mitigation có thể không hiệu quả hoặc làm xấu hơn. Preprocessing không được mặc định là luôn tốt.

## Cách dùng

Giữ test sạch, tránh leakage, dùng nhiều metric, phân tích confidence khi có thời gian, dùng Grad-CAM, và báo kết quả âm nếu một cách xử lý ảnh làm điểm giảm.
