# AI-Driven Durian Leaf Disease Classification 2026

Klangbunrueang, R., Chansanam, W., Iam-On, N. và Boongoen, T. Applied Sciences, 16(9), 4062.

DOI: https://doi.org/10.3390/app16094062

## Vai trò trong đồ án

Nguồn mạnh nhất trong ba bài sầu riêng để thiết kế benchmark Phase 3: nhiều kiến trúc, một protocol, macro và per-class, mất cân bằng lớp. 98% của ConvNeXt không có nghĩa ConvNeXt chắc chắn là reference model trên dataset của đồ án.

## Bài toán

Benchmark tương đối công bằng giữa nhiều CNN, xử lý mất cân bằng lớp, và chứng minh triển khai web.

## Dữ liệu

6.119 ảnh, 6 lớp: Leaf_Healthy, Leaf_Colletotrichum, Leaf_Algal, Leaf_Phomopsis, Leaf_Blight, Leaf_Rhizoctonia. Gộp từ Durian Leaf Disease Dataset, Vietnamese durian leaf disease benchmark và ảnh field bổ sung. Ảnh field có biến đổi illumination, góc camera, khoảng cách lấy nét và nền. Ảnh bổ sung chụp smartphone dưới ánh sáng tự nhiên.

Split stratified: train 4.279, validation 739, test 1.101.

Healthy và Blight lớn hơn Colletotrichum và Rhizoctonia. Weighted cross-entropy tăng phạt lớp thiểu số. Đây là xử lý ở loss, không tăng số lượng hay độ đa dạng của mẫu thiểu số.

## Tiền xử lý và train

Resize theo input của từng kiến trúc, padding giữ tỉ lệ khung, không kéo méo. Chuẩn hóa ImageNet mean và std.

Augmentation lúc đọc: lật ngang, lật dọc, xoay ±30°, color jitter brightness ±0,2 và contrast ±0,2, random resized crop. Số ảnh vật lý vẫn 6.119.

Sáu model: ConvNeXt, ResNet, DenseNet201, InceptionV3, EfficientNet-B3, MobileNetV3. ImageNet pretrained. Giai đoạn 1 đóng backbone, train head 5 epoch. Giai đoạn 2 mở toàn mạng. Batch 10, FP16, early stopping patience 5 theo validation loss.

Metric: accuracy, precision, recall, F1 macro và weighted, confusion matrix, số tham số, GFLOPs.

## Kết quả

ConvNeXt 98,00% accuracy, weighted F1 0,98. ResNet 96,82%. DenseNet201 96,09%. Nhóm thiên efficiency khoảng 91%.

Leaf_Healthy tương đối dễ. Leaf_Colletotrichum có F1 thấp hơn và ít mẫu hơn. Có cụm nhầm Leaf_Algal, Leaf_Blight và Leaf_Phomopsis vì triệu chứng gần nhau.

Hạn chế: chủ yếu hai nguồn, geographic và seasonal diversity còn hạn chế. Performance có thể giảm với cảm biến độ phân giải thấp, nén mạnh, màu kém, white balance hoặc exposure khác. Future work nêu illumination-invariant augmentation và domain adaptation.

## Lấy và không lấy

Lấy cách chọn nhiều kiến trúc, transfer learning ImageNet, split stratified, weighted loss, macro F1, per-class F1, confusion matrix, độ phức tạp model và một protocol train chung.

Không lấy 98% làm mục tiêu của đồ án. Test của đồ án phải khóa và không dùng để chọn model.
