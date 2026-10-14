# Domain generalized plant leaf disease recognition 2025

DOI: https://doi.org/10.1016/j.engappai.2025.111168

Repository: https://github.com/seabearlmx/DG-PLDR

Phân tích này dùng thông tin đã xác nhận từ nhà xuất bản và repository chính thức. Full PDF chưa có trong tập file lúc viết review.

## Vai trò trong đồ án

Dùng cho đánh giá generalization trên dataset ngoài và cho cách nghĩ khi chọn nhiều họ backbone trước khi chốt reference model. Không dùng để chọn mức blur, noise hay brightness.

## Vấn đề

Model học tốt trên dataset train chưa chắc nhận diện đúng khi ảnh đến từ môi trường mới. Nhiều bài chia train và test từ cùng một dataset, nên background, màu, camera và style có thể giống nhau. Accuracy cao chưa chứng minh model học đúng triệu chứng.

Shortcut là dấu hiệu phụ model dùng để đoán nhãn thay vì đặc điểm bệnh. Ví dụ một lớp trong train thường đi với một kiểu nền, model có thể dùng nền để đoán.

## Việc tác giả làm

Benchmark theo source domain và unseen domain cho nhiều nhóm cây. So sánh CNN, Transformer và self-supervised vision foundation model. Foundation model học representation từ nhiều ảnh mà không cần nhãn thủ công cho mọi ảnh.

Phương pháp chính là SDPM. Phần style-sensitive distillation làm feature bớt phụ thuộc style của từng domain. Phần prototype-based matching hướng classifier tới thông tin dùng được ở nhiều domain.

Repository có cấu hình source, target và domain generalization, hỗ trợ pretrained model và frozen stages. Frozen stage là giữ một phần trọng số, không cập nhật cả model khi train.

## Insight

Accuracy cao trên một dataset chưa chứng minh robust ngoài thực tế.

Background và style có thể thành shortcut. Khi đánh giá cần hỏi model đang dựa vào vùng nào của ảnh.

External dataset phải thật sự unseen. Nếu xem kết quả external rồi chỉnh model hoặc preprocessing theo đó, tập này không còn là phép thử độc lập.

Augmentation không đồng nghĩa domain generalization. Ảnh biến đổi thêm có thể giúp train, nhưng vẫn cần một domain độc lập để kiểm tra.

## Không lấy

Mức severity của blur, noise hay brightness. Đó không phải benchmark của bài này.
