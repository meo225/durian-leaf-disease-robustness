# Durian Disease Robustness

## Tổng quan

Đây là đồ án nghiên cứu học phần CS406 Xử lý ảnh và ứng dụng. Đề tài nghiên cứu nhận diện bệnh trên lá sầu riêng bền vững khi chất lượng ảnh biến thiên trong điều kiện thực tế.

## Hướng nghiên cứu

Bài toán không dừng ở phân loại ảnh thông thường. Hướng nghiên cứu gồm nhận diện bệnh trên lá sầu riêng. Biến thiên chất lượng ảnh thực tế gồm ánh sáng độ sáng độ tương phản mờ và nhiễu. Nghiên cứu còn gồm xử lý ảnh nhằm tăng tính bền vững. Đánh giá độ bền vững của mô hình. Đánh giá trên tập dữ liệu ngoài khi phân loại lớp và phân bố dữ liệu phù hợp.

Phương pháp nghiên cứu kiến trúc mô hình kỹ thuật tiền xử lý tham số suy giảm ảnh và phương pháp đề xuất vẫn đang được phát triển. Các quyết định này chưa được chốt.

## Quy trình nghiên cứu dự kiến

Kiểm tra dữ liệu
→ Mô hình cơ sở
→ Chuẩn suy giảm ảnh có kiểm soát
→ Chuẩn xử lý ảnh
→ Phương pháp đề xuất
→ Nghiên cứu loại bỏ thành phần
→ Đánh giá độ bền vững
→ Đánh giá tập dữ liệu ngoài
→ Ứng dụng minh họa

## Cấu trúc kho mã nguồn

- `configs`: file cấu hình thí nghiệm sẽ được thêm sau
- `src`: mã nguồn tái sử dụng gồm dữ liệu mô hình xử lý ảnh đánh giá và tiện ích chung
- `notebooks`: sổ thí nghiệm nghiên cứu
- `outputs`: checkpoint hình vẽ nhật ký và kết quả sinh ra khi chạy thí nghiệm
- `app`: ứng dụng minh họa sẽ được triển khai sau
- `tests`: kiểm thử tự động cho các mô-đun tái sử dụng

Đường dẫn dữ liệu sẽ được cung cấp qua cấu hình hoặc notebook. Mã nguồn tái sử dụng không gắn cứng đường dẫn môi trường thí nghiệm.

## Dữ liệu

Kho mã nguồn không lưu tập dữ liệu. Tập đang xem xét là DurianLDD khoảng 4437 ảnh. Các tập sầu riêng khác nếu dùng sau này chỉ phục vụ đánh giá ngoài và không được gộp tự động vào tập huấn luyện vì phân loại lớp và phân bố dữ liệu có thể khác nhau.

## Trạng thái

Đang chuẩn bị khung nghiên cứu và bước kiểm tra dữ liệu.
