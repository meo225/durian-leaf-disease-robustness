# Durian Disease Robustness

## Tổng quan

Đồ án học phần CS406, Xử lý ảnh và ứng dụng. Đề tài khảo sát phân loại bệnh trên ảnh lá sầu riêng khi chất lượng ảnh thay đổi.

## Hướng nghiên cứu

Công việc tập trung vào các điểm sau:

- phân loại bệnh trên ảnh lá sầu riêng
- ảnh hưởng của thay đổi chất lượng ảnh, gồm ánh sáng, độ tương phản, mờ và nhiễu
- so sánh một số kỹ thuật xử lý ảnh trên cùng thiết lập thí nghiệm
- đo độ ổn định của mô hình trên ảnh bị suy giảm có kiểm soát
- kiểm tra thêm trên bộ dữ liệu ngoài nếu hệ nhãn và phân bố dữ liệu tương thích

Kiến trúc mô hình, tiền xử lý, tham số suy giảm ảnh và phương pháp đề xuất chưa được chọn. Các lựa chọn này sẽ dựa trên kết quả thí nghiệm.

## Quy trình dự kiến

Kiểm tra dữ liệu
→ Mô hình cơ sở
→ Đánh giá trên ảnh suy giảm có kiểm soát
→ So sánh kỹ thuật xử lý ảnh
→ Phương pháp đề xuất
→ Ablation
→ Đánh giá độ ổn định
→ Đánh giá bộ dữ liệu ngoài
→ Demo

## Cấu trúc repo

- `configs`: cấu hình thí nghiệm
- `src`: mã nguồn dùng lại, gồm dữ liệu, mô hình, xử lý ảnh, đánh giá và tiện ích
- `notebooks`: notebook thí nghiệm
- `outputs`: checkpoint, hình, log và kết quả chạy thí nghiệm
- `app`: demo
- `tests`: kiểm thử các module dùng lại

Đường dẫn dữ liệu đưa vào qua cấu hình hoặc notebook. Phần mã trong `src` không gắn cứng đường dẫn máy cục bộ hay Kaggle.

## Dữ liệu

Repo không lưu dữ liệu. Bộ dữ liệu đang xem xét là DurianLDD, khoảng 4437 ảnh.

Bộ dữ liệu khác nếu dùng sau này chỉ để đánh giá ngoài. Không gộp vào tập huấn luyện vì hệ nhãn và phân bố có thể khác.

## Trạng thái

Đang dựng repo. Chưa kiểm tra dữ liệu.
