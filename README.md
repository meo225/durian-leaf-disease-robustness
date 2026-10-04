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
- `manifests`: siêu dữ liệu phân chia tập dữ liệu cố định (train, val, test)
- `src`: mã nguồn dùng lại
  - `data`: tải dữ liệu, kiểm tra dữ liệu, chia tập và tiền xử lý chuẩn
  - `models`: định nghĩa mô hình
  - `training`: vòng lặp huấn luyện và lưu checkpoint
  - `corruption`: tạo suy giảm chất lượng ảnh có kiểm soát (ánh sáng, độ tương phản, mờ, nhiễu)
  - `processing`: kỹ thuật xử lý/khôi phục ảnh thử nghiệm (Gamma, CLAHE, Retinex, v.v.)
  - `evaluation`: độ đo đánh giá, phân tích độ ổn định và Grad-CAM
  - `utils`: tiện ích chung (đường dẫn, seed, logging)
- `notebooks`: notebook điều phối thí nghiệm trên Kaggle
- `outputs`: checkpoint, hình ảnh, log và kết quả chạy thí nghiệm
- `app`: ứng dụng demo
- `tests`: kiểm thử đơn vị cho các module dùng lại

Đường dẫn dữ liệu đưa vào qua cấu hình hoặc notebook. Phần mã trong `src` không gắn cứng đường dẫn máy cục bộ hay Kaggle.

## Dữ liệu

Repo không lưu dữ liệu. Bộ dữ liệu chính là DurianLDD, 4.437 ảnh.

Bộ dữ liệu khác nếu dùng sau này chỉ để đánh giá ngoài. Không gộp vào tập huấn luyện vì hệ nhãn và phân bố có thể khác.

## Kiểm tra dữ liệu

DurianLDD đã được kiểm kê. Báo cáo nằm ở `reports/data_quality/durian_ldd_data_quality_report.md`. Notebook điều phối là `notebooks/01_durian_ldd_eda.ipynb`.

```text
python -m src.data.quality_audit --dataset-root datasets/durian-ldd --output-dir reports/data_quality
```

Ảnh không được commit. Đặt bản giải nén vào `datasets/durian-ldd`.

## Trạng thái

Đã kiểm tra DurianLDD. Bộ dữ liệu đủ điều kiện sang bước chia tập. Split có sẵn của nhà phát hành chưa được khóa.
