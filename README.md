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

## Cài đặt nhanh

Cài thư viện:

```text
python -m pip install -r requirements.txt
```

Repo không lưu ảnh. Bộ dữ liệu chính là DurianLDD, 4.437 ảnh, giấy phép CC BY 4.0:

https://www.kaggle.com/datasets/cthng123/durian-leaf-disease-dataset

Tải bằng Kaggle CLI, sau khi đã có token trong biến môi trường `KAGGLE_API_TOKEN`:

```text
python -m kaggle datasets download -d cthng123/durian-leaf-disease-dataset -p datasets/durian-ldd --unzip
```

Có thể tải zip trên trang Kaggle rồi giải nén vào cùng chỗ. Thư mục cần có là `datasets/durian-ldd/DLD_FinalDataset_224_spit`, bên trong là `train`, `val` và `test`.

Bộ dữ liệu khác nếu dùng sau này chỉ để đánh giá ngoài. Không gộp vào tập huấn luyện vì hệ nhãn và phân bố có thể khác.

## Kiểm tra dữ liệu

DurianLDD đã được kiểm kê. Báo cáo nằm ở `reports/data_quality/durian_ldd_data_quality_report.md`. Notebook điều phối là `notebooks/01_durian_ldd_eda.ipynb`.

```text
python -m src.data.quality_audit --dataset-root datasets/durian-ldd --output-dir reports/data_quality
```

## Data pipeline

Pipeline nạp dữ liệu PyTorch cho DurianLDD được cấu hình tại `configs/pipeline.yaml` và kiểm thử điều phối qua `notebooks/02_data_pipeline.ipynb`:
- **Vị trí dữ liệu:** Đặt thư mục `DLD_FinalDataset_224_spit` vào `datasets/durian-ldd/` (đường dẫn tương đối `image_root: datasets/durian-ldd`).
- **Huấn luyện (Train):** Áp dụng tiền xử lý tăng cường hình học (RandomHorizontalFlip $p=0.5$, RandomVerticalFlip $p=0.5$, RandomRotation $\pm 15^\circ$), ToTensor và chuẩn hóa ImageNet.
- **Đánh giá (Val & Test):** Tuyệt đối **không dùng augmentation** (chỉ Resize về 224 khi cần, ToTensor và chuẩn hóa). Tập `test` được bảo vệ bằng khóa an toàn và chặn truy cập trừ khi bật `FINAL_EVAL=1`.
- **Kỹ thuật xử lý/khôi phục ảnh:** Tuyệt đối không dùng Gamma, CLAHE, Retinex, khử nhiễu (denoising) hay làm nét (sharpening) trong pipeline chuẩn; các kỹ thuật này chỉ được thử nghiệm ở Phase 4.
- **Kết quả kiểm chứng trên ảnh thật:**
  - Toàn bộ 4.437/4.437 tệp ảnh trong manifest tồn tại 100% trên đĩa.
  - Phân chia: Train (3.105 ảnh, 98 batches), Val (444 ảnh, 14 batches), Test (888 ảnh - khóa an toàn).
  - 100% ảnh thuộc tập Train và Val đều đạt chuẩn kích thước 224x224 và hệ màu RGB (không ảnh lỗi, không cần nội suy resize lại).
  - Biểu đồ mẫu kiểm tra đã được xuất tại `outputs/figures/pipeline_samples.png`.

## Trạng thái

Đã kiểm tra chất lượng dữ liệu DurianLDD và khóa tập phân chia cố định (train: 3.105, val: 444, test: 888 ảnh). Đã hoàn thiện pipeline nạp dữ liệu (Dataset, Transforms, DataLoader) sẵn sàng cho giai đoạn huấn luyện mô hình.
