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

## Chạy trên Kaggle

Mọi notebook trong `notebooks/` chỉ chạy trên Kaggle. Cell đầu ghi tên file notebook. Cell thứ hai kiểm tra môi trường, kéo nhánh GitHub, rồi gắn DurianLDD public vào `datasets/durian-ldd`.

Nhánh cần kéo nằm ở dòng này trong cell thứ hai:

```python
GIT_BRANCH = "main"
```

Đổi giá trị đó rồi chạy lại cell nếu nhóm dùng nhánh khác. Cell in ra nhánh, mã commit và đường dẫn ảnh.

### Tài khoản

1. Đăng nhập https://www.kaggle.com.
2. Mở https://www.kaggle.com/settings, mục Phone Verification.
3. Thêm số điện thoại và nhập mã xác nhận.

GPU và Internet chỉ bật được sau bước này. Hạn mức GPU miễn phí khoảng 30 giờ mỗi tuần, có thể cao hơn khi máy rảnh, và reset vào 07:00 sáng thứ Bảy giờ Việt Nam. Một phiên chạy tối đa 12 giờ. Phiên đang sửa notebook tắt sau khoảng 20 phút nếu không chạy cell và không sửa nội dung.

### Token kéo repo private

Repo là https://github.com/meo225/durian-leaf-disease-robustness.

1. Mở https://github.com/settings/personal-access-tokens và tạo fine-grained token.
2. Resource owner chọn `meo225`. Chỉ cấp cho repo này.
3. Permissions: Contents là Read-only, Metadata là Read.
4. Trong notebook Kaggle, mở Add-ons, chọn Secrets.
5. Thêm secret tên `GITHUB_TOKEN`. Value là token vừa tạo.

Không dán token vào notebook và không commit token.

### Đưa file notebook lên Kaggle

Mỗi file ipynb tương ứng một notebook private. Tạo một lần, các lần sau dùng lại notebook đó. Share và mời Thái với quyền Can edit.

Đưa file lên lần đầu, và đưa lại mỗi khi sửa cell trong file ipynb. Sửa file trong `src/`, config hoặc manifest thì không đưa lại ipynb: trên Kaggle chạy lại cell thứ hai là đủ.

Trên máy, sau khi có Kaggle CLI và `KAGGLE_API_TOKEN`:

```powershell
$slug = "TEN_KAGGLE/durian-ldd-data-pipeline"
$nb = "02_data_pipeline.ipynb"
$dir = Join-Path $env:TEMP "kaggle-push"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Copy-Item "notebooks\$nb" (Join-Path $dir $nb) -Force
$meta = Join-Path $dir "kernel-metadata.json"
@"
{
  "id": "$slug",
  "title": "$slug",
  "code_file": "$nb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_internet": "true",
  "dataset_sources": ["cthng123/durian-leaf-disease-dataset"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
"@ | python -c "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')" $meta
kaggle kernels push --no-run -p $dir
```

`TEN_KAGGLE` là username trên profile Kaggle. Đổi `$slug` và `$nb` cho từng notebook:

| File | Slug gợi ý | Accelerator |
|---|---|---|
| `notebooks/01_durian_ldd_eda.ipynb` | `TEN_KAGGLE/durian-ldd-eda` | None |
| `notebooks/01_split_durianldd.ipynb` | `TEN_KAGGLE/durian-ldd-split` | None |
| `notebooks/02_data_pipeline.ipynb` | `TEN_KAGGLE/durian-ldd-data-pipeline` | None |

Ba notebook hiện tại không bật GPU. Notebook train hoặc Grad-CAM sau này mới chọn GPU trong cùng file metadata, hoặc trong Session options.

### Dataset

Dataset public dùng chung:

https://www.kaggle.com/datasets/cthng123/durian-leaf-disease-dataset

Không upload lại 4.437 ảnh. Lệnh push ở trên đã khai báo dataset này trong `dataset_sources`, nên không cần bấm Add Input nếu push thành công.

Nếu tab Input chưa có dataset: Add Input, tìm `cthng123/durian-leaf-disease-dataset`, rồi Add. Gắn một lần cho notebook đó. Lần chạy sau vẫn còn. Notebook mới thì gắn lại.

Folder `train`, `val`, `test` trong dataset Kaggle là folder gốc của DurianLDD. Split của nhóm là cột `split` trong `manifests/split_manifest.csv`.

### Internet và Accelerator

Trong Session options của notebook:

- Internet: On. Tắt thì cell thứ hai không kéo được GitHub.
- Accelerator: None cho ba notebook hiện tại. GPU khi notebook đó train hoặc chạy Grad-CAM.
- Đổi Internet hoặc Accelerator sẽ khởi động lại phiên. Chạy lại từ cell đầu.

### Xem kết quả mà chưa lưu version

1. Mở notebook và bấm Edit.
2. Internet On. Accelerator đúng loại notebook cần.
3. Run cell thứ hai. Đọc nhánh, commit và đường dẫn ảnh.
4. Run các cell sau, hoặc Run All trong editor.

Chữ hiện ngay dưới cell. File tạo ra nằm ở tab Output, trong `/kaggle/working`. Cách này chưa tạo version. Tắt phiên thì file của phiên tương tác mất nếu chưa Save Version.

### Chạy và lưu version

1. Save Version.
2. Chọn Save & Run All.
3. Đặt tên version ngắn, ví dụ `pipeline-check`.
4. Có thể đóng trang. Phiên này chạy từ cell đầu trên máy sạch, tối đa 12 giờ.

Log nằm ở tab Logs của version đó. File nằm ở tab Output. Thư mục repo trong phiên là `/kaggle/working/durian-leaf-disease-robustness/`.

Quick Save chỉ chụp những gì đang có trên editor, không chạy lại từ đầu. Dùng Quick Save sau khi đã Run trong editor và muốn giữ output đó.

### Chỗ giữ kết quả

| Loại | Xem trên Kaggle | Giữ lâu dài |
|---|---|---|
| Số in trong cell | Dưới cell, hoặc tab Logs | Ghi vào CSV hoặc JSON và commit lên GitHub |
| Hình, CSV, JSON nhỏ | Tab Output | Tải về và commit lên GitHub |
| Checkpoint `.pth` | Tab Output | Dataset private trên Kaggle |

Tạo dataset từ output: mở version đã chạy xong, tab Output, New Dataset, để Private, rồi mời Thái trên trang dataset. Quyền của notebook không tự sang dataset. Notebook sau gắn dataset đó bằng Add Input.

Hugging Face chỉ dùng khi nhóm đã chọn model cuối và muốn phát hành bản đó. Checkpoint trung gian và bảng metric không đưa lên Hugging Face.

Ảnh suy giảm và ảnh sau xử lý sinh lúc chạy từ code. Không upload thành dataset ảnh mới, trừ khi một notebook sau phải đọc lại đúng các file đó.

Tập test chỉ mở ở notebook đánh giá cuối, với `FINAL_EVAL=1`. Notebook chọn model chỉ dùng train và val.

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
