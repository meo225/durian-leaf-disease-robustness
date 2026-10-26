# Durian Disease Robustness

## Tổng quan

Đồ án CS406 - Xử lý ảnh và ứng dụng. Đề tài khảo sát phân loại bệnh trên ảnh lá sầu riêng khi chất lượng ảnh thay đổi.

## Hướng nghiên cứu

- Phân loại bệnh trên ảnh lá sầu riêng.
- Ảnh hưởng của thay đổi chất lượng ảnh: ánh sáng, độ tương phản, mờ và nhiễu.
- So sánh các kỹ thuật xử lý ảnh trên cùng thiết lập thí nghiệm.
- Độ ổn định của mô hình trên ảnh bị suy giảm có kiểm soát.
- Đánh giá trên bộ dữ liệu ngoài khi hệ nhãn và phân bố tương thích.

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

Một lần thí nghiệm đi qua ba nơi.

1. Notebook đọc ảnh từ `datasets/`, cấu hình từ `configs/` và split từ `manifests/`.
2. File sinh ra trong lần chạy được ghi vào `outputs/`: checkpoint, hình, log và bảng thô.
3. Sau khi nhóm kiểm tra, bản dùng cho báo cáo chuyển sang `reports/` và đi cùng code khi commit. Checkpoint của lần train được giữ trên một dataset Kaggle. Ảnh gốc ở lại `datasets/`.

Git theo dõi code, cấu hình, split, tài liệu và `reports/`. `outputs/` và `datasets/` ở lại máy chạy.

```text
durian-leaf-disease-robustness/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── docs/
│   └── sources/
├── manifests/            # split đã khóa
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── corruption/       # suy giảm ảnh có kiểm soát
│   ├── processing/       # kỹ thuật xử lý để so sánh
│   └── evaluation/
├── tests/
├── reports/              # kết quả đã chốt, commit cùng code
├── outputs/              # kết quả một lần chạy, không commit
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   └── results/
├── app/
└── datasets/             # ảnh local, không commit
```

Đường dẫn dữ liệu đưa vào qua cấu hình hoặc notebook. Phần mã trong `src` không gắn cứng đường dẫn máy cục bộ hay Kaggle.

## Dữ liệu

Dataset chính là DurianLDD, 4.437 ảnh, năm lớp, giấy phép CC BY 4.0:

https://www.kaggle.com/datasets/cthng123/durian-leaf-disease-dataset

Ảnh không nằm trong Git. Trên máy local, giải nén vào `datasets/durian-ldd/` sao cho có thư mục `DLD_FinalDataset_224_spit`. Trên Kaggle, notebook gắn dataset này làm input. Cell kéo code tìm thư mục đó và nối vào `datasets/durian-ldd`.

Split của nhóm là cột `split` trong `manifests/split_manifest.csv`: train 3.105, val 444, test 888. Folder `train`, `val`, `test` trong bản Kaggle là chia sẵn của dataset, không dùng để train.

Dataset sầu riêng khác, nếu dùng, chỉ để đánh giá ngoài sau khi pipeline đã khóa. Không gộp vào train, val hay test.

## Cài đặt

```text
python -m pip install -r requirements.txt
```

Trên Kaggle, image mặc định đã có các thư viện chính. Cài thêm trong notebook khi một thí nghiệm cần gói chưa có.

## Chạy trên Kaggle

Notebook trong `notebooks/` chạy trên Kaggle. Cell đầu ghi tên file. Cell thứ hai clone repo theo `GIT_BRANCH`, rồi gắn DurianLDD vào `datasets/durian-ldd`.

```python
GIT_BRANCH = "main"
```

Đổi `GIT_BRANCH` khi chạy nhánh khác, rồi chạy lại cell. Cell in nhánh, commit và đường dẫn ảnh.

Sửa `src/`, `configs/` hoặc `manifests/` thì chạy lại cell thứ hai. Sửa cell trong file ipynb thì cập nhật lại notebook trên Kaggle.

### Tài khoản

1. Đăng nhập https://www.kaggle.com.
2. Xác nhận số điện thoại tại https://www.kaggle.com/settings, mục Phone Verification.

GPU và Internet dùng được sau bước này. Hạn mức GPU miễn phí khoảng 30 giờ mỗi tuần, reset 07:00 thứ Bảy giờ Việt Nam. Một phiên chạy tối đa 12 giờ. Phiên interactive kết thúc sau khoảng 20 phút không có thao tác.

### Đưa notebook lên Kaggle

Mỗi file ipynb tương ứng một notebook trong group `CS406-durian-leaf-disease-robustness`, tạo một lần rồi dùng lại.

`title` là tên hiện trong danh sách Kaggle, đặt theo tên file để giữ thứ tự. `id` là địa chỉ `username/slug`. Kaggle chỉ nhận chữ thường và dấu `-` trong slug, nên `_` trong tên file thành `-` trong `id`.

Cần Kaggle CLI và biến môi trường `KAGGLE_API_TOKEN`. Copy file ipynb vào một thư mục tạm, đặt cạnh `kernel-metadata.json`, rồi chạy:

```text
kaggle kernels push --no-run -p <thư-mục-tạm>
```

`kernel-metadata.json` cho `01_durian_ldd_eda.ipynb`:

```json
{
  "id": "TEN_KAGGLE/01-durian-ldd-eda",
  "title": "01_durian_ldd_eda",
  "code_file": "01_durian_ldd_eda.ipynb",
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
```

`TEN_KAGGLE` là username trên profile Kaggle.

| File | `title` | `id` |
|---|---|---|
| `notebooks/01_durian_ldd_eda.ipynb` | `01_durian_ldd_eda` | `TEN_KAGGLE/01-durian-ldd-eda` |
| `notebooks/01_split_durianldd.ipynb` | `01_split_durianldd` | `TEN_KAGGLE/01-split-durianldd` |
| `notebooks/02_data_pipeline.ipynb` | `02_data_pipeline` | `TEN_KAGGLE/02-data-pipeline` |
| `notebooks/03_train_resnet50.ipynb` | `03_train_resnet50` | `TEN_KAGGLE/03-train-resnet50` |
| `notebooks/03_train_hog_svm.ipynb` | `03_train_hog_svm` | `TEN_KAGGLE/03-train-hog-svm` |
| `notebooks/03_train_mobilenetv3_small.ipynb` | `03_train_mobilenetv3_small` | `TEN_KAGGLE/03-train-mobilenetv3-small` |
| `notebooks/03_train_convnext_tiny.ipynb` | `03_train_convnext_tiny` | `TEN_KAGGLE/03-train-convnext-tiny` |
| `notebooks/03_select_reference_model.ipynb` | `03_select_reference_model` | `TEN_KAGGLE/03-select-reference-model` |
| `notebooks/04_corruption_benchmark.ipynb` | `04_corruption_benchmark` | `TEN_KAGGLE/04-corruption-benchmark` |

Ba notebook kiểm tra dữ liệu và `03_train_hog_svm` dùng Accelerator None. `03_train_resnet50`, `03_train_mobilenetv3_small`, `03_train_convnext_tiny`, `03_select_reference_model` và `04_corruption_benchmark` bật GPU trong metadata hoặc trong Session options. Nếu tab Input chưa có dataset, thêm `cthng123/durian-leaf-disease-dataset`. Dataset gắn một lần cho mỗi notebook.

### Phiên chạy

Internet để On. Notebook train dùng GPU. Ba notebook kiểm tra dữ liệu để Accelerator là None. Đổi một trong hai tùy chọn này sẽ khởi động lại phiên.

Chạy tương tác: Edit, chạy cell thứ hai, rồi chạy các cell sau. Output hiện dưới cell và trong tab Output tại `/kaggle/working`. Quick Save lưu trạng thái editor tại thời điểm đó.

Lưu version: Save Version, chọn Save & Run All, đặt tên version. Phiên chạy từ cell đầu trên máy sạch. Log ở tab Logs. File ở tab Output, trong `/kaggle/working/durian-leaf-disease-robustness/`.

### Kết quả

| Sản phẩm | Trong phiên | Sau khi chốt |
|---|---|---|
| Bảng và hình cho báo cáo | tab Output | `reports/`, rồi commit |
| Checkpoint | tab Output | dataset Kaggle riêng, chia sẻ trên trang dataset |

Model đưa lên Hugging Face khi đã chọn bản phát hành. Ảnh suy giảm và ảnh sau xử lý sinh lúc chạy. Tập test chỉ mở ở lần đánh giá cuối, với `FINAL_EVAL=1`. Chọn model trên train và val.

## Cấu hình

| Nội dung | File |
|---|---|
| Split | `configs/split.yaml`, `manifests/split_manifest.csv` |
| Pipeline ảnh | `configs/pipeline.yaml` |
| Train ResNet-50 | `configs/train_resnet50.yaml` |
| Train MobileNetV3-Small | `configs/train_mobilenetv3_small.yaml` |
| Train ConvNeXt-Tiny | `configs/train_convnext_tiny.yaml` |
| Train HOG + SVM (CPU) | `configs/train_hog_svm.yaml` |
| Chọn reference model | `configs/reference_selection.yaml` |
| Suy giảm ảnh | `configs/corruption.yaml` |
| Kiểm tra dữ liệu | `configs/data_audit.yaml` |

Pipeline chuẩn: train dùng lật ngang, lật dọc, xoay ±15°, rồi chuẩn hóa ImageNet. Val và test resize về 224 khi cần, rồi chuẩn hóa. Gamma, CLAHE, Retinex, khử nhiễu và làm nét không nằm trong pipeline này.
