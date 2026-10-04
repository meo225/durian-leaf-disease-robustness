# Kiểm định chất lượng DurianLDD

DurianLDD đủ điều kiện tiếp tục thực nghiệm.

Điều kiện khi sang bước chia tập và tiền xử lý:
- Split có sẵn chứa cặp gần trùng xuyên train, validation hoặc test. Không dùng split này làm split đã khóa. Bước chia tập phải giữ mỗi nhóm gần trùng trong một tập.
- Mọi ảnh đọc được cùng một kích thước. Thống kê độ phân giải mô tả bản phát hành, không mô tả ảnh lúc chụp.
- Không có thời điểm chụp trong EXIF. Lần kiểm tra này không truy được cùng một phiên chụp khi hai ảnh đã khác nhau rõ.

## Phạm vi

Lần kiểm tra này chỉ dùng DurianLDD làm dữ liệu chính. Bộ Vietnamese durian leaf, AI-Driven và Ten durian diseases không nằm trong lần chạy này. Split train, validation và test có sẵn được đo để tìm rò rỉ. Lần kiểm tra không tạo split mới và không khóa test.

Ảnh gần trùng là cặp có hash khác biệt 64 bit cách nhau không quá 5 bit, kể cả khi một ảnh là bản lật ngang của ảnh kia. Ngưỡng này ưu tiên độ chính xác: ảnh nén lại hoặc chỉnh nhẹ được gom, còn ảnh cùng lá nhưng đổi góc chụp mạnh có thể không bị gom. Bảng độ nhạy ở khoảng cách 0, 5, 8 và 10 dùng để xem kết luận có phụ thuộc vào đúng một ngưỡng hay không.

Độ sáng là trung bình kênh xám. Độ tương phản là độ lệch chuẩn kênh xám. Độ sắc nét là phương sai Laplacian. Độ bão hòa là trung bình kênh S trong HSV. Năng lượng tần số cao là độ lệch chuẩn phần dư sau làm mờ Gaussian. Chỉ số cuối trộn texture vết bệnh với nhiễu, nên chỉ dùng để mô tả, không dùng để loại ảnh.

## Nguồn và cấu trúc

Nguồn là bản Kaggle `cthng123/durian-leaf-disease-dataset`, giấy phép CC BY 4.0. Thư mục phát hành là `DLD_FinalDataset_224_spit`. Có 4.437 tệp ảnh, khớp mức 4.437 ảnh đã chọn.

Ảnh đọc được: 4.437. Tệp hỏng hoặc rỗng: 0 (0,0%). Ảnh còn thời điểm chụp trong EXIF: 0.

## Phân bố lớp

| Lớp | Số ảnh | Tỉ lệ |
| --- | --- | --- |
| Algal leaf spot | 733 | 16,5% |
| Allocaridara attack | 913 | 20,6% |
| Healthy leaf | 976 | 22,0% |
| Leaf blight | 937 | 21,1% |
| Phomopsis leaf spot | 878 | 19,8% |
| Tổng | 4.437 | 100,0% |

| Lớp | train | val | test | Tổng |
| --- | --- | --- | --- | --- |
| Algal leaf spot | 513 | 73 | 147 | 733 |
| Allocaridara attack | 639 | 91 | 183 | 913 |
| Healthy leaf | 683 | 97 | 196 | 976 |
| Leaf blight | 655 | 94 | 188 | 937 |
| Phomopsis leaf spot | 614 | 88 | 176 | 878 |
| Tổng | 3.104 | 443 | 890 | 4.437 |

Split có sẵn gồm 70,0% train, 10,0% validation và 20,1% test. Từng lớp cũng nằm sát tỉ lệ này.

Năm tên lớp giữ nguyên theo thư mục phát hành. `ALLOCARIDARA_ATTACK` là tên do nhà phát hành đặt. Lần kiểm tra này không đổi tên và không gộp lớp.

## Kích thước và tỉ lệ khung

Mọi ảnh đọc được đều có kích thước 224x224, tỉ lệ khung hình 1,000. Bản phát hành đã được resize. Thống kê này không mô tả độ phân giải lúc chụp. Hình: `reports/data_quality/figures/resolutions.png`.

## Phân bố chất lượng ảnh

| Chỉ số | Nhỏ nhất | Phân vị 5 | Trung vị | Phân vị 95 | Lớn nhất |
| --- | --- | --- | --- | --- | --- |
| Độ sáng | 43.3 | 85.8 | 118.8 | 138.4 | 163.1 |
| Độ tương phản | 19.1 | 33.3 | 43.9 | 57.9 | 75.6 |
| Độ sắc nét | 119.1 | 254.4 | 520.3 | 1007.8 | 2843.4 |
| Độ bão hòa | 42.7 | 68.1 | 109.8 | 155.3 | 229.5 |
| Năng lượng tần số cao | 2.5 | 3.6 | 5.1 | 7.0 | 11.7 |

Ngưỡng rà soát được đặt trước khi xem phân bố: độ sáng dưới 40, độ sáng trên 220, độ tương phản dưới 20 và độ sắc nét dưới 50. Số ảnh vượt ngưỡng lần lượt là 0, 0, 1 và 0. Đây là cờ để xem lại, không phải quy tắc loại ảnh.

Hình phân bố nằm ở `reports/data_quality/figures/quality_distributions.png` và `reports/data_quality/figures/quality_by_class.png`.

## Tệp hỏng, trùng lặp và gần trùng

Ở ngưỡng Hamming 5 có 6 cặp. Trong đó cặp trùng khớp tuyệt đối, trùng pixel hoặc là bản lật ngang đúng pixel: 1. Phân loại cặp: near 5, identical_file 1. Số nhóm có từ hai ảnh trở lên: 4. Nhóm lớn nhất có 3 ảnh. Số ảnh nằm trong các nhóm đó: 9.

| Ngưỡng Hamming | Số cặp | Số ảnh liên quan | Cặp xuyên split | Cặp khác lớp |
| --- | --- | --- | --- | --- |
| 0 | 2 | 4 | 1 | 0 |
| 5 | 6 | 9 | 3 | 0 |
| 8 | 10 | 16 | 4 | 3 |
| 10 | 39 | 68 | 15 | 20 |

## Nhãn và rò rỉ

Cặp gần trùng khác lớp: 0. Ảnh trong cặp trùng khớp tuyệt đối nhưng khác nhãn: 0. Cặp gần trùng xuyên split: 3. Ảnh test có bạn gần trùng trong train hoặc validation: 2 trên 890 (0,2%). Ảnh validation có bạn gần trùng trong train: 0 trên 443 (0,0%). Cặp có mã `to_label_` liền nhau: 4.407, trong đó xuyên split 2.004 cặp và xuyên split đồng thời gần trùng 1 cặp. Cặp ID liền nhau nhưng khác lớp: 3. Trung vị khoảng cách Hamming của các cặp ID kề nhau là 32.0.

Mã `to_label_` liền nhau có trung vị khoảng cách Hamming quanh mức ngẫu nhiên. Số mã xuyên split vì vậy không phải bằng chứng rò rỉ. Chỉ các cặp vừa liền mã vừa gần trùng, hoặc các cặp gần trùng dù không liền mã, mới được tính là cùng một mẫu.

Các cặp gần trùng xuyên split:

- `DLD_FinalDataset_224_spit/test/ALGAL_LEAF_SPOT/to_label_197.jpg` và `DLD_FinalDataset_224_spit/train/ALGAL_LEAF_SPOT/to_label_196.jpg`, Hamming 4.
- `DLD_FinalDataset_224_spit/test/HEALTHY_LEAF/to_label_2663.jpg` và `DLD_FinalDataset_224_spit/train/HEALTHY_LEAF/to_label_2682.jpg`, Hamming 0.
- `DLD_FinalDataset_224_spit/test/HEALTHY_LEAF/to_label_2663.jpg` và `DLD_FinalDataset_224_spit/train/HEALTHY_LEAF/to_label_2683.jpg`, Hamming 4.

Đối chiếu ảnh cho thấy các cặp xuyên split là cùng một cảnh lá, lệch nhẹ khung hình. Hình các cặp gần trùng: `reports/data_quality/figures/near_duplicate_pairs.png`.

Có một cặp trùng byte. Cặp này đang nằm cùng một split nên không làm rò rỉ split hiện tại, nhưng vẫn phải đi cùng nhau khi chia tập lại.

Nới ngưỡng lên 8 làm xuất hiện 3 cặp khác lớp. Đối chiếu ảnh cho thấy đó là các lá khác nhau, cùng kiểu một lá lớn trên nền vườn. Các cặp này không được tính là gán nhầm. Ngưỡng 10 làm số cặp khác lớp tăng tiếp, nên bước chia tập giữ ngưỡng 5.

Mã nhóm của mọi ảnh nằm ở cột `group_id` trong `reports/data_quality/tables/image_inventory.csv`. Các nhóm có từ hai ảnh trở lên được tách ở `reports/data_quality/tables/near_duplicate_groups.csv`. Bước chia tập dùng các mã này để giữ cả nhóm trong một tập.

Tiền tố `to_label_` là cách đặt tên của toàn bộ bản phát hành. Mọi tệp đều nằm trong một thư mục lớp, nên tiền tố này không được xem là ảnh chưa gán nhãn.

Ảnh mẫu mỗi lớp nằm ở `reports/data_quality/figures/class_examples.png`. Lưới này chỉ phục vụ rà soát định tính. Kết luận về xung đột nhãn dựa trên cặp trùng và gần trùng khác lớp.

## Giới hạn

Hash gần trùng không chứng minh hai ảnh khác hash là hai lá khác nhau. Không có mã phiên chụp và không có EXIF thời gian nên ảnh cùng lá, khác góc hoặc khác thời điểm có thể còn sót. Độ sáng, tương phản và độ sắc nét được đo trên bản đã resize, nên không suy ngược chất lượng ống kính lúc thu thập. Lần kiểm tra không huấn luyện mô hình và không đánh giá độ tách biệt giữa các bệnh.

## Cách chạy lại

```text
python -m src.data.quality_audit --dataset-root datasets/durian-ldd --output-dir reports/data_quality
```

Ảnh gốc không được đưa vào Git. Thư mục `datasets/` nằm trong `.gitignore`.
