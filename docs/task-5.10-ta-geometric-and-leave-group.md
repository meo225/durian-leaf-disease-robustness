# Task 5.10. TrivialAugment hình học và leave-group-out

## Câu hỏi

5.9 cho biết các nhánh khác nhau bao nhiêu khi giao thức cân bằng. Hai câu hỏi còn lại cần hai nhánh riêng:

1. **TrivialAugment có phải đối chứng "tổng quát" sạch không?** TrivialAugment Wide có Brightness, Color, Contrast, Sharpness với biên 0,01 đến 2. Các phép này trùng họ brightness, contrast, color_cast (brightness_bright mức 3 là 2,0, đúng biên của TA). Vậy phần cải thiện của TA ở 5.6 có thể là do phép quang học trùng họ chứ không phải do augmentation nói chung. Nhánh **TA hình học** bỏ hết phép quang học để tách hai khả năng này.
2. **Augmentation suy giảm có chuyển sang họ chưa từng thấy cùng cơ chế không?** Geirhos (R32) bỏ một loại suy giảm khỏi train thì loại đó chỉ đạt 11 đến 14%. Nhánh **leave-group-out** bỏ cả một nhóm họ cùng cơ chế khỏi train và đo chính nhóm đó.

## Thiết kế

Giao thức chung như 5.9 (`configs/controls/protocol.yaml`), mỗi nhánh **một seed (42)**.

### TA hình học (`trivialaugment_geometric.yaml`)

`GeometricTrivialAugment` giữ cách chọn của TrivialAugment (mỗi ảnh một phép, cường độ ngẫu nhiên, dấu ngẫu nhiên) nhưng chỉ chọn trong `Identity, ShearX, ShearY, TranslateX, TranslateY, Rotate`. Bỏ Brightness, Color, Contrast, Sharpness, Posterize, Solarize, AutoContrast, Equalize. Identity giữ lại để mỗi ảnh vẫn có xác suất không bị đổi như TA gốc. Baseline đã có lật và xoay ±15°, nên TA hình học chỉ thêm biến dạng hình học mạnh hơn.

### Leave-group-out (`degradation_leave_group.yaml`)

Bỏ khỏi augmentation suy giảm một **nhóm họ chọn bằng số**, không chọn theo tên: họ mỏ neo `blur` cộng hai họ gần nó nhất theo ma trận khoảng cách đặc trưng giữa 14 họ đã thấy (`seen_pairwise_distance` trong `reports/unseen_v2/msd.json`, do task 5.7 ghi ra). Mỏ neo và kích thước nhóm (3) khóa ở file cấu hình trước khi thấy ma trận. Hòa khoảng cách thì xếp theo tên. Nhóm cụ thể, cùng khoảng cách trong nhóm so với ra ngoài nhóm, được tính lúc chạy và ghi vào JSON kết quả (`leave_group`).

Một khác biệt có chủ ý so với 5.9: nhánh này chọn checkpoint trên validation hỗn hợp **đã bỏ các họ bị bỏ**. Giữ chúng thì checkpoint được chọn bằng chính họ cần coi là chưa thấy, làm điểm trên nhóm đó lạc quan. Số họ trong validation chọn ghi ở cột `val_families`, và điểm `mixed_val_f1` của nhánh này không so trực tiếp với nhánh khác.

## Đánh giá

Cùng thước đo 5.9 (`objective_42`) cộng, cho nhánh leave-group, `objective_group` (trung bình trên các họ bị bỏ) và `objective_rest` (các họ còn lại). Một seed không có SD riêng, nên SD lấy từ nhánh tham chiếu ba seed của 5.9, quy tắc **2,5 lần SD** áp như ở 5.9. Vì vậy 5.10 chỉ chạy được trọn vẹn sau 5.9 và sau khi nhánh `unseen-families-v2` có `msd.json`.

## Cách đọc kết quả (khai báo trước khi chạy)

| Kết quả | Cách diễn giải |
|---|---|
| TA hình học gần baseline (chưa kết luận) và TA kém hơn TA gốc hơn ngưỡng | Cải thiện của TA ở 5.9 đến từ phép quang học trùng họ, không phải từ augmentation nói chung |
| TA hình học hơn baseline vượt ngưỡng | Biến dạng hình học tự nó đã giúp. Phần của TA gốc phải trừ đi phần này trước khi gán cho phép quang học |
| TA hình học gần TA gốc | Phép quang học không phải nguồn chính, và TA ở 5.9 đã là đối chứng tổng quát khá sạch |
| Leave-group: `objective_group` kém suy giảm đầy đủ vượt ngưỡng, và gần baseline | Augmentation suy giảm không chuyển sang họ cùng cơ chế đã bị bỏ (đúng hướng Geirhos). Lợi thế của 4.5 phần lớn nằm ở chỗ nhắm đúng họ |
| Leave-group: `objective_group` kém suy giảm đầy đủ vượt ngưỡng nhưng vẫn hơn baseline vượt ngưỡng | Chuyển được một phần |
| Leave-group: `objective_group` gần suy giảm đầy đủ (chưa kết luận) | Các họ còn lại đã phủ được nhóm bị bỏ. Cơ chế nhóm theo MSD chưa đủ tách |
| Leave-group: `objective_rest` giảm | Bỏ nhóm làm yếu cả các họ còn lại, cần xét riêng |

## Không làm

- Không đổi mỏ neo, kích thước nhóm hay danh sách phép sau khi thấy kết quả. Thử thêm thì ghi là phân tích thăm dò.
- Không mở tập test.

## Cách chạy

`notebooks/15_retrain_controls.ipynb`, `STAGE = "5.10"`. Cần `reports/unseen_v2/msd.json` trên nhánh (merge nhánh `unseen-families-v2` trước, hoặc chép file vào) và kết quả 5.9 (thêm output phiên 5.9 làm input). Hai lần train, khoảng 1,5 giờ GPU T4. Dòng lệnh:

```
python -m src.training.run_control configs/controls/trivialaugment_geometric.yaml --seed 42
python -m src.training.run_control configs/controls/degradation_leave_group.yaml --seed 42
```

## Giới hạn

- Một seed cho mỗi nhánh. SD mượn từ nhánh tham chiếu, nên kết luận yếu hơn 5.9.
- Nhóm bỏ chọn bằng khoảng cách đặc trưng của Swin-Tiny reference trên cùng dữ liệu. Nhóm theo số vẫn có thể không trùng nhóm theo cơ chế vật lý.
- Nhóm gồm ba họ trong mười bốn, bỏ ít hơn thì họ còn lại che phủ nhiều hơn, bỏ nhiều hơn thì tập train suy giảm nghèo đi. Mức 3 là lựa chọn khai báo trước, không dò theo kết quả.
- TA hình học vẫn gồm Rotate, trùng một phần với phép xoay ±15° của baseline.
