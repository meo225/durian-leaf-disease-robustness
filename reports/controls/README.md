# Kết quả đối chứng huấn luyện lại, tasks 5.9 và 5.10

Fine-tune Swin-Tiny với các kiểu augmentation khác nhau theo **một giao thức chung** (`configs/controls/protocol.yaml`): 30 epoch, không dừng sớm, chọn checkpoint bằng macro F1 trên validation hỗn hợp (ảnh sạch cộng 14 họ ở mức 2), cùng optimizer, seed 42, 43, 44. Thiết kế và bảng cách đọc kết quả khai báo trước nằm ở `docs/task-5.9-retrain-controls.md` và `docs/task-5.10-ta-geometric-and-leave-group.md`.

Mọi số đo trên **validation** (444 ảnh). **Tập test chưa mở.** Đây là kết quả trung gian, đo cuối trên bộ họ mới và test thuộc 5.11 và P6.1.

| Nội dung | Vị trí |
|---|---|
| Mỗi lần chạy: chỉ số train, recall từng lớp, ma trận nhầm, lịch sử | `swin_tiny_*_seed*.json`, `*_confusion_matrix.png`, `*_history.png` |
| Macro F1 theo 42 điều kiện của từng lần chạy | `swin_tiny_*_val_conditions.csv` |
| Bảng từng lần chạy (5.9 và 5.10) | `runs_5_9.csv`, `runs_5_9_5_10.csv` |
| SD seed và so sánh ba nhánh (5.9) | `seed_stats_5_9.csv`, `comparison_5_9.csv` |
| So sánh hai nhánh 5.10 với nhánh tham chiếu | `comparison_5_10.csv` |
| Notebook đã chạy trên Kaggle, có log | `logs/15_retrain_controls_stage_5.9.ipynb` (chạy ở commit `b0a589a`), `logs/15_retrain_controls_stage_5.10.ipynb` (commit `fe0d333`) |

Seed 42 của nhánh suy giảm và TrivialAugment dùng lại từ lần chạy cũ (`source = reused` trong `runs_5_9.csv`). Code kiểm hyperparameter, số epoch, quy tắc chọn checkpoint và augmentation trước khi cho dùng lại, nên không có file mới cho hai lần chạy này.

Thước đo chính `objective_42` là macro F1 trung bình 42 điều kiện đã thấy (14 họ, 3 mức). Quy tắc kết luận khai báo trước: chỉ kết luận khi chênh lệch lớn hơn 2,5 lần SD seed.

## Kết luận ngắn

1. **Baseline đúng giao thức cao hơn baseline cũ.** Ba seed cho 0.8908 ± 0.0043, trong khi baseline dừng sớm của v1 là 0.8607. Chênh khoảng 0.030 đến từ việc bỏ dừng sớm ở epoch 12 và chọn checkpoint trên validation hỗn hợp thay vì validation sạch.
2. **Cả hai cách huấn luyện lại đều hơn baseline vượt ngưỡng, và suy giảm hơn TrivialAugment vượt ngưỡng.** Suy giảm 0.9607, TrivialAugment 0.9349, baseline 0.8908. TrivialAugment lấy được 63% mức tăng của suy giảm (v1 ghi 73%).
3. **TrivialAugment chỉ giữ phép hình học không cải thiện so với baseline.** 0.8827, thấp hơn baseline 0.0081, chưa kết luận (1,87 lần SD). Theo bảng khai báo trước, phần cải thiện của TrivialAugment gốc nhiều khả năng đến từ các phép quang học trùng họ hỏng của benchmark, không phải từ augmentation nói chung. Chỉ một seed nên kết luận này yếu hơn 5.9.
4. **Bỏ nhóm họ blur, lowres, motion_blur khỏi augmentation suy giảm: chuyển được một phần.** Trên đúng nhóm bị bỏ, điểm là 0.8917, kém suy giảm đầy đủ (0.9625) vượt ngưỡng nhưng hơn baseline (0.8349) vượt ngưỡng. Lấy lại khoảng 45% khoảng cách giữa baseline và suy giảm đầy đủ trên nhóm đó. Trên các họ còn lại điểm không đổi (0.9594 so với 0.9602).

## Task 5.9: ba nhánh, ba seed

| Nhánh | Số seed | objective_42 | SD seed | Validation sạch |
|---|---|---|---|---|
| baseline (lật, xoay) | 3 | 0.8908 | 0.0043 | 0.9686 |
| TrivialAugment Wide | 3 | 0.9349 | 0.0029 | 0.9791 |
| Suy giảm (4.5) | 3 | 0.9607 | 0.0019 | 0.9761 |

| So sánh | Hiệu | SD gộp | Số lần SD | Kết luận |
|---|---|---|---|---|
| TrivialAugment trừ baseline | +0.0441 | 0.0037 | 12,1 | vượt |
| Suy giảm trừ baseline | +0.0699 | 0.0033 | 20,9 | vượt |
| Suy giảm trừ TrivialAugment | +0.0258 | 0.0024 | 10,6 | vượt |

Tỉ lệ: TrivialAugment chiếm 0.0441 trên 0.0699, tức 63%, và phần còn lại 37%. So với v1 (73% và 27%), tỉ lệ này tính trên baseline đúng giao thức và ba seed. Không chênh lệch nào nằm trong vùng nhiễu seed.

## Task 5.10: hai nhánh, một seed

SD lấy từ nhánh tham chiếu ba seed của 5.9, vì mỗi nhánh 5.10 chỉ có một seed.

| Nhánh | objective_42 | Trên nhóm bị bỏ | Trên các họ còn lại | Validation sạch |
|---|---|---|---|---|
| baseline (3 seed) | 0.8908 | 0.8349 | 0.9060 | 0.9686 |
| TrivialAugment Wide (3 seed) | 0.9349 | 0.9201 | 0.9389 | 0.9791 |
| Suy giảm đầy đủ (3 seed) | 0.9607 | 0.9625 | 0.9602 | 0.9761 |
| **TrivialAugment hình học** (seed 42) | 0.8827 | 0.8228 | 0.8990 | 0.9787 |
| **Suy giảm bỏ nhóm** (seed 42) | 0.9449 | 0.8917 | 0.9594 | 0.9766 |

Nhóm bị bỏ gồm `blur`, `lowres`, `motion_blur`, chọn bằng khoảng cách MSD từ họ mỏ neo `blur` (khoảng cách tới mỏ neo 0.947 và 0.951). Khoảng cách trung bình trong nhóm là 1.158, ra ngoài nhóm 4.374, và họ gần nhất ngoài nhóm cách `blur` 3.126, nên nhóm tách rõ.

| Nhánh | So với | Thước đo | Hiệu | Số lần SD | Kết luận |
|---|---|---|---|---|---|
| TrivialAugment hình học | baseline | objective_42 | -0.0081 | 1,9 | chưa kết luận |
| TrivialAugment hình học | TrivialAugment gốc | objective_42 | -0.0522 | 18,3 | vượt |
| Suy giảm bỏ nhóm | baseline | trên nhóm bị bỏ | +0.0568 | 15,1 | vượt |
| Suy giảm bỏ nhóm | suy giảm đầy đủ | trên nhóm bị bỏ | -0.0708 | 30,9 | vượt |
| Suy giảm bỏ nhóm | TrivialAugment gốc | trên nhóm bị bỏ | -0.0284 | 2,48 | chưa kết luận |
| Suy giảm bỏ nhóm | suy giảm đầy đủ | trên các họ còn lại | -0.0007 | 0,4 | chưa kết luận |

Các dòng còn lại (objective_42 của nhánh bỏ nhóm so với ba nhánh tham chiếu) có trong `comparison_5_10.csv`.

### Đọc theo bảng khai báo trước

| Dòng của bảng | Có xảy ra | Diễn giải |
|---|---|---|
| TA hình học gần baseline (chưa kết luận) và kém TA gốc vượt ngưỡng | Có | Cải thiện của TA ở 5.9 đến từ phép quang học trùng họ, không phải từ augmentation nói chung |
| TA hình học hơn baseline vượt ngưỡng | Không | |
| Nhóm bị bỏ: kém suy giảm đầy đủ vượt ngưỡng và gần baseline | Không | |
| Nhóm bị bỏ: kém suy giảm đầy đủ vượt ngưỡng nhưng vẫn hơn baseline vượt ngưỡng | Có | Chuyển được một phần |
| Nhóm bị bỏ gần suy giảm đầy đủ | Không | |
| Các họ còn lại giảm | Không | Bỏ nhóm không làm yếu các họ còn lại |

## Điều cần thận trọng

- **Một seed ở hai nhánh 5.10.** SD mượn từ nhánh tham chiếu. Dòng so nhóm bị bỏ với TrivialAugment gốc ở 2,48 lần SD, sát ngưỡng 2,5, nên không đọc thành khác biệt hay giống nhau.
- **Điểm là validation, và checkpoint chọn trên cùng ảnh.** Validation hỗn hợp dùng mức 2 của chính 14 họ đang đo, nên `objective_42` hơi lạc quan cho mọi nhánh. So giữa các nhánh vẫn công bằng vì cùng quy tắc, nhưng số tuyệt đối chưa phải số báo cáo cuối. Nhánh bỏ nhóm chọn checkpoint trên validation đã bỏ ba họ đó (`val_families = 11`), nên `mixed_val_f1` của nó không so trực tiếp với nhánh khác.
- **`objective_42` của nhánh bỏ nhóm vẫn gồm cả ba họ bị bỏ** (9 trong 42 điều kiện), vì cần cùng thước đo với các nhánh khác. Điểm riêng của nhóm bị bỏ nằm ở cột `objective_group`.
- **Thăm dò, không khai báo trước:** TrivialAugment hình học thấp hơn baseline 0.0121 trên nhóm blur (khoảng 3,2 lần SD của baseline), cho thấy phép hình học có thể không giúp, thậm chí hơi hại, độ bền với làm mờ. Chỉ một seed.
- **Số trên slide cũ chưa cập nhật.** Baseline 0.8607 trong 5.3, 5.4 và bộ chọn là bản dừng sớm của v1. Bộ chọn 0.8802 thấp hơn baseline huấn luyện lại đúng giao thức (0.8908), nhưng hai số này áp lên hai checkpoint khác nhau (bộ chọn xử lý ảnh trên Swin-Tiny đóng băng, baseline mới là model huấn luyện lại). Mức tăng +0.0195 của bộ chọn cần đo lại trên checkpoint baseline mới trước khi đưa vào báo cáo cuối.
