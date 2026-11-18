# Task 5.9. Huấn luyện lại baseline, suy giảm và TrivialAugment theo một giao thức

## Câu hỏi

TN7 (task 5.6) so ba cách huấn luyện: baseline, TrivialAugment, augmentation suy giảm. Rà soát độ chặt tìm ra bốn chỗ làm so sánh đó chưa khách quan, và tách phần cải thiện thành 73% do augmentation nói chung và 27% do nhắm đúng họ chưa đứng vững:

| Chỗ lệch ở TN7 | Cách sửa ở 5.9 |
|---|---|
| Mỗi nhánh một seed, chênh 0,011 và 0,026 nằm sát dao động giữa các lần train | Ba seed (42, 43, 44) cho mỗi nhánh, kết luận theo độ lệch chuẩn seed |
| Ngân sách epoch lệch: baseline dừng ở epoch 12 (tốt nhất epoch 5), hai nhánh kia chạy 30 epoch. Geirhos (R32) cho thấy riêng việc train lâu hơn đã làm kết quả tốt lên | Mọi nhánh chạy đủ 30 epoch, `early_stopping: false` |
| Quy tắc chọn checkpoint lệch: baseline chọn trên validation sạch, hai nhánh kia chọn trên validation hỗn hợp chứa chính các họ đang đo | Mọi nhánh chọn trên cùng validation hỗn hợp (ảnh sạch cộng 14 họ ở mức 2) |
| Tỉ lệ tách 73/27 phụ thuộc thứ tự tách | Không báo tỉ lệ tách một chiều. Báo từng hiệu cặp kèm SD seed |

## Thiết kế

Giao thức chung khóa trong `configs/controls/protocol.yaml` và nạp cho mọi nhánh. File từng nhánh (`configs/controls/<nhánh>.yaml`) chỉ khai báo tên và augmentation, và code từ chối file nhánh đặt lại khóa nào của giao thức. Nhờ đó khác biệt duy nhất giữa các nhánh là augmentation lúc train.

| Nhánh | Augmentation thêm sau lật và xoay | Tên model |
|---|---|---|
| baseline | không | `swin_tiny_ctrl_baseline` |
| degradation | suy giảm ngẫu nhiên của benchmark 4.1, xác suất 0,5, 14 họ, giống task 4.5 | `swin_tiny_aug` |
| trivialaugment | TrivialAugment Wide, giống task 5.6 | `swin_tiny_trivialaugment` |

Model, optimizer (AdamW, lr 2e-4, weight decay 0,01), scheduler cosine, 30 epoch, batch 32, split và validation hỗn hợp giống nhau ở cả ba. Baseline đổi tên thành `swin_tiny_ctrl_baseline` để không lẫn với `swin_tiny` cũ (chọn trên validation sạch, dừng sớm), vì hai lần chạy đó khác giao thức.

### Số lần train

Chín lần chạy (ba nhánh nhân ba seed), trong đó **bảy lần train mới**. Seed 42 của suy giảm (4.5) và TrivialAugment (5.6) đã chạy đủ 30 epoch, dừng sớm với patience 7 không kích hoạt, nên quá trình train giống hệt khi tắt dừng sớm. `run_control.reusable_problems` kiểm từng điều kiện trước khi dùng lại: seed, model, hyperparameter, số epoch thực chạy, quy tắc chọn checkpoint và tham số augmentation. Lệch một điều kiện thì code báo lỗi và yêu cầu train lại, không âm thầm dùng. Baseline seed 42 cũ không dùng lại được (dừng ở epoch 12, chọn trên validation sạch) nên train lại cả ba seed.

Bảng điều kiện của hai lần chạy cũ lấy từ `reports/augmentation/augmentation_val_conditions.csv` và `reports/generic_augmentation/generic_augmentation_val_conditions.csv`, không cần checkpoint cũ. Checkpoint seed 42 chỉ cần ở 5.11, khi đo lại trên bộ họ mới.

## Đánh giá

Mỗi lần train kết thúc bằng một bảng điểm ảnh sạch cộng 42 điều kiện đã thấy trên validation (444 ảnh). Thước đo chính của 5.9 là **macro F1 trung bình 42 điều kiện** (`objective_42`), cùng thước đo với 4.5 và 5.6.

Ba hiệu khai báo trước: TrivialAugment trừ baseline, suy giảm trừ baseline, suy giảm trừ TrivialAugment.

Quy tắc kết luận: lấy hiệu trung bình ba seed. Chỉ kết luận khi |hiệu| lớn hơn **2,5 lần SD seed** (SD mẫu, hai nhánh gộp bằng căn bậc hai của trung bình hai phương sai). Nhỏ hơn thì ghi "chưa kết luận", không đọc thành "bằng nhau". Với ba seed SD tự nó rất nhiễu, nên ngưỡng này là mức tối thiểu để dám nói, không phải kiểm định thống kê. Chênh 0,011 như ở TN7 nhiều khả năng vẫn chưa kết luận được.

## Cách đọc kết quả (khai báo trước khi chạy)

| Kết quả trên `objective_42` | Cách diễn giải |
|---|---|
| TrivialAugment hơn baseline (vượt) và suy giảm hơn TrivialAugment (vượt) | Cả augmentation nói chung lẫn nhắm đúng họ đều đóng góp. Phần của TrivialAugment còn phải xét thêm ở 5.10, vì nó có phép quang học trùng họ |
| TrivialAugment hơn baseline (vượt), suy giảm hơn TrivialAugment (chưa kết luận) | Chưa tách được hai nguồn. Hai cách tương đương trong giới hạn của ba seed |
| TrivialAugment không hơn baseline (chưa kết luận), suy giảm hơn baseline (vượt) | Cải thiện đến từ việc nhắm đúng họ |
| Suy giảm không hơn baseline | Không có bằng chứng augmentation suy giảm giúp khi giao thức đã cân bằng. Kết luận của 4.5 phải xem lại |
| Ảnh sạch giảm so với baseline | Báo như một đánh đổi cùng kết quả bền vững |

Mỗi kết luận nêu rõ: ba seed, validation, và `objective_42` là điểm đo trên các họ có mặt trong validation chọn checkpoint của cả ba nhánh, nên không phải điểm báo cáo cuối.

## Không làm

- **Nhánh "TrivialAugment với p=1"** bỏ khỏi thiết kế. Mintun (R33) cho thấy chỉ cần một phần nhỏ mẫu giống corruption là đủ.
- **Không chỉnh mức, nhóm họ hay cấu hình sau khi thấy kết quả.** Chỉ sửa khi gặp lỗi hạ tầng hoặc lỗi mã. Muốn thử thêm thì ghi rõ là phân tích thăm dò.
- **Không mở tập test.** Test đã mở 6 lần. Đo trên test thuộc P6.1.

## Cách chạy

1. **Chạy thử một lần** (baseline seed 42, `STAGE = "smoke"` ở notebook) để bắt lỗi cấu hình trước khi tốn giờ GPU.
2. Chạy `STAGE = "5.9"`. Mỗi lần train khoảng 40 đến 45 phút, bảy lần train khoảng 5 đến 6 giờ GPU T4. Chia theo seed qua `SEEDS` nếu cần nhiều phiên.
3. Notebook: `notebooks/15_retrain_controls.ipynb`. GPU T4, Internet On, input `cthng123/durian-leaf-disease-dataset`. Muốn tiếp tục từ phiên trước, thêm output của phiên đó làm input.
4. Dòng lệnh: `python -m src.training.run_control configs/controls/baseline.yaml --seed 42`.
5. Sau khi xong, chép `reports/controls/` từ file nén, commit, rồi lưu phiên (Save Version) và đưa checkpoint thành dataset Kaggle cho 5.11.

Code: `src/training/run_control.py`, `src/evaluation/seed_summary.py`. Test: `tests/test_retrain_controls.py`, `tests/test_training.py`.

## Giới hạn

- Ba seed cho ước lượng SD thô. Kết luận "vượt" là mức tối thiểu, "chưa kết luận" không có nghĩa là bằng nhau.
- Validation hỗn hợp chứa chính 14 họ đang đo, dùng chung cho cả ba nhánh để cân bằng quy tắc chọn. Điều đó làm điểm tuyệt đối lạc quan, nhưng không thiên vị riêng nhánh nào ngoài việc nhánh suy giảm được train đúng các họ này.
- Một kiến trúc (Swin-Tiny fine-tune từ ImageNet). Geirhos train từ đầu ResNet-50, nên không chuyển nguyên kết quả của họ sang đây.
- Seed 42 của hai nhánh suy giảm và TrivialAugment được dùng lại từ lần chạy cũ, đã qua kiểm giao thức nhưng dùng `num_workers` và phần cứng của lần chạy đó. Khác biệt chỉ có thể nhỏ, vì thứ tự dữ liệu cố định theo seed.
