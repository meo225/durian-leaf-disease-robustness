# Task 5.7. Khoảng cách MSD giữa họ suy giảm và augmentation

Nhánh `unseen-families-v2-5.5-5.7-5.8`. Mã ở `src/evaluation/msd.py`, `msd_plots.py`, `ta_overlap.py`, `intervals.py`. Notebook `14_msd_family_distance.ipynb`. Chỉ suy luận, không mở test.

## Câu hỏi

"Gần" và "xa" ở bộ v1 là chia theo tên. Câu hỏi của 5.7: một họ suy giảm xa hay gần với thứ mô hình đã được huấn luyện cùng, **đo bằng số trên mô hình**, và TrivialAugment (đối chứng 5.6) thực sự phủ những họ nào.

## Phương pháp

Theo Mintun, Kirillov, Xie (NeurIPS 2021, tài liệu R33), phần 3 và phụ lục.

- Đặc trưng chuyển đổi của phép `t` trên mô hình `f̂`: `f(t) = E_x[f̂(t(x)) − f̂(x)]`, `f̂` là lớp ẩn cuối trước đầu phân loại (`forward_head(..., pre_logits=True)` của timm).
- Mỗi mẫu augmentation là một cặp (phép, cường độ) cố định áp lên cùng K ảnh. Đặc trưng của một họ suy giảm là trung bình đặc trưng ba mức của nó.
- `d_MSD(A, C) = min_a ‖f(a) − f(C)‖₂` trên các mẫu `a` của `A`.
- Mô hình là Swin-Tiny reference, huấn luyện chỉ với lật và xoay, đúng điều kiện bài báo yêu cầu (mô hình chưa học augmentation nào).

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Ảnh | 100 ảnh validation, hoán vị seed 42 | Bài báo cũng dùng 100 ảnh |
| Mẫu mỗi bộ | 1000 | Bài báo dùng tới hàng chục nghìn. Giảm vì mỗi mẫu là 100 lần suy luận và 100 lần làm hỏng ảnh trên CPU |
| Bộ augmentation | TrivialAugment Wide (31 mức); augmentation suy giảm của 4.5 (42 điều kiện, jitter ±10%, đúng `jitter_value` dùng khi train) | Bỏ phép Identity: đặc trưng 0 làm họ nhẹ nào cũng "gần" |
| Khoảng tin cậy | Bootstrap 200 lần, lấy lại mẫu theo ảnh, cùng chỉ số ảnh cho mọi mẫu | Phản ánh sai số do chỉ có 100 ảnh |

Lấy mẫu TA bám đúng `TrivialAugmentWide.forward` của torchvision (phép đều, cường độ đều trong lưới 31, dấu đều) và gọi cùng `_apply_op` nội suy NEAREST, fill mặc định. Chạy trên ảnh uint8 dạng tensor, khác đường PIL lúc train tối đa 1 mức xám ở Color và Contrast (đã đo). `_augmentation_space` và `_apply_op` là API nội bộ của torchvision, nên có test khóa tên 14 phép.

## Nhóm gần/xa

Ngưỡng `τ` = MSD leave-one-out lớn nhất trong 14 họ đã thấy: mỗi họ đã thấy đo tới augmentation suy giảm đã bỏ chính họ đó. Họ mới `near` nếu `MSD ≤ τ`, `far` nếu lớn hơn, `ambiguous` nếu khoảng bootstrap của nó chứa `τ`.

Quy tắc này nhạy với họ ngoại lai trong 14 họ (một họ rất lạ kéo `τ` lên). Vì vậy mọi MSD liên tục được báo cùng, biểu đồ `msd_to_degradation_aug.png` vẽ cả ngưỡng và khoảng tin cậy, và người đọc không phải chấp nhận nhãn nhị phân. `τ` là quy tắc đăng ký trước, không chỉnh sau khi thấy phân bố.

## Các kết quả phụ

- `msd_v1_families.csv`: bộ v1 đo cùng thước, so nhóm theo tên với nhóm theo số.
- `msd_seen_to_trivialaugment.csv`: MSD của 14 họ đã thấy tới TrivialAugment, cho biết TA phủ họ nào.
- `ta_overlap.py`: đếm xác suất một lần gọi TA cho hệ số trong ±10% quanh mức của họ cùng quy ước hệ số (`brightness`, `contrast`, `saturate`). Là cận dưới của chồng lấn, chưa tính Sharpness với mờ hay Posterize với jpeg.
- `seen_pairwise_distance.csv` và `.png`: ma trận khoảng cách giữa 14 họ đã thấy. Task 5.10 dùng nó để chọn nhóm họ bỏ theo số thay vì theo tên.
- `intervals.spearman_with_ci`: Spearman kèm khoảng bootstrap. Với 14 đến 20 họ khoảng rất rộng và phải báo như vậy. Dùng ở 5.11, khi đã có số của các hệ thống huấn luyện lại, không dùng ở 5.7.

## Khác bài báo

- Một mô hình, một tập 5 lớp, ảnh lá. Bài báo xét ImageNet với nhiều mô hình.
- 1000 mẫu thay vì 100 000. MSD là cực tiểu nên ít mẫu cho giá trị lớn hơn thật, thiên về "xa". Điều này làm `τ` và MSD của họ mới cùng lệch về một phía, nhưng độ lệch không bằng nhau, nên chưa thể bảo đảm nhãn gần/xa không đổi khi tăng mẫu. Nếu có thời gian, chạy lại với `n_samples` lớn hơn như phân tích thăm dò và so nhãn.
- Bài báo báo Spearman giữa MSD và lỗi của mô hình trên ImageNet-C. Ở đây chưa kiểm chứng MSD dự đoán được lỗi trên bộ lá sầu riêng, nên nhãn gần/xa chỉ là thước đo khoảng cách đặc trưng, chưa phải bằng chứng về độ khó.

## Kết quả

Đã chạy trên Kaggle với Swin-Tiny seed 42, 100 ảnh validation, 1000 mẫu, bootstrap 200 lần. Hai bộ augmentation mất 903 giây và 643 giây. Ngưỡng gần xa là 3.14 do jpeg quyết định. Chín họ của bộ v2 chia thành 4 gần, 3 xa, 2 chưa rõ. Số liệu, bảng độ nhạy ngưỡng và so sánh nhãn v1 ở `reports/unseen_v2/README.md`.

Bảng `msd_threshold_sensitivity.csv` do `src/evaluation/v2_summary.py` tạo từ file đã commit. Đây là phân tích thăm dò, nhãn đã khóa không đổi.

Spearman giữa MSD và kết quả hệ thống chưa tính vì chưa có kết quả hệ thống, chờ task 5.11.
