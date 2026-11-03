# Kết quả đối chứng augmentation tổng quát, task 5.6

Task này fine-tune Swin-Tiny (reference model, seed 42) chỉ với TrivialAugment, không có phép làm hỏng nhắm vào benchmark, để tách phần cải thiện do augmentation nói chung khỏi phần do augmentation nhắm đúng các họ suy giảm của task 4.5. Thiết kế và bảng diễn giải đã đăng ký trước nằm ở `docs/task-5.6-generic-augmentation-control.md`.

Mọi số liệu đo trên **tập validation**, 444 ảnh, một seed. Tập test chưa mở. Khoảng tin cậy 95% lấy từ paired bootstrap 1000 lần, dùng chung chỉ số ảnh cho mọi cách. Kiểm định từng điều kiện là McNemar chính xác, hiệu chỉnh Holm.

| Nội dung | Vị trí |
|---|---|
| Notebook trong repo, có log | `notebooks/12_generic_augmentation_control.ipynb`, chạy trên Kaggle (GPU Tesla T4) từ nhánh `generic-augmentation-control` |
| Cấu hình | `configs/train_swin_tiny_trivialaugment.yaml` |
| Chỉ số train và recall từng lớp | `swin_tiny_trivialaugment_seed42.json` |
| Điểm từng điều kiện | `generic_augmentation_val_conditions.csv`, `generic_augmentation_unseen_val_conditions.csv` |

Model train 30 epoch, checkpoint tốt nhất ở epoch 26 (2603 giây). Macro F1 trên validation hỗn hợp dùng để chọn checkpoint là 0.948. Quy tắc chọn giống task 4.5 nên chỉ khác ở augmentation lúc train. Reference chọn checkpoint trên validation sạch, đây là khác biệt đã ghi ở tài liệu thiết kế.

## Kết luận ngắn

1. **TrivialAugment tự nó đã cải thiện rõ.** Trên 42 điều kiện đã thấy, macro F1 trung bình tăng từ 0.861 lên 0.932 (+0.072, khoảng 0.059 đến 0.086). 33/42 điều kiện tốt hơn reference có ý nghĩa sau Holm, không điều kiện nào kém hơn.
2. **Nhắm đúng họ vẫn cho thêm một phần có ý nghĩa.** Augmentation suy giảm đạt 0.958 (+0.098), hơn TrivialAugment +0.026 (khoảng 0.017 đến 0.036). Theo hiệu so với reference, khoảng 73% mức cải thiện của task 4.5 trên validation đạt được bằng augmentation tổng quát, 27% còn lại đến từ việc nhắm đúng họ.
3. **Phần thêm tập trung ở vài họ.** Lợi thế của augmentation suy giảm lớn nhất ở jpeg (+0.102), blur (+0.069), lowres (+0.050), lowlight (+0.035) và motion_blur (+0.033). Ở contrast_low, glare, brightness_bright và contrast_high hai cách gần như bằng nhau, vì TrivialAugment có sẵn phép đổi độ sáng và độ tương phản.
4. **Họ chưa thấy: lợi thế thu hẹp còn khoảng 0.011.** TrivialAugment đạt 0.921, augmentation suy giảm 0.932, hiệu +0.011 (khoảng gần 0.001 đến 0.021, sát 0). Phần lợi thế dồn vào hai họ gần blur và noise, còn ở họ xa thì TrivialAugment bằng hoặc hơn.
5. **Không thấy chi phí trên ảnh sạch.** Macro F1 ảnh sạch là 0.966 (reference), 0.979 (TrivialAugment), 0.971 (suy giảm).
6. **rain_streak là ngoại lệ chung.** Cả hai cách retrain đều thấp hơn reference ở họ này (0.870 và 0.856 so với 0.901).

## Họ đã thấy, 42 điều kiện

| Cách | Ảnh sạch | Trung bình 42 điều kiện | Khoảng 95% | Hiệu so với reference | Khoảng của hiệu | Mức 1 | Mức 2 | Mức 3 |
|---|---|---|---|---|---|---|---|---|
| reference | 0.966 | 0.861 | 0.841 đến 0.876 | | | 0.930 | 0.878 | 0.774 |
| TrivialAugment | 0.979 | 0.932 | 0.919 đến 0.944 | +0.072 | 0.059 đến 0.086 | 0.965 | 0.946 | 0.886 |
| Augmentation suy giảm (4.5) | 0.971 | 0.958 | 0.945 đến 0.970 | +0.098 | 0.085 đến 0.114 | 0.970 | 0.964 | 0.941 |

Tách đóng góp, tất cả có ý nghĩa theo khoảng tin cậy khác 0:

| So sánh | Hiệu | Khoảng 95% | Tỉ lệ so với tổng cải thiện |
|---|---|---|---|
| TrivialAugment trừ reference | +0.072 | 0.059 đến 0.086 | 73% |
| Suy giảm trừ TrivialAugment | +0.026 | 0.017 đến 0.036 | 27% |
| Suy giảm trừ reference | +0.098 | 0.085 đến 0.114 | 100% |

Theo từng điều kiện, augmentation suy giảm hơn TrivialAugment có ý nghĩa ở 7/42 điều kiện và không điều kiện nào kém có ý nghĩa, dù 10/42 điều kiện có hiệu âm nhỏ. Bảng đầy đủ ở `seen_conditions_*.csv`, bản đồ nhiệt ở `seen_delta_ta_vs_ref.png` và `seen_delta_aug_vs_ta.png`.

Ảnh sạch đổi dự đoán so với reference ở 16 ảnh (TrivialAugment: 10 sai thành đúng, 4 đúng thành sai) và 15 ảnh (suy giảm: 8 và 5). Chênh lệch này nhỏ, không đủ để nói có chi phí hay lợi ích.

## Họ chưa thấy, 18 điều kiện

| Cách | Trung bình 18 điều kiện | Khoảng 95% | Hiệu so với reference | Khoảng của hiệu | Mức 3 |
|---|---|---|---|---|---|
| reference | 0.875 | 0.855 đến 0.891 | | | 0.809 |
| TrivialAugment | 0.921 | 0.907 đến 0.934 | +0.046 | 0.031 đến 0.062 | 0.880 |
| Augmentation suy giảm (4.5) | 0.932 | 0.917 đến 0.945 | +0.057 | 0.043 đến 0.071 | 0.900 |

Macro F1 theo họ, trung bình ba mức:

| Họ | Nhóm | reference | TrivialAugment | Suy giảm | Cách tốt hơn |
|---|---|---|---|---|---|
| defocus_blur | gần | 0.834 | 0.911 | 0.965 | suy giảm |
| chroma_noise | gần | 0.802 | 0.849 | 0.900 | suy giảm |
| fog | xa | 0.927 | 0.973 | 0.968 | bằng nhau |
| spatter | xa | 0.932 | 0.968 | 0.973 | bằng nhau |
| rain_streak | xa | 0.901 | 0.870 | 0.856 | reference |
| saturate | xa | 0.853 | 0.955 | 0.929 | TrivialAugment |

Hiệu giữa augmentation suy giảm và TrivialAugment là +0.054 và +0.051 ở hai họ gần, và từ -0.026 đến +0.005 ở bốn họ xa. Theo điều kiện, TrivialAugment hơn reference có ý nghĩa ở 9/18 và kém hơn ở 1/18 (rain_streak). Augmentation suy giảm hơn TrivialAugment có ý nghĩa ở 3/18 và kém hơn ở 1/18. Bản đồ nhiệt ở `unseen_delta_*.png`.

Recall từng lớp của TrivialAugment ở mức 3 (`unseen_trivialaugment_recall_level3.csv`) vẫn còn các điểm yếu: chroma_noise làm HEALTHY_LEAF còn 0.52, defocus_blur làm ALLOCARIDARA_ATTACK còn 0.48, rain_streak làm PHOMOPSIS_LEAF_SPOT còn 0.65.

## Diễn giải theo bảng đã đăng ký

Kết quả rơi vào dòng "TrivialAugment hơn reference nhưng kém suy giảm rõ rệt" của bảng đã đăng ký: augmentation nói chung đóng góp phần lớn, nhắm đúng họ đóng góp thêm một phần. Trên họ chưa thấy, suy giảm vẫn hơn TrivialAugment nhưng chênh lệch nhỏ và sát 0 về khoảng tin cậy, nên chỉ nói được rằng lợi thế nhắm đúng họ có chuyển sang họ lạ ở mức yếu, và chủ yếu ở họ gần.

Hai hệ quả cho các kết luận trước:

- Con số +0.096 của task 4.5 trên test không thể gán hết cho việc nhắm đúng họ. Trên validation, khoảng ba phần tư mức tăng đạt được bằng augmentation không nhắm họ nào.
- Mức tăng của bộ chọn thích ứng (+0.0195 trên validation, task 5.1 đến 5.3) nhỏ hơn đáng kể so với bất kỳ cách huấn luyện lại nào ở đây. Hai hướng khác nhau về chi phí (bộ chọn không cần huấn luyện lại), nên chúng bổ sung chứ không thay thế nhau, nhưng so sánh trực tiếp phải nêu rõ sự khác biệt này.

## Hạn chế

- Một seed, một lần train mỗi cách. Chênh lệch dưới khoảng 0.01 không đủ để kết luận. Seed 43 và 44 để dành cho task 6.1.
- Chỉ đo trên validation. Cả ba cách đều có checkpoint chọn trên validation, và tập validation đã dùng để khóa mức của họ chưa thấy, nên số tuyệt đối có thể lạc quan hơn trên test.
- TrivialAugment có các phép đổi màu, độ sáng và độ tương phản, trùng một phần với họ brightness, contrast, color_cast và với saturate của bộ chưa thấy. Một phần lợi thế ở các họ này là sự trùng khớp phép biến đổi, không phải khả năng khái quát thật.
- Reference chọn checkpoint trên validation sạch, hai cách retrain chọn trên validation hỗn hợp. Một phần nhỏ chênh lệch với reference có thể đến từ quy tắc chọn.
- Biến thể tùy chọn (cùng kỹ thuật xử lý cho cả ảnh train và ảnh test) không thực hiện, theo quy định cắt đầu tiên của plan.

## Tệp trong thư mục

| Tệp | Nội dung |
|---|---|
| `seen_summary_val.csv`, `seen_decomposition_val.csv`, `seen_clean_val.csv`, `seen_family_val.csv` | Bảng chính họ đã thấy |
| `unseen_summary_val.csv`, `unseen_family_val.csv`, `unseen_trivialaugment_recall_level3.csv` | Bảng chính họ chưa thấy |
| `*_conditions_*.csv` | Hiệu, số ảnh đúng và sai, p và p Holm theo điều kiện |
| `seen_objective.png`, `seen_levels.png`, `seen_family.png`, `seen_scatter.png` | Hình họ đã thấy |
| `unseen_objective.png`, `unseen_levels.png`, `unseen_family.png`, `unseen_family_level3.png` | Hình họ chưa thấy |
| `trivialaugment_examples.png`, `swin_tiny_trivialaugment_seed42_history.png`, `..._confusion_matrix.png` | Ví dụ augmentation, lịch sử train, ma trận nhầm lẫn |
