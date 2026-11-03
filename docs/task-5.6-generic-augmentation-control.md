# Task 5.6. Đối chứng augmentation tổng quát

## Câu hỏi

Ở task 4.5, huấn luyện lại Swin-Tiny bằng augmentation suy giảm nâng macro F1 trung bình 42 điều kiện từ 0,869 lên 0,965 trên test. Con số này chưa trả lời được một câu hỏi: phần cải thiện đến từ việc nhắm đúng 14 họ suy giảm của benchmark, hay chỉ đến từ việc có thêm augmentation nói chung. Task 5.6 đo riêng phần thứ hai bằng TrivialAugment, một augmentation tổng quát không có phép nào mô phỏng cơ chế suy giảm của benchmark.

## Thiết kế

| Hạng mục | Reference | Task 4.5 (suy giảm) | Task 5.6 (TrivialAugment) |
|---|---|---|---|
| Model, optimizer, scheduler, epoch, patience, seed, split | giống nhau | giống nhau | giống nhau |
| Phép hình học (lật, xoay) | có | có | có |
| Augmentation thêm | không | suy giảm ngẫu nhiên, xác suất 0,5, 14 họ | TrivialAugment bản wide, mỗi ảnh một phép |
| Tập chọn checkpoint | validation sạch | validation hỗn hợp, mức 2 | validation hỗn hợp, mức 2 |

Hai điểm cần nói rõ:

- Quy tắc chọn checkpoint giữ giống task 4.5 để khác biệt còn lại chỉ nằm ở augmentation lúc train. Hệ quả là reference chọn bằng validation sạch còn hai model sau chọn bằng validation hỗn hợp, nên so với reference có một phần lợi thế do quy tắc chọn. So giữa TrivialAugment và suy giảm thì quy tắc chọn như nhau.
- TrivialAugment gồm các phép màu, độ sáng, tương phản và độ nét. Một số phép trùng vùng giá trị với vài họ của benchmark (`brightness_dark`, `contrast_low`, `blur`). Vì vậy TrivialAugment không hoàn toàn tách rời khỏi 14 họ, và kết quả là cận trên cho phần augmentation nói chung khi đo trên các họ này. Họ chưa thấy của task 5.5 không bị ảnh hưởng bởi trùng lặp này nhiều bằng, nên phần đó có sức nặng hơn.

## Đánh giá

Chỉ dùng validation (444 ảnh), ghép cặp theo ảnh với reference và với model task 4.5:

1. 42 điều kiện đã thấy của benchmark 4.1.
2. 18 điều kiện của 6 họ chưa thấy ở task 5.5 (cần checkpoint reference và model 4.5 để so ghép cặp, nếu không có thì chỉ so điểm).

Thống kê: khoảng tin cậy bootstrap ghép cặp 95% (1.000 lần, cùng chỉ số ảnh cho mọi cách) cho mục tiêu trung bình và cho hiệu từng cặp. McNemar chính xác từng điều kiện, hiệu chỉnh Holm trên toàn bộ điều kiện của mỗi nhóm. Kết luận từng cặp chỉ gồm ba loại: tốt hơn, kém hơn, chưa phân biệt được, theo vị trí khoảng tin cậy của hiệu so với 0.

Tập test không mở. Việc đo trên test, cùng seed 43 và 44, thuộc P6.1 với cấu hình khóa.

## Cách đọc kết quả (khai báo trước khi chạy)

| Kết quả | Cách diễn giải |
|---|---|
| TrivialAugment gần bằng suy giảm trên họ đã thấy | Phần lớn cải thiện của 4.5 là hiệu ứng của augmentation nói chung, hoặc do trùng lặp phép màu và độ sáng. Cần xét tách theo họ trước khi kết luận |
| TrivialAugment hơn reference nhưng kém suy giảm rõ rệt | Augmentation nói chung đóng góp một phần, phần còn lại do nhắm đúng họ. Tỉ lệ phần đóng góp lấy từ bảng tách |
| TrivialAugment không hơn reference | Augmentation nói chung không giúp. Cải thiện của 4.5 là do nhắm đúng họ |
| TrivialAugment làm giảm ảnh sạch đáng kể | Có đánh đổi, báo cáo cùng kết quả bền vững |
| Họ chưa thấy: suy giảm hơn TrivialAugment | Nhắm đúng họ vẫn khái quát được sang họ lạ |
| Họ chưa thấy: TrivialAugment bằng hoặc hơn suy giảm | Lợi thế của 4.5 chỉ đúng trong phân phối, không chuyển sang họ lạ |

Mỗi kết luận nêu rõ là một seed, trên validation.

## Phần tùy chọn

Biến thể áp cùng kỹ thuật xử lý cho cả ảnh train và ảnh test rồi fine-tune, để kiểm tra nhận định của R21 rằng xử lý ảnh không thêm thông tin, **không thực hiện** ở task này theo quy định cắt đầu tiên của plan khi thiếu thời gian. Phần này đòi hỏi một lần train cho mỗi kỹ thuật và không ảnh hưởng đến câu hỏi chính ở trên.

## Cách chạy

- Notebook: `notebooks/12_generic_augmentation_control.ipynb`. GPU T4, Internet On, input `cthng123/durian-leaf-disease-dataset`. Thêm dataset chứa `swin_tiny_seed42.pt` và `swin_tiny_aug_seed42.pt` nếu muốn so ghép cặp trên họ chưa thấy.
- Dòng lệnh: `python -m src.training.run_generic_aug configs/train_swin_tiny_trivialaugment.yaml`.
- Code: `src/training/run_generic_aug.py`, `src/evaluation/control_compare.py`, `src/evaluation/control_plots.py`. Test: `tests/test_generic_augmentation.py`.

## Giới hạn

- Một seed cho cả ba cách. Chênh lệch nhỏ giữa các cách có thể nằm trong dao động giữa các lần train, mà khoảng tin cậy bootstrap không phản ánh được (bootstrap chỉ đo dao động do chọn ảnh, không do seed).
- TrivialAugment là một đại diện của augmentation tổng quát. RandAugment, AugMix hay cutout có thể cho kết quả khác.
- Validation 444 ảnh, khoảng tin cậy rộng hơn ở từng điều kiện riêng lẻ.
