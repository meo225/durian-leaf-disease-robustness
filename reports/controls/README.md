# Kết quả đối chứng huấn luyện lại, tasks 5.9 và 5.10

**Chưa có kết quả.** Thư mục này nhận file từ `notebooks/15_retrain_controls.ipynb` (mục 7) sau khi chạy trên Kaggle.

Thiết kế, quy tắc kết luận và bảng cách đọc kết quả đăng ký trước nằm ở:

- `docs/task-5.9-retrain-controls.md`: baseline, suy giảm, TrivialAugment, cùng giao thức, ba seed.
- `docs/task-5.10-ta-geometric-and-leave-group.md`: TrivialAugment chỉ phép hình học, và suy giảm bỏ một nhóm họ chọn bằng MSD.

Giao thức chung: `configs/controls/protocol.yaml` (30 epoch, không dừng sớm, chọn checkpoint trên validation hỗn hợp). Mọi số đo trên validation. Tập test không mở.

Sau khi chạy, điền kết quả vào file này theo cùng cấu trúc với `reports/generic_augmentation/README.md`: bảng từng lần chạy (`runs_5_9.csv`), SD seed (`seed_stats_5_9.csv`), ba hiệu và kết luận (`comparison_5_9.csv`), rồi 5.10 (`comparison_5_10.csv`).
