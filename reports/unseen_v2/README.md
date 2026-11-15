# Kết quả bộ họ chưa thấy v2, task 5.7 và 5.8

**Chưa có số liệu cuối.** Lượt hiệu chỉnh đầu theo quy tắc v2.0 giữ 2 trên 20 họ nên quy tắc được sửa thành v2.1 (thêm mức nhẹ cho ImageNet-C, seed theo họ và giá trị). Xem `docs/task-5.8-unseen-families-v2.md`, mục "Bản sửa v2.1". File lượt đầu để ở `pass1_v2.0/` để truy vết. Lượt v2.1 và notebook 14 cần chạy trên Kaggle.

| Bước | Notebook | File ghi vào thư mục này |
|---|---|---|
| 5.8 khớp độ khó | `notebooks/13_unseen_v2_calibration.ipynb` | `calibration.json`, `candidates_surrogate.json`, `realised_difficulty.csv`, `severity_examples_*.png` |
| 5.7 MSD, 5.8 vai trò và khóa | `notebooks/14_msd_family_distance.ipynb` | `msd.json`, `msd_*.csv`, `roles.csv`, `seen_pairwise_distance.*`, `msd_to_degradation_aug.png`, và ghi đè `configs/unseen_families_v2.yaml` (khóa) |

Thiết kế và giới hạn: `docs/task-5.8-unseen-families-v2.md`, `docs/task-5.7-msd-family-distance.md`. Sau khi chạy, viết kết quả vào file này theo mẫu `reports/unseen/README.md`, kể cả họ bị bỏ và lý do.
