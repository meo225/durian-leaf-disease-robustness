# Task 5.11. Chạy lại Thực nghiệm 6 và 7 trên bộ họ chưa thấy v2

Nhánh `tn6-tn7-rerun`. Cấu hình `configs/robustness_v2.yaml`, mã `src/evaluation/robustness_v2.py`, notebook `16_robustness_v2.ipynb`, test `tests/test_robustness_v2.py`. Kết quả sẽ nằm ở `reports/robustness_v2/`.

**Viết trước khi chạy.** Mọi kỳ vọng ở mục 5 được ghi khi chưa có số nào của bộ họ v2 cho bất kỳ hệ nào. Phân tích khác với mục 4 ghi là thăm dò.

## 1. Vì sao làm lại

TN6 (bộ chọn và oracle trên họ chưa thấy v1) và TN7 (huấn luyện lại trên họ chưa thấy v1) có các chỗ chưa khách quan đã nêu ở `docs/task-5.8-unseen-families-v2.md` và `docs/task-5.9-retrain-controls.md`: mức do nhóm chọn tay bằng chính reference, họ do nhóm tự cài, nhóm gần xa theo tên, baseline và hai nhánh augmentation lệch ngân sách, một seed, chưa có khoảng tin cậy cho lớp ít ảnh. 5.7 đến 5.10 đã sửa từng điểm. 5.11 đo lại bốn hệ trên bộ đo mới.

## 2. Phạm vi và quy tắc mở dữ liệu

| Mục | Quyết định |
|---|---|
| Ảnh | Chỉ validation (444 ảnh). **Không mở tập test.** |
| Họ | Chỉ họ vai trò `train` và `val` của bộ v2: `inc_zoom_blur`, `inc_frost`, `cbar_cocentric_sine_waves` (train), `inc_defocus_blur`, `cbar_checkerboard_cutout` (val). Năm họ, 15 điều kiện, ba mức mỗi họ |
| Họ vai trò test | `inc_motion_blur`, `inc_impulse_noise`, `inc_brightness`, `inc_snow` **không đo ở đây**, kể cả trên ảnh validation |
| Cổng | `load_settings` từ chối `split` khác `val` và mọi `roles` có `test`. `evaluation_conditions` kiểm lại. Có test cho cả hai |

**Lý do giữ họ vai trò test.** 5.11 là bước chốt cấu hình (đăng ký ở `reports/unseen_v2/README.md`, mục kế hoạch phân tích, điểm 6: bộ v2 mở test một lần, sau khi 5.9, 5.10 và 5.11 chốt cấu hình). Nếu 5.11 đã thấy kết quả của họ vai trò test thì mọi chỉnh sửa sau đó làm P6.1 mất giá trị. Cái giá phải trả: 5.11 chỉ có năm họ (hai gần, một chưa rõ, hai xa), nên kết quả theo họ là mô tả, không đủ để kiểm định gần so với xa. Một dòng đăng ký trước ở `family_roles.py` ghi họ vai trò test mở ở "5.11, 6.1"; tài liệu này chọn cách đọc chặt hơn (chỉ P6.1) và nêu rõ để người duyệt đổi nếu muốn.

Cùng 444 ảnh validation đã dùng để khớp mức, tính MSD và chọn checkpoint ở 5.9 (tái sử dụng validation, đã nêu ở 5.8). Số ở đây là kết quả trung gian ở mức thiết kế, không phải số báo cáo cuối.

## 3. Bốn hệ

| Hệ | Model nền | Cách đo |
|---|---|---|
| Không xử lý | Reference đóng băng của 4.4 (`swin_tiny_seed42.pt`, chọn checkpoint trên validation sạch) | Dự đoán trên ảnh suy giảm nguyên bản |
| Bộ chọn | Cùng reference | Bộ chọn **đã khóa ở 5.3**, không hiệu chỉnh lại, nhìn đặc trưng chất lượng của ảnh chưa xử lý |
| Oracle | Cùng reference | Ba cận trên dùng nhãn: theo họ, theo điều kiện, theo ảnh. Chỉ cận theo họ và theo điều kiện là cận thực tế, cận theo ảnh bị thổi phồng vì may rủi từng ảnh |
| Huấn luyện lại | Swin-Tiny của 5.9 và 5.10 | Suy giảm, TrivialAugment, baseline (3 seed), TrivialAugment hình học và suy giảm bỏ nhóm (1 seed) |

Bộ chọn và oracle chạy trên reference v1, vì bộ chọn khóa trên đúng model đó. **Hiệu chỉnh lại bộ chọn trên baseline mới sau khi thấy kết quả bộ v2 là phá đăng ký trước**, nên không làm. Để biết mức tăng có chuyển sang model mới không, đo thêm **kiểm tra phụ**: cùng bộ chọn và oracle trên baseline 5.9 seed 42. Kiểm tra phụ chỉ để tham khảo, không dùng để chọn hay sửa bộ chọn.

Vì nền khác nhau (reference v1 so với baseline mới), **không so điểm tuyệt đối giữa bộ chọn và huấn luyện lại**. So hai thứ: mức tăng của mỗi hệ so với nền của chính nó, và phần ΔF1 của nền mà hệ lấy lại được.

## 4. Thước đo và quy tắc kết luận

- Thước đo chính của mỗi họ là macro F1 trung bình ba mức. Báo từng họ riêng, kèm trung bình năm họ (mỗi họ một phiếu) chỉ để tham khảo. Không có con số gộp thay cho bảng theo họ.
- Cạnh mọi kết quả báo ΔF1 thực của nền ở cùng họ (macro F1 sạch trừ macro F1 suy giảm). Phần lấy lại được là `(hệ − nền) / (sạch của nền − nền)`, cả hai vế tính trung bình ba mức. Họ có tổng ΔF1 của nền (ba mức cộng lại) dưới `min_reference_drop = 0.05` ghi NaN thay vì chia cho số gần 0. Giá trị âm nghĩa là hệ làm tệ hơn nền.
- **Bộ chọn và oracle so với không xử lý:** bootstrap theo ảnh ghép cặp (1000 lần, cùng chỉ số cho mọi hệ). Kết luận "tốt hơn" hay "kém hơn" khi khoảng 95% của hiệu không chứa 0, ngược lại "chưa phân biệt được".
- **Nhánh nhiều seed (3 seed):** quy tắc 5.9, chỉ kết luận khi |hiệu trung bình| lớn hơn 2,5 lần SD seed (SD mẫu, hai nhánh gộp bằng căn bậc hai của trung bình hai phương sai). Áp theo từng họ. SD từ ba seed theo họ nhiễu hơn SD của thước đo gộp ở 5.9, nên ngưỡng này là mức tối thiểu để dám nói.
- **Nhánh một seed (5.10):** SD lấy từ nhánh baseline ba seed, như 5.10.
- Khoảng bootstrap của một nhánh ba seed là trung bình của ba khoảng cùng chỉ số ảnh. Đây là sai số lấy mẫu ảnh, tách khỏi SD seed.
- **Phân phối lớp dự đoán** (R32): entropy Shannon của lớp dự đoán chia cho log 5, kèm entropy của nhãn thật để so, lớp bị dự đoán nhiều nhất và tỉ lệ của nó, theo từng điều kiện.
- **Lớp ít ảnh:** recall `ALLOCARIDARA_ATTACK` (46 ảnh) kèm khoảng Wilson 95%. Không đọc thay đổi nằm trong khoảng Wilson.
- **MSD và kết quả** (câu hỏi còn lại của 5.7, `docs/task-5.7-msd-family-distance.md`): Spearman giữa MSD tới TrivialAugment và phần TrivialAugment hơn baseline theo họ, trên 14 họ đã thấy (từ 5.9, trung bình 3 seed) cộng 5 họ v2 trong phạm vi. Spearman giữa MSD tới augmentation suy giảm và phần suy giảm hơn baseline chỉ trên 5 họ v2, vì 14 họ đã thấy nằm trong chính augmentation đó. Khoảng bootstrap theo cặp. Với 5 họ khoảng rất rộng và chỉ mô tả, hệ số dưới 5 họ bị từ chối.
- Mọi phân tích ngoài danh sách này (ví dụ đọc theo lớp ở từng họ, so với kiểm tra phụ) ghi là **thăm dò**.

## 5. Kỳ vọng đăng ký trước

Viết khi chưa có số nào của bộ v2. Căn cứ: kết quả 4.5, 5.3, 5.4, 5.9 và 5.10 trên 14 họ đã thấy, cùng MSD ở 5.7. Mỗi dòng nói cả kết quả ngược lại có nghĩa gì.

| # | Kỳ vọng | Nếu ngược lại |
|---|---|---|
| E1 | Suy giảm hơn baseline vượt ngưỡng ở hai họ gần (`inc_defocus_blur`, `inc_zoom_blur`), vì augmentation có `blur` và `lowres` | Augmentation suy giảm không chuyển sang mờ kiểu khác. Đó là kết quả quan trọng, phải viết vào kết luận |
| E2 | Mức hơn baseline của suy giảm ở hai họ xa (`cbar_cocentric_sine_waves`, `cbar_checkerboard_cutout`) nhỏ hơn ở họ gần | Nếu bằng hoặc lớn hơn, phần hơn không đến từ việc "nhắm đúng họ" mà từ độ bền chung. Điều này đồng nhất với 63% ở 5.9 |
| E3 | TrivialAugment hơn baseline nhỏ hơn suy giảm hơn baseline ở hai họ gần | TrivialAugment (có phép Sharpness, gần `inc_defocus_blur` theo MSD 0,82) đủ cho mờ, mâu thuẫn với 5.10 |
| E4 | TrivialAugment chỉ hình học gần baseline ở cả năm họ (chưa kết luận) | Phép hình học tự có tác dụng lên họ lạ. Khi đó 5.10 phải đọc lại |
| E5 | Suy giảm bỏ nhóm nằm giữa baseline và suy giảm đầy đủ ở họ gần, gần suy giảm đầy đủ ở họ xa | Gần baseline ở họ gần nghĩa là chuyển không đáng kể, ở 5.10 đo trên 14 họ đã thấy chỉ lấy lại 45% |
| E6 | Bộ chọn hơn không xử lý ở `inc_defocus_blur` (luật mờ có bằng chứng trên blur, lowres), không kém hơn ở họ nào (khoảng 95% không nằm dưới 0) | Bộ chọn có hại ở một họ thì ghi họ đó và hành động gây hại (`action_rates`) |
| E7 | Bộ chọn chưa phân biệt được với không xử lý ở hai họ xa. Tỉ lệ ảnh bị xử lý ở hai họ này gần 0 | Nếu tỉ lệ xử lý cao mà không có lợi, luật kích hoạt nhầm trên họ lạ, nghĩa là ngưỡng khớp riêng 14 họ |
| E8 | Oracle theo điều kiện không hơn không xử lý quá 0,02 ở hai họ xa, vì không hành động nào trong bốn hành động có bằng chứng ở đó | Nếu lớn hơn, bản đồ phục hồi 4.5 bỏ sót kỹ thuật dùng được cho họ lạ |
| E9 | Mức hơn nền của huấn luyện lại lớn hơn nhiều mức hơn nền của bộ chọn và oracle ở họ gần, như ở 4.5 (0,096 so với 0,022 trên test) | Bộ chọn lấy lại ít nhất bằng huấn luyện lại ở họ gần là một bất ngờ lớn cần kiểm lại mã |
| E10 | Spearman giữa MSD tới TrivialAugment và mức TrivialAugment hơn baseline là âm (MSD càng xa, TrivialAugment càng ít giúp), như R33. **Đã thấy một nửa dữ liệu**: trên 14 họ đã thấy (số của 5.9, không phải bộ v2) tôi tính thử trước khi chạy, ρ = −0,45, khoảng bootstrap 95% từ −0,69 đến 0,06 | Khoảng chứa 0 hoặc dương trên 19 họ nghĩa là MSD không dự đoán được mức hơn của TrivialAugment trên bệnh lá. Ghi rõ giới hạn của phép đo, theo `docs/task-5.7` |

Không đăng ký kỳ vọng cho entropy lớp dự đoán: chỉ báo, và đọc là thăm dò.

## 6. Cách chạy

1. Chạy `16_robustness_v2.ipynb` trên Kaggle (GPU). Input: dataset ảnh `cthng123/durian-leaf-disease-dataset`, và dataset chứa checkpoint: `swin_tiny_seed42.pt` (reference), 11 checkpoint của 5.9 và 5.10 (`swin_tiny_ctrl_baseline_seed{42,43,44}.pt`, `swin_tiny_aug_seed{42,43,44}.pt`, `swin_tiny_trivialaugment_seed{42,43,44}.pt`, `swin_tiny_trivialaugment_geometric_seed42.pt`, `swin_tiny_aug_leave_group_seed42.pt`).
2. Bảng đặc trưng và dự đoán của bộ chọn tính một lần cho reference và baseline seed 42 (kiểm tra phụ), đọc `reports/selector/selector_calibrated.json` (không hiệu chỉnh lại).
3. Tải zip về, giải vào repo, commit `reports/robustness_v2/`.
4. Đọc kết quả theo mục 5. Viết `reports/robustness_v2/README.md` với từng dòng E1 đến E10 ghi "xảy ra" hay "không", như bảng đọc ở 5.10.

Notebook từ chối chạy nếu `FINAL_EVAL` hoặc `FINAL_FAMILY_EVAL` đang bật.

## 7. Giới hạn đã biết

- Năm họ, một họ chưa rõ, hai họ gần là mờ. Họ gần so với họ xa chỉ mô tả.
- Bộ chọn và oracle trên một model và một seed. Bộ chọn chỉ có bốn hành động, nên oracle theo họ là cận trên của bốn hành động, không phải của xử lý ảnh nói chung.
- Reference v1 và baseline mới khác nhau (dừng sớm, quy tắc chọn checkpoint). Hai hệ trên hai nền khác nhau chỉ so qua mức hơn nền.
- Các bản sửa v2.1 và v2.1-fix của bộ họ đã công khai ở `docs/task-5.8`. Họ ImageNet-C mức nhẹ là trộn alpha, không phải mức của bản gốc.
- Fog, gaussian blur và glass blur không có trong bộ vì lỗi tương thích thư viện. Quyết định không vá sau khi đã thấy kết quả. Bộ v2 vì vậy nghiêng về mờ và nhiễu xung, nhiễu gaussian và fog ngoài vườn không được đại diện.
- Test chưa mở. P6.1 mở một lần, đo cả họ vai trò test, ghi một dòng vào `manifests/SPLIT_PROTOCOL.md` mục 3.4 và cần `FINAL_FAMILY_EVAL=1`.

## 8. Đầu ra

| File | Nội dung |
|---|---|
| `reports/robustness_v2/family_scores.csv` | Macro F1 theo họ và mức của mọi hệ và mọi seed |
| `reports/robustness_v2/systems_vs_none.csv` | Bộ chọn và ba oracle so với không xử lý, theo họ, kèm khoảng |
| `reports/robustness_v2/retrained_comparison.csv` | Các cặp nhánh huấn luyện lại, theo họ, quy tắc 2,5 SD |
| `reports/robustness_v2/recovered_fraction.csv` | Phần ΔF1 của nền mà mỗi hệ lấy lại, theo họ |
| `reports/robustness_v2/diagnostics.csv` | Entropy lớp dự đoán, lớp bị dự đoán nhiều nhất, recall lớp nhỏ kèm Wilson |
| `reports/robustness_v2/action_rates.csv` | Tỉ lệ ảnh mỗi hành động của bộ chọn theo điều kiện |
| `reports/robustness_v2/msd_correlation.csv` | Spearman MSD với mức hơn của TrivialAugment và của suy giảm |
| `reports/robustness_v2/side_check_new_baseline.csv` | Bộ chọn và oracle trên baseline mới (tham khảo) |
