# Task 5.4. Ablation của bộ chọn xử lý ảnh thích ứng

Tài liệu này ghi thiết kế thí nghiệm và các dự đoán đặt trước khi chạy. Kết quả nằm ở `reports/selector_ablation/README.md`.

## 1. Mục tiêu

Bộ chọn đã khóa ở task 5.3 tăng mục tiêu từ 0.8607 lên 0.8802 trên validation. Câu hỏi của task này là mức tăng đó đến từ thành phần nào, và thành phần nào có thể bỏ mà không mất gì.

| Câu hỏi | Nội dung |
|---|---|
| Q1 | Mỗi nhóm đặc trưng đóng góp bao nhiêu. Nhóm độ tự nhiên là proxy BRISQUE có giá trị thêm không. |
| Q2 | Mỗi hành động và mỗi kỹ thuật, unsharp hay bilateral, đóng góp bao nhiêu. |
| Q3 | Luật nhiễu, luật mờ và các mệnh đề chặn đóng góp bao nhiêu. |
| Q4 | Bộ chọn còn cách cận trên theo ảnh bao xa. |

## 2. Thiết lập

- Dữ liệu: bảng đặc trưng và dự đoán của task 5.3, 444 ảnh validation, ảnh sạch và 42 điều kiện. Reference model Swin-Tiny đóng băng. Tập test không được mở.
- Mục tiêu và ràng buộc ảnh sạch giữ nguyên như task 5.3, nên chênh lệch giữa các biến thể chỉ do thành phần bị bỏ.
- Mỗi biến thể có hiệu chỉnh lại dùng đúng `fit` của task 5.3 với cùng lưới ngưỡng, `min_gain` và ràng buộc ảnh sạch.
- Biến thể `full` hiệu chỉnh lại phải trùng bộ chọn đã khóa. Notebook dừng nếu khác, vì khi đó bảng tính lại không còn là bảng của 5.3.

### Các biến thể

| Nhóm | Biến thể | Cách dựng |
|---|---|---|
| Tham chiếu | Không xử lý, bốn hành động cố định | Tra bảng. |
| Cận trên | Theo họ, theo điều kiện, theo ảnh | Dùng nhãn nên không triển khai được. Hai cận đầu chọn một hành động cố định cho cả họ hoặc cả điều kiện. Cận theo ảnh chọn hành động đúng cho từng ảnh. |
| Đầy đủ | `full`, `locked` | Hiệu chỉnh lại và bộ đã khóa. |
| Bỏ nhóm đặc trưng | Sáu biến thể, mỗi nhóm một | Nhóm bị loại khỏi ứng viên và mệnh đề chặn của mọi luật, rồi hiệu chỉnh lại. |
| Bỏ hành động | Bốn biến thể | Hành động bị loại khỏi danh sách của luật, rồi hiệu chỉnh lại. |
| Bỏ kỹ thuật | Unsharp, bilateral | Loại mọi cấu hình của kỹ thuật đó, rồi hiệu chỉnh lại. |
| Chỉ một luật | Luật nhiễu, luật mờ | Giữ một luật của bộ `full`, không hiệu chỉnh lại. |
| Bỏ mệnh đề chặn | Một biến thể | Mỗi luật chỉ giữ mệnh đề chính, không hiệu chỉnh lại. |

### Thống kê

- Khoảng tin cậy 95% bằng bootstrap theo ảnh, 1000 mẫu. Mọi biến thể được chấm trên cùng mẫu lặp, nên chênh lệch có ghép cặp và hẹp hơn so với chấm riêng.
- Cross-validation 5 phần theo lớp cho các biến thể có hiệu chỉnh lại. Hiệu chỉnh trên phần còn lại và chấm trên phần giữ lại.
- Một thành phần chỉ được gọi là có ích khi khoảng tin cậy của mức mất không chứa 0.

## 3. Dự đoán đặt trước

Các dự đoán dưới đây viết từ kết quả 5.3 và cấu hình luật, trước khi chạy ablation.

1. **Nhóm màu và nhóm tương phản có mức mất đúng bằng 0.** Không đặc trưng nào của hai nhóm nằm trong ứng viên hay mệnh đề chặn đang dùng. Chỉ `extra_guard` có `luma_std`, và nó đã bị bỏ ở 5.3. Đây là kết quả theo cấu trúc, không phải bằng chứng rằng màu hay tương phản vô ích.
2. **Nhóm độ tự nhiên là nhóm có mức mất lớn nhất.** Luật mờ chọn `mscn_var`. Bỏ nhóm này thì luật mờ phải dùng `sharpness`, và dự đoán mức mất tối thiểu 0.005.
3. **Bỏ unsharp mất nhiều hơn bỏ bilateral.** Ở 5.3 luật mờ đóng góp 0.0143 và luật nhiễu 0.0052. Mức mất khi bỏ unsharp dự đoán gần 0.014.
4. **Bỏ mệnh đề chặn làm giảm mục tiêu hoặc tăng rủi ro trên ảnh sạch.** Mệnh đề `luma_mean` được chọn vì nó chặn bớt ảnh tối bị bắt nhầm.
5. **Cận theo ảnh thổi phồng khoảng cách.** Ở 5.3 bộ chọn lấy được khoảng 32 phần trăm khoảng cách tới cận theo ảnh. Cận đó chọn hành động đúng cho từng ảnh trong 5 lựa chọn nên được nâng lên bởi may rủi của dự đoán. Dự đoán phần bộ chọn lấy được sẽ cao hơn rõ rệt khi so với cận theo họ hoặc theo điều kiện. Ablation không làm con số tuyệt đối tốt hơn, chỉ cho biết thành phần nào đáng cải thiện.

Dự đoán nào sai cũng được ghi lại như kết quả, không sửa dự đoán sau khi xem.

## 4. Giới hạn đã biết

- Mọi số đo trên validation, nơi ngưỡng đã được hiệu chỉnh. Bootstrap theo ảnh cho biết độ ngẫu nhiên do chọn ảnh, không thay thế được đánh giá trên tập test ở task 6.1.
- Các họ suy giảm vẫn là 14 họ đã thấy. Tổng quát hóa sang họ chưa thấy đo ở task 5.5.
- Hiệu chỉnh tham lam theo thứ tự luật, nên khi bỏ một thành phần luật sau có thể chọn lại hành động khác. Mức mất khi đó là mức mất sau khi bộ chọn đã thích nghi, nhỏ hơn mức mất nếu giữ nguyên ngưỡng. Biến thể chỉ một luật và bỏ mệnh đề chặn không hiệu chỉnh lại nên đo mức mất không thích nghi.
- Nhóm độ tự nhiên là phương sai MSCN, một proxy của BRISQUE, không phải điểm BRISQUE đầy đủ. Kết luận về BRISQUE chỉ áp dụng cho proxy này.
- Hai nhóm chưa có đặc trưng trong luật vẫn có thể hữu ích với bộ luật khác. Ablation chỉ trả lời cho bộ luật của 5.3.

## 5. Chạy

Notebook `notebooks/10_selector_ablation.ipynb` trên Kaggle, GPU T4, khoảng 15 đến 25 phút gồm thu lại bảng. Code ở `src/selector/ablation.py` và `src/selector/ablation_plots.py`, test ở `tests/test_selector_ablation.py`.
