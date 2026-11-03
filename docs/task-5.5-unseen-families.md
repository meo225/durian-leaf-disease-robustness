# Task 5.5. Bộ họ suy giảm chưa thấy

Mục đích: kiểm tra bộ chọn xử lý thích ứng (5.1 đến 5.3) và các mô hình có khái quát ra ngoài 14 họ của benchmark 4.1 hay chỉ khớp với những gì đã thấy khi hiệu chỉnh. Bộ này **chỉ để đánh giá**: không đưa vào augmentation, không dùng để hiệu chỉnh ngưỡng của bộ chọn.

## Sáu họ

| Họ | Nhóm | Cơ chế | Tham số `value` | Họ đã thấy gần nhất |
|---|---|---|---|---|
| `defocus_blur` | gần | Tích chập với đĩa phẳng (khẩu độ tròn) | bán kính, pixel | `blur` (Gauss) |
| `chroma_noise` | gần | Nhiễu màu loang chỉ ở kênh Cb, Cr, độ sáng giữ nguyên | độ lệch chuẩn của nhiễu màu | `noise`, `sensor_noise` (nhiễu cộng lên cả ba kênh) |
| `fog` | xa | Mô hình tán xạ khí quyển `I = J·t + A·(1 − t)`, `t = exp(−β·d)`, độ sâu `d` là trường ngẫu nhiên mượt | β | `contrast_low` (chỉ gần về hiệu ứng) |
| `spatter` | xa | Giọt nước bám ống kính: đốm tròn mép mềm, bên trong nhòe và sáng nhẹ | tỉ lệ diện tích phủ | không có |
| `rain_streak` | xa | Mưa rơi: vệt sáng mảnh gần thẳng đứng, nghiêng ngẫu nhiên tối đa 20°, dài 15 px | xác suất điểm đầu vệt (phủ xấp xỉ giá trị × 15) | không có (khác `spatter` ở cấu trúc đường mảnh có hướng) |
| `saturate` | xa | `gray + f·(ảnh − gray)`, f > 1 là quá bão hòa | f | `color_cast` (khác cơ chế) |

Nhóm gần cho biết bộ chọn có chỉ khớp tham số hay hiểu được cơ chế (mờ, nhiễu). `chroma_noise` còn kiểm tra riêng quy tắc nhiễu của bộ chọn, vì đặc trưng nhiễu đo trên độ sáng nên khó thấy nhiễu màu. Nhóm xa cho biết hành vi khi gặp suy giảm mà không quy tắc nào được thiết kế cho. Nếu thiếu thời gian, bỏ `saturate` và giữ 5 họ (plan cho phép tối thiểu 4).

Không dựng hiệu ứng đèn flash ban đêm: ảnh kiểm tra lá ban đêm hiếm, và flash gần như là tổ hợp của `lowlight`, `glare` và `brightness_bright` đã có.

Cài đặt thuần torch trong `src/corruption/unseen.py`, không phụ thuộc gói ngoài. Registry tách khỏi `basic.CORRUPTIONS`, nên `RandomCorruption` không gọi được các họ này (có test). `corrupt_tensor` nhận cả hai registry, nên `build_loader`, `evaluate_conditions` và `ImageCache` dùng được ngay mà không sửa.

## Quy trình mức và khóa

1. Ba mức mỗi họ, tên `<họ>_s<mức>`, 18 điều kiện, cấu hình ở `configs/unseen_corruption.yaml` (`group` là nhóm gần hoặc xa).
2. Hiệu chỉnh bằng `notebooks/11_unseen_corruption_calibration.ipynb`, giống 4.1: xem ảnh mẫu, đo reference model (Swin-Tiny) trên **validation**. Mức hợp lý làm macro F1 giảm dần theo mức và không về mức đoán bừa ngay từ mức 1.
3. Ghi `reports/unseen/validation_calibration.json` , các biểu đồ (`severity_examples_<lớp>.jpg`, `severity_curves.png`, `level3_vs_seen.png`, `per_class_recall_s3.png`) và bảng kiểm tra mức, chỉnh mức nếu cần, rồi đặt `locked: true` **trước** khi mở test ở 6.1. `load_conditions` từ chối tập test khi `locked: false`.
4. Sau khi khóa, không chỉnh mức theo kết quả test. Lần mở test phải ghi vào `manifests/SPLIT_PROTOCOL.md` mục 3.4.

## Kết quả

Cả 6 họ đạt kiểm tra mức, mức đã khóa ở `configs/unseen_corruption.yaml`. Số liệu, ảnh mẫu và kỳ vọng cho 6.1 nằm ở `reports/unseen/README.md`.

## Giới hạn

- Mức được chọn bằng chính reference model, nên "khó vừa" là khó với Swin-Tiny. Với mô hình khác độ khó có thể khác.
- Chỉ sáu họ. Kết quả là bằng chứng khái quát trên các họ này, không phải trên mọi suy giảm ngoài vườn.
- `fog` và `spatter` dùng trường ngẫu nhiên giả (độ sâu, hình giọt), không dựng từ ảnh thật có sương hay giọt nước.
- Retrain kiểu `exclude_families` (tùy chọn của plan) chưa làm ở task này: `RandomCorruption` đã có tham số `exclude_families`, dùng được khi cần.
