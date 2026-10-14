# Background Recomposition và UDA 2026

DOI: https://doi.org/10.1016/j.ecoinf.2025.103579

## Vai trò trong đồ án

Background là một nhánh phân tích domain shift, không cùng loại với Gaussian noise hay blur. Dùng khi giải thích lỗi và khi kiểm tra artifact. Không ép thành một mức corruption đơn giản.

## Việc tác giả làm

Field-adaptive background recomposition tách lá khỏi ảnh source rồi ghép với nền mang đặc điểm field. Sau đó kết hợp UDA để căn source và target.

## Kết quả

Nền là một yếu tố domain thật. Model quen nền đơn giản có thể bị phân tâm bởi đất, cành, lá khác và cảnh phức tạp.

Đổi nền chưa tái tạo hết field shift. Field còn khác ánh sáng, texture, đặc điểm lá và điều kiện chụp.

Tách và ghép có thể tạo artifact ở biên lá. Artifact là dấu do xử lý tạo ra, không có trong ảnh thật. Xử lý ảnh có thể sửa một vấn đề và tạo một vấn đề khác.

## Cách dùng

Nếu một cách xử lý ảnh tăng điểm, kiểm tra ảnh và Grad-CAM để chắc model không học artifact mới.
