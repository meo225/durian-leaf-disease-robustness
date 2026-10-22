# Task 1.4. Experimental protocol

Nghiên cứu ảnh hưởng của xử lý ảnh đến nhận diện bệnh lá sầu riêng.

## 1. Mục tiêu

Protocol xác định trước cách thực hiện và đánh giá thí nghiệm để các giai đoạn sau dùng cùng một quy trình. Câu hỏi: khi chất lượng ảnh lá sầu riêng suy giảm, khả năng nhận diện bệnh thay đổi thế nào, và kỹ thuật xử lý ảnh có phục hồi phần hiệu năng đã mất hay không.

Đối tượng đánh giá chính là hiệu năng nhận diện bệnh. Chất lượng ảnh nhìn bằng mắt chỉ hỗ trợ phân tích.

## Trọng tâm và đóng góp

Đóng góp chính là một cơ chế xử lý ảnh thích ứng theo chất lượng ảnh, đặt trước bộ phân loại bệnh lá sầu riêng được giữ nguyên. Cơ chế ước lượng dạng và mức suy giảm từ chính ảnh bằng đặc trưng chất lượng nhẹ, rồi chọn kỹ thuật xử lý phù hợp hoặc không xử lý.

Bằng chứng nền là bản đồ phục hồi: kỹ thuật cổ điển nào phục hồi nhận diện, ở họ và mức suy giảm nào, với chi phí bao nhiêu trên ảnh sạch và theo từng lớp. Cơ chế thích ứng được thiết kế từ bản đồ này, không đoán trước.

Benchmark suy giảm, đánh giá ngoài phân phối và Grad-CAM là phương tiện kiểm chứng, không phải đóng góp riêng. Các nghiên cứu trước đã có benchmark suy giảm cho bệnh lá, nên bản thân benchmark không được coi là điểm mới.

Điều kiện so sánh ở đánh giá cuối: không xử lý, kỹ thuật cố định tốt nhất, chọn thích ứng, oracle biết trước loại suy giảm, và đối chứng huấn luyện lại reference model bằng augmentation suy giảm. Nếu chọn thích ứng không vượt kỹ thuật cố định, đóng góp còn lại là bản đồ phục hồi và kết quả âm kèm nó.

Ngoài phạm vi: khôi phục ảnh bằng học sâu, thiết kế kiến trúc phân loại mới, DINOv2 đóng băng.

## 2. Phạm vi

Input là ảnh RGB lá sầu riêng. Output là nhãn bệnh hoặc healthy theo hệ nhãn của dataset chính. Bài toán chính là phân loại ảnh. Localization và segmentation không bắt buộc.

Nhánh A, image-quality robustness: cùng các ảnh kiểm thử, chủ động đổi chất lượng. Bốn nhóm là brightness, contrast, blur và noise. Mỗi loại có nhiều mức.

Nhánh B, external-domain generalization: một dataset sầu riêng độc lập, không tham gia training, tuning hay chọn model.

Hai nhánh báo cáo riêng. Corruption có kiểm soát cho biết model phản ứng với một thay đổi đã biết. External phản ánh nhiều thay đổi tự nhiên cùng lúc.

## 3. Câu hỏi nghiên cứu

- RQ1. Độ sáng, tương phản, mờ, nhiễu và các họ mô phỏng điều kiện ngoài vườn ảnh hưởng thế nào đến hiệu năng khi severity tăng?
- RQ2. Kỹ thuật xử lý ảnh cổ điển nào phục hồi hiệu năng, với họ và mức suy giảm nào, và chi phí trên ảnh sạch và theo từng lớp là bao nhiêu?
- RQ3. Chọn được kỹ thuật phù hợp chỉ từ ảnh, khi không biết trước loại suy giảm, đến mức nào so với kỹ thuật cố định, oracle và huấn luyện lại bằng augmentation?
- RQ4. Xử lý ảnh có làm giảm hiệu năng trên ảnh sạch, làm mất kết cấu vết bệnh hoặc tạo artifact hay không?
- RQ5. Pipeline cuối còn hoạt động thế nào trên dataset sầu riêng độc lập? Đây là kiểm chứng, báo riêng.

## 4. Giả thuyết

- H1. Hiệu năng giảm khi severity tăng, mức giảm khác nhau giữa bốn loại corruption.
- H2. Kỹ thuật phù hợp failure mode có thể phục hồi một phần. Không giả định một kỹ thuật luôn tốt cho mọi corruption.
- H3. Xử lý cố định trên mọi input có thể có tác dụng ngược, nhất là ảnh sạch hoặc tổn thương có texture nhỏ. Chọn thích ứng theo chất lượng ảnh giữ được phần lớn mức phục hồi của oracle mà không làm xấu ảnh sạch.
- H4. Metric nhận diện phản ánh giá trị của preprocessing rõ hơn việc chỉ xem ảnh có đẹp hơn hay không.
- H5. Hiệu năng trên dataset độc lập thấp hơn clean test cùng nguồn nếu domain shift đáng kể.

## 5. Biến

Biến độc lập: loại corruption, severity khóa trước, kỹ thuật xử lý ảnh thuộc illumination, denoising, sharpening hoặc restoration.

Biến phụ thuộc: macro F1 là metric chính. Kèm balanced accuracy, per-class recall, confusion matrix và accuracy. Khi cần, ghi model size và inference latency.

Biến kiểm soát: cùng reference model, cùng sample qua chuỗi clean, corrupted, processed. Split, checkpoint, kích thước input và code đánh giá không đổi giữa các điều kiện so sánh.

## 6. Dataset

Dataset chính chốt sau EDA, kiểm tra trùng và rò rỉ. DurianLDD là ứng viên vì 4.437 ảnh field và 5 lớp.

Train học tham số. Validation chọn architecture, hyperparameter, checkpoint và tham số preprocessing. Test chỉ dùng cho đánh giá cuối. Không dùng test để chọn model, severity hay tham số xử lý ảnh.

Tỉ lệ split chốt sau EDA. Giữ phân bố lớp hợp lý. Ảnh giống hoặc gần giống không nằm xuyên split.

External dataset độc lập với pipeline train. Chỉ đánh giá lớp map được. Không ép nhãn khi taxonomy không khớp.

## 7. Chọn model

Phase 3 benchmark ít nhất ba kiến trúc dưới cùng protocol: một CNN mạnh như ConvNeXt hoặc ResNet, một model nhẹ như DurianLSNet hoặc MobileNetV3 nếu cài được, và một kiến trúc khác để có so sánh.

Reference model chọn chủ yếu bằng macro F1, balanced accuracy, per-class recall và độ ổn định trên validation. Clean accuracy không phải tiêu chí duy nhất. Model size và latency xét khi hiệu năng gần nhau. Test không tham gia chọn model.

## 8. Corruption có kiểm soát

Brightness mô phỏng thiếu sáng hoặc quá sáng. Contrast mô phỏng tương phản thấp hoặc bất thường. Blur mô phỏng mất chi tiết do rung, focus kém hoặc chuyển động. Noise mô phỏng nhiễu cảm biến và điều kiện chụp.

Mỗi loại có nhiều mức từ nhẹ đến mạnh. Giá trị cụ thể chốt ở Phase 4 sau khi xem mẫu validation. Mức phải tạo thay đổi có nghĩa và không biến ảnh thành trường hợp phi thực tế. Sau khi khóa, không chỉnh theo kết quả test.

Mỗi ảnh test sạch có bản corrupted tương ứng. Nhãn không đổi. Nhờ đó biết sample nào chuyển từ đúng sang sai.

## 9. Benchmark xử lý ảnh

Illumination: Gamma, CLAHE, Retinex khi phù hợp. Noise: denoising giữ texture tổn thương. Blur: sharpening hoặc restoration phù hợp. Không bắt buộc mọi kỹ thuật áp cho mọi corruption.

So sánh clean, corrupted và processed bằng nhận diện. Một kỹ thuật hữu ích khi cải thiện dữ liệu suy giảm mà không gây hại không chấp nhận được trên ảnh sạch hoặc một lớp cụ thể. Kết quả âm vẫn báo cáo.

## 10. Metric độ bền

Degradation = F1 sạch trừ F1 sau corruption. Càng lớn, model càng nhạy với corruption đó.

Recovery = F1 sau xử lý trừ F1 sau corruption.

Normalized recovery = Recovery chia Degradation khi F1 sạch lớn hơn F1 sau corruption. Gần 1 là phục hồi phần lớn phần đã mất. Bằng 0 là không cải thiện. Âm là processing làm tệ hơn.

Ngoài metric tổng, xem per-class recall và confusion matrix để tránh một kỹ thuật tăng điểm tổng nhưng làm một bệnh giảm mạnh.

## 11. Phân tích lỗi

Chọn mẫu cùng identity ở ba trạng thái. Dùng Grad-CAM hoặc Grad-CAM++. Xem corruption có kéo attention khỏi tổn thương sang nền hay không, và processing có kéo attention về tổn thương hay tạo artifact. Heatmap hỗ trợ giải thích, không tự nó kết luận model đã học đúng đặc trưng.

## 12. External

Chạy sau khi model và pipeline đã khóa. Không fine-tune và không chọn lại preprocessing bằng tập này. Báo riêng với nhánh corruption. Ghi mức giảm so với clean test chính, kèm ca đúng và ca sai tiêu biểu.

## 13. Thứ tự

1. EDA, trùng, rò rỉ.
2. Khóa dataset và split.
3. Train baseline cùng protocol.
4. Chọn reference model bằng validation.
5. Đánh giá clean test.
6. Tạo corruption từ cùng mẫu test.
7. Đo degradation theo loại, mức và lớp.
8. Áp xử lý ảnh phù hợp.
9. Đo recovery và tác dụng phụ.
10. Confusion matrix, lỗi từng mẫu, Grad-CAM.
11. External sau khi pipeline khóa.

## 14. Tái lập

Lưu seed, split và danh sách sample. Lưu config train, hyperparameter, checkpoint và dependency. Lưu đúng loại corruption, severity và tham số xử lý. Mỗi thí nghiệm có ID và log metric. Không đổi protocol test sau khi đã xem điểm. Nếu phải đổi, ghi lý do và chạy lại các thí nghiệm liên quan theo quy tắc mới.
