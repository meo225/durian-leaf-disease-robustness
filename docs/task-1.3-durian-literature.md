# Task 1.3. Nhận diện bệnh trên lá sầu riêng

Đề tài: Nghiên cứu ảnh hưởng của các kỹ thuật xử lý ảnh đến khả năng nhận diện bệnh trên lá sầu riêng trong điều kiện chất lượng ảnh biến đổi.

Phân tích từng bài nằm ở `docs/sources/`. Accuracy giữa ba bài không so trực tiếp khi dataset, hệ lớp và protocol khác nhau.

## 1. Mục tiêu

Tổng hợp dataset, tiền xử lý, kiến trúc, metric và setting đã công bố cho nhận diện bệnh lá sầu riêng.

Tìm baseline có cơ sở cho Phase 3. Không chọn model chỉ vì một bài báo accuracy cao. Xem dataset, split, lớp, preprocessing, compute và metric của bài đó.

Tìm hạn chế liên quan độ bền. Trọng tâm là khi chất lượng ảnh đổi thì model giảm ra sao, và xử lý ảnh có phục hồi nhận diện hay không.

## 2. Ba nguồn

DurianLSNet 2026 tập trung dataset field Việt Nam và CNN rất nhẹ cho mobile hoặc IoT. 4.437 ảnh, 5 lớp, accuracy 93,82%, khoảng 1,2 triệu tham số. Error analysis ghi thiếu sáng là một nguyên nhân nhận diện sai.

AI-Driven 2026 benchmark 6 CNN trên 6.119 ảnh và 6 lớp. ConvNeXt 98,00% accuracy. Weighted cross-entropy cho mất cân bằng lớp. Hạn chế geographic, mùa và camera.

ETASR 2025 so ResNet-9, Xception và Mask R-CNN trên 4 lớp. Dữ liệu gốc 648 ảnh, preprocessing và augmentation rất mạnh. Mask R-CNN báo 99,61%, nhưng protocol khác đáng kể hai bài kia.

## 3. Từng bài

- [DurianLSNet 2026](sources/durianlsnet-2026.md)
- [AI-Driven 2026](sources/ai-driven-durian-2026.md)
- [ETASR 2025](sources/etasr-durian-2025.md)

## 4. So sánh

DurianLSNet: 4.437 ảnh, 5 lớp, field Việt Nam, ảnh 224×224, không augmentation trong thí nghiệm chính, hướng model nhẹ.

AI-Driven: 6.119 ảnh, 6 lớp, hai dataset công khai cộng ảnh field, resize giữ tỉ lệ, chuẩn hóa ImageNet, augmentation lúc đọc, benchmark 6 CNN cùng protocol.

ETASR: 648 ảnh gốc, 4 lớp, field Yala cộng Roboflow, xóa nền, bilateral sharpening, 400×400, augmentation rất mạnh kể cả test theo mô tả bài, có localization.

Không bài nào trong ba nguồn dựng benchmark brightness, contrast, blur và noise theo mức rồi đo degradation và recovery sau xử lý ảnh.

Ba bài cho thấy đã có dataset field, baseline mạnh và hướng triển khai. Khoảng trống không phải chứng minh CNN nhận diện được bệnh sầu riêng. Khoảng trống là đo độ nhạy với chất lượng ảnh và tác động thật của xử lý ảnh lên nhận diện.

## 5. Baseline rút ra

Không chọn model từ bảng xếp hạng giữa các bài. Benchmark candidate trên cùng dataset, cùng split, cùng metric và cùng protocol train.

Ứng viên CNN mạnh: ConvNeXt hoặc ResNet, từ benchmark 6 lớp của AI-Driven.

Ứng viên nhẹ: DurianLSNet nếu cài lại được, hoặc MobileNetV3 hay EfficientNet khi cần đối chứng nhẹ.

Ứng viên góc khác: Xception nếu đủ compute. Mask R-CNN chỉ khi nghiên cứu cần localization hoặc segmentation.

Reference model chọn sau benchmark trên validation. Dùng macro F1, balanced accuracy, per-class recall, confusion matrix, độ ổn định và compute. Clean accuracy không phải tiêu chí duy nhất.

## 6. Gap

Đã có dataset field, benchmark CNN, model nhẹ, segmentation, xử lý mất cân bằng lớp, triển khai, Grad-CAM và phân tích lỗi.

Chưa rõ: cùng một ảnh sầu riêng đổi brightness, contrast, blur hoặc noise theo mức kiểm soát thì điểm nhận diện giảm bao nhiêu.

Chưa rõ Gamma, CLAHE, Retinex, denoising hay sharpening có phục hồi phân loại hay chỉ làm ảnh trông rõ hơn.

Chưa rõ một preprocessing có giúp ảnh hỏng nhưng làm ảnh sạch tệ đi hoặc mất texture tổn thương hay không.

Chưa rõ processing kéo attention về tổn thương hay tạo artifact và shortcut mới.

Durian literature hiện chủ yếu tối ưu model và accuracy trên dataset field. Thiếu một nghiên cứu có kiểm soát, tái lập được, về suy giảm chất lượng ảnh và kỹ thuật xử lý ảnh đối với nhận diện bệnh sầu riêng.

## 7. Hướng thí nghiệm sau ba bài

1. Chọn dataset chính sau EDA và kiểm tra leakage. DurianLDD 4.437 ảnh là ứng viên mạnh.
2. Benchmark khoảng ba kiến trúc cùng split và protocol. Chọn reference model bằng validation, không tune trên test.
3. Khóa clean test. Từ các mẫu này tạo brightness, contrast, blur và noise ở mức cố định.
4. Đo clean sang corrupted bằng macro F1, balanced accuracy, per-class recall và confusion matrix.
5. Áp xử lý phù hợp: Gamma, CLAHE, Retinex cho ánh sáng, denoising cho nhiễu, sharpening hoặc restoration cho blur.
6. Đo corrupted sang processed bằng cùng reference model. Kiểm tra có làm điểm trên ảnh sạch giảm không.
7. Grad-CAM trên cùng mẫu clean, corrupted, processed.
8. Sau khi chốt pipeline mới đánh giá dataset sầu riêng ngoài nếu map lớp hợp lệ.

## 8. Kết luận

Deep learning nhận diện được bệnh lá sầu riêng. Đã có model nhẹ và model mạnh. Điểm cao trên một dataset chưa trả lời model bền đến đâu khi chất lượng ảnh đổi.

DurianLSNet quan sát lỗi do thiếu sáng. AI-Driven ghi nhận exposure, white balance, nén và cảm biến có thể gây lệch phân phối. ETASR dùng sharpening và xóa nền nhưng không tách tác động từng bước.

Đồ án chuyển trọng tâm từ model nào accuracy cao nhất sang: khi ảnh xuống chất lượng, model giảm thế nào, và kỹ thuật xử lý ảnh nào giữ hoặc lấy lại thông tin bệnh mà classifier cần.

## 9. Nguồn

DurianLSNet 2026. DOI 10.1016/j.displa.2026.103451.

AI-Driven 2026. DOI 10.3390/app16094062.

ETASR 2025. DOI 10.48084/etasr.12461.
