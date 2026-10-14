# Task 1.2. Nhận diện bệnh thực vật trong điều kiện thực địa

Đề tài: Nghiên cứu ảnh hưởng của các kỹ thuật xử lý ảnh đến khả năng nhận diện bệnh trên lá sầu riêng trong điều kiện chất lượng ảnh biến đổi.

Phân tích từng bài nằm ở `docs/sources/`. File này giữ câu hỏi, khái niệm, phần tổng hợp và protocol đọc ra từ các bài đó.

Với DG-PLDR, phân tích chỉ dùng thông tin đã xác nhận từ nhà xuất bản và repository chính thức vì full PDF chưa có trong tập file lúc viết.

## 1. Mục tiêu

Ba câu hỏi. Vì sao model nhận diện bệnh lá thường giảm khi chuyển từ ảnh được kiểm soát sang ảnh thực địa. Illumination, blur, noise, background và style variation ảnh hưởng thế nào. Nhóm xây protocol đánh giá robustness và generalization ra sao.

Nguồn: DG-PLDR 2025, From Laboratory to Field 2023, Reliability Gap 2026, Background Recomposition 2026, Frozen Foundation-Model Features 2026, Richter và Kim 2025.

## 2. Khái niệm

Ảnh controlled hoặc laboratory được chụp trong điều kiện tương đối ổn định. Lá thường rõ, nền đơn giản, ánh sáng ít đổi, góc chụp thuận. PlantVillage là ví dụ hay được dùng.

Ảnh field là ảnh ngoài thực tế. Một ảnh có thể gặp cùng lúc nền phức tạp, bóng, góc khác, lá bị che, blur hoặc camera khác. Field recognition khó hơn phân loại trên ảnh sạch.

Domain shift là phân phối lúc test khác lúc train. Trong nhận diện bệnh cây, shift có thể đến từ illumination, camera, nền, độ phân giải, góc, crop, style, hoặc đặc điểm cây và triệu chứng. Domain shift không đồng nghĩa một ảnh bị noise hay blur. Field shift thường là nhiều thay đổi cùng lúc.

UDA, unsupervised domain adaptation, cho phép dùng ảnh target không nhãn trong lúc thích nghi. Domain generalization là sang miền mới mà không được nhìn target domain khi train. Train và tune hết trên dataset sầu riêng chính, chỉ mở dataset khác ở đánh giá cuối, thì setting đó gần domain generalization hơn UDA.

Robustness trong đồ án là giữ hiệu năng khi chất lượng của cùng loại dữ liệu đổi, ví dụ brightness, contrast, blur, noise. Generalization là hoạt động trên distribution khác, ví dụ train dataset A rồi test dataset B khác camera, nền và cách thu thập. Hai khái niệm liên quan và không thay nhau. Corruption có kiểm soát cho biết model nhạy yếu tố nào. External dataset cho biết model chịu natural domain shift hay không.

## 3. Từng bài

- [DG-PLDR 2025](sources/dg-pldr-2025.md)
- [From Laboratory to Field 2023](sources/from-laboratory-to-field-2023.md)
- [Reliability Gap 2026](sources/reliability-gap-2026.md)
- [Background Recomposition 2026](sources/background-recomposition-2026.md)
- [Frozen Foundation-Model Features 2026](sources/frozen-foundation-features-2026.md)
- [Richter và Kim 2025](sources/richter-kim-2025.md)

## 4. Theo từng yếu tố ảnh

Illumination ngoài thực địa rộng hơn việc cả ảnh sáng hơn hoặc tối hơn. Một lá có thể vừa bóng vừa phản sáng. Brightness và contrast trong đồ án là stress test có kiểm soát: chủ động đổi một yếu tố với tham số biết trước. Gamma, CLAHE và Retinex có thể thử khi lỗi liên quan ánh sáng hoặc tương phản. Tiêu chí là classifier có lấy lại hiệu năng mà không mất chi tiết tổn thương, không phải ảnh nhìn đẹp hơn.

Blur làm mất biên và texture. Nhiều bệnh khác nhau ở pattern nhỏ trên tổn thương. Lấy cùng ảnh sạch rồi tạo nhiều mức blur để đo đúng phần blur làm giảm. Sharpening cần xem cả cải thiện và artifact. Sharpen quá mạnh có thể tạo biên giả.

Các nguồn domain-shift hiện tại chưa có một thang severity noise dùng chung. Nếu dùng Gaussian noise hoặc noise model khác, nhóm tự định nghĩa tham số và mức, ghi trong config, giữ giống nhau mọi thí nghiệm. Denoising là đánh đổi. Khử quá mạnh có thể xóa texture tổn thương.

Background có bằng chứng mạnh nhất trong nhóm bài này. Nó có thể thành shortcut và kéo attention khỏi tổn thương. Đổi nền khác bản chất với blur và noise, vì thường kéo theo cảnh mới. Hợp với phân tích external và Grad-CAM hơn là một mức corruption đơn giản.

Style là khác biệt màu, contrast, crop, nén hoặc pipeline xử lý. Acquisition variation là camera, sensor, góc, khoảng cách, framing, địa điểm và occlusion. Vì vậy benchmark tổng hợp không thay dataset ngoài. Benchmark tổng hợp cô lập từng kiểu lỗi. Dataset ngoài kiểm tra tổ hợp tự nhiên.

## 5. Các bài nhìn cùng vấn đề ở các tầng khác nhau

Richter và Kim trả lời nên benchmark model thế nào. DG-PLDR hỏi có generalize sang unseen domain hay không. From Laboratory to Field hỏi adaptation từ lab sang field khi được nhìn ảnh target không nhãn. Background Recomposition cô lập nền. Frozen Foundation-Model Features kiểm tra representation. Reliability Gap hỏi model còn đáng tin khi sang field hay không.

Clean accuracy rất cao có thể đứng cùng field accuracy thấp. Phase 3 chọn reference model đủ tốt và ổn định. Phase 4 mới hỏi model đó chịu brightness, contrast, blur và noise ra sao.

Nền quan trọng nhưng không phải toàn bộ. From Laboratory to Field còn nêu illumination, blur, sensor, góc và occlusion. Frozen features cho thấy representation đổi cross-domain rất mạnh. External thấp thì không kết luận ngay là do nền.

Không có kỹ thuật nào mặc nhiên cải thiện độ bền. DANN-style adaptation có thể làm xấu. Full fine-tuning có thể làm cross-domain giảm. Một mức Gamma tốt cho ảnh tối chưa chắc tốt cho ảnh sạch. Denoising tốt cho nhiễu mạnh có thể phá texture khi nhiễu ít. Sharpening có thể giúp blur nhẹ và tạo artifact ở mức khác. Vì vậy cần benchmark theo loại corruption và severity.

## 6. Follow hướng nào

Không sao chép nguyên một bài.

Follow Reliability Gap 2026 cho khung đánh giá: đo có hệ thống, protocol chặt, nhiều metric, thống kê ảnh, Grad-CAM, báo kết quả âm. Chuỗi của đồ án là cùng một mẫu ở ba trạng thái: clean, corrupted, processed. Processed là ảnh sau Gamma, CLAHE, Retinex, denoising hoặc sharpening. Đây là bài chính về phương pháp đánh giá.

Follow Richter và Kim 2025 ở Phase 3: khoảng ba kiến trúc, cùng protocol, nhiều seed nếu đủ tài nguyên, rồi chọn một reference model.

Follow DG-PLDR và Frozen Foundation-Model Features ở Phase 6 khi kiểm tra dataset sầu riêng ngoài.

Dùng From Laboratory to Field và Background Recomposition để giải thích lỗi, không nhất thiết triển khai UDA của họ.

Chuỗi tổng: Richter và Kim để chọn reference model, Reliability Gap để dựng cách đánh giá, corruption có kiểm soát và xử lý ảnh là phần của nhóm, DG-PLDR và frozen features để kiểm external, Lab-to-Field và Background Recomposition để giải thích lỗi. Phần giữa phải là nghiên cứu của nhóm.

## 7. Protocol đọc ra từ các bài

Khóa dataset trước: phân bố lớp, trùng, gần trùng, source bias. Chia train, validation, test trước khi chọn model. Test không dùng để tune. Nếu nhiều crop từ cùng một ảnh gốc, cố giữ chúng trong một split để tránh leakage. Leakage là thông tin gần giống nhau ở cả train và test, làm điểm cao giả.

Chọn reference model bằng cùng split và metric. Theo dõi macro F1, balanced accuracy, per-class recall, kích thước model và inference time. Nhiều seed nếu có thể. Reference model là dụng cụ đo: đủ tốt và ổn định để khi input đổi, phần đổi của điểm chủ yếu đến từ điều kiện ảnh hoặc xử lý ảnh.

Giữ clean test cố định. Từ cùng mẫu tạo brightness, contrast, blur và noise ở mức đã định nghĩa. Giữ nhãn và danh tính mẫu. Đo degradation trước khi sửa ảnh. Chưa biết model hỏng ở đâu thì chưa có cơ sở chọn enhancement.

Tune tham số xử lý trên validation. Cùng reference model so corrupted và processed. Câu hỏi là phục hồi bao nhiêu điểm nhận diện và có làm điểm trên ảnh sạch giảm hay không.

Grad-CAM hoặc Grad-CAM++ trên ca đại diện theo chuỗi clean, corrupted, processed. Có thể thêm brightness, contrast, sharpness hoặc entropy để nối chất lượng ảnh với điểm.

External chỉ sau khi model và preprocessing đã chốt, và chỉ nếu map lớp hợp lệ. Không dùng kết quả external để quay lại tune nếu muốn giữ phép thử độc lập. Báo riêng với kết quả corruption tổng hợp.

## 8. Khoảng trống

Các nguồn đã cho thấy field shift có thật và có thể rất nặng. Literature đã có domain adaptation, domain generalization, background bias, foundation representation và reliability.

Chưa trả lời trực tiếp: khi chất lượng ảnh lá sầu riêng bị đổi có kiểm soát, từng kỹ thuật xử lý ảnh ảnh hưởng thế nào tới nhận diện bệnh phía sau. Tiêu chí là kết quả nhận diện sau preprocessing, không phải ảnh trông đẹp hơn.

Hướng: clean, rồi corruption có kiểm soát, rồi xử lý ảnh, rồi reference model cố định, rồi đo degradation và recovery, rồi failure analysis, rồi external.

Contribution nên nằm ở đánh giá có kiểm soát, tái lập được, và ở insight kỹ thuật nào hữu ích trong điều kiện nào. Không đặt contribution là chứng minh một kỹ thuật luôn tốt nhất. Độ bền phụ thuộc domain, representation, lớp và kiểu lỗi.

## 9. Sáu điểm cần nhớ

Accuracy cao trên ảnh sạch không có nghĩa dùng tốt ngoài thực tế.

Field shift không phải một corruption. Đó là tổ hợp illumination, nền, cách chụp, blur, occlusion, style và yếu tố khác.

Xử lý ảnh có thể giúp và có thể hại. Mọi cải thiện phải được xác nhận bằng metric nhận diện.

Shortcut từ nền và lỗi representation đều có thể xảy ra. Không giải thích mọi lỗi bằng một nguyên nhân.

External dataset phải độc lập nếu dùng làm bằng chứng generalization.

Kết quả âm vẫn là kết quả. CLAHE, denoising hoặc sharpening làm giảm điểm ở một số điều kiện thì phải báo, không giấu.
