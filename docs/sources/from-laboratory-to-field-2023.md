# From Laboratory to Field 2023

DOI: https://doi.org/10.34133/plantphenomics.0038

## Vai trò trong đồ án

Dùng để giải thích vì sao ảnh field khó, và vì sao illumination, blur, nền được đưa vào phạm vi. Không suy ra một mức blur hay Gaussian noise cụ thể từ kết quả của bài. Đây là UDA trên domain shift tự nhiên, không phải corruption tổng hợp có kiểm soát.

## Vấn đề

Bài hỏi cách thu hẹp khoảng cách laboratory và field khi được nhìn ảnh target không nhãn. UDA là thích nghi miền không giám sát: source có nhãn, target không nhãn.

Ảnh field khác illumination, sensor, location, background, góc chụp, blur, occlusion và lá không đầy đủ.

Ba nhóm khó khăn:

- Interdomain discrepancy: source và target khác nhau lớn.
- Intraclass discrepancy: cùng một bệnh nhưng hình ảnh đổi theo giai đoạn, tuổi cây hoặc môi trường.
- Fuzzy class boundaries: ranh giới lớp không rõ, hai bệnh có thể trông gần nhau.

## Việc tác giả làm

MSUN gồm ba ý. Multi-representation giữ cả thông tin tổng thể và chi tiết bệnh. Subdomain adaptation dùng LMMD để căn source và target theo cấu trúc lớp, không chỉ ép hai domain giống nhau toàn cục. Uncertainty regularization giảm ảnh hưởng của pseudo-label không chắc. Pseudo-label là nhãn model tự gán cho ảnh target để dùng tạm khi train.

Source là PlantVillage. Target gồm PlantDoc, Plant-Pathology, Corn-Leaf-Diseases và Tomato-Leaf-Diseases.

## Insight

Field shift là tổ hợp nhiều yếu tố. Một brightness corruption không đại diện cả ảnh field.

Chi tiết nhỏ của bệnh quan trọng: texture, biên, hình dạng và pattern tổn thương. Denoising có thể khử nhiễu nhưng làm mờ texture. Sharpening có thể làm biên rõ và cũng có thể tạo artifact. Xử lý ảnh phải được chấm bằng phân loại, không chỉ bằng mắt.

Cùng một corruption có thể ảnh hưởng các lớp khác nhau. Macro F1, per-class recall và confusion matrix có giá trị hơn chỉ accuracy tổng.
