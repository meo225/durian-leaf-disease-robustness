# Frozen Foundation-Model Features 2026

DOI: https://doi.org/10.1016/j.atech.2026.102531

## Vai trò trong đồ án

Dùng khi chọn reference model và khi kiểm tra external. Nếu đủ compute, một baseline frozen như DINOv2 giúp biết kết luận về preprocessing có phụ thuộc backbone hay không. Không biến đồ án thành nghiên cứu DINOv2.

## Việc tác giả làm

So conventional model với foundation representation. Một thí nghiệm dùng DINOv2 làm frozen backbone và chỉ train linear probe. Frozen backbone không cập nhật backbone. Linear probe là classifier nhỏ train trên feature của backbone.

Bài còn thử full fine-tuning, low-rank adaptation và weight-space interpolation.

## Kết quả

Frozen DINOv2 đạt khoảng 51,1% trên PlantDoc trong protocol của bài.

Full fine-tuning làm cross-domain accuracy giảm từ khoảng 50,6% xuống 29,4%.

Taxonomic retention kiểm tra model còn giữ thông tin loại cây khi ảnh sang field hay không. DINOv2 giữ đúng crop khoảng 92% trên lá field. CNN thuần khoảng 21% đến 34%. Model có thể thất bại từ representation trước khi phân biệt bệnh.

Hạn chế do chính bài nêu: PlantDoc trong thí nghiệm khá nhỏ. Kết luận tập trung PlantVillage sang PlantDoc và một dataset táo field. Chỉ RGB ảnh tĩnh, chưa đánh giá đủ bệnh giai đoạn sớm.

## Insight

Fine-tune nhiều hơn không đồng nghĩa generalization tốt hơn.

Clean accuracy tốt chưa chắc là reference model tốt nếu mục tiêu là độ bền.

Nếu backbone rất nhạy với texture hoặc pattern riêng của source, chỉ preprocessing có thể không giải quyết mọi lỗi.
