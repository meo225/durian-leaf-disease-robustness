# Tài liệu nghiên cứu

Đề tài: Nghiên cứu ảnh hưởng của các kỹ thuật xử lý ảnh đến khả năng nhận diện bệnh trên lá sầu riêng trong điều kiện chất lượng ảnh biến đổi.

Không có một bài báo nào trùng câu hỏi nghiên cứu của đồ án. Hướng đang follow là ghép vài nguồn theo vai trò.

| Vai trò | Nguồn | Dùng ở |
|---|---|---|
| Cách đánh giá | Reliability Gap 2026 | Khung thí nghiệm: clean, corrupted, processed |
| Cách chọn baseline | Richter và Kim 2025 | Phase 3 |
| External generalization | DG-PLDR 2025, Frozen Foundation-Model Features 2026 | Phase 6 |
| Giải thích lỗi field | From Laboratory to Field 2023, Background Recomposition 2026 | Failure analysis |
| Bài toán sầu riêng | DurianLSNet 2026, AI-Driven 2026, ETASR 2025 | Dataset, baseline candidate, gap |

## Bản tổng hợp

- [Task 1.2. Nhận diện bệnh thực vật ngoài thực địa](task-1.2-field-recognition.md)
- [Task 1.3. Nhận diện bệnh lá sầu riêng](task-1.3-durian-literature.md)
- [Task 1.4. Experimental protocol](task-1.4-experimental-protocol.md)

## Từng nguồn đã đọc

- [DG-PLDR 2025](sources/dg-pldr-2025.md)
- [From Laboratory to Field 2023](sources/from-laboratory-to-field-2023.md)
- [Reliability Gap 2026](sources/reliability-gap-2026.md)
- [Background Recomposition 2026](sources/background-recomposition-2026.md)
- [Frozen Foundation-Model Features 2026](sources/frozen-foundation-features-2026.md)
- [Richter và Kim 2025](sources/richter-kim-2025.md)
- [DurianLSNet 2026](sources/durianlsnet-2026.md)
- [AI-Driven Durian Leaf Disease Classification 2026](sources/ai-driven-durian-2026.md)
- [Classification of Durian Foliar Diseases 2025](sources/etasr-durian-2025.md)

## Có trong plan, chưa có bản phân tích riêng

Các mục này xuất hiện ở phase sau của plan. Chưa viết thành tài liệu đọc vì chưa nằm trong bản literature review đã chốt.

- LeafCLIP 2026, DOI 10.1016/j.compag.2026.111770. Plan ghi nhận illumination và bóng phức tạp. Không dùng để kết luận Gamma, CLAHE hay Retinex là tối ưu.
- Robust Fusion–Pruning for Apple Leaf Disease 2026, DOI 10.1016/j.rineng.2026.110612. Plan dùng cho cách đo corruption và Grad-CAM++. Không lấy severity của bài đó sang sầu riêng.
- DS-MENet 2022, DOI 10.3389/fpls.2022.884464.
- PlantRefineDet 2023, DOI 10.1016/j.eswa.2023.120717.
- Deep learning for guava anthracnose 2026, DOI 10.1186/s12870-026-09125-1.
