# Giao thức phân chia dữ liệu DurianLDD (Split Protocol)

Tài liệu này quy định nguyên tắc phân chia, quản lý và sử dụng các tập dữ liệu `train`, `val`, `test` cho đồ án phân loại bệnh lá sầu riêng trong điều kiện suy giảm chất lượng ảnh.

---

## 1. Thông số phân chia
- **Bộ dữ liệu:** DurianLDD (4.437 ảnh, 5 lớp nguyên bản: `ALGAL_LEAF_SPOT`, `ALLOCARIDARA_ATTACK`, `HEALTHY_LEAF`, `LEAF_BLIGHT`, `PHOMOPSIS_LEAF_SPOT`). Loại bỏ hoàn toàn bộ dữ liệu ngoài (Vietnamese durian leaf).
- **Hằng số ngẫu nhiên (Seed):** `SEED = 42`.
- **Tỉ lệ mục tiêu:** 70% Train (3.105 ảnh), 10% Validation (444 ảnh), 20% Test (888 ảnh).
- **Phương pháp phân chia:**
  - **Bước 1:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` trên toàn bộ 4.437 ảnh $\rightarrow$ tách 1 fold làm `test` (~20%).
  - **Bước 2:** Trên 80% còn lại, áp dụng `StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=42)` $\rightarrow$ tách 1 fold làm `val` (~10% tổng) và 7 fold làm `train` (~70% tổng).

---

## 2. Quy tắc nhóm gần trùng (Leakage Prevention)
- Dựa trên phân tích mã băm dHash (ngưỡng khoảng cách Hamming $\le 5$), có đúng **4 nhóm gần trùng** gồm tổng cộng **9 ảnh**.
- Để chống rò rỉ dữ liệu (data leakage) giữa các tập, mỗi nhóm được gán một `group_key` cố định để thuật toán đảm bảo **mỗi nhóm gần trùng nằm trọn trong đúng một tập duy nhất**:
  - Nhóm `group_id=26` (2 ảnh, `ALGAL_LEAF_SPOT`): Nằm trọn trong `val`.
  - Nhóm `group_id=517` (3 ảnh, `HEALTHY_LEAF`): Nằm trọn trong `test`.
  - Nhóm `group_id=632` (2 ảnh, `LEAF_BLIGHT`): Nằm trọn trong `train`.
  - Nhóm `group_id=1980` (2 ảnh, `ALLOCARIDARA_ATTACK`): Nằm trọn trong `train`.

---

## 3. Quy tắc cốt lõi khi sử dụng dữ liệu
1. **KHÔNG dùng tập `test` để chọn mô hình, severity hay tham số xử lý ảnh:**
   - Nghiêm cấm sử dụng tập `test` trong bất kỳ quyết định nào liên quan đến lựa chọn kiến trúc mô hình (model selection), mức độ suy giảm chất lượng ảnh (severity), hoặc tinh chỉnh tham số của các kỹ thuật xử lý/khôi phục ảnh (Gamma, CLAHE, Retinex, v.v.).
2. **Mọi lựa chọn thử nghiệm dùng tập `val`:**
   - Tập validation (`val`) là tập duy nhất dùng để so sánh các kỹ thuật xử lý ảnh, chọn siêu tham số, theo dõi quá trình huấn luyện và chọn checkpoint tối ưu.
3. **Tập `test` chỉ chạy một lần ở đánh giá cuối:**
   - Tập `test` được khóa mặc định và bảo vệ bằng hàm `src.data.splits.get_split("test")`. Chỉ khi hoàn thành toàn bộ quá trình phát triển mô hình và bước vào đánh giá tổng kết cuối cùng, cờ môi trường `FINAL_EVAL=1` mới được kích hoạt để chạy đánh giá trên `test` đúng một lần duy nhất.
