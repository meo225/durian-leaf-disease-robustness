# ImageNet-C-bar (vendored)

- Nguồn: https://github.com/facebookresearch/augmentation-corruption, thư mục `imagenet_c_bar/`, commit `ba4d5a5e9132fe98fcd4be1ac90abedbbb188794`.
- Bài báo: Mintun, Kirillov, Xie. *On Interaction Between Augmentations and Corruptions in Natural Corruption Robustness*, NeurIPS 2021 (arXiv 2102.11273). Đây là tài liệu R33 của đồ án.
- Giấy phép: MIT, bản gốc ở `LICENSE` cùng thư mục. Giữ nguyên dòng bản quyền Facebook trong từng file.

## Thay đổi so với bản gốc

1. `corrupt.py` chỉ giữ 10 họ có tag `imagenet_c_bar` (single_frequency_greyscale, cocentric_sine_waves, plasma_noise, perlin_noise, blue_noise_sample, brownish_noise, checkerboard_cutout, sparkles, inverse_sparkles, caustic_refraction) và thêm bảng `IMAGENET_C_BAR_FAMILIES`. Thân các lớp giữ nguyên từng dòng.
2. `base.py` chỉ giữ lớp `Transform` và đường gọi `__call__`. Các hàm `convert_*` bị bỏ vì không dùng và gọi `np.int` (đã bị xóa khỏi numpy 2).
3. Import đổi từ mô-đun cấp cao nhất (`from base import ...`, `from utils.image import ...`) sang import tương đối.
4. `image_utils.py` và `perlin_noise.py` là `utils/image.py` và `utils/perlin_noise.py`, chỉ thêm một dòng chú thích nguồn.
5. `imagenet_c_bar.csv` là bảng năm mức độ nghiêm trọng của bản gốc (thang 0 đến 10).

`tests/test_external_corruption.py` so đầu ra của bản này với mã băm lấy từ mã gốc (cùng ảnh, cùng seed), nên chứng minh các thay đổi trên không đổi kết quả.
