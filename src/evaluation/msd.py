"""Khoảng cách MSD (minimal sample distance) giữa một họ suy giảm và một bộ augmentation (task 5.7).

Theo Mintun, Kirillov, Xie (NeurIPS 2021, R33). Mọi đại lượng đo trên cùng một mô hình tham chiếu, huấn luyện chỉ với augmentation mặc định (Swin-Tiny:
lật và xoay), nên khoảng cách phản ánh thứ mô hình chưa từng học.

- Đặc trưng chuyển đổi (transform feature) của phép t trên mô hình f̂: f(t) = E_x[f̂(t(x)) − f̂(x)], lấy trung bình trên K ảnh sạch, f̂ là lớp ẩn cuối trước đầu phân loại.
  Mỗi mẫu augmentation là một cặp (phép, cường độ) cố định, áp lên cùng K ảnh. Đặc trưng của một họ suy giảm là trung bình đặc trưng của ba mức của nó.
- d_MSD(A, C) = min trên các mẫu a của A của ‖f(a) − f(C)‖₂.
- Bài báo dùng 100 ảnh, tới hàng chục nghìn mẫu augmentation, và báo độ lệch chuẩn khoảng 5% của MSD. Ở đây mặc định 100 ảnh và 1000 mẫu. Khoảng tin cậy bootstrap lấy lại mẫu
  theo ảnh (cùng chỉ số ảnh cho mọi mẫu và cho họ), nên phản ánh sai số do chỉ có K ảnh.
- Ảnh giữ nguyên (đặc trưng 0) không thuộc tập mẫu: nếu có thì họ nhẹ nào cũng "gần" augmentation chỉ vì gần phép không làm gì.

Khác biệt so với bài báo (ghi trong docs/task-5.7-msd-family-distance.md): bài báo xét ImageNet và nhiều mô hình, ở đây một mô hình, 5 lớp, ảnh lá sầu riêng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import TrivialAugmentWide, _apply_op

from src.corruption.augment import jitter_value
from src.corruption.benchmark import ROOT, corrupt_tensor

ImageTransform = Callable[[torch.Tensor], torch.Tensor]

# Phép không làm gì: đặc trưng 0, loại khỏi tập mẫu (xem đầu file).
TA_IDENTITY_OP = "Identity"


def penultimate_features(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Đặc trưng lớp ẩn cuối trước đầu phân loại của model timm (Swin-Tiny, ConvNeXt-Tiny)."""
    if not (hasattr(model, "forward_features") and hasattr(model, "forward_head")):
        raise ValueError("Cần model timm có forward_features và forward_head.")
    return model.forward_head(model.forward_features(batch), pre_logits=True)


class Featurizer:
    """Đặc trưng lớp ẩn cuối của một model, tính theo batch. Ảnh vào là tensor 0 đến 1, chuẩn hóa nằm trong lớp này."""

    def __init__(
        self,
        model: nn.Module,
        mean: Sequence[float],
        std: Sequence[float],
        device: torch.device,
        batch_size: int = 64,
        feature_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] = penultimate_features,
    ) -> None:
        self.model = model.eval().to(device)
        self.device = device
        self.batch_size = int(batch_size)
        self.feature_fn = feature_fn
        self.mean = torch.tensor(list(mean), dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(list(std), dtype=torch.float32).view(1, -1, 1, 1)
        self._clean: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Đặc trưng N×d (float32, trên CPU) của N ảnh 0 đến 1."""
        out = []
        for start in range(0, images.shape[0], self.batch_size):
            batch = ((images[start : start + self.batch_size].float() - self.mean) / self.std).to(self.device)
            out.append(self.feature_fn(self.model, batch).float().cpu())
        return torch.cat(out)

    def clean(self, images: torch.Tensor) -> torch.Tensor:
        """Đặc trưng ảnh sạch, tính một lần cho mỗi tensor ảnh."""
        key = id(images)
        if key not in self._clean:
            self._clean[key] = self.features(images)
        return self._clean[key]

    def deltas(self, images: torch.Tensor, transform: ImageTransform) -> torch.Tensor:
        """f̂(t(x)) − f̂(x) cho từng ảnh, dạng K×d. `transform` nhận một ảnh C×H×W trong 0 đến 1."""
        transformed = torch.stack([transform(image) for image in images])
        return self.features(transformed) - self.clean(images)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Mẫu augmentation


def trivial_augment_ops(num_magnitude_bins: int = 31) -> dict[str, tuple[torch.Tensor, bool]]:
    """Không gian phép của TrivialAugmentWide trong torchvision đang cài: {tên phép: (các cường độ, có dấu)}."""
    return TrivialAugmentWide(num_magnitude_bins=num_magnitude_bins)._augmentation_space(num_magnitude_bins)


def sample_trivial_augment(n: int, seed: int, num_magnitude_bins: int = 31, include_identity: bool = False) -> list[tuple[str, float]]:
    """n mẫu (phép, cường độ có dấu) theo đúng cách `TrivialAugmentWide.forward` lấy ngẫu nhiên: phép đều, cường độ đều trong lưới, dấu đều.

    Mặc định bỏ phép Identity (xem đầu file). Dùng seed riêng nên không phụ thuộc bộ sinh số toàn cục.
    """
    space = trivial_augment_ops(num_magnitude_bins)
    names = [name for name in space if include_identity or name != TA_IDENTITY_OP]
    generator = torch.Generator().manual_seed(int(seed))
    samples = []
    for _ in range(n):
        name = names[int(torch.randint(len(names), (1,), generator=generator))]
        magnitudes, signed = space[name]
        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), generator=generator)]) if magnitudes.ndim > 0 else 0.0
        if signed and int(torch.randint(2, (1,), generator=generator)):
            magnitude = -magnitude
        samples.append((name, magnitude))
    return samples


def apply_trivial_augment(image: torch.Tensor, op_name: str, magnitude: float) -> torch.Tensor:
    """Áp một mẫu TA lên ảnh C×H×W trong 0 đến 1, qua ảnh uint8 như khi nó chạy trên ảnh 8 bit. Nội suy và fill mặc định của TrivialAugmentWide."""
    as_uint8 = torch.round(image.clamp(0.0, 1.0) * 255).to(torch.uint8)
    out = _apply_op(as_uint8, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
    return out.float() / 255.0


def sample_degradation_augmentation(
    conditions: Mapping[str, Mapping[str, Any]],
    n: int,
    seed: int,
    jitter: float = 0.1,
) -> list[dict[str, Any]]:
    """n mẫu từ augmentation suy giảm của đối chứng 4.5: điều kiện chọn đều, giá trị nhân hệ số 1 ± jitter (`jitter_value`, đúng công thức của `RandomCorruption`).

    Mỗi mẫu giữ tên điều kiện và họ gốc, để loại một họ khi đo leave-one-out.
    """
    names = list(conditions)
    if not names:
        raise ValueError("Không có điều kiện nào để lấy mẫu.")
    generator = torch.Generator().manual_seed(int(seed))
    samples = []
    for index in range(n):
        name = names[int(torch.randint(len(names), (1,), generator=generator))]
        spec = conditions[name]
        value = jitter_value(spec, jitter, float(torch.rand(1, generator=generator)))
        samples.append({"type": spec["type"], "value": value, "family": spec["family"], "source_condition": name, "sample": index})
    return samples


# ---------------------------------------------------------------------------------------------------------------------------------------
# MSD


@dataclass
class AugmentationPool:
    """Các mẫu augmentation đã đo: `deltas` là N×K×d, `labels[i]` mô tả mẫu i (họ gốc hoặc phép TA)."""

    name: str
    deltas: torch.Tensor
    labels: list[str]
    meta: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.deltas.ndim != 3 or self.deltas.shape[0] != len(self.labels):
            raise ValueError("deltas phải là N×K×d và labels phải có N phần tử.")


@dataclass
class MSDResult:
    value: float
    nearest: str
    ci_low: float
    ci_high: float
    n_samples: int

    def as_dict(self) -> dict[str, Any]:
        return {"msd": self.value, "nearest": self.nearest, "ci_low": self.ci_low, "ci_high": self.ci_high, "n_samples": self.n_samples}


def build_degradation_pool(
    featurizer: Featurizer,
    images: torch.Tensor,
    conditions: Mapping[str, Mapping[str, Any]],
    n_samples: int,
    seed: int,
    jitter: float = 0.1,
    name: str = "degradation_aug",
) -> AugmentationPool:
    """Đo N mẫu của augmentation suy giảm. Họ của mẫu là nhãn, để loại từng họ khi tính leave-one-out."""
    samples = sample_degradation_augmentation(conditions, n_samples, seed, jitter)
    deltas = torch.stack(
        [featurizer.deltas(images, lambda image, s=s: corrupt_tensor(image, f"msd_{name}_{s['sample']}", s, seed)) for s in samples]
    )
    return AugmentationPool(name, deltas, [s["family"] for s in samples], samples)


def build_trivial_augment_pool(
    featurizer: Featurizer,
    images: torch.Tensor,
    n_samples: int,
    seed: int,
    num_magnitude_bins: int = 31,
) -> AugmentationPool:
    """Đo N mẫu của TrivialAugmentWide (bản wide, 31 mức), không có Identity."""
    samples = sample_trivial_augment(n_samples, seed, num_magnitude_bins)
    deltas = torch.stack([featurizer.deltas(images, lambda image, s=s: apply_trivial_augment(image, s[0], s[1])) for s in samples])
    return AugmentationPool("trivialaugment_wide", deltas, [op for op, _ in samples], [{"op": op, "magnitude": m} for op, m in samples])


def family_deltas(
    featurizer: Featurizer,
    images: torch.Tensor,
    conditions: Mapping[str, Mapping[str, Any]],
    family: str,
    seed: int,
) -> torch.Tensor:
    """Đặc trưng chuyển đổi từng ảnh (K×d) của một họ: trung bình trên các mức của họ đó."""
    members = {name: spec for name, spec in conditions.items() if spec["family"] == family}
    if not members:
        raise KeyError(f"Không có điều kiện nào của họ {family}.")
    levels = [featurizer.deltas(images, lambda image, n=name, s=spec: corrupt_tensor(image, n, s, seed)) for name, spec in members.items()]
    return torch.stack(levels).mean(dim=0)


def _distances(pool_means: torch.Tensor, family_mean: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pool_means - family_mean, dim=1)


def msd(
    pool: AugmentationPool,
    family_delta: torch.Tensor,
    exclude_labels: Sequence[str] = (),
    n_bootstrap: int = 200,
    seed: int = 0,
    confidence: float = 0.95,
) -> MSDResult:
    """MSD từ họ có đặc trưng `family_delta` (K×d) tới `pool`, bỏ các mẫu có nhãn trong `exclude_labels` (leave-one-out).

    Khoảng tin cậy: lấy lại mẫu K ảnh có hoàn lại, cùng chỉ số cho mọi mẫu và cho họ, rồi tính lại MSD (phân vị). `n_bootstrap = 0` bỏ khoảng này (ci = NaN).
    """
    keep = [i for i, label in enumerate(pool.labels) if label not in set(exclude_labels)]
    if not keep:
        raise ValueError("Không còn mẫu augmentation nào sau khi loại.")
    deltas = pool.deltas[keep]
    if family_delta.shape != deltas.shape[1:]:
        raise ValueError(f"family_delta {tuple(family_delta.shape)} không khớp K×d của pool {tuple(deltas.shape[1:])}.")
    distances = _distances(deltas.mean(dim=1), family_delta.mean(dim=0))
    best = int(torch.argmin(distances))
    low = high = float("nan")
    if n_bootstrap > 0:
        generator = torch.Generator().manual_seed(int(seed))
        k = family_delta.shape[0]
        draws = []
        for _ in range(n_bootstrap):
            counts = torch.bincount(torch.randint(k, (k,), generator=generator), minlength=k).float() / k
            draws.append(float(_distances(torch.tensordot(deltas, counts, dims=([1], [0])), counts @ family_delta).min()))
        alpha = (1.0 - confidence) / 2.0
        low, high = float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))
    return MSDResult(float(distances[best]), pool.labels[keep[best]], low, high, len(keep))


def leave_one_out_msd(
    pool: AugmentationPool,
    deltas_by_family: Mapping[str, torch.Tensor],
    n_bootstrap: int = 200,
    seed: int = 0,
) -> dict[str, MSDResult]:
    """MSD của từng họ đã thấy tới augmentation suy giảm đã bỏ chính họ đó: độ xa của một họ "mới" so với 13 họ còn lại.

    Một pool chung rút từ cả 14 họ, bỏ mẫu của họ đang xét, tương đương rút lại từ 13 họ còn lại vì điều kiện được chọn đều.
    """
    return {family: msd(pool, delta, exclude_labels=[family], n_bootstrap=n_bootstrap, seed=seed) for family, delta in deltas_by_family.items()}


def pairwise_distances(family_deltas_by_name: Mapping[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Khoảng cách ‖f(A) − f(B)‖₂ giữa đặc trưng của từng cặp họ (đối xứng, đường chéo 0). Dùng để gom họ thành nhóm cơ chế bằng số (chọn nhóm bỏ ở task 5.10)."""
    means = {name: delta.mean(dim=0) for name, delta in family_deltas_by_name.items()}
    return {a: {b: float(torch.linalg.vector_norm(means[a] - means[b])) for b in means} for a in means}


def near_far_threshold(loo: Mapping[str, MSDResult]) -> float:
    """Ngưỡng gần/xa đăng ký trước: giá trị lớn nhất của MSD leave-one-out trên các họ đã thấy.

    Một họ mới là "gần" nếu không xa augmentation hơn họ đã thấy xa nhất khi bị bỏ ra, tức nằm trong tầm mà tập huấn luyện từng phủ. Quy tắc này nhạy với họ ngoại lai
    nên mọi số MSD liên tục đều được báo cùng, và `classify_near_far` đánh dấu họ có khoảng bootstrap chứa ngưỡng là mơ hồ.
    """
    if not loo:
        raise ValueError("Không có kết quả leave-one-out.")
    return max(result.value for result in loo.values())


def classify_near_far(result: MSDResult, threshold: float) -> str:
    """`near` hoặc `far` theo giá trị MSD. `ambiguous` nếu khoảng bootstrap của MSD chứa ngưỡng (không đủ bằng chứng để xếp nhóm)."""
    if not np.isnan(result.ci_low) and result.ci_low <= threshold <= result.ci_high:
        return "ambiguous"
    return "near" if result.value <= threshold else "far"


def select_images(pipeline_cfg: Mapping[str, Any], split: str, n: int, seed: int) -> torch.Tensor:
    """n ảnh của một split (validation hoặc train, không bao giờ test), chọn bằng hoán vị seed cố định, trả về tensor N×3×H×W trong 0 đến 1."""
    from src.data.dataset import DurianDataset
    from src.data.splits import get_split
    from src.data.transforms import ConditionalResize

    if split == "test":
        raise PermissionError("MSD là phân tích thiết kế, không đo trên tập test. Dùng val hoặc train.")
    root = Path(pipeline_cfg["image_root"])
    dataset = DurianDataset(
        split_df=get_split(split),
        image_root=root if root.is_absolute() else ROOT / root,
        transform=transforms.Compose([ConditionalResize(int(pipeline_cfg["image_size"])), transforms.ToTensor()]),
        classes=pipeline_cfg["classes"],
    )
    if n > len(dataset):
        raise ValueError(f"Chỉ có {len(dataset)} ảnh trong split {split}, cần {n}.")
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(int(seed)))[:n]
    return torch.stack([dataset[int(i)][0] for i in order])
