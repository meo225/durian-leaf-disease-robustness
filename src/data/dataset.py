"""Dataset định dạng PyTorch cho DurianLDD."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DurianDataset(Dataset):
    """Dataset nạp ảnh lá sầu riêng từ manifest và chuyển đổi thành Tensor."""

    def __init__(
        self,
        split_df: pd.DataFrame,
        image_root: str | Path,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        classes: Sequence[str] | None = None,
    ) -> None:
        self.split_df = split_df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

        if classes is not None:
            self.classes = list(classes)
        else:
            self.classes = sorted(self.split_df["class_name"].unique())

        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.split_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.split_df.iloc[idx]
        image_path = str(row["image_path"])
        full_path = self.image_root / image_path

        if not full_path.is_file():
            raise FileNotFoundError(f"Không thể mở ảnh tại: {full_path}")

        with Image.open(full_path) as img:
            rgb_img = img.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(rgb_img)
        else:
            tensor = transforms.functional.to_tensor(rgb_img)

        class_name = str(row["class_name"])
        label_index = self.class_to_idx[class_name]

        return tensor, label_index, image_path
