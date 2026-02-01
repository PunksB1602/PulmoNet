from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CheXpertDataset(Dataset):
    """
    CheXpert dataset for multi-label classification. Handles uncertainty policy and label cleaning.
    """

    DEFAULT_LABELS: List[str] = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]

    def __init__(
        self,
        csv_file: Union[str, Path],
        root_dir: Union[str, Path],
        transform=None,
        labels: Optional[Iterable[str]] = None,
        uncertain_policy: str = "ones",
        path_prefix_to_strip: str = "CheXpert-v1.0-small",
        verify_images: bool = False,
    ):
        self.csv_file = Path(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.labels = list(labels) if labels is not None else list(self.DEFAULT_LABELS)

        df = pd.read_csv(self.csv_file)

        path_col = "Path" if "Path" in df.columns else df.columns[0]
        self._paths_raw = df[path_col].astype(str).tolist()

        missing = [c for c in self.labels if c not in df.columns]
        if missing:
            raise ValueError(f"Missing label columns in CSV: {missing}")

        # Coerce to numeric (optional but robust), then fill blanks
        df[self.labels] = df[self.labels].apply(pd.to_numeric, errors="coerce").fillna(0)


        if uncertain_policy == "ones":
            df[self.labels] = df[self.labels].replace(-1, 1)
        elif uncertain_policy == "zeros":
            df[self.labels] = df[self.labels].replace(-1, 0)
        else:
            raise ValueError("uncertain_policy must be 'ones' or 'zeros'")

        # Ensure labels are valid for BCE loss
        df[self.labels] = df[self.labels].clip(lower=0, upper=1)

        self._y = df[self.labels].to_numpy(dtype="float32")

        self.path_prefix_to_strip = path_prefix_to_strip
        if verify_images:
            self._verify_all_paths()

    def __len__(self) -> int:
        return len(self._paths_raw)

    def _normalize_rel_path(self, p: str) -> str:
        p = p.strip().lstrip("./\\")
        prefix = self.path_prefix_to_strip.strip("/\\")
        if p.startswith(prefix + "/") or p.startswith(prefix + "\\"):
            p = p[len(prefix) + 1 :]
        return p

    def _resolve_img_path(self, raw_path: str) -> Path:
        rel = self._normalize_rel_path(raw_path)
        return self.root_dir / rel

    def _verify_all_paths(self) -> None:
        bad = []
        for i, rp in enumerate(self._paths_raw):
            path = (self._resolve_img_path(rp)).resolve()
            if not path.exists():
                bad.append((i, rp, str(path)))
        if bad:
            example = "\n".join([f"- idx={i} csv='{rp}' -> '{p}'" for i, rp, p in bad[:10]])
            raise FileNotFoundError(
                f"Found {len(bad)} missing images. Examples:\n{example}\n"
                f"Check root_dir='{self.root_dir}' and CSV paths."
            )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self._resolve_img_path(self._paths_raw[idx])
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.from_numpy(self._y[idx]).float()
        return image, labels


def get_transforms(mode: str = "train", image_size: int = 224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
