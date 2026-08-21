from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import build_eval_transform, build_train_transform
from .utils import ensure_dir, save_csv, save_json


@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    class_name: str
    split: str | None = None


def scan_dataset(data_dir: str | Path) -> list[Sample]:
    data_dir = Path(data_dir)
    class_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    class_names = []
    for name in ("georges", "non_georges"):
        if (data_dir / name).is_dir():
            class_names.append(name)
    if not class_names:
        class_names = sorted(p.name for p in class_dirs)
    samples: list[Sample] = []
    for class_name in class_names:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            continue
        label = 1 if class_name == "georges" else 0
        for path in sorted(class_dir.glob("*")):
            if path.is_file():
                samples.append(Sample(path=str(path), label=label, class_name=class_name))
    return samples


def resolve_data_dir(data_dir: str | Path) -> Path:
    path = Path(data_dir).expanduser()
    if path.is_dir():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    alt = (repo_root / path).resolve()
    if alt.is_dir():
        return alt
    raise FileNotFoundError(f"Could not find data directory: {path} or {alt}")


def split_samples(samples: Sequence[Sample], seed: int = 42, val_size: float = 0.1, test_size: float = 0.1):
    labels = [s.label for s in samples]
    train_val, test = train_test_split(
        list(samples),
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    train_labels = [s.label for s in train_val]
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_labels,
        random_state=seed,
    )
    return {"train": list(train), "val": list(val), "test": list(test)}


def compute_mean_std(samples: Sequence[Sample], image_size: int) -> tuple[list[float], list[float]]:
    sums = np.zeros(3, dtype=np.float64)
    sq_sums = np.zeros(3, dtype=np.float64)
    count = 0
    from .transforms import ResizeWithPad, ToTensor

    resize = ResizeWithPad(image_size)
    to_tensor = ToTensor()
    for sample in samples:
        with Image.open(sample.path) as image:
            tensor = to_tensor(resize(image))
        arr = tensor.numpy().reshape(3, -1)
        sums += arr.sum(axis=1)
        sq_sums += (arr ** 2).sum(axis=1)
        count += arr.shape[1]
    mean = sums / count
    var = np.maximum(sq_sums / count - mean ** 2, 1e-6)
    std = np.sqrt(var)
    return mean.tolist(), std.tolist()


class ImageDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, torch.tensor(sample.label, dtype=torch.float32), sample.path


def build_dataloaders(
    data_dir: str | Path,
    image_size: int,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
):
    data_dir = resolve_data_dir(data_dir)
    samples = scan_dataset(data_dir)
    splits = split_samples(samples, seed=seed)
    mean, std = compute_mean_std(splits["train"], image_size)
    train_ds = ImageDataset(splits["train"], transform=build_train_transform(image_size, mean, std))
    val_ds = ImageDataset(splits["val"], transform=build_eval_transform(image_size, mean, std))
    test_ds = ImageDataset(splits["test"], transform=build_eval_transform(image_size, mean, std))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    meta = {
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "class_names": ["non_georges", "georges"],
    }
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "splits": splits,
        "meta": meta,
    }


def export_splits(splits: dict[str, Sequence[Sample]], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    rows = []
    for split_name, split_samples in splits.items():
        for sample in split_samples:
            rows.append(
                {
                    "split": split_name,
                    "path": sample.path,
                    "label": sample.label,
                    "class_name": sample.class_name,
                }
            )
    save_csv(output_dir / "splits.csv", rows, fieldnames=["split", "path", "label", "class_name"])
    save_json(output_dir / "split_counts.json", {k: len(v) for k, v in splits.items()})
