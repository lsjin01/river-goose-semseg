from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.transforms import functional as TF


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
SENSOR_SUFFIXES = (
    "_windshield_vis",
    "_front",
    "_camera_left",
    "_camera_right",
    "_realsense",
)


def _strip_sensor_suffix(stem: str) -> str:
    for suffix in SENSOR_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def find_goose_samples(root: Path, split: str) -> List[Tuple[Path, Path]]:
    image_root = root / "images" / split
    label_root = root / "labels" / split
    samples: List[Tuple[Path, Path]] = []
    image_paths = list(image_root.glob("*/*.png")) + list(image_root.glob("*/*.npy"))
    for image_path in sorted(image_paths):
        label_name = _strip_sensor_suffix(image_path.stem) + "_labelids.png"
        label_path = label_root / image_path.parent.name / label_name
        if label_path.exists():
            samples.append((image_path, label_path))
    if not samples:
        raise FileNotFoundError(f"No GOOSE {split!r} samples found under {root}")
    return samples


def _sample_random_crop_box(
    image_size: Tuple[int, int],
    crop_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    image_width, image_height = image_size
    crop_width, crop_height = crop_size
    if crop_width > image_width or crop_height > image_height:
        raise ValueError(
            f"Crop size {crop_size} must be <= image size {(image_width, image_height)}."
        )
    if crop_width == image_width and crop_height == image_height:
        return (0, 0, crop_width, crop_height)

    max_left = image_width - crop_width
    max_top = image_height - crop_height
    left = random.randint(0, max_left) if max_left > 0 else 0
    top = random.randint(0, max_top) if max_top > 0 else 0
    return (left, top, left + crop_width, top + crop_height)


class GooseSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        resize_size: Optional[Iterable[int]] = None,
        flip_prob: float = 0.0,
        enable_random_crop: bool = False,
        crop_size: Optional[Iterable[int]] = None,
        input_channels: int = 3,
        enable_rare_class_crop: bool = False,
        rare_class_crop_prob: float = 0.0,
        rare_class_ids: Optional[Iterable[int]] = None,
        rare_class_min_pixels: int = 0,
        rare_class_min_ratio: float = 0.0,
        rare_class_crop_attempts: int = 10,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.samples = find_goose_samples(self.root, split)
        self.resize_size = tuple(resize_size) if resize_size is not None else None
        self.flip_prob = float(flip_prob)
        self.enable_random_crop = bool(enable_random_crop)
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        self.input_channels = int(input_channels)
        self.enable_rare_class_crop = bool(enable_rare_class_crop)
        self.rare_class_crop_prob = float(rare_class_crop_prob)
        self.rare_class_ids = tuple(sorted({int(class_id) for class_id in (rare_class_ids or [])}))
        self.rare_class_min_pixels = int(rare_class_min_pixels)
        self.rare_class_min_ratio = float(rare_class_min_ratio)
        self.rare_class_crop_attempts = max(1, int(rare_class_crop_attempts))

        if self.enable_random_crop and self.crop_size is None:
            raise ValueError("crop_size must be provided when enable_random_crop=True.")
        if self.crop_size is not None and len(self.crop_size) != 2:
            raise ValueError(f"crop_size must have 2 elements. Got: {self.crop_size}")
        if self.enable_rare_class_crop and not self.enable_random_crop:
            raise ValueError("enable_rare_class_crop=True requires enable_random_crop=True.")

    def _crop_contains_rare_classes(self, label: Image.Image, crop_box: Tuple[int, int, int, int]) -> bool:
        if not self.rare_class_ids:
            return False

        crop_label = np.array(label.crop(crop_box), copy=False)
        rare_mask = np.isin(crop_label, self.rare_class_ids)
        rare_pixels = int(rare_mask.sum())
        if rare_pixels <= 0:
            return False

        if self.rare_class_min_pixels > 0 and rare_pixels < self.rare_class_min_pixels:
            return False

        if self.rare_class_min_ratio > 0.0:
            crop_area = crop_label.shape[0] * crop_label.shape[1]
            if crop_area <= 0:
                return False
            if float(rare_pixels) / float(crop_area) < self.rare_class_min_ratio:
                return False

        return True

    def _choose_crop_box(self, image_size: Tuple[int, int], label: Image.Image) -> Tuple[int, int, int, int]:
        if not self.enable_random_crop or self.crop_size is None:
            return (0, 0, image_size[0], image_size[1])

        crop_box = _sample_random_crop_box(image_size, self.crop_size)
        use_rare_crop = (
            self.split == "train"
            and self.enable_rare_class_crop
            and self.rare_class_crop_prob > 0.0
            and self.rare_class_ids
            and random.random() < self.rare_class_crop_prob
        )
        if not use_rare_crop:
            return crop_box

        for _ in range(self.rare_class_crop_attempts):
            candidate_box = _sample_random_crop_box(image_size, self.crop_size)
            if self._crop_contains_rare_classes(label, candidate_box):
                return candidate_box
        return crop_box

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[index]
        label = Image.open(label_path).convert("L")

        if image_path.suffix == ".npy":
            array = np.load(image_path, allow_pickle=False)
            if array.ndim != 3 or array.shape[0] != self.input_channels:
                raise ValueError(
                    f"Expected {self.input_channels}-channel CHW data, got {array.shape}: {image_path}"
                )
            image_tensor = torch.from_numpy(array.astype(np.float32, copy=False)) / 65535.0
            image = None
            image_size = (int(array.shape[2]), int(array.shape[1]))
        else:
            image = Image.open(image_path).convert("RGB")
            image_size = image.size

        if self.resize_size is not None:
            if image is None:
                image_tensor = F.interpolate(
                    image_tensor[None], size=(self.resize_size[1], self.resize_size[0]),
                    mode="bilinear", align_corners=False,
                )[0]
                image_size = self.resize_size
            else:
                image = image.resize(self.resize_size, Image.BILINEAR)
                image_size = image.size
            label = label.resize(self.resize_size, Image.NEAREST)

        if self.split == "train" and self.enable_random_crop and self.crop_size is not None:
            crop_box = self._choose_crop_box(image_size, label)
            if image is None:
                left, top, right, bottom = crop_box
                image_tensor = image_tensor[:, top:bottom, left:right]
            else:
                image = image.crop(crop_box)
            label = label.crop(crop_box)

        if self.split == "train" and self.flip_prob > 0.0 and random.random() < self.flip_prob:
            if image is None:
                image_tensor = torch.flip(image_tensor, (-1,))
            else:
                image = TF.hflip(image)
            label = TF.hflip(label)

        if image is not None:
            image_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
            image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        label_tensor = torch.from_numpy(np.array(label, copy=True)).long()
        return image_tensor, label_tensor
