from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch


class Compose:
    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


class ResizeWithPad:
    def __init__(self, size: int, fill: tuple[int, int, int] = (0, 0, 0)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        scale = self.size / max(width, height)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        image = image.resize(new_size, Image.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - image.size[0]) // 2
        top = (self.size - image.size[1]) // 2
        canvas.paste(image, (left, top))
        return canvas


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image


class RandomRotation:
    def __init__(self, degrees: float = 10.0):
        self.degrees = degrees

    def __call__(self, image: Image.Image) -> Image.Image:
        angle = random.uniform(-self.degrees, self.degrees)
        return image.rotate(angle, resample=Image.BICUBIC, fillcolor=(0, 0, 0))


class RandomResizedCrop:
    def __init__(self, size: int, scale=(0.8, 1.0), ratio=(0.9, 1.1)):
        self.size = int(size)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        area = width * height
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect = random.uniform(*self.ratio)
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))
            if 0 < crop_w <= width and 0 < crop_h <= height:
                left = random.randint(0, width - crop_w)
                top = random.randint(0, height - crop_h)
                image = image.crop((left, top, left + crop_w, top + crop_h))
                return image.resize((self.size, self.size), Image.BICUBIC)
        return ResizeWithPad(self.size)(image)


class ColorJitter:
    def __init__(self, brightness=0.15, contrast=0.15, saturation=0.15):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.brightness > 0:
            factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast > 0:
            factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        if self.saturation > 0:
            factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            image = ImageEnhance.Color(image).enhance(factor)
        return image


class ToTensor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


class RandomErasing:
    def __init__(self, p: float = 0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value: float = 0.0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        _, h, w = tensor.shape
        area = h * w
        for _ in range(10):
            target = random.uniform(*self.scale) * area
            aspect = random.uniform(*self.ratio)
            erase_w = int(round(math.sqrt(target * aspect)))
            erase_h = int(round(math.sqrt(target / aspect)))
            if 0 < erase_w < w and 0 < erase_h < h:
                top = random.randint(0, h - erase_h)
                left = random.randint(0, w - erase_w)
                tensor[:, top : top + erase_h, left : left + erase_w] = self.value
                return tensor
        return tensor


def build_train_transform(size: int, mean: Sequence[float], std: Sequence[float]):
    return Compose(
        [
            RandomResizedCrop(size=size),
            RandomHorizontalFlip(),
            RandomRotation(10),
            ColorJitter(),
            ToTensor(),
            Normalize(mean, std),
            RandomErasing(),
        ]
    )


def build_eval_transform(size: int, mean: Sequence[float], std: Sequence[float]):
    return Compose([ResizeWithPad(size), ToTensor(), Normalize(mean, std)])

