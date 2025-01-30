"""Sequence classes for loading and parsing sequences of images."""

from pathlib import Path

import PIL
import PIL.Image
import torch
import torchvision.transforms.functional as F


class Image:
    """Image class for loading and parsing images from a folder."""

    @staticmethod
    def parse(target_size: int, filepath: Path) -> tuple[torch.Tensor, tuple]:
        image = PIL.Image.open(filepath).convert("RGB")
        shape = image.size
        image = F.to_tensor(image)
        image = F.resize(
            image,
            (target_size, target_size),
            interpolation=PIL.Image.Resampling.BILINEAR,
        )
        image = F.normalize(image, mean=0.5, std=0.5)
        return image, shape
