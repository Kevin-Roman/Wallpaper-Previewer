"""Sequence classes for loading and parsing sequences of images."""

import torch
import torchvision.transforms.functional as F
from PIL import Image as PILImage


class ProcessImage:
    """Image class for loading and parsing images from a folder."""

    @staticmethod
    def parse(target_size: int, image: PILImage.Image) -> tuple[torch.Tensor, tuple]:
        image = image.convert("RGB")
        shape = image.size
        image = F.to_tensor(image)
        image = F.resize(
            image,
            (target_size, target_size),
            interpolation=PILImage.Resampling.BILINEAR,
        )
        image = F.normalize(image, mean=0.5, std=0.5)
        return image, shape
