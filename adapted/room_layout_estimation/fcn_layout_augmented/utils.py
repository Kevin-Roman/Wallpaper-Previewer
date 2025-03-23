import torch
import torchvision.transforms.functional as F
from PIL import Image as PILImage


class ProcessImage:
    """Image class for loading and parsing images."""

    @staticmethod
    def parse(target_size: int, image: PILImage.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image_tensor = F.to_tensor(image)
        image_tensor = F.resize(
            image_tensor,
            (target_size, target_size),
        )
        image_tensor = F.normalize(image_tensor, mean=0.5, std=0.5)
        return image_tensor
