from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision.transforms

from .constants import DEVICE, IMAGENET_MEAN, IMAGENET_STD
from .models import SegmentationModule


def segment_image(
    segmentation_module: SegmentationModule,
    img: Path | PIL.Image.Image,
):
    """
    Function for segmenting wall in the input image. The input can be path to image, or
    a loaded image
    """
    pil_to_tensor = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    if isinstance(img, Path):
        img = PIL.Image.open(img)

    img_original = np.array(img)
    img_data = pil_to_tensor(img)
    singleton_batch = {"img_data": img_data[None].to(DEVICE)}
    seg_size = img_original.shape[:2]

    with torch.no_grad():
        scores = segmentation_module(singleton_batch, seg_size=seg_size)

    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()

    return pred
