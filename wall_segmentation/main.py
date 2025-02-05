from pathlib import Path

import cv2
import numpy as np
import PIL
import torch
import torchvision.transforms

from .constants import DEVICE, IMAGENET_MEAN, IMAGENET_STD
from .models import SegmentationModule, build_decoder, build_encoder

WEIGHTS_ENCODER = Path("./wall_segmentation/weights/encoder.pth")
WEIGHTS_DECODER = Path("./wall_segmentation/weights/decoder.pth")


class WallSegmenter:
    def __init__(self) -> None:
        net_encoder = build_encoder(WEIGHTS_ENCODER, encoder_model="resnet101-dilated")
        net_decoder = build_decoder(WEIGHTS_DECODER)

        segmentation_module = SegmentationModule(net_encoder, net_decoder)
        self.segmentation_module = segmentation_module.to(DEVICE).eval()

    def segment_wall(
        self,
        img: Path | PIL.Image.Image,
    ):
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
            scores = self.segmentation_module(singleton_batch, seg_size=seg_size)

        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()

        return pred


if __name__ == "__main__":
    wall_segmenter = WallSegmenter()

    segmentation_mask = wall_segmenter.segment_wall(
        Path("./data/1a98599d3f7d168f2cf53e64ad1dd5c6e95e1b64.jpg")
    )
    cv2.imwrite(
        "./output/wall_segmentation.png", segmentation_mask.astype(np.uint8) * 255
    )
