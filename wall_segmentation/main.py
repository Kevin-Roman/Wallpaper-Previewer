from pathlib import Path

import cv2
import numpy as np

from .constants import DEVICE
from .eval import segment_image
from .models import SegmentationModule, build_decoder, build_encoder

WEIGHTS_ENCODER = Path("./wall_segmentation/weights/encoder.pth")
WEIGHTS_DECODER = Path("./wall_segmentation/weights/decoder.pth")


def predict_wall_segmentation(
    segmentation_module: SegmentationModule,
    path_image: Path,
):
    segmentation_mask = segment_image(segmentation_module, path_image)
    return segmentation_mask


def create_segmentation_module() -> SegmentationModule:
    net_encoder = build_encoder(WEIGHTS_ENCODER, encoder_model="resnet101-dilated")
    net_decoder = build_decoder(WEIGHTS_DECODER)

    segmentation_module = SegmentationModule(net_encoder, net_decoder)
    return segmentation_module.to(DEVICE).eval()


if __name__ == "__main__":
    segmentation_module = create_segmentation_module()

    path_image = Path("./data/1a98599d3f7d168f2cf53e64ad1dd5c6e95e1b64.jpg")
    segmentation_mask = predict_wall_segmentation(segmentation_module, path_image)
    cv2.imwrite(
        "./output/segmentation_mask.jpg", segmentation_mask.astype(np.uint8) * 255
    )
