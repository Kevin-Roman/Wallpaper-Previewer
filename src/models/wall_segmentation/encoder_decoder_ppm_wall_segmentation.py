from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms
from numpy.typing import NDArray
from PIL import Image as PILImage

from adapted.wall_segmentation import EncoderDecoderPPMWallSegmentationPredictor
from constants import (
    ENCODER_DECODER_PPM_WALL_SEGMENTATION_WEIGHTS,
    FCN_AUGMENTATION_ROOM_LAYOUT_ESTIMATION_IMAGENET_MEAN,
    FCN_AUGMENTATION_ROOM_LAYOUT_ESTIMATION_IMAGENET_STD,
    TORCH_DEVICE,
)
from src.interfaces.wall_segmentation import WallSegmenter


class EncoderDecoderPPMWallSegmenter(WallSegmenter):
    """Performs wall segmentation in 2D images by classifying pixels as wall or no-wall
    using an encoder-decoder architecture with a dilated ResNet50/101 network.

    The implementation is the approach described in the paper:
        Bjekic, M., Lazovic, A., K, V., Bacanin, N., Zivkovic, M., Kvascev, G., &
        Nikolic, B. (2025).
        "Wall Segmentation in 2D Images Using Convolutional Neural Networks."
        Everseen, Daon, University of Hradec Králové, Singidunum University, University
        of Belgrade.

    Implementation from https://github.com/bjekic/WallSegmentation. The author's shared
    model weights were used in this project.
    """

    def __init__(self) -> None:
        segmentation_module = EncoderDecoderPPMWallSegmentationPredictor(
            ENCODER_DECODER_PPM_WALL_SEGMENTATION_WEIGHTS.encoder,
            ENCODER_DECODER_PPM_WALL_SEGMENTATION_WEIGHTS.decoder,
        )

        self.segmentation_module = segmentation_module.to(TORCH_DEVICE).eval()

    def model_inference(
        self,
        image: PILImage.Image,
    ) -> NDArray[np.bool_]:
        pil_to_tensor = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=FCN_AUGMENTATION_ROOM_LAYOUT_ESTIMATION_IMAGENET_MEAN,
                    std=FCN_AUGMENTATION_ROOM_LAYOUT_ESTIMATION_IMAGENET_STD,
                ),
            ]
        )

        image_np = np.array(image)
        image_data = pil_to_tensor(image_np)
        seg_size = image_np.shape[:2]

        # Speed up inference by disabling gradient tracking.
        with torch.no_grad():
            scores = self.segmentation_module(
                {"image_data": image_data.unsqueeze(0).to(TORCH_DEVICE)},
                seg_size=seg_size,
            )

        _, predictions = torch.max(scores, dim=1)
        predictions = predictions.cpu()[0].numpy()

        # The model was trained to classify the wall as class 0.
        # Therefore, the prediction is inverted to have the wall as class 1.
        return (1 - predictions).astype(bool)


if __name__ == "__main__":
    wall_segmenter = EncoderDecoderPPMWallSegmenter()

    segmentation_mask = wall_segmenter(
        PILImage.open(Path("./data/1a98599d3f7d168f2cf53e64ad1dd5c6e95e1b64.jpg"))
    )
    cv2.imwrite(
        "./output/wall_segmentation.png", segmentation_mask.astype(np.uint8) * 255
    )
