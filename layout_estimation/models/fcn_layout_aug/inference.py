from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from ...constants import DEVICE, LayoutSegmentationLabels
from ..base_model import BaseLayoutEstimator
from .constants import MODEL_IMAGE_SIZE, WEIGHT_PATH
from .model import LayoutSeg
from .utils import ProcessImage

torch.backends.cudnn.benchmark = True


class FCNLayoutAugLayoutEstimator(BaseLayoutEstimator):
    def __init__(self) -> None:
        self.weight_path = WEIGHT_PATH
        self.predictor = Predictor(self.weight_path)

    def estimate_walls(
        self,
        image: PILImage.Image,
        model_image_size: int = MODEL_IMAGE_SIZE,
    ) -> list[np.ndarray]:
        image, shape = ProcessImage.parse(model_image_size, image)

        label_mask = cv2.resize(
            self.predictor.feed(image), shape, interpolation=PILImage.NEAREST
        )
        walls_masks = [
            np.isin(label_mask, wall_side)
            for wall_side in LayoutSegmentationLabels.walls()
        ]

        return walls_masks

    @staticmethod
    def find_wall_corners(mask: np.ndarray) -> np.ndarray:
        """Find the corners of the wall in the mask.

        Assumes that the wall is the largest closed quadrilateral in the mask."""
        # TODO: explain the output format/order of the corners.
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            assert "No wall found in the image."

        contour = max(contours, key=cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        if len(approx) != 4:
            assert "Wall is not a quadrilateral."

        corners = approx.reshape(-1, 2)

        corners[[0, 1]] = corners[[1, 0]]
        corners[[2, 3]] = corners[[3, 2]]

        return corners


class Predictor:
    """Predictor for layout estimation."""

    def __init__(self, weight_path: Path) -> None:
        # pylint: disable=E1120
        self.model = LayoutSeg.load_from_checkpoint(
            checkpoint_path=weight_path, backbone="resnet101"
        )
        self.model.freeze()
        self.model.to(DEVICE)

    @torch.no_grad()
    def feed(self, image: torch.Tensor) -> np.ndarray:
        """Feed image to the model and return the multi-class label mask."""
        _, outputs = self.model(image.unsqueeze(0).to(DEVICE))
        return outputs.permute(1, 2, 0).cpu().numpy().squeeze(-1)


if __name__ == "__main__":
    layout_estimator = FCNLayoutAugLayoutEstimator()
    walls_masks = layout_estimator.estimate_walls(
        PILImage.open(Path("./data/1a98599d3f7d168f2cf53e64ad1dd5c6e95e1b64.jpg")),
    )

    cv2.imwrite(
        "./output/wall_layout_segmentation_mask.png",
        walls_masks[0].astype(np.uint8) * 255,
    )
