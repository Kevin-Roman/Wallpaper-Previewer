from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from ...common import (
    DEVICE,
    LayoutSegmentationLabels,
    LayoutSegmentationLabelsOnlyWalls,
)
from ..base_model import BaseLayoutEstimator
from .constants import MODEL_IMAGE_SIZE, WEIGHT_PATH
from .model import LayoutSeg
from .utils import ProcessImage

# The model and input size remains the same, therefore, it may benefit from
# enabling this flag which let's cudnn look for the optimal set of algorithms
# for the particular configuration.
torch.backends.cudnn.benchmark = True


class FCNLayoutAugLayoutEstimator(BaseLayoutEstimator):
    """Deep Fully Connected Network with Layout-Degeneration Augmentation

    This implementation is the approach described in the paper:
        Lin, H. J., Huang, S. W., Lai, S. H., & Chiang, C. K. (2018).
        "Indoor Scene Layout Estimation from a Single Image."
        Proceedings of the 24th International Conference on Pattern Recognition (ICPR).

    Implementation from https://github.com/leVirve/lsun-room, and the model weights
    retrained by the authors, shared, and used in this project. Due to the original
    weights not able to be released, the model had to be retrained and unfortunately
    has a lower accuracy than the findings in the paper.
    """

    def __init__(self) -> None:
        self._predictor = Predictor(WEIGHT_PATH)

    def estimate_layout(
        self,
        image: PILImage.Image,
        model_image_size: int = MODEL_IMAGE_SIZE,
    ) -> dict[LayoutSegmentationLabelsOnlyWalls, np.ndarray]:
        """Estimates the layout of the room using the given predictor.

        Returns a dictionary mapping the wall label to the corresponding boolean
        mask of the wall. Maximum number of walls that can be returned is 3.
        """
        image, shape = ProcessImage.parse(model_image_size, image)

        label_mask = cv2.resize(
            self._predictor.feed(image), shape, interpolation=PILImage.NEAREST
        )

        return {
            wall_side: np.isin(label_mask, wall_side)
            for wall_side in LayoutSegmentationLabels.walls()
        }


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
    walls_masks = layout_estimator.estimate_layout(
        PILImage.open(Path("./data/1a98599d3f7d168f2cf53e64ad1dd5c6e95e1b64.jpg")),
    )

    cv2.imwrite(
        "./output/wall_layout_segmentation_mask.png",
        walls_masks[LayoutSegmentationLabels.WALL_CENTER].astype(np.uint8) * 255,
    )
