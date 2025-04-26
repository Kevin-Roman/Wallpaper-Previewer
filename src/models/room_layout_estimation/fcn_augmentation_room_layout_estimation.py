import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from adapted.room_layout_estimation import FCNLayoutAugLayoutEstimationPredictor
from constants import FCN_AUGMENTATION_ROOM_LAYOUT_ESTIMATION_WEIGHTS
from src.common import LayoutSegmentationLabels
from src.interfaces import RoomLayoutEstimator

# The model and input size remains the same, therefore, it may benefit from
# enabling this flag which let's cudnn look for the optimal set of algorithms
# for the particular configuration.
torch.backends.cudnn.benchmark = True

MODEL_IMAGE_SIZE = 320


class FCNAugmentedRoomLayoutEstimator(RoomLayoutEstimator):
    """Deep Fully Connected Network with Layout-Degeneration Augmentation

    This implementation is the approach described in the paper:
        Lin, H. J., Huang, S. W., Lai, S. H., & Chiang, C. K. (2018).
        "Indoor Scene Layout Estimation from a Single Image."
        Proceedings of the 24th International Conference on Pattern Recognition (ICPR).

    Original implementation available at: https://github.com/leVirve/lsun-room.
    Model weights used were retrained by the authors, shared, and used in this project.
    """

    def __init__(self) -> None:
        self._predictor = FCNLayoutAugLayoutEstimationPredictor(
            FCN_AUGMENTATION_ROOM_LAYOUT_ESTIMATION_WEIGHTS
        )

    def model_inference(
        self,
        image: PILImage.Image,
    ) -> dict[LayoutSegmentationLabels, np.ndarray]:
        """Estimates the layout of the room using the given predictor.

        Returns a dictionary mapping the wall label to the corresponding boolean
        mask of the wall. Maximum number of walls that can be returned is 3.
        """
        label_mask = cv2.resize(
            self._predictor.feed(image),
            (image.width, image.height),
            interpolation=PILImage.Resampling.NEAREST,
        )

        return {side: np.isin(label_mask, side) for side in LayoutSegmentationLabels}
