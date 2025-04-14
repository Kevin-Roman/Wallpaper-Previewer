from abc import ABC, abstractmethod

from PIL import Image as PILImage
from numpy.typing import NDArray

import numpy as np


class WallSegmenter(ABC):
    """Base class for wall segmentation estimators (classifying pixels as wall or
    no-wall)."""

    def __call__(self, image: PILImage.Image) -> NDArray[np.bool_]:
        """Classifies pixels of image into either wall or no-wall.
        Returns a bool mask where True represents pixels that are considered walls."""
        return self.model_inference(image)

    @abstractmethod
    def model_inference(
        self,
        image: PILImage.Image,
    ) -> NDArray[np.bool_]:
        """Pass input for inference through the Wall Segmentation model.

        Must returns a bool mask where True represents pixels that are considered walls.
        """
        pass
