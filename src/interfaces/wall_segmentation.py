from abc import ABC, abstractmethod

from cv2.typing import MatLike
from PIL import Image as PILImage


class WallSegmenter(ABC):
    """Base class for wall segmentation estimators (classifying pixels as wall or no-wall)."""

    @abstractmethod
    def __call__(
        self,
        image: PILImage.Image,
    ) -> MatLike:
        """Classifies pixels of image into either wall or no-wall.

        Returns a bool mask where True represents pixels that are considered walls."""
        pass
