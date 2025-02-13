from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image as PILImage


class BaseWallSegmenter(ABC):
    """Base class for wall segmenters (classifying pixels as wall or no-wall)."""

    @abstractmethod
    def segment_wall(
        self,
        image: PILImage.Image,
    ) -> np.ndarray:
        """Classifies pixels of image into either wal or no-wall.

        Returns a bool mask where True represents pixels that are considered walls."""
        pass
