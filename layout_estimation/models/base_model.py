from abc import ABC, abstractmethod

import numpy as np
from PIL import Image as PILImage


class BaseLayoutEstimator(ABC):
    @abstractmethod
    def estimate_walls(self, image: PILImage.Image) -> list[np.ndarray]:
        pass
