import os
from abc import ABC, abstractmethod
from pathlib import Path

import hdrio
import numpy as np
from PIL import Image as PILImage

from adapted.illumination_estimation.StyleLight.evaluation.tonemap import (
    tonemap_hdr_panorama,
)
from adapted.illumination_estimation.StyleLight.warping import warp_hdr_panorama
from constants import TEMP_PATH


class IlluminationEstimator(ABC):
    """Base class for Illumination Estimation from a single LDR low-FOV image."""

    def __call__(self, image: PILImage.Image, save_path: Path) -> None:
        """Estimates a HDR panorama from a single LDR low-FOV image, and applies
        post-processing steps. This image can be used for lighting of 3D scenes.

        The image is saved to the specified path and as a `.exr` file.
        """
        os.makedirs(TEMP_PATH, exist_ok=True)
        hdr_panorama = self.model_inference(image)

        assert hdr_panorama is not None, "HDR panorama is None"

        hdr_panorama_warped = warp_hdr_panorama(hdr_panorama)
        hdr_panorama_tonemapped = tonemap_hdr_panorama(
            hdr_panorama_warped, temp_save_path=save_path
        )
        hdrio.imsave(str(save_path.resolve()), hdr_panorama_tonemapped)

    @abstractmethod
    def model_inference(self, image: PILImage.Image) -> np.ndarray | None:
        """Pass input for inference through the Illumination Estimation model.
        Must generate a HDR panorama from a single LDR low-FOV image which can be
        used for lighting of 3D scenes.
        """
        pass
