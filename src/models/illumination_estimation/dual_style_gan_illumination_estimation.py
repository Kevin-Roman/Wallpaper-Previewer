from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from adapted.illumination_estimation import stylelight_generate_hdr_panorama
from constants import TEMP_PATH
from src.interfaces import IlluminationEstimator


class DualStyleGANIlluminationEstimator(IlluminationEstimator):
    """Estimates a HDR panorama from a single LDR low-FOV image using StyleLight."""

    def model_inference(self, image: PILImage.Image) -> np.ndarray | None:
        """Estimates a HDR panorama from a single LDR low-FOV image using the StyleLight
        method.
        """
        return stylelight_generate_hdr_panorama(image)


if __name__ == "__main__":
    image = PILImage.open(Path("./data/0a578e8af1642d0c1e715aaa04478858ac0aab01.jpg"))
    estimator = DualStyleGANIlluminationEstimator()
    save_path = TEMP_PATH / "test.exr"
    estimator(image, save_path)
