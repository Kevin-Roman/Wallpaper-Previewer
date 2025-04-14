from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from adapted.illumination_estimation import stylelight_generate_hdr_panorama
from constants import TEMP_PATH
from src.interfaces import IlluminationEstimator


class DualStyleGANIlluminationEstimator(IlluminationEstimator):
    """StyleLight: HDR Panorama Generation for Lighting Estimation and Editing

    This implementation follows the methodology described in the paper:
        Wang, G., Yang, Y., Loy, C. C., & Liu, Z. (2022).
        “StyleLight: HDR Panorama Generation for Lighting Estimation and Editing.”
        Proceedings of the European Conference on Computer Vision (ECCV) 2022.

    Original implementation available at: https://github.com/Wanggcong/StyleLight.
    Model weights provided by the authors are used in this project.
    """

    def model_inference(self, image: PILImage.Image) -> np.ndarray | None:
        return stylelight_generate_hdr_panorama(image)


if __name__ == "__main__":
    image = PILImage.open(Path("./data/0a578e8af1642d0c1e715aaa04478858ac0aab01.jpg"))
    estimator = DualStyleGANIlluminationEstimator()
    save_path = TEMP_PATH / "test.exr"
    estimator(image, save_path)
