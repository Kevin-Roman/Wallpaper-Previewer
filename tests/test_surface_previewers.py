import numpy as np
from PIL import Image as PILImage

from src.app.surface_previewer import SurfacePreviewer


def test_standardise_image_dimensions() -> None:
    image = PILImage.new("RGB", (1, 1), color=(255, 0, 0))
    output = SurfacePreviewer.standardise_image_dimensions(
        image, target_longest_size_px=1_500
    )

    assert isinstance(output, PILImage.Image), "Output should be a PILImage"
    assert (
        output.width == 1_500 or output.height == 1_500
    ), f"Unexpected shape: {np.array(output).shape}"
