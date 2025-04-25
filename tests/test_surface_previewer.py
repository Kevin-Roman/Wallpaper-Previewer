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


def test_get_highlights() -> None:
    source_image = np.array([[[0, 0, 0], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
    target_image = np.zeros((1, 3, 3), dtype=np.uint8)
    mask = np.array([[True, True, False]]).astype(np.uint8)

    highlighted_image = SurfacePreviewer.apply_highlights(
        source_image, mask, target_image
    )

    # The mean colour of the masked area of the source image should be
    # (0 + 255) / 2 = 127.5, which is then subtracted from the masked source image,
    # and then added onto the target image.
    assert np.array_equal(
        highlighted_image,
        np.array(
            [
                [
                    [0, 0, 0],
                    [np.round(127.5), np.round(127.5), np.round(127.5)],
                    [0, 0, 0],
                ]
            ]
        ),
    ), "Highlighted image does not match the expected output"
