import numpy as np

from src.app.surface_previewer import WallpaperPreviewer


def test_fill_rectangle_with_pattern() -> None:
    pattern_image = np.full((10, 10, 3), 255, dtype=np.uint8)
    rectangle_width_pixels = 100
    rectangle_height_pixels = 50
    wall_height_m = 3.0
    pattern_height_m = 2.0

    pattern_filled_rectangle = WallpaperPreviewer.fill_rectangle_with_pattern(
        pattern_image,
        rectangle_width_pixels,
        rectangle_height_pixels,
        wall_height_m,
        pattern_height_m,
    )

    assert pattern_filled_rectangle.shape == (50, 100, 3), (
        "Pattern filled rectangle doesn't match the shape of the desired rectangle "
        f"({rectangle_height_pixels}, {rectangle_width_pixels}, 3)"
    )

    assert np.array_equal(
        pattern_filled_rectangle, np.full((50, 100, 3), 255, dtype=np.uint8)
    ), "Pattern filled rectangle isn't filled with the expected pattern"
