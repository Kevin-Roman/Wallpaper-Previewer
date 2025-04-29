import os

# Needs to be set before cv2 is imported.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import time
from pathlib import Path

from PIL import Image as PILImage

from src.app.surface_previewer import TexturePreviewer, WallpaperPreviewer
from src.common import LayoutSegmentationLabels

from .room_layout_estimation.hedau import evaluate_hedau, get_hedau_test_data
from .room_layout_estimation.lsun_room import evaluate_lsun_room
from .utils import calculate_metrics, save_box_plot
from .wall_segmentation.ade20k import evaluate_ade20k


def evaluate_room_layout_estimation() -> None:
    hedau_pixel_errors = evaluate_hedau()
    lsun_pixel_errors = evaluate_lsun_room()

    data = [(hedau_pixel_errors, "Hedau"), (lsun_pixel_errors, "LSUN-Room")]

    save_box_plot(
        data,
        title="Pixel Error Distributions with Bipartite Matching",
        y_label="Pixel Error with Bipartite Matching",
    )

    calculate_metrics(data)


def evaluate_wall_segmentation() -> None:
    ade20k_pixel_errors, ade20k_iou_losses = evaluate_ade20k()
    pixel_error_data = [(ade20k_pixel_errors, "ADE20k")]
    iou_losses_data = [(ade20k_iou_losses, "ADE20k")]

    save_box_plot(
        pixel_error_data,
        title="Pixel Error Distribution",
        y_label="Pixel Error",
        y_max=1.0,
        nbins=20,
    )

    save_box_plot(
        [(ade20k_iou_losses, "ADE20k")],
        title="IoU Loss Distribution",
        y_label="IoU Loss (1 - IoU)",
        y_max=1.0,
        nbins=20,
    )

    print("\nPixel Error Distribution")
    calculate_metrics(pixel_error_data)

    print("\nIoU Loss Distribution")
    calculate_metrics(iou_losses_data)


def evaluate_wallpaper_previewing_pipeline_speed() -> None:
    images = get_hedau_test_data()[0]

    wallpaper_previewer = WallpaperPreviewer()

    elapsed_times: list[float] = []
    for image in images:
        start = time.perf_counter()
        wallpaper_previewer(
            PILImage.fromarray(image),
            PILImage.open(Path("data/wallpapers/white_paint.png")),
            set(LayoutSegmentationLabels.walls()),
        )
        end = time.perf_counter()
        elapsed_times.append(end - start)

    print(f"Average time: {sum(elapsed_times) / len(elapsed_times):.4f} seconds")


def evaluate_texture_previewing_pipeline_speed() -> None:
    images = get_hedau_test_data()[0]

    texture_previewer = TexturePreviewer()

    elapsed_times: list[float] = []
    for image in images[:5]:
        start = time.perf_counter()
        texture_previewer(
            PILImage.fromarray(image),
            set(LayoutSegmentationLabels.walls()),
        )
        end = time.perf_counter()
        elapsed_times.append(end - start)

    print(f"Average time: {sum(elapsed_times) / len(elapsed_times):.4f} seconds")


if __name__ == "__main__":
    # evaluate_room_layout_estimation()
    # evaluate_wall_segmentation()
    # evaluate_wallpaper_previewing_pipeline_speed()
    evaluate_texture_previewing_pipeline_speed()
