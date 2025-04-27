from .room_layout_estimation.hedau import evaluate_hedau
from .room_layout_estimation.lsun_room import evaluate_lsun_room
from .utils import calculate_metrics, save_box_plot
from .wall_segmentation.ade20k import evaluate_ade20k


def evaluate_room_layout_estimation():
    hedau_pixel_errors = evaluate_hedau()
    lsun_pixel_errors = evaluate_lsun_room()

    data = [(hedau_pixel_errors, "Hedau"), (lsun_pixel_errors, "LSUN-Room")]

    save_box_plot(
        data,
        title="Pixel Error Distributions with Bipartite Matching",
        y_label="Pixel Error with Bipartite Matching",
    )

    calculate_metrics(data)


def evaluate_wall_segmentation():
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


if __name__ == "__main__":
    # evaluate_room_layout_estimation()
    evaluate_wall_segmentation()
