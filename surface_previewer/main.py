from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from layout_estimation import (
    FCNLayoutAugLayoutEstimator,
    BaseLayoutEstimator,
    LayoutSegmentationLabels,
    LayoutSegmentationLabelsOnlyWalls,
)
from wall_segmentation import EncoderDecoderPPMWallSegmenter, BaseWallSegmenter


class SurfacePreviewer:
    def __init__(self) -> None:
        self.wall_segmenter: BaseWallSegmenter = EncoderDecoderPPMWallSegmenter()
        self.layout_estimator: BaseLayoutEstimator = FCNLayoutAugLayoutEstimator()

    def apply_wallpaper(
        self,
        room_image_pil: PILImage.Image,
        wallpaper_image_pil: PILImage.Image,
        selected_walls: set[LayoutSegmentationLabelsOnlyWalls],
    ) -> PILImage.Image:
        """Applies a wallpaper to the selected walls, perspective warping the wallpaper
        image to the wall quadrilateral shape, and masking out the non-wall elements.
        """

        output_image = room_image_pil
        for selected_wall in selected_walls:
            selected_wall_mask = self.layout_estimator.estimate_layout(room_image_pil)[
                selected_wall
            ]

            wallpaper_applied_image = self.__apply_wallpaper_to_selected_wall(
                output_image, wallpaper_image_pil, selected_wall_mask
            )

            if not wallpaper_applied_image:
                continue

            output_image = wallpaper_applied_image

        return output_image

    def __apply_wallpaper_to_selected_wall(
        self,
        room_image_pil: PILImage.Image,
        wallpaper_image_pil: PILImage.Image,
        selected_wall_mask: np.ndarray,
    ) -> PILImage.Image | None:
        """Transforms and applies a wallpaper onto a wall region in an image."""
        wall_corners = self.layout_estimator.estimate_wall_corners(selected_wall_mask)

        if not wall_corners:
            return

        corner_coords = (
            wall_corners.get_corners_clockwise_from_second_quadrant().astype(np.float32)
        )

        # CV2 uses BGR whilst PIL use RGB. Therefore, convert RGB images to BGR.
        room_image_cv2 = cv2.cvtColor(np.array(room_image_pil), cv2.COLOR_RGB2BGR)
        wallpaper_image_cv2 = cv2.cvtColor(
            np.array(wallpaper_image_pil), cv2.COLOR_RGB2BGR
        )

        # Determine the width and height of the bounding box of the quadrilateral of
        # the wall.
        # `np.linalg.norm` computes the euclidean distance.
        width, height = int(
            max(
                np.linalg.norm(corner_coords[1] - corner_coords[0]),
                np.linalg.norm(corner_coords[2] - corner_coords[3]),
            )
        ), int(
            max(
                np.linalg.norm(corner_coords[2] - corner_coords[1]),
                np.linalg.norm(corner_coords[3] - corner_coords[0]),
            )
        )

        # Define the source quadrilateral for the perspective warp.
        src_coords = np.array(
            ((0, 0), (width, 0), (width, height), (0, height)), dtype=np.float32
        )

        perspective_warp_transformation = cv2.getPerspectiveTransform(
            src_coords, corner_coords
        )

        resized_wallpaper = cv2.resize(wallpaper_image_cv2, (width, height))
        warped_wallpaper = cv2.warpPerspective(
            resized_wallpaper,
            perspective_warp_transformation,
            (room_image_pil.width, room_image_pil.height),
        )

        # Create a mask for blending.
        mask = np.zeros((room_image_pil.height, room_image_pil.width), dtype=np.uint8)
        cv2.fillPoly(mask, [corner_coords.astype(np.int32)], 1)

        wall_mask = self.wall_segmenter.segment_wall(room_image_cv2).astype(np.uint8)

        # Combine the masks, extend to [0, 255] range, and expand dimensions to
        # match the 3-channel image.
        combined_mask = cv2.merge([cv2.bitwise_and(mask, wall_mask) * 255] * 3)

        # Blend the wallpaper with the original image.
        output_image = room_image_cv2.copy()
        cv2.copyTo(warped_wallpaper, combined_mask, output_image)

        # Convert back to RGB.
        return PILImage.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    previewer = SurfacePreviewer()
    wallpaper_applied = previewer.apply_wallpaper(
        PILImage.open(Path("./data/1a98599d3f7d168f2cf53e64ad1dd5c6e95e1b64.jpg")),
        PILImage.open(Path("./data/wallpaper.png")),
        set([LayoutSegmentationLabels.WALL_CENTER]),
    )
    wallpaper_applied.save("./output/wallpaper_applied.png")
