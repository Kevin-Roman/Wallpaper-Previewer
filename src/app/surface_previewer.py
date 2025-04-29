import os
import shutil
from abc import ABC, abstractmethod

import cv2
import numpy as np
from cv2.typing import MatLike
from PIL import Image as PILImage

from constants import BLENDER_SCENE_PATH, TEMP_PATH
from src.common import LayoutSegmentationLabelsOnlyWalls
from src.interfaces import IlluminationEstimator, RoomLayoutEstimator, WallSegmenter
from src.models.illumination_estimation import DualStyleGANIlluminationEstimator
from src.models.room_layout_estimation import FCNAugmentedRoomLayoutEstimator
from src.models.wall_segmentation import EncoderDecoderPPMWallSegmenter
from src.rendering import estimate_wall_and_render_material


class SurfacePreviewer(ABC):
    """Base class for surface previewers (applying a wallpaper or texture to a room
    image)."""

    def __init__(
        self,
        room_layout_estimator: RoomLayoutEstimator | None = None,
        wall_segmenter: WallSegmenter | None = None,
    ) -> None:
        if room_layout_estimator is None:
            room_layout_estimator = FCNAugmentedRoomLayoutEstimator()

        if wall_segmenter is None:
            wall_segmenter = EncoderDecoderPPMWallSegmenter()

        self.room_layout_estimator = room_layout_estimator
        self.wall_segmenter = wall_segmenter

    @abstractmethod
    def __call__(self, *args, **kwargs) -> PILImage.Image | None:
        """The wallpaper/texture previewing pipeline."""
        pass

    @staticmethod
    def apply_highlights(
        source_image: MatLike, mask: MatLike, target_image: MatLike
    ) -> MatLike:
        """Extracts highlights (lighting/shadows) from source image and overlays them
        over the target image.
        """
        # Calculate the base colour of the selected area from the source image.
        base_colour = cv2.mean(source_image, mask=mask)

        # Remove base colour from the selected area in the image.
        # This should extract the local lightings and shadows.
        extracted_lighting = cv2.subtract(
            source_image, np.array(base_colour, dtype=np.uint8)
        )

        # Apply the extracted shadow and lighting on top of the overlay.
        target_with_overlaid_lighting = cv2.addWeighted(
            target_image, 1, extracted_lighting, 1, 0
        )
        # Remove non mask areas from the overlay with lighting applied image.
        target_with_overlaid_lighting = cv2.bitwise_and(
            target_with_overlaid_lighting, target_with_overlaid_lighting, mask=mask
        )

        return target_with_overlaid_lighting

    @staticmethod
    def standardise_image_dimensions(
        image: PILImage.Image, target_longest_size_px: int = 1_500
    ) -> PILImage.Image:
        """Standardises the dimensions of an image to a target size for it's longest
        dimension."""
        width, height = image.size

        scale_factor = target_longest_size_px / max(width, height)

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        return image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)


class WallpaperPreviewer(SurfacePreviewer):
    def __init__(
        self,
        room_layout_estimator: RoomLayoutEstimator | None = None,
        wall_segmenter: WallSegmenter | None = None,
    ) -> None:
        super().__init__(room_layout_estimator, wall_segmenter)

    def __call__(
        self,
        room_image_pil: PILImage.Image,
        wallpaper_image_pil: PILImage.Image,
        selected_walls: set[LayoutSegmentationLabelsOnlyWalls],
        wall_height_m: float = 2.0,
        pattern_height_m: float = 2.0,
    ) -> PILImage.Image | None:
        """Applies a wallpaper to the selected walls, perspective warping the wallpaper
        image to the quadrilateral-shaped wall, and masking out the non-wall detected
        elements/pixels.
        """
        # Resize to specific target side length as some models (e.g. wall segmenter)
        # perform best at lower resolutions.
        room_image_pil = self.standardise_image_dimensions(room_image_pil)

        segmented_wall_mask = self.wall_segmenter(room_image_pil).astype(np.uint8)

        # CV2 uses BGR whilst PIL use RGB. Therefore, convert RGB images to BGR.
        room_image_cv2 = cv2.cvtColor(np.array(room_image_pil), cv2.COLOR_RGB2BGR)
        wallpaper_image_cv2 = cv2.cvtColor(
            np.array(wallpaper_image_pil), cv2.COLOR_RGB2BGR
        )

        output_image = room_image_cv2
        estimate_room_layout = self.room_layout_estimator(room_image_pil)

        for selected_wall in selected_walls:
            selected_wall_plane_mask = estimate_room_layout[selected_wall]

            wallpaper_applied_image = self.__apply_wallpaper_to_selected_wall(
                output_image,
                wallpaper_image_cv2,
                segmented_wall_mask,
                selected_wall_plane_mask,
                wall_height_m,
                pattern_height_m,
            )
            if wallpaper_applied_image is None:
                continue

            output_image = wallpaper_applied_image

        return PILImage.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

    def __apply_wallpaper_to_selected_wall(
        self,
        room_image_cv2: MatLike,
        wallpaper_image_cv2: MatLike,
        segmented_wall_mask: MatLike,
        selected_wall_plane_mask: MatLike,
        wall_height_m: float = 2.0,
        pattern_height_m: float = 2.0,
        transfer_local_highlights: bool = True,
    ) -> MatLike | None:
        """Transforms and applies a wallpaper onto a wall region in an image."""
        if not (
            estimate_quadrilateral := (
                self.room_layout_estimator.estimate_quadrilateral(
                    selected_wall_plane_mask
                )
            )
        ):
            return

        estimate_quadrilateral_mask, corner_coords = estimate_quadrilateral

        # Determine the width and height of the "bounding box" of the quadrilateral of
        # the wall. `np.linalg.norm` computes the euclidean distance.
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

        tiled_wallpaper = self.fill_rectangle_with_pattern(
            wallpaper_image_cv2, width, height, wall_height_m, pattern_height_m
        )

        perspective_warp_transformation = cv2.getPerspectiveTransform(
            src_coords, corner_coords
        )

        warped_wallpaper = cv2.warpPerspective(
            tiled_wallpaper,
            perspective_warp_transformation,
            (room_image_cv2.shape[1], room_image_cv2.shape[0]),
        )

        combined_mask = cv2.bitwise_and(
            estimate_quadrilateral_mask, segmented_wall_mask
        ).astype(np.uint8)

        if transfer_local_highlights:
            warped_wallpaper = self.apply_highlights(
                room_image_cv2,
                combined_mask,
                warped_wallpaper,
            )

        # Blend the wallpaper with the original image.
        output_image = room_image_cv2.copy()
        cv2.copyTo(warped_wallpaper, combined_mask, output_image)

        return output_image

    @staticmethod
    def fill_rectangle_with_pattern(
        pattern_image: MatLike,
        rectangle_width_pixels: int,
        rectangle_height_pixels: int,
        wall_height_m: float = 2.0,
        pattern_height_m: float = 2.0,
    ):
        """Fills a rectangle with a seamlessly repeatable pattern."""
        pattern_width_m = pattern_height_m * (
            pattern_image.shape[1] / pattern_image.shape[0]
        )

        wall_width_m = (
            rectangle_width_pixels / rectangle_height_pixels
        ) * wall_height_m

        wall_height_pixels = rectangle_height_pixels
        wall_width_pixels = rectangle_width_pixels
        pattern_height_pixels = int(
            wall_height_pixels / (wall_height_m / pattern_height_m)
        )
        pattern_width_pixels = int(wall_width_pixels / (wall_width_m / pattern_width_m))

        pattern_resized = cv2.resize(
            pattern_image, (pattern_width_pixels, pattern_height_pixels)
        )
        # Crop in the edge case where the pattern is large than the wall-rectangle in
        # any of the dimensions.
        pattern_cropped = pattern_resized[
            : min(pattern_resized.shape[0], wall_height_pixels),
            : min(pattern_resized.shape[1], wall_width_pixels),
        ]

        num_repeats_vertical = int(wall_height_pixels // pattern_height_pixels)
        remainder_vertical_pixels = int(wall_height_pixels % pattern_height_pixels)
        num_repeats_horizontal = int(wall_width_pixels // pattern_width_pixels)
        remainder_horizontal_pixels = int(wall_width_pixels % pattern_width_pixels)

        # Repeat pattern within canvas.
        canvas = np.zeros((wall_height_pixels, wall_width_pixels, 3), dtype=np.uint8)

        # Fill first column of canvas with the pattern.
        for i in range(num_repeats_vertical):
            canvas[
                i * pattern_height_pixels : (i + 1) * pattern_height_pixels,
                :pattern_width_pixels,
            ] = pattern_cropped

        # Fill remainder of the first column.
        if remainder_vertical_pixels > 0:
            canvas[-remainder_vertical_pixels:, :pattern_width_pixels] = (
                pattern_resized[-remainder_vertical_pixels:]
            )

        # Repeat first column.
        for i in range(1, num_repeats_horizontal):
            canvas[:, i * pattern_width_pixels : (i + 1) * pattern_width_pixels] = (
                canvas[:, :pattern_width_pixels]
            )

        # Fill remainder of last column.
        if remainder_horizontal_pixels > 0:
            canvas[:, -remainder_horizontal_pixels:] = canvas[
                :, :remainder_horizontal_pixels
            ]

        return canvas


class TexturePreviewer(SurfacePreviewer):
    """Applies a and renders a texture to each selected walls."""

    def __init__(
        self,
        room_layout_estimator: RoomLayoutEstimator | None = None,
        wall_segmenter: WallSegmenter | None = None,
        illumination_estimator: IlluminationEstimator | None = None,
    ):
        super().__init__(room_layout_estimator, wall_segmenter)

        if illumination_estimator is None:
            illumination_estimator = DualStyleGANIlluminationEstimator()

        self.illumination_estimator = illumination_estimator

    def __call__(
        self,
        room_image_pil: PILImage.Image,
        selected_walls: set[LayoutSegmentationLabelsOnlyWalls],
    ) -> PILImage.Image | None:
        room_image_pil = self.standardise_image_dimensions(room_image_pil)

        try:
            os.makedirs(TEMP_PATH, exist_ok=True)
            temp_room_image_path = TEMP_PATH / "room_image.png"
            temp_hdri_path = TEMP_PATH / "hdri.exr"
            temp_render_output_path = TEMP_PATH / "render_output.png"

            # Creates an exr hdr panorama file.
            self.illumination_estimator(room_image_pil, temp_hdri_path)

            output_image = room_image_pil.copy()
            room_layout_estimation = self.room_layout_estimator(room_image_pil)

            # Temporarily save the room image for estimate_wall_and_render_material.
            room_image_pil.save(temp_room_image_path)

            for selected_wall in selected_walls:
                selected_wall_plane_mask = room_layout_estimation[selected_wall]
                if not (
                    corners := self.room_layout_estimator.estimate_wall_corners(
                        selected_wall_plane_mask
                    )
                ):
                    continue

                estimate_wall_and_render_material(
                    BLENDER_SCENE_PATH,
                    temp_room_image_path,
                    temp_hdri_path,
                    temp_render_output_path,
                    corners,
                )

                overlay_image_pil = PILImage.open(temp_render_output_path)
                if (
                    overlay_applied_image := self.__apply_overlay(
                        output_image,
                        overlay_image_pil,
                        selected_wall_plane_mask,
                    )
                ) is None:
                    continue

                output_image = overlay_applied_image

        except Exception as e:
            print(e)
            return
        finally:
            # Delete temporary directory.
            if os.path.exists(TEMP_PATH):
                shutil.rmtree(TEMP_PATH)

        return output_image

    def __apply_overlay(
        self,
        room_image_pil: PILImage.Image,
        overlay_image_pil: PILImage.Image,
        selected_wall_plane_mask: MatLike,
        transfer_local_highlights: bool = False,
    ) -> PILImage.Image | None:
        """Applies an overlay image over a source room image for chosen walls."""
        if not (
            estimate_quadrilateral := (
                self.room_layout_estimator.estimate_quadrilateral(
                    selected_wall_plane_mask
                )
            )
        ):
            return

        quadrilateral_plane_mask, _ = estimate_quadrilateral

        segmented_wall_mask = self.wall_segmenter(room_image_pil).astype(np.uint8)

        # CV2 uses BGR whilst PIL use RGB. Therefore, convert RGB images to BGR.
        room_image_cv2 = cv2.cvtColor(np.array(room_image_pil), cv2.COLOR_RGB2BGR)
        overlay_image_cv2 = cv2.cvtColor(np.array(overlay_image_pil), cv2.COLOR_RGB2BGR)

        mask = cv2.bitwise_and(
            quadrilateral_plane_mask,
            segmented_wall_mask,
        )

        if transfer_local_highlights:
            overlay_image_cv2 = self.apply_highlights(
                room_image_cv2,
                mask,
                overlay_image_cv2,
            )

        output_image = room_image_cv2.copy()
        cv2.copyTo(
            overlay_image_cv2,
            mask,
            output_image,
        )

        return PILImage.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    previewer = WallpaperPreviewer()
