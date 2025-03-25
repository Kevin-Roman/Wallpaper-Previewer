import os
import shutil
from pathlib import Path
from typing import Type

import cv2
import numpy as np
from cv2.typing import MatLike
from PIL import Image as PILImage

from constants import BLENDER_SCENE_PATH, TEMP_PATH
from src.common import LayoutSegmentationLabels, LayoutSegmentationLabelsOnlyWalls
from src.interfaces import IlluminationEstimator, RoomLayoutEstimator, WallSegmenter
from src.models.illumination_estimation import DualStyleGANIlluminationEstimator
from src.models.room_layout_estimation import FCNAugmentedRoomLayoutEstimator
from src.models.wall_segmentation import EncoderDecoderPPMWallSegmenter
from src.rendering import estimate_wall_and_render_material


class SurfacePreviewer:
    def __init__(
        self,
        room_layout_estimator: Type[
            RoomLayoutEstimator
        ] = FCNAugmentedRoomLayoutEstimator,
        wall_segmenter: Type[WallSegmenter] = EncoderDecoderPPMWallSegmenter,
        illumination_estimator: Type[
            IlluminationEstimator
        ] = DualStyleGANIlluminationEstimator,
    ) -> None:
        self.layout_estimator = room_layout_estimator()
        self.wall_segmenter = wall_segmenter()
        self.illumination_estimator = illumination_estimator()

    def apply_wallpaper(
        self,
        room_image_pil: PILImage.Image,
        wallpaper_image_pil: PILImage.Image,
        selected_walls: set[LayoutSegmentationLabelsOnlyWalls],
    ) -> PILImage.Image:
        """Applies a wallpaper to the selected walls, perspective warping the wallpaper
        image to the quadrilateral-shaped wall, and masking out the non-wall detected
        elements/pixels.
        """

        segmented_wall_mask = self.wall_segmenter(room_image_pil).astype(np.uint8)

        # CV2 uses BGR whilst PIL use RGB. Therefore, convert RGB images to BGR.
        room_image_cv2 = cv2.cvtColor(np.array(room_image_pil), cv2.COLOR_RGB2BGR)
        wallpaper_image_cv2 = cv2.cvtColor(
            np.array(wallpaper_image_pil), cv2.COLOR_RGB2BGR
        )

        output_image = room_image_cv2
        estimate_room_layout = self.layout_estimator(room_image_pil)

        for selected_wall in selected_walls:
            selected_wall_plane_mask = estimate_room_layout[selected_wall]

            wallpaper_applied_image = self.__apply_wallpaper_to_selected_wall(
                room_image_cv2,
                wallpaper_image_cv2,
                segmented_wall_mask,
                selected_wall_plane_mask,
            )
            if wallpaper_applied_image is None:
                continue

            output_image = wallpaper_applied_image

        return PILImage.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

    def render_and_apply_surface(
        self,
        room_image_pil: PILImage.Image,
        selected_walls: set[LayoutSegmentationLabelsOnlyWalls],
    ) -> PILImage.Image | None:
        try:
            os.makedirs(TEMP_PATH, exist_ok=True)
            temp_room_image_path = TEMP_PATH / "room_image.png"
            temp_hdri_path = TEMP_PATH / "hdri.exr"
            temp_render_output_path = TEMP_PATH / "render_output.png"

            # Creates an exr hdr panorama file.
            self.illumination_estimator(room_image_pil, temp_hdri_path)

            output_image = room_image_pil.copy()
            room_layout_estimation = self.layout_estimator(room_image_pil)

            # Temporarily save the room image for estimate_wall_and_render_material.
            room_image_pil.save(temp_room_image_path)

            for selected_wall in selected_walls:
                selected_wall_plane_mask = room_layout_estimation[selected_wall]
                if not (
                    corners := self.layout_estimator.estimate_wall_corners(
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
        transfer_local_highlights: bool = True,
    ) -> PILImage.Image | None:
        """Applies an overlay image over a source room image for chosen walls."""
        room_image_pil_resized = room_image_pil.resize(overlay_image_pil.size)
        selected_wall_plane_mask_resized = cv2.resize(
            selected_wall_plane_mask.astype(np.uint8), overlay_image_pil.size
        )

        if not (
            quadrilateral_plane := (
                self.layout_estimator.quadrilateral_plane(
                    selected_wall_plane_mask_resized
                )
            )
        ):
            return

        quadrilateral_plane_mask, _ = quadrilateral_plane

        segmented_wall_mask = self.wall_segmenter(room_image_pil_resized).astype(
            np.uint8
        )

        # CV2 uses BGR whilst PIL use RGB. Therefore, convert RGB images to BGR.
        room_image_cv2 = cv2.cvtColor(
            np.array(room_image_pil_resized), cv2.COLOR_RGB2BGR
        )
        overlay_image_cv2 = cv2.cvtColor(np.array(overlay_image_pil), cv2.COLOR_RGB2BGR)

        mask = cv2.bitwise_and(
            quadrilateral_plane_mask,
            segmented_wall_mask,
        )

        if transfer_local_highlights:
            overlay_image_cv2 = self.transfer_lighting_and_shadows(
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

    def __apply_wallpaper_to_selected_wall(
        self,
        room_image_cv2: MatLike,
        wallpaper_image_cv2: MatLike,
        segmented_wall_mask: MatLike,
        selected_wall_plane_mask: MatLike,
    ) -> MatLike | None:
        """Transforms and applies a wallpaper onto a wall region in an image."""
        if not (
            quadrilateral_plane := (
                self.layout_estimator.quadrilateral_plane(selected_wall_plane_mask)
            )
        ):
            return

        quadrilateral_plane_mask, corner_coords = quadrilateral_plane

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
            (room_image_cv2.shape[1], room_image_cv2.shape[0]),
        )

        # Combine the masks, extend to [0, 255] range, and expand dimensions to
        # match the 3-channel image.
        combined_mask = cv2.merge(
            [cv2.bitwise_and(quadrilateral_plane_mask, segmented_wall_mask) * 255] * 3
        )

        # Blend the wallpaper with the original image.
        output_image = room_image_cv2.copy()
        cv2.copyTo(warped_wallpaper, combined_mask, output_image)

        return output_image

    @staticmethod
    def transfer_lighting_and_shadows(
        source_image: MatLike, mask: MatLike, target_image: MatLike
    ) -> MatLike:
        """Extracts lighting/shadows from source image and overlays them over the target
        image.
        """
        # Calculate the base colour of the selected area from the source image.
        base_color = cv2.mean(source_image, mask=mask)
        base_color_bgr = np.array(base_color, dtype=np.uint8)

        # Remove base color from the selected area in the image.
        # This should extract the local lightings and shadows.
        extracted_lighting = cv2.subtract(source_image, base_color_bgr)

        # Apply the extracted shadow and lighting on top of the overlay.
        target_with_overlaid_lighting = cv2.addWeighted(
            target_image, 1, extracted_lighting, 1, 0
        )
        # Remove non mask areas from the overlay with lighting applied image.
        target_with_overlaid_lighting = cv2.bitwise_and(
            target_with_overlaid_lighting, target_with_overlaid_lighting, mask=mask
        )

        return target_with_overlaid_lighting


if __name__ == "__main__":
    previewer = SurfacePreviewer()
    if surface_applied := previewer.render_and_apply_surface(
        PILImage.open(Path("./data/1a98599d3f7d168f2cf53e64ad1dd5c6e95e1b64.jpg")),
        set(LayoutSegmentationLabels.walls()),
    ):
        surface_applied.save("./output/final.png")
