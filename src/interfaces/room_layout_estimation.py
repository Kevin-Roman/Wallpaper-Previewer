from abc import ABC, abstractmethod

import cv2
import numpy as np
from cv2.typing import MatLike
from PIL import Image as PILImage

from constants import ARC_LENGTH_DEVIATION_TOLERANCE

from ..common import LayoutSegmentationLabelsOnlyWalls, PixelPoint, WallCorners


class RoomLayoutEstimator(ABC):
    """Base class for room layout estimators (predicting the layout of a room, including
    segmenting the different walls/ceiling/floor)
    """

    @abstractmethod
    def __call__(
        self, image: PILImage.Image
    ) -> dict[LayoutSegmentationLabelsOnlyWalls, MatLike]:
        """Estimates the layout of the room using the given predictor.

        Returns a dictionary mapping the wall label to the corresponding boolean
        mask of the wall. Maximum number of walls that can be returned is 3.

        If no quadrilateral plane is detected, None is returned.
        """
        pass

    @staticmethod
    def quadrilateral_plane(
        selected_wall_plane_mask: MatLike,
    ) -> tuple[MatLike, np.ndarray] | None:
        if not (
            wall_corners := RoomLayoutEstimator.estimate_wall_corners(
                selected_wall_plane_mask
            )
        ):
            return

        corner_coords = (
            wall_corners.get_corners_clockwise_from_second_quadrant().astype(np.float32)
        )

        # Create a mask for blending.
        mask = np.zeros_like(selected_wall_plane_mask, dtype=np.uint8)
        cv2.fillPoly(mask, [corner_coords.astype(np.int32)], 1)

        return mask, corner_coords

    @staticmethod
    def estimate_wall_corners(mask: MatLike) -> WallCorners | None:
        """Finds the corners of the wall in a boolean mask.

        Assumes that the wall is the largest closed quadrilateral in the mask.
        """
        # Convert to an image so that it can be passed through cv2 functions.
        image = mask.astype(np.uint8) * 255

        # Finds the boundaries of objects in the given image.
        # - `cv2.RETR_EXTERNAL`: Retrieves only the outermost contours (ignores child
        #   contours).
        # - `cv2.CHAIN_APPROX_SIMPLE`: Compresses the contours, to not store redundant
        #   points.
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return

        # Select the largest contour based on it's area.
        contour = max(contours, key=cv2.contourArea)

        # `cv2.arcLength` calculates the perimeter of the contour which is a closed
        # shape. Epsilon is the tolerance factor for the contour approximation.
        epsilon = ARC_LENGTH_DEVIATION_TOLERANCE * cv2.arcLength(contour, closed=True)
        # Approximates the contour into a polygon.
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True)

        # The polygon must be a quadrilateral to be considered a wall.
        # This applies even if only the wall is only partially visible in the image.
        if len(polygon) != 4:
            assert "Wall is not a quadrilateral."
            return

        # Reshape into columns of (x, y), and then swap to be (y, x).
        corners = tuple(
            PixelPoint(row=pair[1], col=pair[0]) for pair in polygon.reshape(-1, 2)
        )

        center = PixelPoint(
            row=np.mean([corner.row for corner in corners]),
            col=np.mean([corner.col for corner in corners]),
        )

        # Order the corners based on angle between the center and the corner
        # (unit circle [-pi, pi])
        corners_ordered = sorted(
            corners,
            key=lambda corner: np.arctan2(
                corner.row - center.row, corner.col - center.col
            ),
        )

        return WallCorners(
            top_left=corners_ordered[0],
            top_right=corners_ordered[1],
            bottom_right=corners_ordered[2],
            bottom_left=corners_ordered[3],
        )
