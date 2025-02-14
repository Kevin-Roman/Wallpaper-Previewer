from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image as PILImage

from ..common import LayoutSegmentationLabels

EPSILON_DEVIATION = 0.02


@dataclass
class PixelPoint:
    row: int
    col: int

    def __iter__(self) -> tuple[int, int]:
        return iter((self.col, self.row))


@dataclass
class WallCorners:
    top_left: PixelPoint
    top_right: PixelPoint
    bottom_right: PixelPoint
    bottom_left: PixelPoint

    def get_corners_clockwise_from_second_quadrant(self) -> np.ndarray:
        """Returns a numpy array representing the corners, ordered
        clockwise from the second quadrant (top-left)."""
        return np.array(
            [
                tuple(self.top_left),
                tuple(self.top_right),
                tuple(self.bottom_right),
                tuple(self.bottom_left),
            ]
        )


class BaseLayoutEstimator(ABC):
    """Base class for layout estimators (predicting the layout of a room, including
    segmenting the different walls/ceiling/floor)
    """

    @abstractmethod
    def estimate_layout(
        self, image: PILImage.Image
    ) -> dict[LayoutSegmentationLabels, np.ndarray]:
        """Estimates the layout of the room using the given predictor.

        Returns a dictionary mapping the wall label to the corresponding boolean
        mask of the wall. Maximum number of walls that can be returned is 3.
        """
        pass

    @staticmethod
    def estimate_wall_corners(mask: np.ndarray) -> WallCorners | None:
        """Find the corners of the wall in a boolean mask.

        Assumes that the wall is the largest closed quadrilateral in the mask."""
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
            assert "No wall found in the image."
            return

        # Select the largest contour based on it's area.
        contour = max(contours, key=cv2.contourArea)

        # `cv2.arcLength` calculates the perimeter of the contour which is a closed
        # shape. Epsilon is the tolerance factor for the contour approximation.
        epsilon = EPSILON_DEVIATION * cv2.arcLength(contour, closed=True)
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
            bottom_left=corners_ordered[0],
            bottom_right=corners_ordered[1],
            top_right=corners_ordered[2],
            top_left=corners_ordered[3],
        )
