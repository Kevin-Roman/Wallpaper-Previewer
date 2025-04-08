from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Literal

import numpy as np


class LayoutSegmentationLabels(IntEnum):
    WALL_CENTRE = 0
    WALL_LEFT = 1
    WALL_RIGHT = 2
    FLOOR = 3
    CEILING = 4

    @classmethod
    def walls(
        cls,
    ) -> tuple[
        "LayoutSegmentationLabelsOnlyWalls",
        "LayoutSegmentationLabelsOnlyWalls",
        "LayoutSegmentationLabelsOnlyWalls",
    ]:
        return cls.WALL_CENTRE, cls.WALL_LEFT, cls.WALL_RIGHT


LayoutSegmentationLabelsOnlyWalls = Literal[
    LayoutSegmentationLabels.WALL_CENTRE,
    LayoutSegmentationLabels.WALL_LEFT,
    LayoutSegmentationLabels.WALL_RIGHT,
]


@dataclass
class PixelPoint:
    row: int
    col: int

    def __iter__(self) -> Iterator[int]:
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
