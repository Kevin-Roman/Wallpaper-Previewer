from enum import IntEnum
from typing import Literal

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LayoutSegmentationLabels(IntEnum):
    WALL_CENTER = 0
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
        return cls.WALL_CENTER, cls.WALL_LEFT, cls.WALL_RIGHT


LayoutSegmentationLabelsOnlyWalls = Literal[
    LayoutSegmentationLabels.WALL_CENTER,
    LayoutSegmentationLabels.WALL_LEFT,
    LayoutSegmentationLabels.WALL_RIGHT,
]
