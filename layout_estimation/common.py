from enum import IntEnum

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
        "LayoutSegmentationLabels",
        "LayoutSegmentationLabels",
        "LayoutSegmentationLabels",
    ]:
        return cls.WALL_CENTER, cls.WALL_LEFT, cls.WALL_RIGHT
