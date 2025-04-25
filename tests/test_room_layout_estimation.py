from dataclasses import dataclass

import numpy as np
from cv2.typing import MatLike
from pytest_subtests import SubTests

from src.common import PixelPoint, WallCorners
from src.interfaces.room_layout_estimation import RoomLayoutEstimator


def test_estimate_wall_corners(subtests: SubTests) -> None:
    @dataclass(frozen=True)
    class Testcase:
        description: str
        mask: np.ndarray
        expected_wall_corners: WallCorners | None

    testcases = [
        Testcase(
            description="non-rectangular wall",
            mask=np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ).astype(bool),
            expected_wall_corners=WallCorners(
                top_left=PixelPoint(row=3, col=1),
                top_right=PixelPoint(row=1, col=3),
                bottom_left=PixelPoint(row=5, col=1),
                bottom_right=PixelPoint(row=5, col=3),
            ),
        ),
        Testcase(
            description="rectangular wall",
            mask=np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ).astype(bool),
            expected_wall_corners=WallCorners(
                top_left=PixelPoint(row=1, col=1),
                top_right=PixelPoint(row=1, col=3),
                bottom_left=PixelPoint(row=5, col=1),
                bottom_right=PixelPoint(row=5, col=3),
            ),
        ),
        Testcase(
            description="wall on edges",
            mask=np.array(
                [
                    [1, 1, 1],
                    [1, 1, 1],
                ]
            ).astype(bool),
            expected_wall_corners=WallCorners(
                top_left=PixelPoint(row=0, col=0),
                top_right=PixelPoint(row=0, col=2),
                bottom_left=PixelPoint(row=1, col=0),
                bottom_right=PixelPoint(row=1, col=2),
            ),
        ),
        Testcase(
            description="wall with only one corner",
            mask=np.array(
                [
                    [1],
                ]
            ).astype(bool),
            expected_wall_corners=None,
        ),
        Testcase(
            description="no wall",
            mask=np.array(
                [
                    [0],
                ]
            ).astype(bool),
            expected_wall_corners=None,
        ),
        Testcase(
            description="wall without 4 clear corners",
            mask=np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ).astype(bool),
            expected_wall_corners=None,
        ),
    ]

    for testcase in testcases:
        with subtests.test(testcase.description):
            wall_corners = RoomLayoutEstimator.estimate_wall_corners(testcase.mask)

            assert wall_corners == testcase.expected_wall_corners


def test_estimate_quadrilateral(subtests: SubTests) -> None:
    @dataclass(frozen=True)
    class Testcase:
        description: str
        mask: np.ndarray
        expect_estimated_quadrilateral: bool
        expected_estimated_quadrilateral_mask: MatLike | None

    testcases = [
        Testcase(
            description="wall with 4 corners but non straight edges",
            mask=np.array(
                [
                    [1, 1, 1, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                ]
            ).astype(bool),
            expect_estimated_quadrilateral=True,
            expected_estimated_quadrilateral_mask=np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        ),
    ]

    for testcase in testcases:
        with subtests.test(testcase.description):
            estimated_quadrilateral = RoomLayoutEstimator.estimate_quadrilateral(
                testcase.mask
            )

            assert (
                estimated_quadrilateral is None
            ) != testcase.expect_estimated_quadrilateral

            if estimated_quadrilateral is None:
                continue

            assert testcase.expected_estimated_quadrilateral_mask is not None

            estimated_quadrilateral_mask, _ = estimated_quadrilateral

            assert np.array_equal(
                estimated_quadrilateral_mask,
                testcase.expected_estimated_quadrilateral_mask,
            )
