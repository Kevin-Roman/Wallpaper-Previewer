import math
from dataclasses import dataclass

import numpy as np
import pytest
from pytest_subtests import SubTests

from src.rendering.material_rendering import (
    calculate_point_given_x_and_y,
    calculate_point_given_z,
    calculate_point_on_line,
)


# Own implementation of mathutils.Vector is needed due to mathutils being only
# available within Blender.
class TestVector:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.array([self.x, self.y, self.z])))

    def normalized(self) -> "TestVector":
        if (length := self.length) == 0:
            return TestVector(0, 0, 0)

        return TestVector(self.x / length, self.y / length, self.z / length)

    def dot(self, other: "TestVector") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __add__(self, other: "TestVector") -> "TestVector":
        return TestVector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, other: float) -> "TestVector":
        return TestVector(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other: float) -> "TestVector":
        return self.__mul__(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TestVector):
            return NotImplemented

        return (
            math.isclose(self.x, other.x)
            and math.isclose(self.y, other.y)
            and math.isclose(self.z, other.z)
        )

    def __repr__(self) -> str:
        return f"TestVector({self.x}, {self.y}, {self.z})"


def test_calculate_point_on_line(subtests: SubTests) -> None:
    @dataclass(frozen=True)
    class Testcase:
        description: str
        world_position_start: TestVector
        pixel_vector: TestVector
        distance: float
        expected_point_on_line: TestVector | None

    testcases = [
        Testcase(
            description="starting point at origin, pixel vector pointing in positive x"
            "and y axis",
            world_position_start=TestVector(0, 0, 0),
            pixel_vector=TestVector(1, 1, 0),
            distance=5,
            expected_point_on_line=TestVector(
                5 * (1 / math.sqrt(2)), 5 * (1 / math.sqrt(2)), 0
            ),
        ),
        Testcase(
            description="starting point not at origin, pixel vector pointing in positive x"
            "and y axis",
            world_position_start=TestVector(1, 1, 1),
            pixel_vector=TestVector(1, 1, 0),
            distance=5,
            expected_point_on_line=TestVector(1, 1, 1)
            + TestVector(5 * (1 / math.sqrt(2)), 5 * (1 / math.sqrt(2)), 0),
        ),
        Testcase(
            description="zero distance",
            world_position_start=TestVector(0, 0, 0),
            pixel_vector=TestVector(1, 1, 0),
            distance=0,
            expected_point_on_line=None,
        ),
        Testcase(
            description="negative distance",
            world_position_start=TestVector(0, 0, 0),
            pixel_vector=TestVector(1, 1, 0),
            distance=-5,
            expected_point_on_line=None,
        ),
    ]

    for testcase in testcases:
        with subtests.test(testcase.description):
            args = (
                testcase.world_position_start,
                testcase.pixel_vector,
                testcase.distance,
            )

            if testcase.expected_point_on_line is None:
                pytest.raises(ValueError, calculate_point_on_line, *args)
                continue

            assert calculate_point_on_line(*args) == testcase.expected_point_on_line
