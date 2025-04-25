import math
from dataclasses import dataclass
from unittest.mock import patch

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
    def __init__(self, vector: tuple[float, float, float]) -> None:
        self.x = vector[0]
        self.y = vector[1]
        self.z = vector[2]

    @property
    def length_squared(self) -> float:
        return self.x**2 + self.y**2 + self.z**2

    @property
    def length(self) -> float:
        return math.sqrt(self.length_squared)

    def normalized(self) -> "TestVector":
        if (length := self.length) == 0:
            return TestVector((0, 0, 0))

        return TestVector((self.x / length, self.y / length, self.z / length))

    def dot(self, other: "TestVector") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __add__(self, other: "TestVector") -> "TestVector":
        return TestVector((self.x + other.x, self.y + other.y, self.z + other.z))

    def __sub__(self, other: "TestVector") -> "TestVector":
        return TestVector((self.x - other.x, self.y - other.y, self.z - other.z))

    def __mul__(self, other: float) -> "TestVector":
        return TestVector((self.x * other, self.y * other, self.z * other))

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
            world_position_start=TestVector((0, 0, 0)),
            pixel_vector=TestVector((1, 1, 0)),
            distance=5,
            expected_point_on_line=TestVector(
                (5 * (1 / math.sqrt(2)), 5 * (1 / math.sqrt(2)), 0)
            ),
        ),
        Testcase(
            description="starting point not at origin, pixel vector pointing in "
            "positive x and y axis",
            world_position_start=TestVector((1, 1, 1)),
            pixel_vector=TestVector((1, 1, 0)),
            distance=5,
            expected_point_on_line=TestVector((1, 1, 1))
            + TestVector((5 * (1 / math.sqrt(2)), 5 * (1 / math.sqrt(2)), 0)),
        ),
        Testcase(
            description="zero distance",
            world_position_start=TestVector((0, 0, 0)),
            pixel_vector=TestVector((1, 1, 0)),
            distance=0,
            expected_point_on_line=None,
        ),
        Testcase(
            description="negative distance",
            world_position_start=TestVector((0, 0, 0)),
            pixel_vector=TestVector((1, 1, 0)),
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


def test_calculate_point_given_z(subtests: SubTests) -> None:
    @dataclass(frozen=True)
    class Testcase:
        description: str
        world_position_start: TestVector
        direction_vector: TestVector
        target_z: float
        expected_point_give_z: TestVector | None

    testcases = [
        Testcase(
            description="direction vector with non-zero z component",
            world_position_start=TestVector((0, 0, 0)),
            direction_vector=TestVector((1, 1, 1)),
            target_z=5,
            expected_point_give_z=TestVector((5, 5, 5)),
        ),
        Testcase(
            description="direction vector with zero z component",
            world_position_start=TestVector((0, 0, 0)),
            direction_vector=TestVector((1, 1, 0)),
            target_z=5,
            expected_point_give_z=None,
        ),
        Testcase(
            description="direction vector with non-zero z component; negative z target",
            world_position_start=TestVector((0, 0, 0)),
            direction_vector=TestVector((1, 1, 1)),
            target_z=-5,
            expected_point_give_z=TestVector((-5, -5, -5)),
        ),
        Testcase(
            description="direction vector with negative z component",
            world_position_start=TestVector((0, 0, 0)),
            direction_vector=TestVector((-1, -1, -1)),
            target_z=5,
            expected_point_give_z=TestVector((5, 5, 5)),
        ),
        Testcase(
            description="target z of zero",
            world_position_start=TestVector((0, 0, 0)),
            direction_vector=TestVector((1, 1, 1)),
            target_z=0,
            expected_point_give_z=TestVector((0, 0, 0)),
        ),
        Testcase(
            description="direction vector with non-zero z component; non-origin start "
            "point",
            world_position_start=TestVector((1, 1, 1)),
            direction_vector=TestVector((1, 1, 1)),
            target_z=5,
            expected_point_give_z=TestVector((5, 5, 5)),
        ),
    ]

    for testcase in testcases:
        with subtests.test(testcase.description):
            args = (
                testcase.world_position_start,
                testcase.direction_vector,
                testcase.target_z,
            )

            if testcase.expected_point_give_z is None:
                pytest.raises(ValueError, calculate_point_given_z, *args)
                continue

            assert calculate_point_given_z(*args) == testcase.expected_point_give_z


def test_calculate_point_given_x_and_y(subtests: SubTests) -> None:
    @dataclass(frozen=True)
    class Testcase:
        description: str
        world_position_start: TestVector
        direction_vector: TestVector
        target_x: float
        target_y: float
        expected_point_given_x_and_y: TestVector | None

    testcases = [
        Testcase(
            description="direction vector with non-zero x and y components",
            world_position_start=TestVector((0, 0, 0)),
            direction_vector=TestVector((1, 1, 1)),
            target_x=5,
            target_y=5,
            expected_point_given_x_and_y=TestVector((5, 5, 5)),
        ),
        Testcase(
            description="direction vector with zero x component",
            world_position_start=TestVector((0, 0, 0)),
            direction_vector=TestVector((0, 1, 1)),
            target_x=5,
            target_y=5,
            expected_point_given_x_and_y=None,
        ),
        Testcase(
            description="direction vector with zero y component",
            world_position_start=TestVector((0, 0, 0)),
            direction_vector=TestVector((1, 0, 1)),
            target_x=5,
            target_y=5,
            expected_point_given_x_and_y=None,
        ),
        Testcase(
            description="direction vector with zero x and y components",
            world_position_start=TestVector((0, 0, 0)),
            direction_vector=TestVector((0, 0, 1)),
            target_x=5,
            target_y=5,
            expected_point_given_x_and_y=None,
        ),
        Testcase(
            description="direction vector with non-zero x and y components; negative "
            "negative x and y target",
            world_position_start=TestVector((0, 0, 0)),
            direction_vector=TestVector((1, 1, 1)),
            target_x=-5,
            target_y=-5,
            expected_point_given_x_and_y=TestVector((-5, -5, -5)),
        ),
        Testcase(
            description="direction vector with negative x and y components",
            world_position_start=TestVector((0, 0, 0)),
            direction_vector=TestVector((-1, -1, -1)),
            target_x=5,
            target_y=5,
            expected_point_given_x_and_y=TestVector((5, 5, 5)),
        ),
        Testcase(
            description="target x and y of zero",
            world_position_start=TestVector((0, 0, 0)),
            direction_vector=TestVector((1, 1, 1)),
            target_x=0,
            target_y=0,
            expected_point_given_x_and_y=TestVector((0, 0, 0)),
        ),
        Testcase(
            description="direction vector with non-zero x and y components; non-origin "
            "start point",
            world_position_start=TestVector((1, 1, 1)),
            direction_vector=TestVector((1, 1, 1)),
            target_x=5,
            target_y=5,
            expected_point_given_x_and_y=TestVector((5, 5, 5)),
        ),
    ]

    for testcase in testcases:
        with subtests.test(testcase.description):
            with patch("src.rendering.material_rendering.Vector", TestVector):
                args = (
                    testcase.world_position_start,
                    testcase.direction_vector,
                    testcase.target_x,
                    testcase.target_y,
                )

                if testcase.expected_point_given_x_and_y is None:
                    pytest.raises(ValueError, calculate_point_given_x_and_y, *args)
                    continue

                assert (
                    calculate_point_given_x_and_y(*args)
                    == testcase.expected_point_given_x_and_y
                )
