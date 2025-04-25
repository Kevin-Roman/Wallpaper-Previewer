from functools import partial, reduce
from operator import add
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image as PILImage

from src.common import PixelPoint, WallCorners

# Desired distance from the camera to the initial point on the top-right line
INITIAL_DISTANCE = 5


def calculate_point_on_line(
    world_position_start: Vector, pixel_vector: Vector, distance: float
) -> Vector:
    """Calculate the point on a line (position + direction)"""
    if distance <= 0:
        raise ValueError("Distance must be greater than 0.")

    return world_position_start + pixel_vector.normalized() * distance


def calculate_point_given_z(
    world_position_start: Vector, direction_vector: Vector, target_z: float
) -> Vector:
    """Calculate the point on a line (position + direction) given the z-coordinate."""
    if direction_vector.z == 0:
        raise ValueError(
            "The line is parallel to the XY plane. Cannot intersect at a specific Z."
        )

    t = (target_z - world_position_start.z) / direction_vector.z
    point = world_position_start + direction_vector * t
    return point


def calculate_point_given_x_and_y(
    world_position_start: Vector,
    direction_vector: Vector,
    target_x: float,
    target_y: float,
) -> Vector:
    """Calculate the point on a line (position + direction) given the x and y
    coordinates.

    As the line might not have a point at the given x and y coordinates, the closest
    point on the line to the target x and y coordinates is determined."""
    target_point = Vector((target_x, target_y, world_position_start.z))
    line_to_target = target_point - world_position_start
    # Project the line to the target point onto the direction vector, getting the
    # distance from the start point to the closest point on the line
    t = line_to_target.dot(direction_vector) / direction_vector.length_squared
    return world_position_start + t * direction_vector


def point_on_line_through_pixel(
    camera: bpy.types.Object,
    camera_resolution_multiplier: float,
    range_x: np.ndarray,
    range_y: np.ndarray,
    frame_z: float,
    pixel_x: int,
    pixel_y: int,
) -> Vector:
    x = range_x[int(pixel_x * camera_resolution_multiplier)]
    y = range_y[int(pixel_y * camera_resolution_multiplier)]

    # Create pixel vector in camera space
    pixel_vector = Vector((x, y, frame_z))

    # Apply camera rotation to the vector
    pixel_vector.rotate(camera.matrix_world.to_quaternion())

    return calculate_point_on_line(
        camera.matrix_world.translation, pixel_vector, INITIAL_DISTANCE
    )


def setup_scene(
    source_image_path: Path,
    hdri_path: Path,
) -> None:
    source_image = PILImage.open(source_image_path)

    scene = bpy.context.scene
    scene.render.resolution_x = source_image.width
    scene.render.resolution_y = source_image.height
    scene.render.resolution_percentage = 100

    # Apply HDRI.
    world = bpy.context.scene.world
    env_tex_node = None
    for node in world.node_tree.nodes:
        if node.type == "TEX_ENVIRONMENT":
            env_tex_node = node
            break

    if env_tex_node is None:
        raise ValueError("No 'Environment Texture' node found in the World shader.")

    env_tex_node.image = bpy.data.images.load(
        str(hdri_path.resolve()), check_existing=True
    )

    # Apply source image for compositing.
    node_tree = bpy.context.scene.node_tree
    image_node = None
    for node in node_tree.nodes:
        if node.type == "IMAGE":  # For the Image Input node
            image_node = node
            break

    if image_node is None:
        raise ValueError("No 'Image' node found in Compositing.")

    image_node.image = bpy.data.images.load(
        str(source_image_path.resolve()), check_existing=True
    )


def setup_plane(wall_corners_pixels: WallCorners) -> None:
    # Retrieve camera and scene data.
    if (camera := bpy.data.objects.get("Camera")) is None:
        raise ValueError("Camera is not found in the scene.")

    scene = bpy.context.scene

    camera_resolution_multiplier = scene.render.resolution_percentage / 100
    camera_resolution_width = int(
        scene.render.resolution_x * camera_resolution_multiplier
    )
    camera_resolution_height = int(
        scene.render.resolution_y * camera_resolution_multiplier
    )

    frame_top_right, _, frame_bottom_left, frame_top_left = camera.data.view_frame(
        scene=scene
    )

    point_on_line_partial = partial(
        point_on_line_through_pixel,
        camera,
        camera_resolution_multiplier,
        np.linspace(frame_top_left[0], frame_top_right[0], camera_resolution_width),
        np.linspace(frame_top_left[1], frame_bottom_left[1], camera_resolution_height),
        frame_top_left[2],
    )
    top_right_point = point_on_line_partial(
        wall_corners_pixels.top_right.col,
        wall_corners_pixels.top_right.row,
    )
    top_left_point = point_on_line_partial(
        wall_corners_pixels.top_left.col,
        wall_corners_pixels.top_left.row,
    )
    bottom_left_point = point_on_line_partial(
        wall_corners_pixels.bottom_left.col,
        wall_corners_pixels.bottom_left.row,
    )

    # Determine the coordinates of all points, starting given the top-right point.
    # The other points can be known based on the fact that a/some component(s) of
    # the point/vector will be shared across other points.
    x1, y1, z1 = top_right_point.x, top_right_point.y, top_right_point.z
    x2, y2, _ = calculate_point_given_z(
        top_left_point,
        (top_left_point - camera.matrix_world.translation).normalized(),
        z1,
    )
    _, _, z3 = calculate_point_given_x_and_y(
        bottom_left_point,
        (bottom_left_point - camera.matrix_world.translation).normalized(),
        x2,
        y2,
    )

    world_wall_corners = [
        Vector((x1, y1, z1)),  # Top right.
        Vector((x2, y2, z1)),  # Top left.
        Vector((x2, y2, z3)),  # Bottom left.
        Vector((x1, y1, z3)),  # Bottom right.
    ]
    plane_centre = reduce(add, world_wall_corners) / 4

    bpy.ops.object.select_all(action="DESELECT")

    # Get the existing plane.
    if (plane := bpy.data.objects["Plane"]) is None:
        raise ValueError("Default plane not found in the scene.")

    # Activate and select the plane.
    bpy.context.view_layer.objects.active = plane
    plane.select_set(True)

    # Update the plane's centre in world space.
    plane.location = plane_centre

    # Update global space determine coordinates to local space.
    plane_inverse_world_matrix = plane.matrix_world.inverted()
    plane.data.vertices[0].co = plane_inverse_world_matrix @ (world_wall_corners[0])
    plane.data.vertices[1].co = plane_inverse_world_matrix @ (world_wall_corners[1])
    plane.data.vertices[2].co = plane_inverse_world_matrix @ (world_wall_corners[3])
    plane.data.vertices[3].co = plane_inverse_world_matrix @ (world_wall_corners[2])


"""
def apply_material(material_name: str = "Poliigon_BrickReclaimedRunning_7787") -> None:
    # Get the existing plane.
    if (plane := bpy.data.objects.get("Plane")) is None:
        raise ValueError("Default plane not found in the scene.")

    # Activate and select the plane.
    bpy.context.view_layer.objects.active = plane
    plane.select_set(True)

    # Apply material
    bpy.ops.poliigon.poliigon_active(
        mode="mat", asset_type="TEXTURE", data="Poliigon_BrickReclaimedRunning_7787_2K"
    )
    bpy.ops.poliigon.poliigon_apply(asset_id=7787, name_material=material_name)
    bpy.ops.poliigon.poliigon_material(
        tooltip=f"{material_name}\n(Apply Material)",
        asset_id=7787,
        size="2K",
        mapping="UV",
        scale=1,
        mode_disp="NORMAL",
        displacement=0.2,
        use_16bit=True,
        reuse_material=True,
    )
"""


def render_scene(save_path: Path) -> None:
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = str(save_path.resolve())
    bpy.ops.render.render(write_still=True)


def estimate_wall_and_render_material(
    blender_scene_path: Path,
    source_image_path: Path,
    hdri_path: Path,
    save_path: Path,
    wall_corners_pixels: WallCorners,
) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(blender_scene_path.resolve()))

    # Sometimes the plane isn't placed immediately correctly, and requires a retry to
    # apply it correctly. Very weird bug but for now just temporarily run the placing
    # algorithm twice.
    for _ in range(2):
        setup_scene(source_image_path, hdri_path)
        setup_plane(wall_corners_pixels)
    render_scene(save_path)

    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    wall_corners_pixels = WallCorners(
        top_right=PixelPoint(119, 995),
        top_left=PixelPoint(200, 573),
        bottom_left=PixelPoint(424, 573),
        bottom_right=PixelPoint(497, 995),
    )

    source_image_path = Path("./data/0a578e8af1642d0c1e715aaa04478858ac0aab01.jpg")
    hdri_path = Path("./temp/test.exr")
    blender_scene_path = Path("./src/rendering/render_wall_with_material.blend")
    save_path = Path("./temp/final.png")

    estimate_wall_and_render_material(
        blender_scene_path, source_image_path, hdri_path, save_path, wall_corners_pixels
    )
