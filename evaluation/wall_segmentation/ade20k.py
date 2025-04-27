"""
Code adapted from: https://github.com/bjekic/WallSegmentation/blob/main/models/dataset.py
Bjekic, M. and Lazovic, A. and K, V. and Bacanin, N. and Zivkovic, M. and Kvascev, G. and Nikolic, B. (2023).
Wall segmentation in 2D images using convolutional neural networks.
https://doi.org/10.7717/peerj-cs.1565
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from src.interfaces import WallSegmenter
from src.models.wall_segmentation import EncoderDecoderPPMWallSegmenter

from ..utils import pixel_error, intersection_over_union_loss

LIST_SCENES = [
    "bathroom",
    "bedroom",
    "kitchen",
    "living_room",
    "art_gallery",
    "art_studio",
    "attic",
    "auditorium",
    "shop",
    "ballroom",
    "bank_indoor",
    "banquet_hall",
    "bar",
    "basement",
    "bookstore",
    "childs_room",
    "classroom",
    "room",
    "closet",
    "clothing_store",
    "computer_room",
    "conference_room",
    "corridor",
    "office",
    "darkroom",
    "dentists_office",
    "diner_indoor",
    "dinette_home",
    "dining_room",
    "doorway_indoor",
    "dorm_room",
    "dressing_room",
    "entrance_hall",
    "galley",
    "game_room",
    "garage_indoor",
    "gymnasium_indoor",
    "hallway",
    "home_office",
    "hospital_room",
    "hotel_room",
    "jail_cell",
    "kindergarden_classroom",
    "lecture_room",
    "library_indoor",
    "lobby",
    "museum_indoor",
    "nursery",
    "playroom",
    "staircase",
    "television_studio",
    "utility_room",
    "waiting_room",
    "warehouse_indoor",
    "youth_hostel",
]


def create_scene_dict(path_to_scene_categories: Path) -> dict:
    dict_scene = {}

    with open(path_to_scene_categories, "r") as f:
        for line in f:
            temp = line.split(" ")
            scene = temp[1].strip()
            img_name = temp[0]
            dict_scene[img_name] = scene

    return dict_scene


def get_ade20k_validation_data() -> list[tuple[PILImage.Image, PILImage.Image]]:
    dataset_path = Path("datasets/ade20k")
    odgt_path = dataset_path / "validation.odgt"
    scene_categories_path = dataset_path / "sceneCategories.txt"

    scene_dict = create_scene_dict(scene_categories_path)

    with open(odgt_path, "r") as f:
        records = [json.loads(line.strip()) for line in f]

    validation_data: list[tuple[PILImage.Image, PILImage.Image]] = []

    for record in records:
        image_path = dataset_path / record["fpath_img"]
        segmentation_path = dataset_path / record["fpath_segm"]

        record_name = os.path.splitext(os.path.basename(record["fpath_img"]))[0]

        if scene_dict.get(record_name, None) not in LIST_SCENES:
            continue

        image = PILImage.open(image_path).convert("RGB")
        segmentation = PILImage.open(segmentation_path)

        validation_data.append((image, segmentation))

    return validation_data


def evaluate_ade20k() -> tuple[list[float], list[float]]:
    validation_data = get_ade20k_validation_data()

    wall_segmenter: WallSegmenter = EncoderDecoderPPMWallSegmenter()

    pixel_errors: list[float] = []
    iou_losses: list[float] = []
    errors = 0
    for i, (image, segmentation) in enumerate(validation_data):
        mask_map = wall_segmenter.model_inference(image)
        mask_map = np.array(mask_map).astype(np.uint8)
        # Invert the mask mask to match the segmentation mask labels.
        mask_map = 1 - mask_map

        segmentation = np.array(segmentation) - 1
        segmentation[segmentation > 0] = 1

        if mask_map.shape != segmentation.shape:
            errors += 1
            continue

        pixel_error_value = pixel_error(mask_map, segmentation)
        iou_loss_value = intersection_over_union_loss(mask_map, segmentation)

        pixel_errors.append(pixel_error_value)
        iou_losses.append(iou_loss_value)

        print(
            f"Image {i + 1}/{len(validation_data)}: "
            f"Pixel error: {pixel_error_value:.2%}, "
            f"IoU Loss: {iou_loss_value:.2%}"
        )

    print(
        f"\nMean pixel error: {np.mean(pixel_errors):.2%}\n"
        f"Mean IoU: {np.mean(iou_losses):.2%}\n"
        f"Excluded images: {errors}"
    )

    return pixel_errors, iou_losses


if __name__ == "__main__":
    evaluate_ade20k()
