from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from scipy.io import loadmat

from src.interfaces import RoomLayoutEstimator
from src.models.room_layout_estimation import FCNAugmentedRoomLayoutEstimator

from ..utils import (
    max_bipartite_matching_score,
    merge_masks,
    pixel_error,
)


def get_lsun_room_validation_data() -> tuple[list[np.ndarray], list[np.ndarray]]:
    dataset_path = Path("datasets/lsun_room/relabelled")
    image_folder = dataset_path / "images"
    label_folder = dataset_path / "layout_seg"

    # Validation set is the test set used in the 'Large-scale Scene Understanding
    # Challenge: Room Layout Estimation', as the test set isn't labelled.
    # https://web.archive.org/web/20161002144418/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/LSUN_RoomLayout.pdf
    index_file = dataset_path / "validation.mat"

    test_filenames = sorted(
        [entry.squeeze()[0][0] for entry in loadmat(index_file)["validation"].squeeze()]
    )

    image_filenames = sorted([path.stem for path in sorted(image_folder.glob("*.jpg"))])
    label_filenames = sorted([path.stem for path in sorted(label_folder.glob("*.mat"))])

    filtered_test_filenames = [
        filename for filename in test_filenames if filename in set(image_filenames)
    ]

    images = [
        np.array(PILImage.open(image_folder / f"{filename}.jpg"))
        for filename in filtered_test_filenames
    ]

    labels = [
        np.array(loadmat(label_folder / f"{filename}.mat")["layout"])
        for filename in filtered_test_filenames
        if filename in label_filenames
    ]

    mapping = {
        0: -1,
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: -1,
    }

    map = np.vectorize(lambda label: mapping[label])
    labels = [map(array) for array in labels]
    return images, labels


def evaluate_lsun_room() -> list[float]:
    images, grount_truth = get_lsun_room_validation_data()

    room_layout_estimator: RoomLayoutEstimator = FCNAugmentedRoomLayoutEstimator()

    pixel_errors: list[float] = []
    scores: list[float] = []
    errors = 0
    for i, test_image in enumerate(images):
        mask_map = room_layout_estimator.model_inference(PILImage.fromarray(test_image))
        merged_mask = merge_masks(mask_map)

        if merged_mask.shape != grount_truth[i].shape:
            errors += 1
            continue

        pixel_error_value = pixel_error(merged_mask, grount_truth[i])
        score = max_bipartite_matching_score(merged_mask, grount_truth[i])

        pixel_errors.append(pixel_error_value)
        scores.append(score)

        print(
            f"Image {i + 1}/{len(images)}: "
            f"Pixel error: {pixel_error_value:.2%}, "
            f"Score: {score:.2%}"
        )

    print(
        f"\nMean pixel error: {np.mean(pixel_errors):.2%}\n"
        f"Mean Matching complement score: {np.mean(scores):.2%}\n"
        f"Excluded images: {errors}"
    )

    return scores
