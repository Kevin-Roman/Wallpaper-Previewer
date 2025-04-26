from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from scipy.io import loadmat

from src.models.room_layout_estimation import FCNAugmentedRoomLayoutEstimator

from .common import (
    max_bipartite_matching_score,
    merge_masks,
    pixel_error,
)


def hedau_testing() -> list[float]:
    dataset_path = Path("datasets/hedau")
    image_folder = dataset_path / "images"
    label_folder = dataset_path / "layout_seg"
    index_file = dataset_path / "traintestind.mat"

    # MATLAB to Python: 1-based -> 0-based
    indices = loadmat(index_file)["testind"].squeeze() - 1

    images_paths = sorted(image_folder.glob("*.jpg"))
    labels_paths = sorted(label_folder.glob("*.mat"))

    test_images = [np.array(PILImage.open(images_paths[i])) for i in indices]
    test_labels = [np.array(loadmat(labels_paths[i])["fields"]) for i in indices]

    # Hedau dataset required mapping of labels to the standard set out in
    # `src/common.py`.
    mapping = {
        0: -1,
        1: 3,
        2: 0,
        3: 2,
        4: 1,
        5: 4,
        6: -1,
    }

    map_func = np.vectorize(lambda label: mapping[label])
    test_labels = [map_func(arr) for arr in test_labels]

    room_layout_estimator = FCNAugmentedRoomLayoutEstimator()

    pixel_errors: list[float] = []
    scores: list[float] = []
    errors = 0
    for i, test_image in enumerate(test_images):
        mask_map = room_layout_estimator.model_inference(PILImage.fromarray(test_image))
        merged_mask = merge_masks(mask_map)

        if merged_mask.shape != test_labels[i].shape:
            errors += 1
            continue

        pixel_error_value = pixel_error(merged_mask, test_labels[i])
        score = max_bipartite_matching_score(merged_mask, test_labels[i])

        pixel_errors.append(pixel_error_value)
        scores.append(score)

        print(
            f"Image {i + 1}/{len(test_images)}: "
            f"Pixel error: {pixel_error_value:.2%}, "
            f"Score: {score:.2%}"
        )

    print(
        f"\nMean pixel error: {np.mean(pixel_errors):.2%}\n"
        f"Mean Matching complement score: {np.mean(scores):.2%}\n"
        f"Excluded images: {errors}"
    )

    return scores
