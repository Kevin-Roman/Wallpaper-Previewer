from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from src.common import LayoutSegmentationLabels


def merge_masks(label_masks: dict[LayoutSegmentationLabels, np.ndarray]) -> np.ndarray:
    merged_mask = np.zeros(label_masks[0].shape, dtype=int)

    for label, mask in label_masks.items():
        merged_mask[mask] = label

    return merged_mask


def pixel_error(
    predicted: np.ndarray, ground_truth: np.ndarray, ignore_label: int = -1
):
    valid_mask = ground_truth != ignore_label
    correct_pixels = (predicted == ground_truth) & valid_mask
    num_valid = np.sum(valid_mask)

    if num_valid == 0:
        return 0.0

    return 1 - (np.sum(correct_pixels) / num_valid)


def max_bipartite_matching_score(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    num_labels: int = 5,
    ignore_label: int = -1,
) -> float:
    """Optimally determines and assigns labels to the predicted mask.
    The result is subtracted from 1 (the complement) to get the error.

    Code adapted from:
    https://github.com/leVirve/OneGan/blob/e0d5f387c957fbf599919078d8c6277740015336/onegan/metrics/semantic_segmentation.py#L48
    """
    predicted = predicted.flatten()
    ground_truth = ground_truth.flatten()

    valid_mask = ground_truth != ignore_label

    predicted = predicted[valid_mask]
    ground_truth = ground_truth[valid_mask]

    cost = np.zeros((num_labels, num_labels))

    for i in range(num_labels):
        for j in range(num_labels):
            pred_mask = predicted == i
            gt_mask = ground_truth == j
            cost[i, j] = -np.sum(pred_mask & gt_mask)

    row_ind, col_ind = linear_sum_assignment(cost)

    score = -cost[row_ind, col_ind].sum()
    # Subtract from 1 to get the error.
    return 1 - (score / len(predicted))


def intersection_over_union_loss(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> float:
    """
    Code adapted from: https://github.com/bjekic/WallSegmentation/blob/main/models/dataset.py
    Bjekic, M. and Lazovic, A. and K, V. and Bacanin, N. and Zivkovic, M. and Kvascev, G. and Nikolic, B. (2023).
    Wall segmentation in 2D images using convolutional neural networks.
    https://doi.org/10.7717/peerj-cs.1565
    """
    intersection = np.sum((predicted == 0) & (ground_truth == 0))
    union = np.sum((predicted == 0) | (ground_truth == 0))
    return 1 - (intersection / (union + 1e-15))


def calculate_metrics(data: list[tuple[list[float], str]]) -> None:
    for values, dataset_name in data:
        print(
            f"\n{dataset_name}"
            f"\nMean: {np.mean(values):.4f}"
            f"\nStd: {np.std(values):.4f}"
            f"\nMin: {np.min(values):.4f}"
            f"\nQ1: {np.percentile(values, 25):.4f}"
            f"\nMedian: {np.median(values):.4f}"
            f"\nQ3: {np.percentile(values, 75):.4f}"
            f"\nMax: {np.max(values):.4f}"
        )


def save_box_plot(
    data: list[tuple[list[float], str]],
    title: str,
    y_label: str,
    y_max: float = 0.5,
    nbins: int = 10,
) -> None:
    Path("output/evaluation").mkdir(parents=True, exist_ok=True)

    values_list = [data_point[0] for data_point in data]
    labels = [data_point[1] for data_point in data]

    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.boxplot(values_list)

    plt.xlabel("Datasets")
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)

    plt.ylabel(y_label)
    plt.ylim(0, y_max)
    plt.locator_params(axis="y", nbins=nbins)
    plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)

    plt.savefig(f"output/evaluation/box_plot_{title}.svg", format="svg")
    plt.close()
