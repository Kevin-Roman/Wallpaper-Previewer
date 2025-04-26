from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from .hedau import hedau_testing
from .lsun_room import lsun_room_testing


def calculate_metrics(data: list[tuple[list[float], str]]) -> None:
    for values, dataset_name in data:
        print(
            f"\n{dataset_name}"
            f"\nMean: {np.mean(values):.4f}"
            f"\nMedian: {np.median(values):.4f}"
            f"\nStd: {np.std(values):.4f}"
            f"\nMin: {np.min(values):.4f}"
            f"\nMax: {np.max(values):.4f}"
            f"\nQ1: {np.percentile(values, 25):.4f}"
            f"\nQ3: {np.percentile(values, 75):.4f}"
        )


def save_box_plot(
    data: list[tuple[list[float], str]], title: str, y_label: str
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
    plt.ylim(0, 0.5)
    plt.locator_params(axis="y", nbins=10)
    plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)

    plt.savefig("output/evaluation/box_plot.svg", format="svg")
    plt.close()


if __name__ == "__main__":
    hedau_pixel_errors = hedau_testing()
    lsun_pixel_errors = lsun_room_testing()

    data = [(hedau_pixel_errors, "Hedau"), (lsun_pixel_errors, "LSUN-Room")]

    save_box_plot(
        data,
        title="Pixel Error Distributions with Bipartite Matching",
        y_label="Pixel Error with Bipartite Matching",
    )

    calculate_metrics(data)
