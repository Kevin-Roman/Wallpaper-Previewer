import numpy as np
from scipy.optimize import linear_sum_assignment


def merge_masks(label_masks):
    merged_mask = np.zeros(label_masks[0].shape, dtype=int)

    for label, mask in label_masks.items():
        merged_mask[mask] = label

    return merged_mask


def pixel_error(pred, gt, ignore_label=-1):
    valid_mask = gt != ignore_label
    correct_pixels = (pred == gt) & valid_mask
    num_valid = np.sum(valid_mask)

    if num_valid == 0:
        return 0.0

    return 1 - (np.sum(correct_pixels) / num_valid)


def max_bipartite_matching_score(
    predicted, ground_truth, num_labels=5, ignore_label=-1
):
    """Optimally determines and assigns labels to the predicted mask.
    https://github.com/leVirve/OneGan/blob/e0d5f387c957fbf599919078d8c6277740015336/onegan/metrics/semantic_segmentation.py#L48
    The result is subtracted from 1 (the complement) to get the error.
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
