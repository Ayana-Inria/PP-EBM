from typing import Union, List, Dict

import numpy as np
from scipy.spatial import cKDTree
from torch import Tensor

from base.mappings import ValueMapping


def compute_configuration_metrics(gt: Union[np.ndarray, Tensor], pred: Union[np.ndarray, Tensor],
                                  maximum_matching_distance: float, mappings: List[ValueMapping],
                                  match_mode: str = 'distance'):
    if type(gt) is Tensor:
        gt = gt.detach().cpu().numpy()
    if type(pred) is Tensor:
        pred = pred.detach().cpu().numpy()

    if match_mode == 'distance':
        gt_pos = cKDTree(data=gt[:, :2])
        dd, ii = gt_pos.query(
            pred[:, :2], distance_upper_bound=maximum_matching_distance)
    else:
        raise NotImplementedError

    matched_pred = pred[~np.isinf(dd)]
    matching_gt = gt[ii[~np.isinf(dd)]]
    distances = dd[~np.isinf(dd)]

    mse_per_mark = {}
    for i, m in enumerate(mappings):
        mse = np.mean(
            np.square(matched_pred[:, 2 + i] - matching_gt[:, 2 + i]))
        mse_per_mark["MSE_" + m.name] = mse

    mse_dist = np.mean(
        np.sum(np.square(matched_pred[:, [0, 1]] - matching_gt[:, [0, 1]]), axis=-1))

    unique_detections = np.unique(ii[~np.isinf(dd)])

    # pred_gt_match_index = ii.copy()
    # pred_gt_match_index[np.isinf(dd)] = -1
    matched_pred_mask = ii < len(gt)
    matched_pred_indices = np.arange(len(pred))[matched_pred_mask]
    matched_pred_gt_indices = ii[matched_pred_mask]

    res = {
        "n_gt": len(gt),
        "n_pred": len(pred),
        "true_positives": len(matched_pred),
        "false_positives": np.isinf(dd).sum(),
        "false_negatives": len(gt) - len(np.unique(ii[~np.isinf(dd)])),
        "unique_detections": len(unique_detections),
        "overlapping_detections": len(matched_pred) - len(unique_detections),
        "MSE_pos": mse_dist,
        **mse_per_mark,
        'matched_pred_mask': matched_pred_mask,
        'matched_pred_indices': matched_pred_indices,
        'matched_pred_gt_indices': matched_pred_gt_indices
        # "pred_to_gt_match_index": ii,
        # "pred_to_gt_match_distance": dd
    }

    return res


def reduce_metrics(metrics: Dict):
    return {
        **{k: v for k, v in metrics.items() if 'MSE' in k},
        'cardinality_error': metrics['n_gt'] - metrics['n_pred'],
        # 'precision': metrics['true_positives'] / metrics['n_pred'],
        # 'recall': metrics['true_positives'] / metrics['n_gt']
        'precision': 1 - metrics['false_positives'] / metrics['n_pred'],
        'recall': 1 - metrics['false_negatives'] / metrics['n_gt']
    }
