import glob
import json
import logging
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
import shapely.geometry
from tqdm import tqdm

from base.state_ops import parse_dota_df_to_state

DOTA_GT_FILE_HEADER = ['x1', 'y1', 'x2', 'y2',
                       'x3', 'y3', 'x4', 'y4', 'category', 'difficult']
DOTA_DET_FILE_HEADER = ['patch_id', 'score', 'x1',
                        'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
AVAILABLE_IOU_THRESHOLDS = [0.05, 0.10, 0.25, 0.50, 0.75]


def compute_iou(poly_pair: np.ndarray):
    assert poly_pair.shape == (2, 4, 2)
    poly_1 = shapely.geometry.Polygon(poly_pair[0])
    poly_2 = shapely.geometry.Polygon(poly_pair[1])
    inter = poly_1.intersection(poly_2).area
    union = poly_1.union(poly_2).area
    return inter / union


def load_dota_pr(base_inference_dir: str, dataset: str, model_name: str, load_metrics: bool = True):
    inference_dir = os.path.join(base_inference_dir, dataset, 'val')
    results_dir = os.path.join(inference_dir, model_name)

    dota_det_file = os.path.join(results_dir, 'dota', 'det', 'vehicle.txt')
    dota_image_set_file = os.path.join(results_dir, 'dota', 'imageSet.txt')
    dota_gt_files = glob.glob(os.path.join(results_dir, 'dota', 'gt', '*.txt'))

    patch_re_txt = re.compile(r'.*([0-9]{4}).txt')
    dota_det_df = pd.read_csv(dota_det_file, sep=' ',
                              header=None, names=DOTA_DET_FILE_HEADER)

    dota_image_set_df = pd.read_csv(
        dota_image_set_file, header=None, names=['patch_id'])
    dota_gt_df = []
    for p in dota_gt_files:
        patch_id = int(patch_re_txt.match(p).group(1))
        patch_gt = pd.read_csv(p, sep=' ', header=None,
                               names=DOTA_GT_FILE_HEADER)
        patch_gt['patch_id'] = patch_id
        dota_gt_df.append(patch_gt)
    dota_gt_df = pd.concat(dota_gt_df, ignore_index=True)

    # metrics
    dota_det_df.sort_values('score', inplace=True, ascending=False)
    # dota_metrics_df = pd.DataFrame()
    # dota_metrics_df['score'] = np.sort(dota_det_df.score)[::-1]
    if load_metrics:
        for thresh in AVAILABLE_IOU_THRESHOLDS:
            dota_metric_file = os.path.join(
                results_dir, 'dota', f'metrics{thresh:.2f}.json')
            with open(dota_metric_file, 'r') as f:
                dota_metrics = json.load(f)['vehicle']

            # dota_metrics_df = pd.DataFrame([dota_metrics['precision'], dota_metrics['recall']],
            #                                index=['precision', 'recall']).T
            dota_det_df[f'precision_{thresh:.2f}'] = dota_metrics['precision']
            dota_det_df[f'recall_{thresh:.2f}'] = dota_metrics['recall']
            if 'tp' in dota_metrics:
                dota_det_df[f'tp_{thresh:.2f}'] = dota_metrics['tp']
            if 'fp' in dota_metrics:
                dota_det_df[f'fp_{thresh:.2f}'] = dota_metrics['fp']
            if 'iou' in dota_metrics:
                dota_det_df[f'iou'] = dota_metrics['iou']

    dota_det_df['model'] = model_name

    return dota_det_df, dota_gt_df


def compute_dota_pr_from_file(results_dir: str, **kwargs):
    dota_det_file = os.path.join(results_dir, 'dota', 'det', 'vehicle.txt')
    dota_image_set_file = os.path.join(results_dir, 'dota', 'imageSet.txt')
    dota_gt_files = glob.glob(os.path.join(results_dir, 'dota', 'gt', '*.txt'))

    patch_re_txt = re.compile(r'.*([0-9]{4}).txt')

    dota_det_df = pd.read_csv(dota_det_file, sep=' ',
                              header=None, names=DOTA_DET_FILE_HEADER)
    dota_image_set_df = pd.read_csv(
        dota_image_set_file, header=None, names=['patch_id'])
    dota_gt_df = []
    for p in dota_gt_files:
        patch_id = int(patch_re_txt.match(p).group(1))
        patch_gt = pd.read_csv(p, sep=' ', header=None,
                               names=DOTA_GT_FILE_HEADER)
        patch_gt['patch_id'] = patch_id
        dota_gt_df.append(patch_gt)
    dota_gt_df = pd.concat(dota_gt_df, ignore_index=True)

    dota_det_df = compute_prec_rec(
        dota_det_df, dota_gt_df,
        unique_patches=dota_image_set_df.patch_id,
        prefix='', score_key='score',
        **kwargs
    )

    return dota_det_df, dota_gt_df


def compute_prec_rec(dota_det_df, dota_gt_df, unique_patches, prefix: str, score_key: str, iou_thresholds: list,
                     ignore_difficulty: bool = False, use_cached_iou: bool = False):
    # metrics
    dota_det_df = dota_det_df.sort_values(
        score_key, inplace=False, ascending=False)
    # dota_det_df['model'] = model_name
    iou_key = prefix + 'iou'
    match_key = prefix + 'match_index'

    dota_det_df[iou_key] = None
    # dota_det_df['distance'] = None
    dota_det_df[match_key] = None
    # dota_det_df['match_distance_index'] = None

    # compute IOU and distance
    for patch_id in tqdm(unique_patches, desc='computing IOUs'):
        det_polys = parse_dota_df_to_state(
            dota_det_df[dota_det_df.patch_id == patch_id], poly=True)
        gt_polys = parse_dota_df_to_state(
            dota_gt_df[dota_gt_df.patch_id == patch_id], poly=True)
        n_det, n_gt = len(det_polys), len(gt_polys)
        ii, jj = np.meshgrid(np.arange(n_det), np.arange(n_gt), indexing='ij')
        ii = ii.ravel()
        jj = jj.ravel()
        pair_array = np.stack((det_polys[ii], gt_polys[jj]), axis=1)
        with Pool() as p:
            iou_flat = p.map(compute_iou, pair_array)
        iou = np.array(iou_flat).reshape(n_det, n_gt)
        match_index = np.argmax(iou, axis=1)
        max_ious = iou[np.arange(n_det), match_index]
        dota_det_df.loc[dota_det_df.patch_id == patch_id, [iou_key]] = max_ious
        dota_det_df.loc[dota_det_df.patch_id == patch_id, [match_key]] = \
            np.where(max_ious > 0,
                     dota_gt_df[dota_gt_df.patch_id ==
                                patch_id].index.values[match_index],
                     None)

    for iou_thresh in iou_thresholds:
        tp_key = f'{prefix}tp_{iou_thresh:.2f}'
        fp_key = f'{prefix}fp_{iou_thresh:.2f}'
        dota_det_df[tp_key] = np.zeros(len(dota_det_df))
        dota_det_df[fp_key] = np.zeros(len(dota_det_df))
        dota_det_df[f'{prefix}precision_{iou_thresh:.2f}'] = np.nan
        dota_det_df[f'{prefix}recall_{iou_thresh:.2f}'] = np.nan
        gt_match_key = f'{prefix}matched_{iou_thresh:.2f}'
        dota_gt_df[gt_match_key] = False
        sorted_idx = dota_det_df.index[np.argsort(
            -dota_det_df[score_key].values)]

        for i in sorted_idx:
            iou = dota_det_df.loc[i, iou_key]
            match_index = dota_det_df.loc[i, match_key]
            if iou >= iou_thresh:
                if ignore_difficulty or dota_gt_df.loc[match_index, 'difficult'] == 0:
                    if not dota_gt_df.loc[match_index, gt_match_key]:
                        dota_det_df.loc[i, tp_key] = 1
                        dota_gt_df.loc[match_index, gt_match_key] = True
                    else:
                        dota_det_df.loc[i, fp_key] = 1
            else:
                dota_det_df.loc[i, fp_key] = 1

        fp_cs = np.cumsum(dota_det_df.loc[sorted_idx, fp_key].values)
        tp_cs = np.cumsum(dota_det_df.loc[sorted_idx, tp_key].values)

        if ignore_difficulty:
            npos = len(dota_gt_df)
        else:
            npos = np.sum(dota_gt_df['difficult'] == 0)
        rec = tp_cs / float(npos)
        prec = tp_cs / np.maximum(tp_cs + fp_cs, np.finfo(np.float64).eps)

        dota_det_df.loc[sorted_idx,
                        f'{prefix}precision_{iou_thresh:.2f}'] = prec
        dota_det_df.loc[sorted_idx, f'{prefix}recall_{iou_thresh:.2f}'] = rec

    return dota_det_df


def compute_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
