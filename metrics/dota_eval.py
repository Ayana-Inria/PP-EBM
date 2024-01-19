import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from base.data import get_inference_path
from base.files import NumpyEncoder

# task 1 : https://captain-whu.github.io/DOTA/tasks.html , det with OBB
# code adapted from https://github.com/CAPTAIN-WHU/DOTA_devkit/blob/master/dota-v1.5_evaluation_task1.py

IOU_THRESHOLDS = [0.05, 0.10, 0.25, 0.50, 0.75]


def dota_eval(det_type: str, legacy_mode: bool, postfix='', model_dir: str = None, dataset: str = None,
              subset: str = None,
              inference_path=None, save_subdir: str = None, ignore_difficulty: bool = False):
    assert det_type in ['obb', 'hbb']
    if inference_path is None:
        assert model_dir is not None
        assert dataset is not None
        assert subset is not None
        model_name = os.path.split(model_dir)[1]
        results_dir = get_inference_path(
            model_name=model_name, dataset=dataset, subset=subset)
    else:
        results_dir = inference_path
    dota_files_path = os.path.join(results_dir, 'dota' + postfix)
    if save_subdir is None:
        save_dir = dota_files_path
    else:
        save_dir = os.path.join(dota_files_path, save_subdir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # det_path = os.path.join(dota_files_path, 'det', r'{:s}.txt')
    # annot_path = os.path.join(dota_files_path, 'gt', r'{:s}.txt')
    # image_set_file = os.path.join(dota_files_path, 'imageSet.txt')

    classnames = ['vehicle']
    results = {}

    if not legacy_mode:

        if det_type != 'obb':
            raise NotImplementedError(
                f"does not support det_type={det_type} with legacy_mode=False")

        if len(classnames) > 1:
            raise NotImplementedError("Only work with one class for now")

        from metrics.custom_dota_pr import compute_dota_pr_from_file, compute_ap

        dota_det_df, dota_gt_df = compute_dota_pr_from_file(
            results_dir=results_dir, iou_thresholds=IOU_THRESHOLDS,
            ignore_difficulty=ignore_difficulty
        )
        dota_det_df = dota_det_df.sort_values(
            'score', inplace=False, ascending=False)
        for iou_t in IOU_THRESHOLDS:
            prec = dota_det_df[f'precision_{iou_t:.2f}'].values
            rec = dota_det_df[f'recall_{iou_t:.2f}'].values
            ap = compute_ap(rec, prec)

            results[f'{iou_t:.2f}'] = {
                classnames[0]: {
                    'ap': ap,
                    'difficult_rate': np.mean(dota_gt_df['difficult']),
                    'precision': prec,
                    'recall': rec,
                    'tp': dota_det_df[f'tp_{iou_t:.2f}'].values,
                    'fp': dota_det_df[f'fp_{iou_t:.2f}'].values,
                }
            }
        dota_det_df.to_pickle(os.path.join(save_dir, 'dota_det_df.pkl'))
        dota_gt_df.to_pickle(os.path.join(save_dir, 'dota_gt_df.pkl'))
        # with open(os.path.join(dota_files_path))
    else:
        if ignore_difficulty:
            raise NotImplementedError(
                "ignore_difficulty=True is not supported with legacy=True")
        for iou_t in IOU_THRESHOLDS:
            print(f"IOU thresh = {iou_t}")
            # classaps = []
            # mean_ap = 0
            results[f'{iou_t:.2f}'] = {}

            for classname in classnames:
                print(f'doing class {classname}')
                if det_type == 'obb':
                    sys.path.append(os.path.join(
                        os.getcwd(), 'data/DOTA_devkit'))
                    from data.DOTA_devkit.dota_evaluation_task1 import voc_eval
                    print('using OBB')
                    rec, prec, ap, extra = voc_eval(
                        detpath=os.path.join(
                            dota_files_path, 'det', r'{:s}.txt'),
                        annopath=os.path.join(
                            dota_files_path, 'gt', r'{:s}.txt'),
                        imagesetfile=os.path.join(
                            dota_files_path, 'imageSet.txt'),
                        classname=classname,
                        ovthresh=iou_t,
                        use_07_metric=False,
                        return_extra=True
                    )
                else:  # call task 2 for non oriented BB
                    # todo this one does not work since the gt is not handeled correcly, should either use task 1 or fix
                    sys.path.append(os.path.join(
                        os.getcwd(), 'data/DOTA_devkit'))
                    from data.DOTA_devkit.dota_evaluation_task2 import voc_eval
                    print('using HBB')
                    rec, prec, ap = voc_eval(
                        detpath=os.path.join(
                            dota_files_path, 'det', r'{:s}.txt'),
                        annopath=os.path.join(
                            dota_files_path, 'gt', r'{:s}.txt'),
                        imagesetfile=os.path.join(
                            dota_files_path, 'imageSet.txt'),
                        classname=classname,
                        ovthresh=iou_t,
                        use_07_metric=False
                    )
                    extra = {}
                    # mean_ap = mean_ap + ap
                # classaps.append(ap)
                print(f"ap@{iou_t:.2f} : {ap}")

                results[f'{iou_t:.2f}'][classname] = {
                    'ap': ap,
                    'precision': prec,
                    'recall': rec,
                    **extra
                }

    for iou_t in IOU_THRESHOLDS:
        try:
            for c in classnames:
                r = results[f'{iou_t:.2f}'][c]
                prec = r[f'precision']
                rec = r[f'recall']
                plt.figure(figsize=(8, 4))
                plt.xlabel('recall')
                plt.ylabel('precision')
                plt.plot(rec, prec)
                plt.savefig(os.path.join(
                    save_dir, f'prec_rec_curve_{c}_{iou_t:.2f}.png'))
                plt.close('all')
        except Exception as e:
            print("error occured while displaying figures")
            print(e)

        # mean_ap = mean_ap / len(classnames)
        # print('map:', mean_ap)
        # classaps = np.array(classaps)
        # print('classaps: ', classaps)

        with open(os.path.join(save_dir, f'metrics{iou_t:.2f}.json'), 'w') as f:
            json.dump(results[f'{iou_t:.2f}'], f, cls=NumpyEncoder, indent=1)


def main():
    frcnn_model = '/workspaces/nef/home/jmabon/models/fasterrcnn/fasterRCNN_dota_20/'
    shapenet_model = '/workspaces/nef/home/jmabon/models/shapenet/shape_dota_22/'

    dota_eval(
        model_dir=shapenet_model,
        dataset='DOTA_gsd50',
        subset='val',
        det_type='obb',
        legacy_mode=False
    )


if __name__ == '__main__':
    main()
