import os
import pickle
import shutil

import numpy as np
import yaml
from matplotlib import pyplot as plt, patches
from tqdm import tqdm

from base.files import make_if_not_exist
from base.misc import save_figure
from base.state_ops import crop_image_and_state
from display.draw_on_ax import draw_shapes_on_ax
from metrics.custom_dota_pr import load_dota_pr
from tools.common_tools import parse_df_to_state

BETA_VALUES = [0.5, 1.0, 2.0]

CONFIG_FILE = 'figures_results.yaml'

def draw_one_patch(patch, state, **draw_args):
    n_rows, n_cols = 1, 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
    axs.axis('off')
    axs.imshow(patch)
    draw_shapes_on_ax(axs, state, **draw_args)


def main():
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    save_dir = config['save_path']
    make_if_not_exist(save_dir)

    shutil.copy(CONFIG_FILE, os.path.join(save_dir, 'config.yaml'))

    cache_file = os.path.join(save_dir, 'cache.pkl')
    if config['use_cache'] and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    datasets = config['datasets']
    models = config['models']

    for model in tqdm(models.keys(), desc='loading model results'):
        for dataset, dataset_args in datasets.items():
            if model not in cache \
                    or dataset not in cache[model] \
                    or not (all([k in cache[model][dataset] for k in ['det', 'gt']])):
                try:
                    dota_det_df, dota_gt_df = load_dota_pr(base_inference_dir=config['inference_path'],
                                                           dataset=dataset,
                                                           model_name=model,
                                                           load_metrics=True)
                except KeyError as e:
                    print(e)
                    dota_det_df, dota_gt_df = load_dota_pr(base_inference_dir=config['inference_path'],
                                                           dataset=dataset,
                                                           model_name=model,
                                                           load_metrics=False)

                if model not in cache:
                    cache[model] = {}
                if dataset not in cache[model]:
                    cache[model][dataset] = {}
                cache[model][dataset].update({
                    'det': dota_det_df, 'gt': dota_gt_df
                })
                if config['use_cache']:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache, f)

    for model, model_args in models.items():
        if 'threshold' not in model_args:
            for dataset, dataset_args in datasets.items():
                res_save_dir = os.path.join(save_dir, dataset_args['name'])
                make_if_not_exist(res_save_dir)
                det_df = cache[model][dataset]['det']
                det_df = det_df.sort_values('score', inplace=False, ascending=False)
                iou_thresh = 0.25
                pr = det_df[f"precision_{iou_thresh:.2f}"].values
                rc = det_df[f"recall_{iou_thresh:.2f}"].values

                fig, axs = plt.subplots(1, 1)
                cmap = plt.get_cmap('tab10')
                for j, beta in enumerate(BETA_VALUES):
                    f = (1 + beta ** 2) * (pr * rc) / ((beta ** 2 * pr) + rc)  # https://en.wikipedia.org/wiki/F-score
                    argmax_f = np.nanargmax(f)
                    scores = det_df['score'].values
                    s = scores[argmax_f]
                    axs.plot(scores, f, color=cmap(j), label=f'F{beta:.1f}')
                    axs.scatter(s, f[argmax_f], color=cmap(j))
                    cache[model][dataset][f'max_f{beta:.1f}_threshold'] = s
                axs.legend()
                axs.set_xlabel('score threshold')
                axs.set_ylabel('F_')
                save_figure(res_save_dir, f"argmaxfscore_{model_args['name']}")

                # f2 = 5 * (pr * rc) / ((4 * pr) + rc)  # https://en.wikipedia.org/wiki/F-score
                # argmax_f2 = np.nanargmax(f2)
                # cache[model][dataset]['max_f2_threshold'] = scores[argmax_f2]
        else:
            for dataset, dataset_args in datasets.items():
                for beta in BETA_VALUES:
                    cache[model][dataset][f'max_f{beta:.1f}_threshold'] = model_args['threshold']

    draw_args = config['draw_args']

    pbar = tqdm(total=len(datasets) * len(config['patches']) * len(models), desc='making figures')
    for dataset, dataset_args in datasets.items():
        res_save_dir = os.path.join(save_dir, dataset_args['name'])
        make_if_not_exist(res_save_dir)
        gt_df = cache[list(cache.keys())[0]][dataset]['gt']
        for patch_id, patch_args in config['patches'].items():

            image_file = f'/home/jmabon/Documents/data/datasets/{dataset}/val/images/{patch_id:04}.png'
            image = plt.imread(image_file)[..., :3]

            crop_args = patch_args['crop']
            h, w = image.shape[:2]
            cx, cy = h * crop_args[0], w * crop_args[1]
            s = h * crop_args[2]
            x, y = cx - s / 2, cy - s / 2
            n_rows, n_cols = 1, 1
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
            axs.axis('off')
            axs.imshow(image)
            axs.add_patch(patches.Rectangle(
                (y, x), s, s, fill=False, color='red'
            ))
            save_figure(res_save_dir, f"p{patch_id:04}_cropinfo.png", tight=True)

            state_gt = parse_df_to_state(gt_df[gt_df.patch_id == patch_id], patch_id=patch_id)
            patch, state_gt_p = crop_image_and_state(
                image, state_gt.astype(float), patch_args['crop'])
            local_draw_args = draw_args.copy()
            local_draw_args.update(config['gt']['draw_args'])
            draw_one_patch(patch, state_gt_p, **local_draw_args)
            save_figure(res_save_dir, f"p{patch_id:04}_GT.png", tight=True)

            for model, model_args in models.items():
                if model_args is None:
                    model_args = {}

                det_df = cache[model][dataset]['det']
                det_df = det_df[det_df.patch_id == patch_id]
                # threshold = model_args['threshold']

                threshold = cache[model][dataset]['max_f1.0_threshold']

                det_df = det_df[det_df.score > threshold]
                state_det = parse_df_to_state(det_df)
                patch, state_det_p = crop_image_and_state(
                    image, state_det, patch_args['crop'])

                plt.imsave(os.path.join(res_save_dir,f"p{patch_id:04}.png"),patch)

                local_draw_args = draw_args.copy()
                local_draw_args.update(model_args.get('draw_args', {}))
                draw_one_patch(patch, state_det_p, **local_draw_args)
                save_figure(res_save_dir, f"p{patch_id:04}_{model_args['name']}.png", tight=True)

                pbar.update(1)

    iou_thresholds = config['iou_thresholds']
    if not config.get('skip_metrics', False):
        pbar = tqdm(total=len(datasets) * len(models) * len(iou_thresholds), desc='plotting PR curves')
        for dataset, dataset_args in datasets.items():
            res_save_dir = os.path.join(save_dir, dataset_args['name'])
            make_if_not_exist(res_save_dir)
            for iou_thresh in iou_thresholds:
                n_rows, n_cols = 1, 1
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), squeeze=True)
                for model, model_args in models.items():
                    det_df = cache[model][dataset]['det']
                    det_df = det_df.sort_values('score', inplace=False, ascending=False)
                    precision = det_df[f"precision_{iou_thresh:.2f}"][8:]
                    recall = det_df[f"recall_{iou_thresh:.2f}"][8:]
                    fig_args = {
                        'color': model_args['draw_args']['color'],
                        **config['pr_draw_args']
                    }
                    axs.plot(recall, precision, label=model_args['name'], **fig_args)
                    if config['show_thresh_pr']:
                        threshold = cache[model][dataset]['max_f1_threshold']
                        f1 = (precision * recall) / (precision + recall)
                        argmax_f1 = np.nanargmax(f1)
                        axs.scatter(recall[argmax_f1], precision[argmax_f1], s=20.0, color=model_args['draw_args']['color'])
                    pbar.update(1)
                axs.set_xticks(config['pr_limits']['x'])
                axs.set_yticks(config['pr_limits']['y'])
                axs.set_xlabel('recall')
                axs.set_ylabel('precision')
                # axs.legend()
                save_figure(res_save_dir, f'PR-{iou_thresh:.2f}.png', tight=True)

        with open(config['metrics_source'], 'r') as f:
            metrics = yaml.load(f, Loader=yaml.Loader)

        metrics_table = config['metrics_table']
        metrics_models = metrics_table['models']
        metrics_keys = metrics_table['keys']
        results_latex = [
            ' & '.join(['Method'] + [k['name'] for k in metrics_keys]) + r'\\' + '\n' + '\hline'
        ]

        for model, model_name in metrics_models.items():
            metrics_values = [
                metrics[model][k['dataset']][f"AP_{k['thresh']:.2f}"] for k in metrics_keys
            ]
            results_latex.append(
                ' & '.join([model_name] + [f"{v:.2f}" for v in metrics_values]) + r'\\'
            )

        with open(os.path.join(save_dir, 'results_table.tex'), 'w') as f:
            print('\n'.join(results_latex), file=f)
            print('\n'.join(results_latex))

    thresholds = {
        model: {
            dataset: {
                f'max_f{beta:.1f}': float(cache[model][dataset][f'max_f{beta:.1f}_threshold'])
                for beta in BETA_VALUES
            }
            for dataset in datasets.keys()
        }
        for model in models.keys()
    }
    with open(os.path.join(save_dir, 'thresholds.yaml'), 'w') as f:
        yaml.dump(thresholds, f, sort_keys=False)


if __name__ == '__main__':
    main()
