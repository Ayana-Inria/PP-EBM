import os
import pickle

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from scipy import interpolate
from tqdm import tqdm

from base.files import make_if_not_exist
from base.misc import save_figure
from base.state_ops import crop_image_and_state
from display.draw_on_ax import draw_shapes_on_ax
from display.fancy_colors import colormaker

KOLOR = ['R', 'G', 'B']
CONFIG_FILE = 'figures_results.yaml'


def main():
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    save_dir = config['save_path']
    make_if_not_exist(save_dir)

    cache_file = os.path.join(save_dir, 'cache_expl.pkl')
    if config['use_cache'] and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    def update_cache():
        if config['use_cache']:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)

    with open(os.path.join(save_dir, 'thresholds.yaml'), 'r') as f:
        thresholds = yaml.load(f, Loader=yaml.Loader)

    datasets = config['datasets']
    models = config['models']

    draw_args = config['sub_scores']['draw_args']

    col_anchors = []
    col_values = []
    for d in config['sub_scores']['colors']:
        col_anchors.append(d['anchor'])
        col_values.append(colormaker(d['color']))
    col_values = np.array(col_values)
    col_anchors = np.array(col_anchors)
    colormode = config['sub_scores']['color_mode']

    def colorize(xi, **kwargs):
        if colormode == 'anchor':
            c0 = interpolate.griddata(col_anchors, col_values[:, 0], xi, method=config['sub_scores']['color_interp'],
                                      fill_value=0.5)
            c1 = interpolate.griddata(col_anchors, col_values[:, 1], xi, method=config['sub_scores']['color_interp'],
                                      fill_value=0.5)
            c2 = interpolate.griddata(col_anchors, col_values[:, 2], xi, method=config['sub_scores']['color_interp'],
                                      fill_value=0.5)
            ccc = np.stack((c0, c1, c2), axis=-1)
        elif colormode == 'RB':
            c0 = xi[:, 0]
            c1 = np.ones_like(xi[:, 0]) * 0.5
            c2 = xi[:, 1]
            ccc = np.stack((c0, c1, c2), axis=-1)
        elif colormode == 'threshold':
            ccc = np.ones((len(xi), 4))
            ccc[xi[:, 0] > xi[:, 1]] = colormaker('blue')
            ccc[xi[:, 0] < xi[:, 1]] = colormaker('yellow')
            ccc[(xi[:, 0] + xi[:, 1]) < kwargs['q1']] = colormaker('purple')
            ccc[(xi[:, 0] + xi[:, 1]) > kwargs['q2']] = colormaker('green')
        else:
            raise NotImplementedError

        return np.clip(ccc, 0, 1)

    res = 256
    pts = np.stack(np.meshgrid(np.linspace(0, 1, res), np.linspace(1, 0, res)), axis=-1).reshape(-1, 2)
    color = colorize(pts[:, [1, 0]], q1=0.1, q2=1.9)
    color_plot = color.reshape((res, res, -1))
    plt.imsave(os.path.join(save_dir, f"colorscope.png"), color_plot)

    pbar = tqdm(total=len(datasets) * len(config['patches']) * 2, desc='making figures')
    for dataset, dataset_args in datasets.items():
        res_save_dir = os.path.join(save_dir, dataset_args['name'])
        make_if_not_exist(res_save_dir)

        for patch_id, patch_args in config['patches'].items():
            image_file = f'/home/jmabon/Documents/data/datasets/{dataset}/val/images/{patch_id:04}.png'
            image = plt.imread(image_file)[..., :3]
            for model, model_args in models.items():
                if 'mppe' in model:
                    # print(f"{dataset}/{patch_id:04}/{model}:")
                    with open(os.path.join(config['inference_path'], dataset, 'val', model,
                                           f"{patch_id:04}_results_details.pkl"), 'rb') as f:
                        res = pickle.load(f)
                    state = res['state'].numpy()

                    det_df = pd.DataFrame(
                        data=np.concatenate(
                            (state, res['log_papangelou'].reshape((-1, 1)), res['energies_deltas']),
                            axis=1),
                        columns=['i', 'j', 'a', 'b', 'alpha', 'score'] + [f"{e}_delta" for e in res['energies']]
                    )

                    data_energies = ['position', 'width', 'length', 'angle']
                    prior_energies = [e for e in res['energies'] if e not in data_energies]

                    for cluster in ['data', 'prior']:
                        values = np.zeros(len(det_df))
                        if cluster == 'data':
                            group = data_energies
                        elif cluster == 'prior':
                            group = prior_energies
                        else:
                            raise NotImplementedError
                        for e in group:
                            values += res['weights'][f"weight_{e}"] * det_df[f"{e}_delta"]

                        det_df[f"{cluster}_delta"] = values
                        det_df[f"{cluster}_papangelou"] = np.exp(values)

                        if config['sub_scores']['log_scores']:
                            v = values

                        else:
                            v = np.exp(values)
                        # mi, ma = np.min(v), np.max(v)
                        mi, ma = np.quantile(v, q=[0.05, 0.95])
                        det_df[f"{cluster}_score"] = (v - mi) / (ma - mi)

                    scores = det_df[['data_score', 'prior_score']].to_numpy().sum(axis=-1)
                    q1, q2 = np.quantile(scores, q=[0.15, 0.75])
                    # print(np.quantile(scores, q=[0.1, 0.5, 0.9]))

                    colors = colorize(det_df[['data_score', 'prior_score']].to_numpy(), q1=q1, q2=q2)
                    for i, k in enumerate(KOLOR):
                        det_df[k] = colors[:, i]

                    t = thresholds[model][dataset]['max_f1.0']
                    # t = det_df['score'].min()
                    patch, state_p, oob = crop_image_and_state(
                        image,
                        det_df[['i', 'j', 'a', 'b', 'alpha']].to_numpy(),
                        patch_args['crop'], return_oob=True)
                    oob = oob.numpy()
                    state_p = state_p[(det_df['score'][~oob] > t).to_numpy()]
                    # mask = (~oob) & (res['score'] > t)
                    # print(f"{len(res['state'])} det -> {np.sum(~oob)} cropped")

                    masked_df = det_df[(~oob) & (det_df['score'] > t)]
                    n_rows, n_cols = 1, 1
                    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
                    axs.axis('off')

                    axs.imshow(patch)
                    draw_shapes_on_ax(axs, state_p, colors=masked_df[KOLOR].to_numpy(), **draw_args)

                    save_figure(res_save_dir, f"p{patch_id:04}_{model_args['name']}_details.png", tight=True)

                    e = 0.05
                    n_rows, n_cols = 1, 1
                    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
                    axs: plt.Axes
                    pts = det_df[['data_score', 'prior_score']].to_numpy()
                    color = det_df[KOLOR].to_numpy()
                    res = 256
                    gridpts = np.stack(np.meshgrid(np.linspace(0, 1, res), np.linspace(1, 0, res)), axis=-1).reshape(-1, 2)
                    colorgrid = colorize(gridpts[:, [1, 0]], q1=q1, q2=q2)
                    color_plot = colorgrid.reshape((res, res, -1))
                    axs.imshow(color_plot, alpha=0.65, extent=[0 - e, 1 + e, 0 - e, 1 + e])
                    axs.scatter(pts[:, 1], pts[:, 0], color='black', marker='+')
                    axs.set_xlim(-e, 1 + e)
                    axs.set_ylim(-e, 1 + e)
                    axs.set_xticks([0, 1.0])
                    axs.set_yticks([0, 1.0])
                    axs.set_xlabel('prior')
                    axs.set_ylabel('data')
                    axs.spines['right'].set_visible(False)
                    axs.spines['top'].set_visible(False)
                    save_figure(res_save_dir, f"p{patch_id:04}_{model_args['name']}_scoresplot.png",
                                bbox_inches='tight', pad_inches=0.05)

                    pbar.update(1)


if __name__ == '__main__':
    main()
