import datetime
import json
import os
import pickle
import shutil
from typing import Dict

import numpy as np
import yaml
from matplotlib import pyplot as plt
from skimage.draw import draw
from tqdm import tqdm

from base.data import get_dataset_base_path
from base.files import make_if_not_exist, NumpyEncoder
from data.synthetic import textures, display_annotations
from display.draw_on_img import draw_shapes_on_img


def make_synth_dataset(config: Dict, overwrite: bool):
    data_dir = get_dataset_base_path()
    name = config['name']
    dataset_dir = os.path.join(data_dir, name)
    if os.path.exists(dataset_dir) and overwrite:
        shutil.rmtree(dataset_dir)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    subsets = ['val', 'train']
    subdirs = ['images', 'annotations', 'metadata', 'images_w_annotations']

    for subset in subsets:
        for subd in subdirs:
            d = os.path.join(dataset_dir, subset, subd)
            make_if_not_exist(d, recursive=True)

    with open(os.path.join(dataset_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    rng = np.random.default_rng(config['seed'])

    simulator_mode = config.get('simulator_mode', 'reject')
    if simulator_mode == 'reject':
        from data.synthetic.reject_sim import RejectSimulator
        simulator = RejectSimulator(config)
    elif simulator_mode == 'energy':
        from data.synthetic.energy_sim import EnergySimulator
        simulator = EnergySimulator(config)
    elif 'custom.' in simulator_mode:
        from data.synthetic import custom_sims
        simulator = getattr(custom_sims, simulator_mode.split('.')[1])(config)
    else:
        raise ValueError

    shape = tuple(config['shape'])
    for subset in subsets:
        nb_images = config[f'nb_{subset}']
        for i in tqdm(range(nb_images), desc=f'Making {subset} images'):

            image_mode = config.get('image', None)
            if type(image_mode) is str:
                image = getattr(textures, image_mode)(
                    shape=shape, rng=rng, **config.get('image_args', {}))
            elif image_mode is None:
                image = np.zeros(shape)
            else:
                raise ValueError

            state = simulator.make_points(rng=rng, image=image)

            if len(image.shape) == 2:  # handle greyscale img
                image = np.stack([image] * 3, axis=-1)

            if config['draw_shape'] == 'circle':
                draw_args = config.get('draw_shape_args', {})
                if 'colormap' in draw_args:
                    cmap = plt.get_cmap(draw_args['colormap'])
                    draw_args.pop('colormap')
                    draw_args['colormap_func'] = lambda _: cmap(rng.random())
                image = draw_shapes_on_img(
                    image, states=state, draw_method='circle', fill=True, **draw_args)
            elif config['draw_shape'] == 'none':
                pass
            elif config['draw_shape'] == 'colordot':
                draw_args = config['draw_shape_args']
                size = draw_args['size']
                color = draw_args['color']
                for s in state:
                    mask = draw.disk(s[[0, 1]], size, shape=shape)
                    image[mask] = color
            elif config['draw_shape'] == 'colorrect':
                def colormap_func(_): return rng.uniform((0, 0, 0), (1, 1, 1))
                image = draw_shapes_on_img(image, states=state, draw_method='rectangle',
                                           colormap_func=colormap_func, fill=True)
            elif config['draw_shape'] == 'rectangle':
                draw_args = config.get('draw_shape_args', {})

                if 'colormap' in draw_args:
                    cmap = plt.get_cmap(draw_args['colormap'])
                    draw_args.pop('colormap')
                    draw_args['colormap_func'] = lambda _: cmap(rng.random())

                image = draw_shapes_on_img(
                    image, states=state, draw_method='rectangle', fill=True, **draw_args)
            elif config['draw_shape'] == 'blob':
                raise NotImplementedError
            else:
                raise NotImplementedError

            image = image + config['noise_fac'] * rng.normal(size=image.shape)
            image = np.clip(image, 0, 1)
            if len(image.shape) == 2:  # handle greyscale img
                image = np.stack([image] * 3, axis=-1)

            labels_dict = {
                'centers': state[:, :2],
                'parameters': state[:, 2:],
                'difficult': np.zeros(len(state), dtype=int),
                'categories': np.zeros(len(state), dtype=int)
            }

            metadata = {
                'shape': shape,
                'n_objects': len(state),
                'source': 'synthetic',
                'date': datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
            }

            plt.imsave(os.path.join(dataset_dir, subset,
                       'images', f'{i:04}.png'), image)

            with open(os.path.join(dataset_dir, subset, 'annotations', f'{i:04}.pkl'), 'wb') as f:
                pickle.dump(labels_dict, f)

            with open(os.path.join(dataset_dir, subset, 'metadata', f'{i:04}.json'), 'w') as f:
                json.dump(metadata, f, cls=NumpyEncoder)

            draw_annot = config.get('display_annotations', None)
            if draw_annot is not None:
                drawn_annot = draw_shapes_on_img(
                    image, state,
                    draw_method=draw_annot,
                    **config.get('display_annotations_args', {})
                )
                # drawn_annot = getattr(display_annotations, draw_annot)(
                #     image, state, **config.get('display_annotations_args', {})
                # )
                plt.imsave(os.path.join(dataset_dir, subset,
                           'images_w_annotations', f'{i:04}.png'), drawn_annot)
