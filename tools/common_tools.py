import logging
import os
import pickle
import re
from typing import Union, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from torch import Tensor

from base.array_operations import make_batch_indices, invert_permutation
from base.geometry import poly_to_rect_array
from base.images import extract_patch
from base.misc import append_lists_in_dict, deep_update
from base.parse_config import ConfigParser
from energies.generic_energy_model import GenericEnergyModel
from samplers.mic_set_utils import slice_state_to_context_cubes, state_to_context_cube


def load_mpp_model(model_path, override_params=None):
    model_config_file = os.path.join(model_path, 'config.yaml')
    with open(model_config_file, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    if override_params is not None and len(override_params) > 0:
        model_config = deep_update(model_config, override_params)
    model_config = ConfigParser(
        config=model_config, model_type=model_config['arch'], load_model=True, overwrite=False,
        debug_mode=False, init_if_none=False, save_path=model_path)
    model = GenericEnergyModel(model_config)
    resume_path = str(model_config.resume)
    logging.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path, )
    model.load_state_dict(checkpoint['state_dict'])
    logging.info(f'loading of {model_config["name"]} done')
    model.eval()
    return model


def image_and_gt(dataset_base_dir: str, dataset: str, patch_id: int, subset: str = 'val',return_difficult:bool=False):
    dataset_dir = os.path.join(dataset_base_dir, dataset, subset, )
    image_file = os.path.join(dataset_dir, 'images', f'{patch_id:04}.png')
    image = plt.imread(image_file)[..., :3]

    label_file = os.path.join(dataset_dir, 'annotations', f'{patch_id:04}.pkl')
    with open(label_file, 'rb') as f:
        label_dict = pickle.load(f)
    state_gt = np.concatenate([label_dict['centers'], label_dict['parameters']], axis=-1)
    if return_difficult:
        return image, state_gt, label_dict['difficult']
    return image, state_gt


def crop_image_and_gt(image: np.ndarray, state_gt: np.ndarray, patch_size: int, center=None):
    if center is None:
        rng = np.random.default_rng(0)
        center_anchor = rng.choice(state_gt)[:2]
    elif type(center) is np.ndarray:
        center_anchor = center
    elif type(center) is int:
        center_anchor = state_gt[center]
    else:
        raise TypeError

    patch, tl_anchor, centers_offset = extract_patch(image, center_anchor=center_anchor.astype(int),
                                                     patch_size=patch_size,
                                                     minimal_padding=True)

    state_gt_cropped = state_gt.copy()
    state_gt_cropped[:, :2] = state_gt_cropped[:, :2] - tl_anchor + centers_offset
    oob = np.any((state_gt_cropped[..., :2] < 0.0) | (state_gt_cropped[..., :2] > 256.0), axis=-1)
    state_gt_cropped = state_gt_cropped[~oob]
    return patch, state_gt_cropped


def parse_df_to_state(df: pd.DataFrame,poly: bool = False,patch_id=None):
    if patch_id is not None:
        df = df[df.patch_id == patch_id]
    all_boxes = np.stack((df[['y1', 'y2', 'y3', 'y4']].values,
                          df[['x1', 'x2', 'x3', 'x4']].values), axis=-1)
    if poly:
        return all_boxes
    else:
        centers = np.mean(all_boxes, axis=1)
        parameters_2 = poly_to_rect_array(all_boxes)
        return np.concatenate((centers, parameters_2), axis=-1)


def compute_energy_on_image(model: GenericEnergyModel, state: Union[np.ndarray, Tensor], image: np.ndarray,
                            large_image: bool, batch_size: int = None,
                            return_weighted: bool = False, return_as_tensor: bool = False) -> Dict[str, np.ndarray]:
    image_t = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(dim=0)
    if type(state) is np.ndarray:
        state = torch.from_numpy(state)
    energy_infer_args = {"cut_and_stitch": True}
    pos_e_m, marks_e_m = model.energy_maps_from_image(image_t.to(model.device), **energy_infer_args)
    if large_image:
        if return_as_tensor:
            raise NotImplementedError  # todo
        assert batch_size is not None
        cube, cube_m, bounds, cube_i = slice_state_to_context_cubes(state=state,
                                                                    cell_size=2 * model.max_interaction_distance,
                                                                    image_shape=image.shape[:2],
                                                                    return_original_index=True)

        batch_indices = make_batch_indices(n_items=len(cube), batch_size=batch_size)
        batch_results = {}
        for bi in batch_indices:
            res = model.forward(cube[bi].to(model.device), cube_m[bi].to(model.device),
                                pos_e_m.to(model.device), [m.to(model.device) for m in marks_e_m],
                                compute_context=False)
            values = {}
            for k, v in res.items():
                if k not in ['energy_per_subset', 'total_energy', 'state']:
                    values[k] = v[cube_m[bi][:, 0, 0]].detach().cpu().numpy()
                else:
                    values[k] = v.detach().cpu().numpy()
            # values['state'] = cube[bi][:, 0, 0][cube_m[bi][:, 0, 0]]
            values['index'] = cube_i[bi][:, 0, 0][cube_m[bi][:, 0, 0]].detach().cpu().numpy().astype(int)
            append_lists_in_dict(batch_results, values)

        perm = np.concatenate(batch_results['index'], axis=0)
        i_perm = invert_permutation(perm)
        result = {}
        for k, v in batch_results.items():
            if k not in ['energy_per_subset', 'total_energy']:
                result[k] = np.concatenate(v, axis=0)[i_perm]
            elif k == 'energy_per_subset':
                result[k] = np.concatenate(v, axis=0)
            elif k == 'total_energy':
                result[k] = np.sum(v)
            else:
                raise KeyError
    else:
        cube, cube_m = state_to_context_cube(state)
        result = model.forward(cube.to(model.device), cube_m.to(model.device),
                               pos_e_m.to(model.device), [m.to(model.device) for m in marks_e_m],
                               compute_context=False)
        if not return_as_tensor:
            result = {k: v.detach().cpu().numpy() for k, v in result.items()}

    if return_weighted:
        weights = model.combination_module_weights
        for k, w in weights.items():
            if k != 'bias':
                ek = re.match(r'weight_(.*)', k).group(1)
                result[f'w_{ek}'] = w * result[ek]
        result['w_bias'] = weights['bias']

    return result
