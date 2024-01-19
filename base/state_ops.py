import logging
import traceback
from typing import List

import numpy as np
import pandas as pd
import torch
from numpy.random import Generator
from skimage.feature import peak_local_max
from torch import Tensor

from base.geometry import poly_to_rect_array
from base.images import extract_patch
from base.mappings import ValueMapping
from base.sampler2d import sample_point_2d

# @torch.jit.script


def clip_state_to_bounds(state: Tensor, min_bound: Tensor, max_bound: Tensor, cyclic: Tensor):
    """
    clips a state to bounds while respecting cyclic marks
    :param state: tensor state of shape (...,D). With D = number of marks + 2
    :param min_bound: minium bounds on marks (and position), tensor of size D
    :param max_bound: maximum bounds on marks (and position), tensor of size D
    :param cyclic: tensor of size D, element is true if corresponding marks (or coordinate) is cyclic
    :return:
    """
    n_marks = state.shape[-1]
    assert min_bound.shape == (n_marks,)
    assert max_bound.shape == (n_marks,)
    assert cyclic.shape == (n_marks,)

    # if state.device != min_bound.device:
    #     e = f"state ({state.device}) and min_bound ({min_bound.device}) are on different devices"
    #     # raise RuntimeError(e)
    #     logging.warning(e)
    #     min_bound = min_bound.detach().to(state.device)
    #     max_bound = max_bound.detach().to(state.device)
    #     cyclic = cyclic.detach().to(state.device)

    return torch.where(
        cyclic.broadcast_to(state.shape),
        torch.remainder(state - min_bound,
                        max_bound - min_bound) + min_bound,
        torch.clip(state, min_bound, max_bound)
    )


def clip_state_to_bounds_np(state: np.ndarray, min_bound: np.ndarray, max_bound: np.ndarray, cyclic: np.ndarray):
    """
    clips a state to bounds while respecting cyclic marks
    :param state: tensor state of shape (...,D). With D = number of marks + 2
    :param min_bound: minium bounds on marks (and position), tensor of size D
    :param max_bound: maximum bounds on marks (and position), tensor of size D
    :param cyclic: tensor of size D, element is true if corresponding marks (or coordinate) is cyclic
    :return:
    """
    n_marks = state.shape[-1]
    assert min_bound.shape == (n_marks,)
    assert max_bound.shape == (n_marks,)
    assert cyclic.shape == (n_marks,)

    return np.where(
        np.broadcast_to(cyclic, state.shape),
        np.remainder(state - min_bound,
                     max_bound - min_bound) + min_bound,
        np.clip(state, min_bound, max_bound)
    )


def check_inbound_state(state: Tensor, min_bound: Tensor, max_bound: Tensor, cyclic: Tensor, clip_if_oob: bool,
                        masked_state=None):
    """
    checks if state is inbound, if not throws and error when clip_if_oob is False. If true the state is clipped and returned
    :param state:
    :param min_bound:
    :param max_bound:
    :param cyclic:
    :param clip_if_oob:
    :param masked_state: if not None, will check the bounds only on those
    :return:
    """

    if state.device != min_bound.device:
        e = f"state ({state.device}) and min_bound ({min_bound.device}) are on different devices"
        # raise RuntimeError(e)
        logging.warning(e)
        min_bound = min_bound.detach().to(state.device)
        max_bound = max_bound.detach().to(state.device)
        cyclic = cyclic.detach().to(state.device)

    if masked_state is None:
        masked_state = state
    if len(masked_state) > 0:
        oob = torch.any((masked_state > max_bound) |
                        (masked_state < min_bound), dim=-1)
        if torch.any(oob):
            e = f"Values out of bounds :\n" \
                f"Min: {min_bound=}\n" \
                f"Max :{max_bound=}\n" \
                f"Values: \n{masked_state[oob]}\n" \
                f"CLIPPING !"

            if clip_if_oob:
                logging.warning(e)
                state = clip_state_to_bounds(
                    state, min_bound, max_bound, cyclic)
            else:
                raise RuntimeError(e)
    return state


def maps_local_max_state(position_map: np.ndarray, mark_maps: List[np.ndarray], mappings: List[ValueMapping],
                         local_max_distance=5, local_max_thresh=0.5):
    assert len(position_map.shape) == 2
    assert all(mark_maps[i].shape[0] ==
               mappings[i].n_classes for i in range(len(mappings)))
    xy = peak_local_max(
        position_map, min_distance=local_max_distance, threshold_abs=local_max_thresh)
    return np.concatenate([
        xy,
        np.stack([
            mapping.class_to_value(
                np.argmax(mm[:, xy[:, 0], xy[:, 1]], axis=0))
            for mm, mapping in zip(mark_maps, mappings)
        ], axis=-1)
    ], axis=-1)


def maps_sample_state(position_density: np.ndarray, mark_maps: List[np.ndarray], mappings: List[ValueMapping],
                      n_points: int, rng: Generator, argmax_marks: bool):
    xy = sample_point_2d(img_shape=position_density.shape,
                         density=position_density, rng=rng, size=n_points)
    if argmax_marks:
        return np.concatenate([
            xy,
            np.stack([
                mapping.class_to_value(
                    np.argmax(mm[:, xy[:, 0], xy[:, 1]], axis=0))
                for mm, mapping in zip(mark_maps, mappings)
            ], axis=-1)
        ], axis=-1)
    else:
        raise NotImplementedError


def crop_image_and_state(image, state, crop=None, center_anchor=None, target_size: int = None,
                         return_list: bool = False, return_oob: bool = False, minimal_padding: bool = True):
    if type(state) is not list:
        state = [state]
    states_r = []
    for s in state:
        if type(s) is Tensor:
            states_r.append(s.clone())
        elif type(s) is np.ndarray:
            states_r.append(torch.tensor(s))
        else:
            raise TypeError

    if crop is None and center_anchor is None:
        return (image, *states_r)
    if crop is not None:
        h, w = image.shape[:2]
        target_size = int(h * crop[2])
        center_anchor = np.array([h * crop[0], w * crop[1]])
    elif center_anchor is not None:
        # center_anchor = center_anchor
        # target_size = target_size
        pass
    else:
        raise RuntimeError

    patch, tl_anchor, centers_offset = extract_patch(image, center_anchor=center_anchor.astype(int),
                                                     patch_size=target_size,
                                                     minimal_padding=minimal_padding)
    states_cropped = []
    oob_s = []
    for s in states_r:
        s[:, :2] = s[:, :2] - tl_anchor + centers_offset
        oob = torch.any((s[..., :2] < 0.0) | (
            s[..., :2] > target_size), dim=-1)
        states_cropped.append(s[~oob])
        oob_s.append(oob)

    if return_list:
        r = patch, states_cropped
        if return_oob:
            r = r + (oob_s,)
    else:
        r = (patch, *states_cropped)
        if return_oob:
            r = r + (*oob_s,)

    return r


def parse_dota_df_to_state(df: pd.DataFrame, poly: bool = False):
    all_boxes = np.stack((df[['y1', 'y2', 'y3', 'y4']].values,
                          df[['x1', 'x2', 'x3', 'x4']].values), axis=-1)

    if poly:
        return all_boxes
    else:
        centers = np.mean(all_boxes, axis=1)
        parameters_2 = poly_to_rect_array(all_boxes)

        return np.concatenate((centers, parameters_2), axis=-1)
