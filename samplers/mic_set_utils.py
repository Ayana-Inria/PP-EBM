from typing import Union

import numpy as np
import torch
from numba import jit
from torch import Tensor

from samplers.mic_sets import make_cell_bounds


def pad_to_size(arr: np.ndarray, shape, **pad_kwargs):
    c_shape = arr.shape
    if len(c_shape) == 2:
        return np.pad(arr, ((0, shape[0] - c_shape[0]), (0, shape[1] - c_shape[1])), **pad_kwargs)
    elif len(c_shape) == 3:
        return np.pad(arr, ((0, shape[0] - c_shape[0]), (0, shape[1] - c_shape[1]), (0, 0)), **pad_kwargs)
    else:
        raise NotImplementedError


def slice_maps_to_bounds(arr: np.ndarray, bounds: np.ndarray, **pad_kwargs):
    slice_size = np.max(np.diff(bounds, axis=1), axis=0)[0]
    patches = np.stack(
        [pad_to_size(arr[tl[0]:br[0], tl[1]:br[1]], slice_size,
                     **pad_kwargs) for tl, br in bounds],
        axis=0
    )
    return patches


@jit(nopython=True)
def random_to_grid_samples(rd: np.ndarray, flat_p_cumdist, flat_densities, flat_grid, n_cells: int):
    samples = np.sum(flat_p_cumdist < rd, axis=1)
    cells_range = np.arange(n_cells)
    # grid_samples = flat_grid[cells_range, samples]
    # grid_samples = np.take_along_axis(flat_grid, samples.reshape((n_sets, 1, 1)), axis=1).reshape((n_sets, 2))
    samples_o = samples + cells_range * flat_grid.shape[1]
    grid_samples = flat_grid.reshape((-1, 2))[samples_o]
    samples_densities = flat_densities.reshape((-1,))[samples_o]
    return grid_samples, samples_densities


# @jit(nopython=True)
def sample_marks_values(marks_cum_density_map: np.ndarray, positions: np.ndarray, rd_values: np.ndarray,
                        mapping: np.ndarray):
    """

    :param marks_cum_density_map: (M,H,W,C)
    :param positions: : (N,2)
    :param rd_values: (M,N)
    :param mapping: (M,C)
    :return:
    """
    n_marks = mapping.shape[0]
    n_points = positions.shape[0]
    n_classes = mapping.shape[1]

    assert marks_cum_density_map.shape[3] == mapping.shape[1]
    cum_densities = marks_cum_density_map[:, positions[:, 0].astype(
        int), positions[:, 1].astype(int)]  # (M,N,C)
    samples = np.sum(cum_densities < rd_values.reshape(
        (n_marks, n_points, 1)), axis=-1)  # (M,N)
    samples_flat = (np.expand_dims(samples, -1) +
                    np.arange(n_marks).reshape((-1, 1, 1)) * n_classes).ravel()
    mapping_flat = mapping.ravel()
    return mapping_flat[samples_flat].reshape((n_marks, n_points))  # (M,N)


def state_to_context_cube(state: Union[np.ndarray, Tensor]):
    if type(state) is np.ndarray:
        state = torch.tensor(state)
    marks_and_pos = state.shape[-1]
    n_points = state.shape[0]
    context_cube = torch.zeros(
        (1, 3, 3, n_points, marks_and_pos), device=state.device)
    context_cube_mask = torch.zeros(
        (1, 3, 3, n_points), device=state.device).bool()

    context_cube[0, 0, 0, :n_points] = state
    context_cube_mask[0, 0, 0, :n_points] = True

    return context_cube, context_cube_mask


def slice_state_to_context_cubes(state: Union[np.ndarray, Tensor],
                                 bounds: np.ndarray = None, cell_size=None, image_shape=None,
                                 return_original_index: bool = False):
    return_bounds = bounds is None
    if bounds is None:
        assert cell_size is not None
        assert image_shape is not None
        bounds = make_cell_bounds(cell_size=cell_size, support_shape=image_shape, n_frames=1, temporal=False,
                                  constant_cell_size=False, return_ndivs=False)

    if type(state) is np.ndarray:
        state = torch.from_numpy(state)

    img_tl_corner = torch.from_numpy(
        np.min(bounds[:, 0], axis=0)).to(state.device)
    img_br_corner = torch.from_numpy(
        np.max(bounds[:, 1], axis=0)).to(state.device)
    oob = ~ torch.all((state[:, :2] >= img_tl_corner)
                      & (state[:, :2] < img_br_corner), dim=-1)
    if torch.any(oob):
        raise RuntimeError(
            f"Points out of bounds (with {img_tl_corner=},{img_br_corner=}):\n{state[oob]}")

    if type(state) is np.ndarray:
        state = torch.tensor(state)

    marks_and_pos = state.shape[-1]
    n_points = state.shape[0]
    state_index = torch.arange(0, n_points)
    n_cubes = len(bounds)
    max_points = 0
    points = []
    points_indices = []
    for tl_corner, br_corner in bounds:
        mask = torch.all((state[:, :2] >= torch.from_numpy(tl_corner)) &
                         (state[:, :2] < torch.from_numpy(br_corner)), dim=-1)
        local_state = state[mask]
        n = len(local_state)
        if n > max_points:
            max_points = n
        points.append(local_state)
        points_indices.append(state_index[mask])

    context_cube = torch.zeros(
        (n_cubes, 3, 3, max_points, marks_and_pos), device=state.device)
    context_cube_mask = torch.zeros(
        (n_cubes, 3, 3, max_points), device=state.device).bool()
    points_indices_cube = torch.empty(
        (n_cubes, 3, 3, max_points), device=state.device)

    for i, (s, si) in enumerate(zip(points, points_indices)):
        n = len(s)
        context_cube[i, 0, 0, :n] = s
        context_cube_mask[i, 0, 0, :n] = True
        points_indices_cube[i, 0, 0, :n] = si

    # all_indices = points_indices_cube[context_cube_mask].numpy().astype(int)
    # if len(all_indices) != len(state_index):
    #     missed_points = []
    #     for i in state_index.numpy():
    #         if i not in all_indices:
    #             missed_points.append(i)
    #     pts_coord = ', '.join([f"({state[i, 0]},{state[i, 1]})" for i in missed_points])
    #     raise RuntimeError(f"Missing points {missed_points}, at {pts_coord}")

    return_tuple = (context_cube, context_cube_mask)
    if return_bounds:
        return_tuple = return_tuple + (bounds,)
    if return_original_index:
        return_tuple = return_tuple + (points_indices_cube,)
    return return_tuple
