import logging
import traceback
import warnings
from dataclasses import dataclass
from typing import Union, Dict

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from base.array_operations import make_batch_indices
from base.images import map_range, map_range_auto
from base.timer import Timer
from energies.generic_energy_model import GenericEnergyModel
from samplers.mic_set_utils import state_to_context_cube
from samplers.mic_sets import TorchTimeMicSets


def image_to_tensor(image: Union[Tensor, np.ndarray]):
    if type(image) is Tensor:
        if len(image.shape) == 4:
            assert image.shape[0] == 1
            return image
        else:
            return torch.unsqueeze(image, dim=0)
    else:
        return torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(dim=0)


def compute_papangelou(states: Tensor, model: GenericEnergyModel, log_values: bool = False,
                       image: Union[Tensor, np.ndarray] = None, pos_e_m=None,
                       marks_e_m=None):
    if pos_e_m is None or marks_e_m is None:
        image_t = image_to_tensor(image)
        pos_e_m, marks_e_m = model.energy_maps_from_image(
            image_t.to(model.device))
    else:
        if len(pos_e_m.shape) == 3:
            pos_e_m = pos_e_m.unsqueeze(dim=0)
        if len(marks_e_m[0].shape) == 3:
            marks_e_m = [m.unsqueeze(dim=0) for m in marks_e_m]

    cube, cube_m = state_to_context_cube(states)
    output = model.forward(cube.to(model.device), cube_m.to(model.device),
                           pos_e_m.to(model.device), [
        m.to(model.device) for m in marks_e_m],
        compute_context=False)
    energy_0 = output['total_energy'].detach().cpu().numpy()

    n_pts = cube.shape[3]
    papangelou_all = []

    for i in range(n_pts):
        cube_m[:, 0, 0, i] = False
        output = model.forward(cube.to(model.device), cube_m.to(model.device),
                               pos_e_m.to(model.device), [
            m.to(model.device) for m in marks_e_m],
            compute_context=False)
        energy_1 = output['total_energy'].detach().cpu().numpy()
        delta_u = - (energy_0 - energy_1)
        if log_values:
            papangelou = delta_u
        else:
            # papangelou(y;Y)=exp(-(U(Y \cup {y}) - U(Y)) / T)
            papangelou = np.exp(delta_u)
        papangelou_all.append(papangelou)
        cube_m[:, 0, 0, i] = True

    return np.array(papangelou_all)


def scores_scaling(scores, scaling):
    if scaling is None:
        return scores
    elif scaling == 'bound':
        return map_range_auto(scores, min_out=0, max_out=1)
    elif scaling == 'log':
        return np.log(scores + 1e-8)
    elif scaling == 'logbound':
        logging.warning("Don´t do that !")
        return map_range_auto(np.log(scores + 1e-8), min_out=0, max_out=1)
    else:
        raise ValueError(
            f"scaling {scaling} not valid: use None, 'bound', 'log' or 'logbound'")


def compute_papangelou_scoring(states: Tensor, model: GenericEnergyModel, log_values: bool, verbose=0,
                               image: Union[Tensor, np.ndarray] = None, pos_e_m=None,
                               marks_e_m=None, default_to_simple: bool = True, **kwargs):
    if type(states) is np.ndarray:
        states = torch.tensor(states)
    try:
        return efficient_papangelou_scoring(image=image, pos_e_m=pos_e_m, marks_e_m=marks_e_m, states=states,
                                            model=model, verbose=verbose, log_values=log_values, **kwargs)
    except Exception as e:
        if default_to_simple:
            logging.error(f"compute_papangelou_scoring failed with error {e} defulating to other method:"
                          f"\n{traceback.format_exc()}")

            return simple_papangelou_scoring(image=image, pos_e_m=pos_e_m, marks_e_m=marks_e_m, states=states,
                                             model=model,
                                             verbose=verbose, log_values=log_values)
        else:
            raise e


def simple_papangelou_scoring(states: Tensor, model: GenericEnergyModel, log_values: bool, verbose=0,
                              image: Union[Tensor, np.ndarray] = None, pos_e_m=None,
                              marks_e_m=None):
    if marks_e_m is None or pos_e_m is None:
        image_t = image_to_tensor(image)
        pos_e_m, marks_e_m = model.energy_maps_from_image(
            image_t.to(model.device))

    n_pts = len(states)
    state_mask = np.ones(n_pts, dtype=bool)
    state_index = np.arange(n_pts)
    scores = np.empty(n_pts)

    if verbose > 0:
        pbar = tqdm(range(n_pts), desc='computing papangelou scores')
    else:
        pbar = range(n_pts)

    for i in pbar:
        assert np.sum(state_mask) > 0
        papangelou_scores_masked = compute_papangelou(
            states=states[state_mask], model=model, pos_e_m=pos_e_m, marks_e_m=marks_e_m, log_values=log_values
        )
        argmin_masked = np.argmin(papangelou_scores_masked)
        argmin = state_index[state_mask][argmin_masked]
        scores[argmin] = papangelou_scores_masked[argmin_masked]
        state_mask[argmin] = False

    return scores


def efficient_papangelou_scoring(states: Tensor, model: GenericEnergyModel, log_values: bool, verbose=0,
                                 image: Union[Tensor, np.ndarray] = None, pos_e_m=None,
                                 marks_e_m=None, use_buffer: bool = False, return_sub_energies: bool = False):
    if marks_e_m is None or pos_e_m is None:
        image_t = image_to_tensor(image)
        pos_e_m, marks_e_m = model.energy_maps_from_image(
            image_t.to(model.device))
    else:
        if len(pos_e_m.shape) == 3:
            pos_e_m = pos_e_m.unsqueeze(dim=0)
        if len(marks_e_m[0].shape) == 3:
            marks_e_m = [m.unsqueeze(dim=0) for m in marks_e_m]
    if type(states) is np.ndarray:
        states = torch.tensor(states)

    states = states.to(model.device)

    n_pts = len(states)
    # h, w = image_t.shape[-2:]
    h, w = pos_e_m.shape[-2:]

    make_mic = True
    max_points_per_pixel = 0.01
    loops = 0
    mic = None
    while make_mic:
        mic = TorchTimeMicSets(
            support_shape=(h, w),
            spatial_interaction_distance=model.max_interaction_distance,
            move_distance=0,
            temporal=False,
            constant_cell_size=False,
            n_marks=len(model.mappings),
            max_points_per_pixel=max_points_per_pixel,
            device=model.device
        )
        try:
            mic.add_points(states, raise_full_cell_error=True)
        except RuntimeError as e:
            if loops > 8:
                raise e
            torch.cuda.empty_cache()
            warnings.warn(
                f"mic points add failed, increasing points per pixel. failed with :\n{e}")
            max_points_per_pixel = max_points_per_pixel + 0.005
            loops += 1
            continue
        make_mic = False
    # states[permut] ==  mic.all_points
    if n_pts == 0:
        return np.zeros(0)
    states_scores = np.empty(n_pts)
    n_enr = len(model.sub_energies)
    if return_sub_energies:
        state_energies = np.empty((n_pts, n_enr))
    else:
        state_energies = None

    energy_func = model.energy_func_wrapper(
        position_energy_map=pos_e_m, marks_energy_maps=marks_e_m, compute_context=True
    )
    device = model.device

    if verbose > 0:
        pbar = tqdm(range(n_pts), desc='computing papangelou scores')
    else:
        pbar = range(n_pts)

    buffer: Dict = {}

    for i in pbar:
        if len(mic) == 0:
            assert not np.any(np.isnan(states_scores))
            break

        scores_per_cell = [np.empty(n)
                           for c, n in enumerate(mic.cells_point_number)]
        energies_delta_per_cell = [
            np.empty((n, n_enr)) for c, n in enumerate(mic.cells_point_number)]
        # find minimal score
        for cell_class in mic.available_sets:
            current_cells = mic.cells_indices_per_set[cell_class]

            if use_buffer:
                cells_to_compute = []
                for ic, c in enumerate(current_cells):
                    if c in buffer:
                        # load buffer data
                        scores_per_cell[c] = buffer[c]['score']
                        energies_delta_per_cell[c] = buffer[c]['energies']
                    else:
                        cells_to_compute.append(c)
                cells_to_compute = np.array(cells_to_compute)
            else:
                cells_to_compute = current_cells
            if len(cells_to_compute) > 0:
                cells_to_compute_batches = [cells_to_compute[bi]
                                            for bi in make_batch_indices(len(cells_to_compute), batch_size=16)]
                for cells_batch in cells_to_compute_batches:
                    context_cube, context_cube_mask, _ = mic.points_and_context(
                        cell_indicies=cells_batch, reduce=True, keep_one=True)
                    res_0 = energy_func(
                        context_cube.to(device), context_cube_mask.to(device)
                    )
                    energy_per_set_0 = res_0['energy_per_subset']
                    # sub_energies_per_set_0 = np.stack(
                    #         [res_0[k].detach().cpu().numpy() for k in model.sub_energies]
                    #         , axis=-1)
                    max_pts = np.max(mic.cells_point_number[cells_batch])
                    for point_index in range(max_pts):
                        non_empty_cells_mask = mic.masks[cells_batch,
                                                         point_index]
                        non_empty_cells = cells_batch[non_empty_cells_mask.bool(
                        ).cpu().numpy().astype(bool)]
                        # nb_sets = len(non_empty_cells)
                        removal_indices = point_index
                        context_cube_mask[non_empty_cells_mask,
                                          0, 0, removal_indices] = False

                        res_1 = energy_func(
                            context_cube.to(
                                device), context_cube_mask.to(device)
                        )
                        energy_per_set_1 = res_1['energy_per_subset']

                        energy_delta = (
                            energy_per_set_1[non_empty_cells_mask] -
                            energy_per_set_0[non_empty_cells_mask]
                        ).detach().cpu().numpy()

                        sub_energies_delta = np.stack(
                            [
                                torch.sum(res_1[k][non_empty_cells_mask] - res_0[k][non_empty_cells_mask],
                                          dim=-1).detach().cpu().numpy()
                                for k in model.sub_energies], axis=-1)

                        if log_values:
                            scores = energy_delta
                        else:
                            scores = np.exp(energy_delta)

                        # put it back !
                        context_cube_mask[non_empty_cells_mask,
                                          0, 0, removal_indices] = True

                        for j, c in enumerate(non_empty_cells):
                            scores_per_cell[c][point_index] = scores[j]
                            energies_delta_per_cell[c][point_index] = sub_energies_delta[j]

                    if use_buffer:
                        for c in cells_batch:
                            buffer[c] = {
                                'score': scores_per_cell[c],
                                'energies': energies_delta_per_cell[c]
                            }

        # find minimal score
        argmin_per_cell = [np.argmin(s) if mic.cells_point_number[k] > 0 else None for k, s in
                           enumerate(scores_per_cell)]
        min_per_cell = [s[a] if a is not None else np.inf for a,
                        s in zip(argmin_per_cell, scores_per_cell)]
        argmin_cell = np.argmin(min_per_cell)
        min_score = min_per_cell[argmin_cell]

        # remove minimal score point
        pt_set, pt_index = argmin_cell, argmin_per_cell[argmin_cell]
        pt_state = mic.cells[pt_set, pt_index]
        # corresp = torch.all(states == pt_state, dim=-1)
        n_corresp = np.inf
        corresp = None
        ii = 4
        flag = True
        while flag:
            fuzz_factor = np.power(10.0, -ii)
            corresp = torch.all(torch.abs(states - pt_state)
                                < fuzz_factor, dim=-1)
            n_corresp = torch.sum(corresp)
            assert n_corresp > 0
            if n_corresp == 1:
                flag = False
            elif (n_corresp > 1 and ii > 8):
                flag = False
            if n_corresp == 0:
                raise RuntimeError(f"Failed to match point {pt_state} ({pt_state=},{pt_set=}) in {states=} \n"
                                   f"found {n_corresp} match(es) with {fuzz_factor=}")
            ii += 1
        if n_corresp == 1:
            arg = torch.argmax(corresp.float(), dim=0)
            states_scores[arg] = min_score
        else:
            logging.error(
                f"found two corresponding points for {pt_state}, putting the score on the two points")
            for arg in np.where(corresp)[0]:  # todo DO NOT DO THIS !
                states_scores[arg] = min_score
        mic.remove_one_point(pt_index, pt_set)
        if use_buffer:
            changed_cells = mic.get_cell_neighbors(int(pt_set))
            for c in changed_cells:
                buffer.pop(c)
        if return_sub_energies:
            min_energies = energies_delta_per_cell[argmin_cell][argmin_per_cell[argmin_cell]]
            state_energies[arg] = min_energies
    if return_sub_energies:
        return states_scores, state_energies
    return states_scores


def papangelou_score_scale(values, log: bool):
    if type(values) is list:
        assert type(values[0]) is np.ndarray
        all_values = np.concatenate(values)
    else:
        assert type(values) is np.ndarray
        all_values = values

    if not log:
        scores = np.exp(all_values)
    else:
        scores = all_values

    v_min = np.min(scores)
    v_max = np.max(scores)

    if type(values) is list:
        result = [
            (v - v_min) / (v_max - v_min) for v in values
        ]
    else:
        result = (values - v_min) / (v_max - v_min)

    return result


def papangelou_score_eq(values, uniform_bins=False):
    N_BINS_MAX = int(1e6)
    rng = np.random.default_rng(0)
    if type(values) is list:
        assert type(values[0]) is np.ndarray
        all_values = np.concatenate(values)
    else:
        assert type(values) is np.ndarray
        all_values = values

    n_values = len(all_values)
    # n_bins = n_values
    #
    #     n_bins = N_BINS_MAX

    with Timer() as timer:
        if uniform_bins:
            if n_values > N_BINS_MAX:
                logging.warning(
                    f"len(values)={n_values}, clipping uniform n_bins from {n_values} to {N_BINS_MAX}")
            bins = min(N_BINS_MAX, n_values)
        else:
            if n_values < N_BINS_MAX:
                bins = np.sort(all_values)
            else:
                logging.warning(
                    f"len(values)={n_values}, clipping uniform n_bins from {n_values} to {N_BINS_MAX}")
                v_max = np.max(values)
                v_min = np.min(values)

                flag = True
                repeat = 0
                while flag:
                    mid_bins = rng.choice(
                        values, size=N_BINS_MAX - 2, replace=False)
                    if v_max not in mid_bins:
                        if v_min not in mid_bins:
                            flag = False
                    if repeat > 16:
                        raise RuntimeError()
                    repeat += 1

                bins = np.concatenate(
                    [[v_min], np.sort(mid_bins), [v_max]]
                )

        histogram, bins = np.histogram(all_values, bins=bins)
        cdf = histogram.cumsum()
        cdf = cdf / cdf[-1]
        cdf = np.pad(cdf, (1, 0), constant_values=0.0)
        if type(values) is list:
            result = [
                np.interp(v, bins, cdf) for v in values
            ]
        else:
            result = np.interp(values, bins, cdf)
    logging.info(f"scaled papangelou scores in {timer():.1e}s")
    return result
