import logging
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from base.state_ops import clip_state_to_bounds
from display.draw_on_ax import draw_shapes_on_ax
from display.mpp_model_results import display_position_energy_map, display_mark_energy_map, display_state_w_energies, \
    display_state_w_papangelou, display_energy_details
from energies.generic_energy_model import GenericEnergyModel
from samplers.mic_set_utils import state_to_context_cube
from samplers.rjmcmc import ParallelRJMCMC, Kernel

# ENERGY_CMAP = 'coolwarm'
# ENERGY_CMAP ='PRGn'
ENERGY_CMAP = 'PiYG'
SCORE_CMAP = 'magma'


def plot_weights(model: GenericEnergyModel):
    d = model.energy_combination_module.describe()
    values = np.array(list(d.values()))
    fig, axs = plt.subplots(1, 1)
    axs.bar(np.arange(len(values)), values, color=plt.get_cmap(
        'viridis')(np.abs(values) / np.max(np.abs(values))))
    axs.set_xticks(np.arange(len(values)))
    axs.set_xticklabels(list(d.keys()), rotation=90)
    return fig


def plot_maps(model: GenericEnergyModel, image: np.ndarray, state_gt: Tensor, interpolate: bool = True,
              energy_infer_args=None):
    if energy_infer_args is None:
        energy_infer_args = {}
    image_t = torch.from_numpy(np.stack(image, axis=0)).permute(
        (2, 0, 1)).unsqueeze(dim=0)
    n_rows, n_cols = 1, 2 + len(model.mappings)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6), squeeze=False, sharex='all',
                            sharey='all')

    pos_map, marks_maps = model.energy_maps_from_image(
        image_t.to(model.device), **energy_infer_args)

    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Image')
    h, w = image.shape[:2]

    display_position_energy_map(axs[0, 1], pos_map[0], states=state_gt, h=h, w=w, labels=False, cbar=True,
                                colormap=ENERGY_CMAP)
    axs[0, 1].set_title("Position energy map")
    i = 2
    for mapping, mark_map in zip(model.mappings, marks_maps):
        display_mark_energy_map(axs[0, i], state_gt, mark_map, mapping, model, h, w, i - 2, labels=False, cbar=True,
                                colormap=ENERGY_CMAP, norm='on_state', interpolate=interpolate)
        axs[0, i].set_title(f"Mark energy map: {mapping.name}")
        i += 1
    del pos_map, marks_maps
    return fig


def plot_rjmcmc_log(log):
    # keys = list(log.keys())
    # keys.append('energy per point')

    keys = ['temperature', 'dt', 'n_points', 'energy', 'energy per point']

    n_rows, n_cols = 1, len(keys)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(
        n_cols * 6, n_rows * 6), squeeze=True)
    for i, k in enumerate(keys):
        if k == 'temperature':
            axs[i].plot(log[k])
            axs[i].set_yscale("log")
        elif k == 'dt':
            dts = np.stack(log[k], axis=0)
            for j in range(dts.shape[-1]):
                lgd = 'dt_' + ['i', 'j', 'a', 'b', 'alpha'][j]
                axs[i].plot(dts[:, j], label=lgd)
            axs[i].set_yscale("log")
            axs[i].legend()
        elif k == 'energy per point':
            values = np.array(log['energy']).astype(float)
            n_points = np.array(log['n_points'])
            axs[i].plot(values / np.maximum(1, n_points))
            # axs[i].set_yscale("log")
        # elif k == 'parallel_cells':
        #     values = log[k]
        #     bins = np.arange(np.min(values), np.max(values) + 2)
        #     axs[i].hist(values, bins=bins, align='left')
        else:
            axs[i].plot(log[k])
        axs[i].set_title(k)

    return fig


def plot_rjmcmc_kernels_log(log):
    def expand_nans(s: pd.Series):
        # print(s)
        cells = s['current_cells']
        n = len(cells)
        for k in s.keys():
            if k != 'kernel':
                if s[k] is None or s[k] is np.nan:
                    s[k] = np.full(n, np.nan)
        return s

    if type(log) is list:
        kernel_logs = []
        for l in log:
            kernel_logs.append(pd.DataFrame(l['kernel_log']))
        kernel_logs = pd.concat(kernel_logs, ignore_index=True)
    else:
        kernel_logs = pd.DataFrame(log['kernel_log'])
    kernel_logs_nd = kernel_logs[kernel_logs.kernel != Kernel.DIFFUSION]
    kernel_logs_nd.apply(expand_nans, axis=1)
    explode_keys = ['accepted', 'delta_energy']
    used_keys = ['kernel'] + explode_keys
    kernel_logs_nd = kernel_logs_nd[used_keys].explode(
        column=explode_keys,
        ignore_index=True
    ).fillna(value=np.nan)

    n_rows, n_cols = 1, 5
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(
        n_cols * 6, n_rows * 6), squeeze=True)
    hue_order_1 = [Kernel.BIRTH, Kernel.DEATH]
    hue_order_2 = [Kernel.BIRTH, Kernel.DEATH, Kernel.DIFFUSION]

    sns.histplot(
        kernel_logs_nd.replace({'accepted': {True: 'true', False: 'false', np.nan: 'failed'}}).sort_values(
            by="accepted"),
        x='accepted', hue='kernel', multiple='dodge', shrink=0.8, ax=axs[0], stat='proportion',
        hue_order=hue_order_1, common_norm=False
    )

    sns.histplot(
        kernel_logs_nd,
        x='delta_energy', hue='kernel', ax=axs[1], stat='density', element='step',
        hue_order=hue_order_1
    )

    kernel_logs['n_cells'] = kernel_logs.current_cells.apply(len)
    sns.histplot(
        data=kernel_logs, x='n_cells', hue='kernel', hue_order=hue_order_2,
        multiple='dodge', common_norm=False, stat='proportion', ax=axs[2]
    )

    sns.histplot(
        kernel_logs,
        x='time', hue='kernel', ax=axs[3], stat='density', element='step',
        hue_order=hue_order_2, log_scale=True
    )
    kernel_logs['time_per_cell'] = kernel_logs['time'] / kernel_logs['n_cells']

    sns.histplot(
        kernel_logs,
        x='time_per_cell', hue='kernel', ax=axs[4], stat='density', element='step',
        hue_order=hue_order_2, log_scale=True
    )

    # kernel_logs_nd['green_ratio'] = np.where(
    #     np.isnan(kernel_logs_nd.log_green_ratio),
    #     np.nan,
    #     np.minimum(1, np.exp(kernel_logs_nd.log_green_ratio)))
    # kernel_logs_nd['log_temperature'] = np.where(
    #     np.isnan(kernel_logs_nd.temperature),
    #     np.nan,
    #     np.log(kernel_logs_nd.temperature + 1e-8))
    #
    # kernel_logs_nd_f = kernel_logs_nd[~kernel_logs_nd.green_ratio.isna()]
    # kernel_logs_nd_f = kernel_logs_nd_f[['kernel', 'log_temperature', 'green_ratio']]
    # fig = sns.jointplot(data=kernel_logs_nd_f, x='log_temperature', y='green_ratio', hue='kernel')

    return fig


def plot_config_energy(model: GenericEnergyModel, state: Tensor, image: np.ndarray, draw_method: str,
                       energy_infer_args=None):
    if energy_infer_args is None:
        energy_infer_args = {}
    image_t = torch.from_numpy(np.stack(image, axis=0)).permute(
        (2, 0, 1)).unsqueeze(dim=0)
    n_rows, n_cols = 1, 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 8), squeeze=True, sharex='all',
                            sharey='all')

    pos_e_m, marks_e_m = model.energy_maps_from_image(
        image_t.to(model.device), **energy_infer_args)
    # pos_e_m, marks_e_m = model.energy_maps_from_image(image_t.to(model.device))
    ax = axs
    cube, cube_m = state_to_context_cube(state)
    output = model.forward(cube.to(model.device), cube_m.to(model.device),
                           pos_e_m.to(model.device), [
        m.to(model.device) for m in marks_e_m],
        compute_context=False)
    display_state_w_energies(ax=ax, image=image, states=state, draw_method=draw_method, output=output,
                             cube_m=cube_m, colormap=ENERGY_CMAP)
    ax.axis('off')
    return fig


def plot_papangelou(model: GenericEnergyModel, image: np.ndarray, state_gt: Tensor, draw_method: str,
                    energy_infer_args=None):
    if energy_infer_args is None:
        energy_infer_args = {}
    overlaps = True

    state_dim = state_gt.shape[1]
    if overlaps:
        state = state_gt.numpy()
        copy_points = min(len(state) // 2, 20)
        new_pts = state[:copy_points] + \
            np.random.normal(0, 0.5, size=(copy_points, state_dim))
        state = np.concatenate([state, new_pts], axis=0)
        state = torch.tensor(state)
    else:
        state = state_gt
        new_pts = np.empty((0, state_dim))
    new_pts = torch.tensor(new_pts)

    h, w = image.shape[:2]
    bound_min = torch.tensor(
        [0, 0] + [m.v_min for m in model.mappings], device=state.device)
    bound_max = torch.tensor(
        [h, w] + [m.v_max for m in model.mappings], device=state.device)
    cyclic = torch.tensor(
        [False, False] + [m.is_cyclic for m in model.mappings], device=state.device)
    state = clip_state_to_bounds(
        state, bound_min, bound_max, cyclic=cyclic
    )

    n_rows, n_cols = 1, 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 8), squeeze=True, sharex='all',
                            sharey='all')

    alpha = 0.25
    image_faded = alpha * image + (1 - alpha) * np.zeros_like(image)

    axs[0].imshow(image_faded)
    axs[0].set_title("Ground truth")
    draw_shapes_on_ax(axs[0], state_gt, fill=False,
                      color='green', draw_method=draw_method)

    axs[1].imshow(image_faded)
    axs[1].set_title("Added points")
    draw_shapes_on_ax(axs[1], new_pts, fill=False,
                      color='red', draw_method=draw_method)

    image_t = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(dim=0)

    pos_e_m, marks_e_m = model.energy_maps_from_image(
        image_t.to(model.device), **energy_infer_args)

    display_state_w_papangelou(ax=axs[2], image=image_faded, states=state, draw_method=draw_method, pos_e_m=pos_e_m,
                               marks_e_m=marks_e_m, model=model, show_delta_u=False, colormap=ENERGY_CMAP)
    axs[2].set_title(r'Papangelou $\lambda(y;Y \backslash \{y\})$')

    display_state_w_papangelou(ax=axs[3], image=image_faded, states=state, draw_method=draw_method, pos_e_m=pos_e_m,
                               marks_e_m=marks_e_m, model=model, show_delta_u=True, colormap=ENERGY_CMAP)
    axs[3].set_title(r'$\Delta U(Y \backslash \{y\} \rightarrow Y)$')

    return fig


def plot_config_score(state: Tensor, score: np.ndarray, image: np.ndarray):
    n_rows, n_cols = 1, 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 8), squeeze=True, sharex='all',
                            sharey='all')
    cmap = plt.get_cmap(SCORE_CMAP)
    score_s = score
    if len(score_s) > 0:
        score_s = (score_s - score_s.min()) / (score_s.max() - score_s.min())
    colors = cmap(score_s)
    labels = [f"{s:.2f}" for s in score]
    axs.imshow(image)
    draw_shapes_on_ax(ax=axs, states=state, colors=colors,
                      labels=labels, fill=False)
    axs.axis('off')
    return fig


def plot_energy_details_2(states: List[Tensor], states_labels: List[str], model: GenericEnergyModel, image: np.ndarray,
                          draw_method: str, summarise: bool = False, energy_infer_args=None):
    if energy_infer_args is None:
        energy_infer_args = {}
    image_t = torch.from_numpy(np.stack(image, axis=0)).permute(
        (2, 0, 1)).unsqueeze(dim=0)

    pos_e_m, marks_e_m = model.energy_maps_from_image(
        image_t.to(model.device), **energy_infer_args)

    cache = {}

    for j, state in enumerate(states):
        cube, cube_m = state_to_context_cube(state)
        output = model.forward(cube.to(model.device), cube_m.to(model.device),
                               pos_e_m.to(model.device), [
            m.to(model.device) for m in marks_e_m],
            compute_context=False)
        cache[j] = {
            'state': state,
            'label': states_labels[j],
            'output': output,
            'cube': cube, 'cube_m': cube_m
        }

    if not summarise:
        n = len(model.sub_energies)
        n_cols = n
        n_rows = len(states)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6), squeeze=False, sharex='all',
                                sharey='all')
        [ax.axis('off') for ax in axs.ravel()]
        weights = model.combination_module_weights

        DISP_WEIGHED = True

        for i, k in enumerate(model.sub_energies):

            for j, sd in cache.items():

                ax: plt.Axes = axs[j, i]
                if DISP_WEIGHED:
                    weight = weights[f'weight_{k}']
                else:
                    weight = 1.0

                display_energy_details(ax, image, states=sd['state'], cube_m=sd['cube_m'], output=sd['output'],
                                       energy_key=k,
                                       show_radiuses=True, model=model,
                                       draw_method=draw_method, colormap=ENERGY_CMAP, weight=weight)
                if DISP_WEIGHED:
                    ax.set_title(f"{k} energy on {sd['label']}  (weighted)")
                else:
                    ax.set_title(
                        f"{k} energy on {sd['label']}  (w={weights[f'weight_{k}']:.2f})")

    else:
        # raise NotImplementedError
        priors_names = [m.name for m in model.prior_modules]
        data_names = model.data_energies.sub_energies

        n_cols = 2
        n_rows = len(states)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6), squeeze=False, sharex='all',
                                sharey='all')
        [ax.axis('off') for ax in axs.ravel()]
        weights = model.combination_module_weights

        def add_energies(state, output, cube_m, enr_keys):
            tot_enr = np.zeros(len(state))
            mask = cube_m[:, 0, 0]
            for k in enr_keys:
                weight = weights[f'weight_{k}']
                v = output[k][mask].cpu().detach().numpy() * weight
                tot_enr += v
            return tot_enr

        for j, sd in cache.items():
            prior_enr = add_energies(
                sd['state'], sd['output'], sd['cube_m'], priors_names)
            data_enr = add_energies(
                sd['state'], sd['output'], sd['cube_m'], data_names)

            sd.update({
                'prior_enr': prior_enr,
                'data_enr': data_enr
            })

            for ax, title, values in zip((axs[j, 0], axs[j, 1]),
                                         (f"data energy on {sd['label']}",
                                          f"prior energy on {sd['label']}"),
                                         (data_enr, prior_enr)
                                         ):
                display_energy_details(ax, image, states=sd['state'], values=values, model=model,
                                       draw_method=draw_method,
                                       colormap=ENERGY_CMAP)
                ax.set_title(title)

    del cache
    return fig


def plot_energy_details(state_gt: Tensor, last_state: Tensor, model: GenericEnergyModel, image: np.ndarray,
                        draw_method: str, summarise: bool = False, energy_infer_args=None):
    if energy_infer_args is None:
        energy_infer_args = {}
    image_t = torch.from_numpy(np.stack(image, axis=0)).permute(
        (2, 0, 1)).unsqueeze(dim=0)
    state_1 = state_gt
    state_2 = last_state

    pos_e_m, marks_e_m = model.energy_maps_from_image(
        image_t.to(model.device), **energy_infer_args)

    cube_1, cube_m_1 = state_to_context_cube(state_1)
    output_1 = model.forward(cube_1.to(model.device), cube_m_1.to(model.device),
                             pos_e_m.to(model.device), [
        m.to(model.device) for m in marks_e_m],
        compute_context=False)

    cube_2, cube_m_2 = state_to_context_cube(state_2)
    output_2 = model.forward(cube_2.to(model.device), cube_m_2.to(model.device),
                             pos_e_m.to(model.device), [
        m.to(model.device) for m in marks_e_m],
        compute_context=False)

    if not summarise:
        n = len(model.sub_energies)
        n_cols = n
        n_rows = 2
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6), squeeze=False, sharex='all',
                                sharey='all')
        [ax.axis('off') for ax in axs.ravel()]
        weights = model.combination_module_weights

        DISP_WEIGHED = True

        for i, k in enumerate(model.sub_energies):
            ax: plt.Axes = axs[0, i]
            if DISP_WEIGHED:
                weight = weights[f'weight_{k}']
            else:
                weight = 1.0

            display_energy_details(ax, image, states=state_1, cube_m=cube_m_1, output=output_1, energy_key=k,
                                   show_radiuses=True, model=model,
                                   draw_method=draw_method, colormap=ENERGY_CMAP, weight=weight)
            if DISP_WEIGHED:
                ax.set_title(f"{k} energy on GT  (weighted)")
            else:
                ax.set_title(
                    f"{k} energy on GT  (w={weights[f'weight_{k}']:.2f})")

            ax: plt.Axes = axs[1, i]
            display_energy_details(ax, image, states=state_2, cube_m=cube_m_2, output=output_2, energy_key=k,
                                   show_radiuses=True, model=model,
                                   draw_method=draw_method, colormap=ENERGY_CMAP, weight=weight)
            if DISP_WEIGHED:
                ax.set_title(f"{k} energy on inferred (weighted)")
            else:
                ax.set_title(
                    f"{k} energy on inferred  (w={weights[f'weight_{k}']:.2f})")
    else:
        # raise NotImplementedError
        priors_names = [m.name for m in model.prior_modules]
        data_names = model.data_energies.sub_energies

        n_cols = 2
        n_rows = 2
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6), squeeze=False, sharex='all',
                                sharey='all')
        [ax.axis('off') for ax in axs.ravel()]
        weights = model.combination_module_weights

        def add_energies(state, output, cube_m, enr_keys):
            tot_enr = np.zeros(len(state))
            mask = cube_m[:, 0, 0]
            for k in enr_keys:
                weight = weights[f'weight_{k}']
                v = output[k][mask].cpu().detach().numpy() * weight
                tot_enr += v
            return tot_enr

        prior_enr_1 = add_energies(state_1, output_1, cube_m_1, priors_names)
        data_enr_1 = add_energies(state_1, output_1, cube_m_1, data_names)
        prior_enr_2 = add_energies(state_2, output_2, cube_m_2, priors_names)
        data_enr_2 = add_energies(state_2, output_2, cube_m_2, data_names)

        tab = [
            (axs[0, 0], 'data energy on GT', state_1, data_enr_1),
            (axs[1, 0], 'data energy on inferred', state_2, data_enr_2),
            (axs[0, 1], 'prior energy on GT', state_1, prior_enr_1),
            (axs[1, 1], 'prior energy on inferred', state_2, prior_enr_2),
        ]

        for ax, title, state, values in tab:
            display_energy_details(ax, image, states=state, values=values, model=model, draw_method=draw_method,
                                   colormap=ENERGY_CMAP)
            ax.set_title(title)

    del output_1, output_2, cube_m_1, cube_1, cube_m_2, cube_2, pos_e_m, marks_e_m
    return fig


def plot_prior_only(last_state: Tensor, image: np.ndarray, draw_method: str):
    h, w = image.shape[:2]
    image_t = torch.ones((1, 3, h, w)) * 0.5
    image_gray = image_t[0].permute(1, 2, 0).numpy()

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    axs.imshow(image_gray)

    draw_shapes_on_ax(axs, last_state, fill=False,
                      color='white', draw_method=draw_method)
    return fig


def plot_sampler(model: GenericEnergyModel, image: np.ndarray, draw_method: str, config, rjmcmc_overrides,
                 energy_infer_args=None):
    if energy_infer_args is None:
        energy_infer_args = {}
    image_t = torch.from_numpy(np.stack(image, axis=0)).permute(
        (2, 0, 1)).unsqueeze(dim=0)
    rng = np.random.default_rng(0)

    pos_e_map, marks_e_map = model.energy_maps_from_image(
        image_t.to(model.device), **energy_infer_args)

    pos_e_map = pos_e_map[0]
    marks_e_map = [m[0] for m in marks_e_map]

    shape = pos_e_map.shape[1:]
    pos_density_map, marks_density_maps = model.densities_from_energy_maps(
        pos_e_map, marks_e_map)

    config_rjmcmc = config['RJMCMC_params']

    position_birth_density = pos_density_map.detach().cpu().numpy()
    logging.info(
        f"birth density (sum over image) = {position_birth_density.sum()} (should be 1.0)")

    # intensity_tot = len(state_gt)
    # intensity_tot = 10.0

    # intensity = float(intensity_tot / np.prod(shape))
    intensity = 0.001
    intensity_tot = intensity * np.prod(shape)
    logging.info(
        f"intensity * |S| = {intensity_tot} | intensity = {intensity}")

    logging.info("original config : ")
    logging.info(config_rjmcmc)

    overrides = rjmcmc_overrides.copy()

    # if config_rjmcmc.get('scale_temperature',False):
    #     overrides['end_temperature'] = 1e-8

    for k, w in overrides.items():
        config_rjmcmc[k] = w

    logging.info("updated config : ")
    logging.info(config_rjmcmc)

    pert_mc = ParallelRJMCMC(
        support_shape=shape,
        device=model.device,
        max_interaction_distance=model.max_interaction_distance,
        rng=rng,
        energy_func=model.energy_func_wrapper(position_energy_map=pos_e_map,
                                              marks_energy_maps=marks_e_map,
                                              compute_context=True),
        position_birth_density=position_birth_density,
        mark_birth_density=marks_density_maps,
        mappings=model.mappings,
        debug_mode=False,
        intensity=intensity,
        # energy_scaling='auto',
        **config_rjmcmc
    )

    n_rows, n_cols = 1, 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6), squeeze=True, sharex='all',
                            sharey='all')

    ax: plt.Axes = axs[0]
    ax.imshow(image)
    ax.set_title("image")

    ax: plt.Axes = axs[1]
    ax.imshow(pert_mc.sampler.get_density(norm_per_cell=False))
    pert_mc.mic_points.display_mic_set(ax, fill=False, color="white")
    ax.set_title("position density")

    ax: plt.Axes = axs[2]
    ax.imshow(image)
    cmap = plt.get_cmap('tab10')
    pert_mc.mic_points.display_mic_set(ax, fill=False, color="white")
    n = 1024
    all_pts = []
    for k in range(n):
        set_class, current_cells, current_cells_pick_p = pert_mc.pick_current_cells()
        pts = pert_mc.sampler.sample(current_cells=current_cells, rng=rng)
        draw_shapes_on_ax(ax=ax, states=pts, color=cmap(
            set_class), fill=False, alpha=0.5, draw_method=draw_method)
        all_pts.append(pts)
    ax.set_title(f"{n} samples")
    all_pts = np.concatenate(all_pts)

    h, w = image.shape[:2]
    n_bins = 32
    ax: plt.Axes = axs[3]
    bins_x = np.linspace(0, w, int(n_bins * shape[1] / 256))
    bins_y = np.linspace(0, h, int(n_bins * shape[0] / 256))
    # ax.hist2d(x=all_pts[:, 1], y=all_pts[:, 0],
    # bins=[bins_x, bins_y])
    hist, _, _ = np.histogram2d(
        x=all_pts[:, 1], y=all_pts[:, 0], bins=[bins_x, bins_y])
    ax.imshow(hist.T, cmap='viridis', extent=(-0.5,
              shape[1] - 0.5, shape[0] - 0.5, -0.5))
    ax.set_title(f"Distribution of {n} samples (2d histogram)")

    return fig
