import logging
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import patches, colors, colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.random import Generator
from torch import Tensor

from base.images import map_range_auto
from base.mappings import ValueMapping
from base.misc import all_logging_disabled
from display.draw_on_ax import draw_shapes_on_ax
from energies.base import BaseEnergyModel
from energies.compute_distances import compute_distances_in_sets
from energies.generic_energy_model import GenericEnergyModel
from energies.sub_energies_modules.base import PriorModule
from modules.interpolation import interpolate_position, interpolate_marks
from samplers.mic_set_utils import state_to_context_cube

# DEFAULT_ENR_COLORMAP = 'coolwarm'
DEFAULT_ENR_COLORMAP = 'PiYG'


def analyse_results(model: BaseEnergyModel, rng: Generator, image, state, gt_state, pos_e_map, marks_e_map, save_file,
                    anchors=None, n_crops: int = 4, size: int = 128, **figure_kwargs):
    shape = image.shape[:2]

    # TODO compute total energy and display it somewhere

    if type(state) is Tensor:
        state = state.detach().cpu().numpy()

    images = []
    pos_e_map_crop = []
    marks_e_map_crop = []
    states = []
    gt_states = []
    image_padded = np.pad(
        image, ((0, max(0, size - shape[0])), (0, max(0, size - shape[1])), (0, 0)))
    shape_padded = image_padded.shape[:2]
    if anchors is None:
        anchors = []
        use_anchors = False
    else:
        use_anchors = True
    if shape_padded[0] > size or shape_padded[1] > size:
        for i in range(n_crops):
            if use_anchors:
                anchor = anchors[i]
            else:
                if len(state) > 0:
                    anchor = rng.choice(state[..., :2], axis=0).astype(int)
                elif len(gt_state) > 0:
                    anchor = rng.choice(gt_state[..., :2], axis=0).astype(int)
                else:
                    anchor = rng.uniform(
                        (0, 0), shape_padded, size=2).astype(int)
                anchors.append(anchor)
            tl_anchor = np.clip(anchor - size // 2, (0, 0),
                                (shape_padded[0] - size, shape_padded[1] - size))
            br_anchor = tl_anchor + size

            images.append(
                torch.from_numpy(
                    image_padded[tl_anchor[0]:br_anchor[0], tl_anchor[1]:br_anchor[1]]).permute(
                    (2, 0, 1)))

            last_state_crop = state[np.all(state[..., :2] >= tl_anchor, axis=-1) &
                                    np.all(state[..., :2] <= br_anchor, axis=-1)]
            last_state_crop[..., :2] = last_state_crop[..., :2] - tl_anchor
            states.append(last_state_crop)
            if gt_state.shape[0] > 0:
                gt_state_crop = gt_state[np.all(gt_state[..., :2] >= tl_anchor, axis=-1) &
                                         np.all(gt_state[..., :2] <= br_anchor, axis=-1)]
                gt_state_crop[..., :2] = gt_state_crop[..., :2] - tl_anchor
            else:
                gt_state_crop = np.zeros_like(gt_state)
            gt_states.append(gt_state_crop)
            if len(pos_e_map.shape) == 4:
                pos_e_map_crop.append(
                    pos_e_map[..., tl_anchor[0]:br_anchor[0], tl_anchor[1]:br_anchor[1]])
                marks_e_map_crop.append(
                    [m[..., tl_anchor[0]:br_anchor[0], tl_anchor[1]:br_anchor[1]] for m in
                     marks_e_map])
            else:
                pos_e_map_crop.append(
                    pos_e_map[:, tl_anchor[0]:br_anchor[0], tl_anchor[1]:br_anchor[1]].unsqueeze(dim=0))
                marks_e_map_crop.append(
                    [m[:, tl_anchor[0]:br_anchor[0], tl_anchor[1]:br_anchor[1]].unsqueeze(dim=0) for m in
                     marks_e_map])
    elif size > shape[0] or size > shape[0]:
        images = [torch.from_numpy(image).permute((2, 0, 1))]
        states = [state]
        gt_states = [gt_state]
        pos_e_map_crop = [pos_e_map.unsqueeze(dim=0)]
        marks_e_map_crop = [[m.unsqueeze(dim=0) for m in marks_e_map]]
    else:
        images = [torch.from_numpy(image_padded).permute((2, 0, 1))]
        states = [state]
        gt_states = [gt_state]
        pos_e_map_crop = [pos_e_map.unsqueeze(dim=0)]
        marks_e_map_crop = [[m.unsqueeze(dim=0) for m in marks_e_map]]

    make_figure(
        model=model,
        save_file=save_file,
        images=images,
        states=[torch.from_numpy(s) for s in states],
        gt_states=[torch.from_numpy(s) for s in gt_states],
        pos_e_maps=pos_e_map_crop,
        marks_e_maps=marks_e_map_crop,
        **figure_kwargs
    )

    return anchors


def display_state_w_energies(ax: plt.Axes, image: np.ndarray, states: Tensor,
                             draw_method: str, output, cube_m, colormap=DEFAULT_ENR_COLORMAP):
    mask = cube_m[:, 0, 0]
    values = output['energy_per_point'][mask].detach().cpu().detach().numpy()
    ax.imshow(image)
    if len(values) > 0:
        norm_values = 0.5 * values / np.max(np.abs(values)) + 0.5
        colors = plt.get_cmap(colormap)(norm_values)
        labels = [f"{v:.2f}" for v in values]
        draw_shapes_on_ax(ax, states=states, fill=False, colors=colors,
                          labels=labels, draw_method=draw_method)


def display_state_w_papangelou(ax: plt.Axes, image: np.ndarray, states: Tensor, draw_method: str, model, pos_e_m,
                               marks_e_m, show_delta_u: bool = False, colormap=DEFAULT_ENR_COLORMAP):
    cube, cube_m = state_to_context_cube(states)

    output = model.forward(cube.to(model.device), cube_m.to(model.device),
                           pos_e_m.to(model.device), [
        m.to(model.device) for m in marks_e_m],
        compute_context=False)
    energy_0 = output['total_energy'].detach().cpu().numpy()
    assert torch.all(cube_m[:, 0, 0])

    n_pts = cube.shape[3]
    papangelou_all = []

    for i in range(n_pts):
        cube_m[:, 0, 0, i] = False
        output = model.forward(cube.to(model.device), cube_m.to(model.device),
                               pos_e_m.to(model.device), [
            m.to(model.device) for m in marks_e_m],
            compute_context=False)
        energy_1 = output['total_energy'].detach().cpu().numpy()
        if show_delta_u:
            papangelou = energy_0 - energy_1
        else:
            # papangelou(y;Y)=exp(-(U(Y \cup {y}) - U(Y)) / T)
            papangelou = np.exp(- (energy_0 - energy_1))
        papangelou_all.append(papangelou)
        cube_m[:, 0, 0, i] = True

    values = np.array(papangelou_all)

    ax.imshow(image)
    if len(values) > 0:
        if show_delta_u:
            norm_values = 0.5 * (values) / np.max(np.abs(values)) + 0.5
        else:
            norm_values = 0.5 * (1 - values) / np.max(np.abs(1 - values)) + 0.5
        colors = plt.get_cmap(colormap)(norm_values)
        labels = [f"{v:.2f}" for v in values]
        draw_shapes_on_ax(ax, states=states, fill=False, colors=colors,
                          labels=labels, draw_method=draw_method)


def display_position_energy_map(ax: plt.Axes, pos_e_map, states: Tensor, h, w, labels=True, cbar: bool = False,
                                norm: bool = True, colormap=DEFAULT_ENR_COLORMAP):
    n_points = len(states)
    # assert len(pos_e_map.shape) == 2  # (H,W)
    pos_e_np = pos_e_map.detach().cpu().numpy().squeeze()
    # pos_e_np_norm = 0.5 * pos_e_np / np.max(np.abs(pos_e_np)) + 0.5
    vmax = np.max(np.abs(pos_e_np))
    if norm:
        norm = colors.Normalize(vmin=-vmax, vmax=vmax, clip=False)
    else:
        norm = colors.Normalize(clip=False)

    im = ax.imshow(pos_e_np, cmap=colormap, norm=norm)
    if labels:
        for k in range(n_points):
            y, x = states[k][0], states[k][1]
            ax.scatter(x, y, color='green', s=1.0)
            ax.text(
                x, y, f"{pos_e_np[min(int(y), h - 1), min(int(x), w - 1)]:.2f}", color='green')
    if cbar:
        divider = make_axes_locatable(ax)
        fig = ax.get_figure()
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


def display_mark_energy_map(ax: plt.Axes, states: Tensor, mark_e_map, mapping: ValueMapping, model, h, w, mark_index,
                            upscale: float = 1.0, labels=True, cbar: bool = False, norm: bool = True,
                            colormap=DEFAULT_ENR_COLORMAP, interpolate: bool = True):
    n_points = len(states)
    if len(states) > 0:
        values = states.detach().cpu().numpy()[..., 2 + mark_index]
        if interpolate:
            dists_t = interpolate_position(
                positions=states[:, :2].view(
                    (1, 1, n_points, 2)).float().to(model.device),
                image=mark_e_map.view(
                    (1, mapping.n_classes, h, w)).float().to(model.device)
            ).view((mapping.n_classes, n_points)).T
            dists = dists_t.detach().cpu().numpy()
            e_value = interpolate_marks(
                positions=states[:, :2].view(
                    (1, 1, 1, n_points, 2)).float().to(model.device),
                image=mark_e_map.view(
                    (1, 1, mapping.n_classes, h, w)).float().to(model.device),
                mapping=mapping,
                marks=states[..., 2 + mark_index].view((1, 1, 1, n_points)).float().to(
                    model.device)
            ).view(n_points)

            resolution = int(upscale * mapping.n_classes)

            check_pos = states[:, :2].view((1, 1, 1, n_points, 2)).tile(
                (1, 1, resolution, 1, 1)).float()
            check_marks = torch.linspace(mapping.v_min, mapping.v_max, resolution).view(
                (1, 1, resolution, 1)).tile((1, 1, 1, n_points)).float()

            dists_check = interpolate_marks(
                positions=check_pos.to(model.device).float(),
                image=mark_e_map.view(
                    (1, 1, mapping.n_classes, h, w)).float().to(model.device),
                mapping=mapping,
                marks=check_marks.to(model.device),
                mode='bilinear'
            )[0, 0, 0].permute((1, 0)).detach().cpu().numpy()

            dists_interp_mse = np.mean(np.square(dists - dists_check))
            assert np.isnan(dists_interp_mse) or dists_interp_mse < 1e-2
        else:
            positions = states[:, :2].cpu().numpy().astype(int)
            dists = mark_e_map.cpu().numpy()[
                0, :, positions[:, 0], positions[:, 1]]

    else:
        dists = np.array([])
        values = np.array([])
        e_value = torch.tensor([])

    if norm == 'on_state':
        vmax = np.max(dists)
        norm = colors.Normalize(vmin=-vmax, vmax=vmax, clip=False)
    elif norm:
        vmax = np.max(np.abs(mark_e_map.detach().cpu().numpy()))
        norm = colors.Normalize(vmin=-vmax, vmax=vmax, clip=False)
    else:
        norm = colors.Normalize(clip=False)

    if len(dists) > 0:
        # dists_norm = 0.5 * (dists / np.max(np.abs(dists))) + 0.5
        dists_norm = norm(dists)
        dist_colors = plt.get_cmap(colormap)(dists_norm)
        n_classes = dists_norm.shape[1]
        xx = np.linspace(0, h + 1, num=n_points + 1, dtype=int)
        yy = np.linspace(0, w + 1, num=n_classes + 1, dtype=int)
        dist_plot = np.empty((h, w) + (4,))
        for k in range(n_points):

            for l in range(n_classes):
                dist_plot[xx[k]:min(xx[k + 1], h), yy[l]: yy[l + 1]] = dist_colors[k, l]

            if labels:
                value_int = int(
                    (w - 1) * (values[k] - mapping.v_min) / mapping.range)
                value_int = np.clip(value_int, 0, w - 1)
                dist_plot[xx[k]:min(xx[k + 1], h),
                          value_int] = np.array([0.0, 1.0, 0.0, 1.0])
                # dist_plot[xx[k], :] = np.zeros(4)
                ax.text(value_int, h * (k + 0.5) / n_points,
                        f"{e_value[k].detach().cpu().numpy():.2f}",
                        color='green')

    else:
        dist_plot = np.zeros((h, w) + (4,))

    im = ax.imshow(dist_plot)
    if cbar:
        divider = make_axes_locatable(ax)
        fig = ax.get_figure()
        cax = divider.append_axes('right', size='5%', pad=0.05)
        colorbar.ColorbarBase(ax=cax, cmap=plt.get_cmap(
            colormap), orientation='vertical', norm=norm)
        # fig.colorbar(norm, cax=cax, orientation='vertical')


def display_energy_details(ax: plt.Axes, image: np.ndarray, states: Tensor, model, energy_key: str = None,
                           cube_m: Tensor = None, output=None,
                           show_radiuses: bool = False, draw_method: str = 'rectangle', weight: float = 1.0,
                           values=None,
                           colormap=DEFAULT_ENR_COLORMAP):
    ax.imshow(image)
    n_points = len(states)
    if n_points > 0:
        if values is None:
            v = output[energy_key]
            mask = cube_m[:, 0, 0]
            values = v[mask].cpu().detach().numpy() * weight
        else:
            pass

        # norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))
        abs_max = np.max(np.abs(values))
        if abs_max != 0.0:
            norm_values = 0.5 * values / abs_max + 0.5
        else:
            norm_values = values + 0.5
        colors = plt.get_cmap(colormap)(norm_values)
        labels = [f"{v:.2f}" for v in values]

        if isinstance(model, GenericEnergyModel) and show_radiuses:
            model: GenericEnergyModel
            current_prior: PriorModule = None
            for p in model.prior_modules:
                if p.name == energy_key:
                    current_prior = p
            if current_prior is not None:
                distance = current_prior.maximum_distance
                if distance > 0:
                    for p in states.detach().cpu().numpy():
                        c = patches.Circle(p[[1, 0]], radius=distance, alpha=0.4,
                                           color='yellow', fill=False,
                                           ls='--')
                        ax.add_patch(c)

        draw_shapes_on_ax(ax, states=states, fill=False, colors=colors,
                          labels=labels, draw_method=draw_method)


def display_interactions(ax: plt.Axes, image: np.ndarray, states: Tensor, prior: PriorModule, cube_m: Tensor,
                         draw_method: str, show_radius: bool = False, image_alpha: float = 1.0, colormap='viridis',
                         focus: int = None, bg=0.0, labels=False):
    ax.imshow(np.ones_like(image) * bg)
    ax.imshow(image, alpha=image_alpha)
    n_points = len(states)
    if n_points > 0:
        draw_shapes_on_ax(ax, states=states, fill=False,
                          color='white', draw_method=draw_method)
        if focus is not None:
            draw_shapes_on_ax(ax, states=states[[
                              focus]], fill=False, color='white', draw_method=draw_method, lw=4.0)
        n_sets = cube_m.shape[0]
        points = states.unsqueeze(dim=0)
        points_mask = cube_m[:, 0, 0].view((n_sets, -1))

        dist = compute_distances_in_sets(
            points=points,
            points_mask=points_mask,
            others=points,
            others_mask=points_mask,
            maximum_dist=prior.maximum_distance,
            marks_diff=False
        )

        weights = prior.interaction_weight(
            points=points.float(),
            points_mask=points_mask,
            context_points=points.float(),
            context_points_mask=points_mask,
            distance_matrix=dist.float(),
        )
        if show_radius:
            if focus is not None:
                states_radius = states.detach().cpu().numpy()[[focus]]
            else:
                states_radius = states.detach().cpu().numpy()
            for p in states_radius:
                c = patches.Circle(p[[1, 0]], radius=prior.maximum_distance, alpha=0.4,
                                   color='red', fill=False,
                                   ls='--')
                ax.add_patch(c)

        w_np = weights[0].detach().cpu().numpy()
        d_np = dist[0].detach().cpu().numpy()
        colors = plt.get_cmap(colormap)(map_range_auto(w_np, 0, 1))
        states_np = states.detach().cpu().numpy()
        # if focus is not None:
        #     loop_i = states_np[[focus]]
        # else:
        #     loop_i = states_np
        for i, s1 in enumerate(states_np):
            for j, s2 in enumerate(states_np):
                if focus is None or i == focus:
                    if d_np[i, j] < prior.maximum_distance:
                        if focus is not None:
                            s_mid = s2[:2]
                        else:
                            s_mid = 0.5 * (s1[:2] + s2[:2])
                        s_text = 0.5 * (s1[:2] + s_mid[:2])
                        ax.plot([s1[1], s_mid[1]], [s1[0], s_mid[0]], color=colors[i, j], alpha=1.0,
                                lw=1.0 * w_np[i, j] + 2.0
                                )
                        if labels:
                            ax.text(
                                s_text[1], s_text[0], s=f"{w_np[i, j]:.2f}", clip_on=True, color=colors[i, j])


def make_figure(model: BaseEnergyModel, images, states, gt_states, save_file: str,
                pos_e_maps=None, marks_e_maps=None, show_e_maps=True, show_energy_detail=True,
                draw_method='rectangle', show_radiuses=False, show_interaction_weights=False):
    n_fig = len(images)
    max_points = 512
    legacy_gt = False

    states = [s.detach().cpu() for s in states]

    if pos_e_maps is None or marks_e_maps is None:
        logging.warning(f"did not provide pos_e_map, or marks_e_maps: regenerating "
                        f"-- this might cause display inconsistencies --")
        images_t = images.to(model.device) if type(images) is Tensor \
            else torch.tensor(images).to(model.device)
        pos_e_m, marks_e_m = model.energy_maps_from_image(
            images_t,
            as_energies=True, large_image=False)
        pos_e_maps = [pos_e_m[[i]] for i in range(len(images))]
        marks_e_maps = [[marks_e_m[j][[i]]
                         for j in range(len(marks_e_m))] for i in range(len(images))]

    n_marks = len(marks_e_maps[0])

    c = 2
    if show_e_maps:
        c = c + 1 + n_marks
    if show_energy_detail:
        c = c + len(model.sub_energies)
    if show_interaction_weights:
        assert isinstance(model, GenericEnergyModel)
        model: GenericEnergyModel
        for p in model.prior_modules:
            p: PriorModule
            if p.maximum_distance > 0:
                c = c + 1

    n_rows, n_cols = n_fig, c
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False, sharex='all',
                            sharey='all')
    [ax.axis('off') for ax in axs.ravel()]

    for i in range(n_fig):
        # display GT
        img_np = images[i].detach().cpu().permute((1, 2, 0)).numpy()
        h, w = img_np.shape[:2]
        pos_e_m = pos_e_maps[i]
        marks_e_m = marks_e_maps[i]

        if not legacy_gt:
            cube, cube_m = state_to_context_cube(gt_states[i][:max_points])

            output = model.forward(
                cube.to(model.device), cube_m.to(model.device),
                pos_e_m.to(model.device), [m.to(model.device)
                                           for m in marks_e_m],
                compute_context=False
            )
            display_state_w_energies(
                ax=axs[i, 0],
                image=img_np,
                states=gt_states[i][:max_points],
                draw_method=draw_method,
                output=output, cube_m=cube_m
            )
        else:
            axs[i, 0].imshow(img_np)
            draw_shapes_on_ax(axs[i, 0], states=gt_states[i][:max_points], fill=False, draw_method=draw_method,
                              color='tab:green')

        if i == 0:
            axs[i, 0].set_title('positive samples energies')
        if len(states[i]) > max_points:
            logging.info(
                f"too much points to display : showing {max_points} out of {len(states[i])}")

        cube, cube_m = state_to_context_cube(states[i][:max_points])

        output = model.forward(
            cube.to(model.device), cube_m.to(model.device),
            pos_e_m.to(model.device), [m.to(model.device) for m in marks_e_m],
            compute_context=False
        )

        j = 1
        # display energies
        display_state_w_energies(
            ax=axs[i, j],
            image=img_np,
            states=states[i][:max_points],
            draw_method=draw_method,
            output=output, cube_m=cube_m
        )

        if i == 0:
            axs[i, j].set_title(f'negative samples energies')

        j += 1

        if show_e_maps:

            display_position_energy_map(
                ax=axs[i, j],
                pos_e_map=pos_e_m,
                states=states[i][:max_points],
                h=h, w=w
            )
            if i == 0:
                axs[i, j].set_title('position energy map')

            j += 1

            # Draw marks energies distributions for each object position
            pos = states[i][:max_points].detach(
            ).cpu().numpy()[..., :2].astype(int)
            for i_m, mapping in enumerate(model.mappings):
                display_mark_energy_map(
                    ax=axs[i, j], states=states[i][:max_points],
                    mark_e_map=marks_e_m[i_m], mapping=mapping,
                    model=model, h=h, w=w, mark_index=i_m
                )

                if i == 0:
                    axs[i, j].set_title(f'{mapping.name} mark energies hist.')
                j += 1

        if show_energy_detail:

            # Draw objets with sub-energies (using output)
            for k, v in output.items():
                if k in model.sub_energies:

                    display_energy_details(
                        ax=axs[i, j], image=img_np, states=states[i][:max_points],
                        cube_m=cube_m, output=output, energy_key=k, show_radiuses=show_radiuses,
                        model=model, draw_method=draw_method
                    )

                    if i == 0:
                        axs[i, j].set_title(f'{k} energy')

                    j += 1

        if show_interaction_weights:
            model: GenericEnergyModel
            for p in model.prior_modules:
                p: PriorModule
                if p.maximum_distance > 0:
                    display_interactions(
                        ax=axs[i, j], image=img_np, states=states[i][:max_points], prior=p,
                        cube_m=cube_m, draw_method=draw_method
                    )

                    if i == 0:
                        axs[i, j].set_title(f'{p.name} interaction weights')

                    j += 1

    fig.tight_layout()

    if save_file is None:
        return fig
    else:
        plt.savefig(save_file, bbox_inches='tight')
        plt.close('all')
        logging.info(f"saved file at {save_file}")


# def energies_cross_plot(sub_energies: List[str], output: Dict[str, Tensor], cube_mask: Tensor):
def energies_pair_plot(model: GenericEnergyModel, image: Union[Tensor, List[Tensor]], states: List[Tensor], colors=None,
                       labels=None, **kwargs):
    if colors is None:
        colors = [plt.get_cmap('tab10')(i) for i in range(len(states))]
    if labels is None:
        labels = [f'state {i}' for i in range(len(states))]

    pos_e_m, marks_e_m = None, None
    if type(image) is not list:
        pos_e_m, marks_e_m = model.energy_maps_from_image(
            image.to(model.device))
    outputs = []
    masks = []
    with all_logging_disabled():
        for i, s in enumerate(states):
            if type(image) is list:
                pos_e_m, marks_e_m = model.energy_maps_from_image(
                    image[i].to(model.device))
            cube, cube_m = state_to_context_cube(s)
            output = model.forward(cube.to(model.device), cube_m.to(model.device),
                                   pos_e_m.to(model.device), [
                m.to(model.device) for m in marks_e_m],
                compute_context=False)
            outputs.append(output)
            masks.append(cube_m[0, 0, 0])
    sub_energies = model.sub_energies
    n_energies = len(sub_energies)
    all_df = []
    for i, out in enumerate(outputs):
        if len(out['energy_per_point'][0]) == 0:  # empy config:
            values = np.empty((0, len(sub_energies)))
        else:
            try:
                values = np.stack([out[k].cpu().detach().numpy()[
                                  0, masks[i]] for k in sub_energies], axis=-1)
            except Exception as e:
                print(out)
                print(masks[i])
                print(e)
                values = np.empty((0, len(sub_energies)))
        df = pd.DataFrame(values, columns=sub_energies)
        df['label'] = labels[i]
        all_df.append(df)

    big_df = pd.concat(all_df, ignore_index=True)

    sns.pairplot(big_df, hue='label', **kwargs)
    return None

    # n_rows, n_cols = n_energies, n_energies
    # fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex='col', sharey='row')
    #
    # for i, ki in enumerate(sub_energies):
    #     for j, kj in enumerate(sub_energies):
    #         ax: plt.Axes = axs[j, i]
    #
    #         for k, o in enumerate(outputs):
    #             values_i = o[ki].cpu().detach().numpy()[0, masks[k]]
    #             values_j = o[kj].cpu().detach().numpy()[0, masks[k]]
    #
    #             ax.scatter(values_i, values_j, color=colors[k],s=10.0,alpha=0.65)
    #
    #         if i == 0:
    #             ax.set_ylabel(kj + ' energy')
    #         if j == n_energies - 1:
    #             ax.set_xlabel(ki + ' energy')
    #
    # fig.tight_layout()

    # return fig
