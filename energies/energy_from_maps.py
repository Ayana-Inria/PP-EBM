import logging
import warnings
from typing import List, Dict

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import Module

from base.data import get_model_config_by_name
from base.mappings import mappings_from_config
from base.parse_config import ConfigParser
from base.state_ops import clip_state_to_bounds, check_inbound_state
from energies import energy_combinators, quality_functions
from energies.base import BaseEnergyModel
from energies.compute_distances import compute_distances_in_sets
from energies.energy_combinators import BaseEnergyCombinator
from energies.intersect_modules import RectangleIntersect
from models.mpp_data_net.model import MPPDataModel
from modules.interpolation import interpolate_position, interpolate_marks
from samplers.mic_set_utils import state_to_context_cube
from samplers.types import EnergyFunc


class EnergyFromMaps(BaseEnergyModel, Module):

    def __init__(self, config, device=None):
        super(EnergyFromMaps, self).__init__()
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        logging.info(f"device is {self.device}")
        self.config = config
        self.check_gradients = self.config['model']['check_gradients']
        if self.check_gradients:
            logging.warning(
                "check_gradients is True ! this may slow down execution")
        # self.position_energy_map = position_energy_map.unsqueeze(dim=0).unsqueeze(dim=0).to(self.device).float()
        # self.marks_energy_maps = [m.unsqueeze(dim=0).to(self.device).float() for m in marks_energy_maps]

        self.max_dist = self.config['model']['maximum_distance']
        self.energy_from_densities = self.config['model']['energy_from_densities']

        maps_model = self.config['model'].get('maps_module')
        self.maps_module = None
        if maps_model is not None:
            map_model_config = get_model_config_by_name(maps_model)
            with open(map_model_config, 'r') as f:
                map_model_config = yaml.load(f, Loader=yaml.SafeLoader)
            map_model_config = ConfigParser(
                map_model_config, 'MPPDataModel', load_model=True, resume=True)
            self.maps_module: MPPDataModel = MPPDataModel(map_model_config)
            checkpoint = torch.load(map_model_config.resume, )
            self.maps_module.load_state_dict(checkpoint['state_dict'])

            self.maps_module.training = False
            logging.debug(
                f"loaded MPPDataModel with config :\n{repr(map_model_config)}")
            self.mappings = mappings_from_config(self.maps_module.config)
        else:
            self.mappings = mappings_from_config(self.config)

        self._sub_energies = [
            'position', 'width', 'length', 'angle', 'overlap', 'align', 'area'
        ]

        self.energy_combination_module: BaseEnergyCombinator = \
            getattr(energy_combinators, self.config['model']['energy_combinator'])(
                sub_energies_names=self._sub_energies, device=self.device,
                **self.config['model'].get('energy_combinator_params', {})
            ).to(self.device)

        self.eps = torch.tensor(1e-8)

        for attr in ['area_q_func', 'align_q_func', 'overlap_q_func']:
            fun = getattr(quality_functions, self.config['model'][attr]['method'])(
                **self.config['model'][attr].get('params', {})
            )
            setattr(self, attr, fun)

        self.rect_intersect = RectangleIntersect(
            approx=self.config['model'].get(
                'rect_intersect_method', 'ellipsis'),
            norm_p=10
        )

    @property
    def max_interaction_distance(self) -> float:
        return self.max_dist

    @property
    def sub_energies(self) -> List[str]:
        return self._sub_energies

    def train(self, mode: bool = True):
        super(EnergyFromMaps, self).train()
        self.maps_module.eval()

    def forward(self, context_cube: Tensor, context_cube_mask: Tensor,
                position_energy_map: Tensor, marks_energy_maps: List[Tensor],
                compute_context: bool) -> Dict[str, Tensor]:
        """
                :param marks_energy_maps:
                :param position_energy_map:
                :param context_cube: tensor of shape (B,3,3,N,D)
                :param context_cube_mask: tensor of shape (B,3,3,N)
                :param compute_context: if true then the energies are computed on the whole context cube,
                otherwise if is computed only for the center of the context cube
                :return: dict of energies
        """
        check_grad = False
        if self.check_gradients:
            torch.autograd.set_detect_anomaly(True)
            if context_cube.requires_grad:
                check_grad = True

        n_sets = context_cube.shape[0]
        n_points = context_cube.shape[3]
        n_dims = context_cube.shape[4]
        h, w = position_energy_map.shape[2:]

        bound_min = torch.tensor(
            [0, 0] + [m.v_min for m in self.mappings], device=context_cube.device)
        bound_max = torch.tensor(
            [h, w] + [m.v_max for m in self.mappings], device=context_cube.device)
        cyclic = torch.tensor(
            [False, False] + [m.is_cyclic for m in self.mappings], device=context_cube.device)

        context_cube = check_inbound_state(
            context_cube, bound_min, bound_max, cyclic, clip_if_oob=True, masked_state=context_cube[context_cube_mask])

        sub_energies_per_point = {}

        if n_points != 0:
            context_points = context_cube.view(
                (n_sets, -1, n_dims))  # points center + 8 neighbors
            context_points_mask = context_cube_mask.view((n_sets, -1))
            n_others = context_points.shape[1]
            if compute_context:
                points = context_points
                points_mask = context_points_mask
                n_points = n_others
            else:
                points = context_cube[:, 0, 0]  # points at center set
                points_mask = context_cube_mask[:, 0, 0]

            dist, marks_diff = compute_distances_in_sets(
                points=points,
                points_mask=points_mask,
                others=context_points,
                others_mask=context_points_mask,
                maximum_dist=self.max_dist,
                marks_diff=True
            )

            # overlap prior
            overlap = self.rect_intersect.forward(
                state=points,
                state_mask=points_mask,
                state_other=context_points,
                distance_matrix=dist)
            overlap = self.overlap_q_func(overlap)
            overlap_energy = (dist != 0.0) * overlap
            overlap_energy = overlap_energy * points_mask.view((n_sets, n_points, 1)) * context_points_mask.view(
                (n_sets, 1, n_others))
            overlap_energy = torch.max(overlap_energy, dim=2)[0]

            sub_energies_per_point['overlap'] = overlap_energy

            pi = torch.tensor(np.pi)
            angles_dist = torch.remainder(marks_diff[..., 2], pi)
            angles_dist = torch.minimum(angles_dist, pi - angles_dist)

            align = self.align_q_func(angles_dist)
            align = (dist != 0.0) * (dist < self.max_dist) * align
            align_energy = torch.min(align, dim=2)[0]

            sub_energies_per_point['align'] = align_energy

            # area prior
            areas = points[:, :, 2] * points[:, :, 3]
            areas_prior = self.area_q_func(areas)
            sub_energies_per_point['area'] = areas_prior * points_mask

            # position term
            positions = points[..., :2].view((1, n_sets, n_points, 2))
            position_energies = interpolate_position(
                positions=positions,
                image=position_energy_map
            ).view((n_sets, n_points))
            position_energies = position_energies * points_mask
            position_energies = position_energies.view((n_sets, n_points))

            sub_energies_per_point['position'] = position_energies

            # marks terms
            for i, (mark_energy_map, mapping) in enumerate(zip(marks_energy_maps, self.mappings)):
                mark_energies = interpolate_marks(
                    positions=positions.view((1, 1, n_sets, n_points, 2)),
                    marks=points[..., 2 + i].view((1, 1, n_sets, n_points)),
                    image=mark_energy_map.view(
                        (1, 1, mapping.n_classes, h, w)),
                    mapping=mapping
                ).view((n_sets, n_points))
                sub_energies_per_point[mapping.name] = mark_energies * points_mask

            energy_vector = torch.stack(
                [sub_energies_per_point[k] for k in self._sub_energies], dim=-1)
            energies = self.energy_combination_module.forward(
                energy_vector) * points_mask
            if check_grad:
                for k, v in sub_energies_per_point.items():
                    assert v.requires_grad
        else:
            points_mask = context_cube_mask[:, 0, 0]
            energies = torch.zeros(
                (n_sets, n_points), device=context_cube.device) * points_mask

        per_subset = torch.sum(energies, dim=-1)
        total = torch.sum(per_subset)

        return {
            'energy_per_point': energies,
            'energy_per_subset': per_subset,
            'total_energy': total,
            **sub_energies_per_point
        }

    @property
    def combination_module_weights(self) -> Dict[str, float]:
        return self.energy_combination_module.describe()

    def forward_state(self, state: Tensor,
                      position_energy_map: Tensor, marks_energy_maps: List[Tensor],
                      compute_context=False) -> Dict[str, Tensor]:
        context_cube, context_cube_mask = state_to_context_cube(state)
        return self.forward(
            context_cube=context_cube, context_cube_mask=context_cube_mask,
            position_energy_map=position_energy_map, marks_energy_maps=marks_energy_maps,
            compute_context=compute_context
        )

    @torch.no_grad()
    def energy_maps_from_image(self, image: Tensor, as_energies: bool, large_image: bool):
        self.maps_module.eval()
        if large_image:
            assert len(image.shape) == 3
            output = self.maps_module.infer_on_large_image(
                image.to(self.device), margin=68 // 2, patch_size=128)
        else:
            if len(image.shape) == 3:
                image = image.unsqueeze(dim=0)
            output = self.maps_module.forward(image)
        if self.energy_from_densities:
            pos_energy_map = -2 * torch.sigmoid(output['center_heatmap']) + 1
            marks_energy_maps = [-torch.softmax(
                output[f'mark_{m.name}'], dim=0) for m in self.mappings]
        else:
            pos_energy_map = -output['center_heatmap']
            marks_energy_maps = [-output[f'mark_{m.name}']
                                 for m in self.mappings]

        if as_energies:
            return pos_energy_map, marks_energy_maps
        else:
            return self.densities_from_energy_maps(pos_energy_map, marks_energy_maps)

    def densities_from_energy_maps(self, pos_energy_map, marks_energy_maps):
        if self.energy_from_densities:
            pos_density_map = - 0.5 * pos_energy_map + 0.5
            marks_density_maps = [-m for m,
                                  mm in zip(marks_energy_maps, self.mappings)]
        else:
            pos_density_map = torch.sigmoid(-pos_energy_map)
            marks_density_maps = [
                torch.softmax(-m, dim=0) for m in marks_energy_maps]
        return pos_density_map, marks_density_maps

    # def density_to_energy(self, pos_density_map, marks_density_maps):
    #     raise DeprecationWarning
    #     position_energy_map = - 2 * pos_density_map - 1
    #     marks_energy_maps = [-m + 0.5 for m in marks_density_maps]
    #     return position_energy_map, marks_energy_maps

    def energy_func_wrapper(self, image=None, position_energy_map: Tensor = None,
                            marks_energy_maps: List[Tensor] = None, compute_context: bool = False) -> EnergyFunc:
        if position_energy_map is None or marks_energy_maps is None:
            assert image is not None and self.maps_module is not None
            position_energy_map, marks_energy_maps = self.energy_maps_from_image(
                image, as_energies=True)

        if len(position_energy_map.shape) == 3:
            position_energy_map = position_energy_map.unsqueeze(dim=0)
        if len(marks_energy_maps[0].shape) == 3:
            marks_energy_maps = [m.unsqueeze(dim=0) for m in marks_energy_maps]

        position_energy_map = position_energy_map.to(self.device)
        marks_energy_maps = [m.to(self.device) for m in marks_energy_maps]

        def func(context_cube: Tensor, context_cube_mask: Tensor) -> Dict[str, Tensor]:
            res = self.forward(
                context_cube, context_cube_mask, position_energy_map, marks_energy_maps,
                compute_context=compute_context
            )
            return res

        return func
