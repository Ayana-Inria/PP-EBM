import logging
from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from base.mappings import ValueMapping
from base.misc import append_lists_in_dict
from base.state_ops import check_inbound_state, maps_local_max_state
from energies import energy_combinators
from energies.base import BaseEnergyModel
from energies.compute_distances import compute_distances_and_marks_diffs_in_sets
from energies.energy_combinators import BaseEnergyCombinator
from energies.sub_energies_modules import priors, likelihood
from energies.sub_energies_modules.base import PriorModule, LikelihoodModule
from energies.sub_energies_modules.likelihood import CNNEnergies
from models.mpp_data_net.model import MPPDataModel
from modules.ellipsis_overlap import dividing_axis_gap
from samplers.mic_set_utils import state_to_context_cube


class GenericEnergyModel(BaseEnergyModel, Module):

    def __init__(self, config, device=None):
        super(GenericEnergyModel, self).__init__()
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        logging.info(f"device is {self.device}")
        self.config = config
        self.resolution = self.config['model'].get('resolution', 1.0)
        self.shape = self.config['model'].get('shape', 'rectangle')
        self.check_gradients = self.config['model'].get(
            'check_gradients', False)
        if self.check_gradients:
            logging.warning(
                "check_gradients is True ! this may slow down execution")

        self.data_energies: LikelihoodModule = getattr(likelihood, self.config['model'].get('data_energy'))(
            **self.config['model'].get('data_energy_params', {}), config=self.config
        )

        prior_config = self.config['model']['priors']

        if type(prior_config) is dict:
            priors_list = []
            for k, v in prior_config.items():
                priors_list.append((k, v))
        elif type(prior_config) is list:
            if len(prior_config) == 0:
                priors_list = []
            elif type(prior_config[0]) is dict:
                priors_list = []
                for d in prior_config:
                    priors_list.append(list(d.items())[0])
            else:
                raise RuntimeError("expects priors to be a dict of {prior:argument_dict}, or a list of"
                                   "{prior:argument_dict}")
        else:
            raise RuntimeError("expects priors to be a dict of {prior:argument_dict}, or a list of"
                               "{prior:argument_dict}")

        self.prior_modules: ModuleList[PriorModule] = ModuleList([
            getattr(priors, k)(mappings=self.mappings,
                               device=self.device, resolution=self.resolution, **v)
            for k, v in priors_list
        ])

        self.max_dist = self.config['model'].get('maximum_distance', None)
        if self.max_dist is None:
            self.max_dist = max(
                [0] + [p.maximum_distance for p in self.prior_modules])
            logging.info(
                f"automatically found the maximum interaction distance to be {self.max_dist}")

        self._sub_energies = self.data_energies.sub_energies + \
            [p.name for p in self.prior_modules]

        self.energy_combination_module: BaseEnergyCombinator = \
            getattr(energy_combinators, self.config['model']['energy_combinator'])(
                sub_energies_names=self._sub_energies, device=self.device,
                **self.config['model'].get('energy_combinator_params', {})
            ).to(self.device)

        self.bound_min = torch.tensor(
            [0, 0] + [m.v_min for m in self.mappings], device=self.device)
        self.bound_max = torch.tensor(
            [1, 1] + [m.v_max for m in self.mappings], device=self.device)
        self.cyclic = torch.tensor(
            [False, False] + [m.is_cyclic for m in self.mappings], device=self.device)

        self.eps = torch.tensor(1e-8)

        self.debug_mode = self.config['model'].get('debug_mode', False)
        if self.debug_mode:
            logging.warning("activated debug mode, may slow things down !")

    def parameters(self, recurse: bool = True):
        if 'lr_per_group' in self.config['trainer']:
            lr_per_group = self.config['trainer']['lr_per_group']
            likelihood_params = self.data_energies.parameters()
            prior_params = self.prior_modules.parameters()
            weights_params = self.energy_combination_module.parameters()
            return [
                {'params': likelihood_params,
                    'lr': lr_per_group['likelihood']},
                {'params': prior_params, 'lr': lr_per_group['prior']},
                {'params': weights_params, 'lr': lr_per_group['weights']}
            ]
        else:
            return Module.parameters(self, recurse)

    @property
    def max_interaction_distance(self) -> float:
        return self.max_dist

    @property
    def sub_energies(self) -> List[str]:
        return self._sub_energies

    @property
    def mappings(self) -> List[ValueMapping]:
        return self.data_energies.mappings

    @property
    def trainable_maps(self) -> bool:
        return self.data_energies.is_trainable

    def forward(self, context_cube: Tensor, context_cube_mask: Tensor, position_energy_map: Tensor,
                marks_energy_maps: List[Tensor], compute_context: bool) -> Dict[str, Tensor]:
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

        if len(position_energy_map.shape) != 4:
            raise RuntimeError(
                f"expected position_energy_map of shape (1,1,H,W) got {position_energy_map.shape}")
        if any([len(m.shape) != 4 for m in marks_energy_maps]):
            raise RuntimeError(
                f"expected marks_energy_maps of shapes (1,1,H,W) got {[m.shape for m in marks_energy_maps]}")

        n_sets = context_cube.shape[0]
        n_points = context_cube.shape[3]
        n_dims = context_cube.shape[4]
        h, w = position_energy_map.shape[2:]
        self.bound_max[0] = h
        self.bound_max[1] = w

        context_cube = check_inbound_state(
            context_cube, self.bound_min, self.bound_max, self.cyclic, clip_if_oob=True,
            masked_state=context_cube[context_cube_mask])

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

            dist, diff_tensor = compute_distances_and_marks_diffs_in_sets(
                points=points,
                points_mask=points_mask,
                others=context_points,
                others_mask=context_points_mask,
                maximum_dist=self.max_dist
            )

            if self.shape == 'rectangle':
                gap_matrix = dividing_axis_gap(
                    state_1=points,
                    state_2=context_points, std_intersect=True)
            elif self.shape == 'circle':
                gap_matrix = None  # todo get proper values
            else:
                raise NotImplementedError(self.shape)

            interactions_mask = (dist != 0) & (dist < self.max_dist)

            # data energies
            data_e_dict = self.data_energies.forward(
                points=points,
                points_mask=points_mask,
                position_energy_map=position_energy_map,
                marks_energy_maps=marks_energy_maps,
            )
            sub_energies_per_point.update(data_e_dict)

            # prior energies
            for prior in self.prior_modules:
                sub_energies_per_point[prior.name] = prior.forward(
                    points=points, points_mask=points_mask,
                    context_points=context_points, context_points_mask=context_points_mask,
                    distance_matrix=dist,
                    diff_tensor=diff_tensor,
                    gap_matrix=gap_matrix,
                    interactions_mask=interactions_mask,
                    global_max_dist=self.max_dist
                )

            energy_vector = torch.stack(
                [sub_energies_per_point[k] for k in self._sub_energies], dim=-1)
            energy_per_point = self.energy_combination_module.forward(
                energy_vector.float()) * points_mask
            if check_grad:
                for k, v in sub_energies_per_point.items():
                    assert v.requires_grad
        else:
            points_mask = context_cube_mask[:, 0, 0]
            energy_per_point = torch.zeros(
                (n_sets, n_points), device=context_cube.device) * points_mask
            sub_energies_per_point = {k: torch.zeros((n_sets, n_points), device=context_cube.device)
                                      for k in self.sub_energies}

        energy_per_subset = torch.sum(energy_per_point, dim=-1)
        total_energy = torch.sum(energy_per_subset)
        if torch.isnan(total_energy):
            err_msg = f"resulting energy is NaN: {total_energy} !\n" \
                      f"with model weights: {self.energy_combination_module.describe()}"
            if self.debug_mode:
                raise ValueError(err_msg)
            logging.error(err_msg)

        return_dict = {
            'energy_per_point': energy_per_point,
            'energy_per_subset': energy_per_subset,
            'total_energy': total_energy,
            **sub_energies_per_point
        }
        if compute_context:
            # energies tensor is of shape (n_sets, 9 * n_points),
            # energies[:,:n_points] should correspond to the points in context_cube[:, 0, 0]
            per_point_inner = energy_per_point[:, :n_points]
            return_dict['energy_per_point_inner'] = per_point_inner
            per_subset_inner = per_point_inner.sum(dim=-1)
            return_dict['energy_per_subset_inner'] = per_subset_inner

        return return_dict

    def energy_maps_from_image(self, image: Tensor, **kwargs):
        weights = self.energy_combination_module.describe()
        return self.data_energies.energy_maps_from_image(image, energy_weights=weights, **kwargs)

    def densities_from_energy_maps(self, pos_energy_map, marks_energy_maps):
        weights = self.energy_combination_module.describe()
        return self.data_energies.densities_from_energy_maps(pos_energy_map, marks_energy_maps,
                                                             energy_weights=weights)

    @property
    def combination_module_weights(self) -> Dict[str, float]:
        return self.energy_combination_module.describe()

    def forward_state(self, state: Tensor, position_energy_map: Tensor, marks_energy_maps: List[Tensor],
                      compute_context=False) -> Dict[str, Tensor]:
        context_cube, context_cube_mask = state_to_context_cube(state)
        return self.forward(
            context_cube=context_cube, context_cube_mask=context_cube_mask,
            position_energy_map=position_energy_map, marks_energy_maps=marks_energy_maps,
            compute_context=compute_context
        )

    def infer_from_cnn(self, images: Tensor, pad_to_size: bool = False, large_image: bool = False,
                       cut_and_stitch: bool = False, lm_distance=5, lm_thresh=0.5, threshold: float = 0.0):
        if len(images.shape) != 4:
            raise RuntimeError(
                f"Expected image(s) for shape (B,3,H,W) and got {tuple(images.shape)}")
        split_info = {}

        assert type(self.data_energies) is CNNEnergies
        maps_module: MPPDataModel = self.data_energies.maps_module

        with torch.no_grad():
            if large_image:
                raise DeprecationWarning(
                    "large_image is no longer supported, use cut_and_stitch instead")
            elif cut_and_stitch:
                if len(images.shape) != 4 or images.shape[0] != 1:
                    raise RuntimeError(f"energy_maps_from_image with large_image only works on one image, "
                                       f"image shape is {images.shape}")

                output = maps_module.infer_on_large_image(images[0].to(self.device),
                                                          restore_batch_dim=True)
            elif pad_to_size:
                from base.images import pad_to_valid_size, unpad_to_size
                output = maps_module.forward(
                    pad_to_valid_size(images.to(self.device), 64))
                output = unpad_to_size(output, images.shape[2:])
            else:
                output = maps_module.forward(images.to(self.device))

        center_heatmap = torch.sigmoid(output['center_heatmap'])
        marks_maps = [output[f'mark_{m.name}'].detach(
        ).cpu().numpy() for m in self.mappings]

        results = {}

        for i in range(len(images)):
            # local maxima
            heatmap = center_heatmap[i].detach().cpu().numpy()
            proposed_state = maps_local_max_state(
                position_map=heatmap[0],
                mark_maps=[m[i] for m in marks_maps],
                mappings=self.mappings,
                local_max_thresh=lm_thresh, local_max_distance=lm_distance
            )
            proposal_scores = heatmap[0, proposed_state[:, 0].astype(
                int), proposed_state[:, 1].astype(int)]

            append_lists_in_dict(results, {
                'heatmap': heatmap,
                'proposals': proposed_state,
                'scores': proposal_scores,
                'state': proposed_state[proposal_scores > threshold],
                'state_score': proposal_scores[proposal_scores > threshold]
            })

        return results
