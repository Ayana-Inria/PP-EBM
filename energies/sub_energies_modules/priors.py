import logging
import warnings
from typing import Dict, List, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Module, functional

from base.images import map_range
from base.mappings import ValueMapping
from energies import quality_functions, quality_function_modules
from energies.intersect_modules import RectangleIntersect, CircleIntersect
from energies.quality_function_modules import Sigmoid, Cosine
from energies.sub_energies_modules.base import PriorModule
from modules.custom import MaskedAttention, NegPosSigmoid, Scale, TrueAttention, masked_softmax
from modules.mlp import MLP
from modules.position_embeddings import PositionEmbedding

ANGLE_MARK_INDEX = 4
EPSILON = 1e-4


class Overlap(PriorModule, Module):
    def __init__(self, quality_function: str,
                 quality_function_args: Dict, mappings: List[ValueMapping], prior_name='overlap',
                 shape: str = 'rectangle',
                 rect_intersect_method: str = None, norm_p: float = 10, **kwargs):
        super(Overlap, self).__init__()

        raise DeprecationWarning("this method is deprecated")

        self._name = prior_name

        if shape == 'rectangle':
            self.intersect_module = RectangleIntersect(
                approx=rect_intersect_method,
                norm_p=norm_p
            )
            r = 0
            for m in mappings:
                if m.name in ['height', 'width']:
                    if m.v_max > r:
                        r = m.v_max
            self.max_dist = r
        elif shape == 'circle':
            assert mappings[0].name == 'radius'
            self.max_dist = mappings[0].v_max * 2
            self.intersect_module = CircleIntersect()
        else:
            raise ValueError

        self.q_func = getattr(quality_functions, quality_function)(
            **quality_function_args)

    @property
    def maximum_distance(self) -> float:
        return self.max_dist

    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor, distance_matrix: Tensor, **kwargs) -> Tensor:
        overlap_energy, argmax = self.forward(points, points_mask, context_points, context_points_mask, distance_matrix,
                                              return_argmax=True, **kwargs)
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]
        weights = torch.zeros((n_sets, n_points, n_others),
                              device=overlap_energy.device)
        weights[:, argmax] = 1.0
        return weights

    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, return_argmax: bool = False, **kwargs) -> Tensor:
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]

        overlap = self.intersect_module.forward(
            state=points,
            state_mask=points_mask,
            state_other=context_points,
            distance_matrix=distance_matrix)
        overlap = self.q_func(overlap)
        overlap_energy = (distance_matrix != 0.0) * overlap
        overlap_energy = overlap_energy * points_mask.view((n_sets, n_points, 1)) * context_points_mask.view(
            (n_sets, 1, n_others))
        overlap_energy, argmax = torch.max(overlap_energy, dim=2)
        if not return_argmax:
            return overlap_energy
        else:
            return overlap_energy, argmax

    @property
    def name(self) -> str:
        return self._name


class Align(PriorModule, Module):
    def __init__(self, max_dist: float, reduce: str, no_neighbors_energy: float, quality_function: str,
                 quality_function_args: Dict,
                 prior_name='align', **kwargs):
        super(Align, self).__init__()
        self.max_dist = max_dist
        self._name = prior_name
        self.pi = torch.tensor(np.pi)
        self.q_func = getattr(quality_functions, quality_function)(
            **quality_function_args)
        self.reduce = reduce
        self._angle_id = ANGLE_MARK_INDEX
        self.no_neighbors_value = no_neighbors_energy

    @property
    def maximum_distance(self) -> float:
        return self.max_dist

    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor, distance_matrix: Tensor, **kwargs) -> Tensor:
        align_energy, arg_reduce = self.forward(points, points_mask, context_points, context_points_mask,
                                                distance_matrix,
                                                return_arg_reduce=True, **kwargs)

        if self.reduce == 'mean':
            arg_reduce = arg_reduce / \
                torch.sum(arg_reduce, dim=2, keepdim=True)
            arg_reduce = torch.nan_to_num(arg_reduce, 0)
            return arg_reduce
        elif self.reduce in ['min', 'max']:
            n_sets = points.shape[0]
            n_points = points.shape[1]
            n_others = context_points.shape[1]
            weights = torch.zeros(
                (n_sets, n_points, n_others), device=align_energy.device)
            weights[:, arg_reduce] = 1.0
            return weights
        else:
            raise NotImplementedError

    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, interactions_mask: Tensor, global_max_dist: float,
                return_arg_reduce: bool = False,
                **kwargs) -> Tensor:

        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]
        if 'diff_tensor' in kwargs:
            angles_diff = kwargs['diff_tensor'][..., self._angle_id]
        else:
            points_angle = points[..., self._angle_id].reshape(
                (n_sets, n_points, 1))
            others_angle = context_points[..., self._angle_id].reshape(
                (n_sets, 1, n_others))
            angles_diff = points_angle - others_angle

        angles_dist = torch.remainder(angles_diff, self.pi)
        angles_dist = torch.minimum(angles_dist, self.pi - angles_dist)

        align = self.q_func(angles_dist)
        ib_points = interactions_mask
        if global_max_dist > self.max_dist:
            ib_points = ib_points & (distance_matrix <= self.max_dist)

        if self.reduce == 'min':
            w = torch.where(ib_points, align, torch.tensor(
                [torch.inf], device=align.device, dtype=align.dtype))
            align_energy, arg_reduce = torch.min(w, dim=2)
        elif self.reduce == 'max':
            w = torch.where(ib_points, align, torch.tensor(
                [-torch.inf], device=align.device, dtype=align.dtype))
            align_energy, arg_reduce = torch.max(w, dim=2)
        elif self.reduce == 'mean':
            w = torch.where(ib_points, align, torch.tensor(
                [torch.nan], device=align.device, dtype=align.dtype))
            align_energy = torch.nanmean(w, dim=2)
            arg_reduce = ib_points
        else:
            raise ValueError
        align_energy = torch.nan_to_num(
            align_energy, self.no_neighbors_value, self.no_neighbors_value, self.no_neighbors_value
        )
        if not return_arg_reduce:
            return align_energy
        else:
            return align_energy, arg_reduce

    @property
    def name(self) -> str:
        return self._name


class Area(PriorModule, Module):
    def __init__(self, quality_function: str, quality_function_args: Dict, prior_name='area', **kwargs):
        super(Area, self).__init__()
        self._name = prior_name
        self.q_func = getattr(quality_functions, quality_function)(
            **quality_function_args)

    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor, distance_matrix: Tensor, **kwargs) -> Tensor:
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]
        return torch.zeros((n_sets, n_points, n_others), device=torch.device)

    @property
    def maximum_distance(self) -> float:
        return 0

    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, **kwargs) -> Tensor:
        areas = points[:, :, 2] * points[:, :, 3]
        areas_prior = self.q_func(areas)
        return areas_prior * points_mask

    @property
    def name(self) -> str:
        return self._name


class Repulsive(PriorModule, Module):
    def __init__(self, max_dist: float, quality_function: str, device: torch.device, prior_name='repulsive',
                 falloff=2.0,
                 use_gap: bool = False, threshold: float = 0.0, learn_threshold: bool = False, attractive: bool = False,
                 slope: float = 1.0, learn_slope: bool = False, bias: float = 0.0, quality_function_args: Dict = None,
                 **kwargs):
        super(Repulsive, self).__init__()
        self.max_dist = max_dist
        self._name = prior_name
        self.use_gap = use_gap
        self.threshold = None
        self.attractive = attractive
        if self.attractive and quality_function not in ['relu', 'sigmoid', 'Relu', 'Sigmoid']:
            logging.warning(
                f"attractive not implemented for quality_function={quality_function}")
        # self.q_func = getattr(quality_functions, quality_function)(**quality_function_args)
        if quality_function == 'linear':
            self.q_func = lambda x: map_range(
                x, 0, max_dist, 1, 0, clip=True) + bias
        elif quality_function == 'falloff':
            b = 1 / np.power(1 + max_dist, falloff)
            self.q_func = lambda x: torch.clip(
                map_range(1 / torch.pow(1 + x, falloff), b, 1, 0, 1), 0, 1.0) + bias
        elif quality_function in ['Relu', 'Sigmoid']:
            ops = [
                Scale(a=1 / max_dist),
                getattr(quality_function_modules, quality_function)(
                    device=device, **quality_function_args)
            ]
            if self.attractive:
                ops.append(Scale(a=1.0, b=-1.0))
            else:
                ops.append(Scale(a=-1.0, b=1.0))

            self.q_func = nn.Sequential(*ops)
        elif quality_function in ['relu', 'softplus']:
            warnings.warn(f"please use Relu (with soft for softplus) instead")
            if learn_threshold:
                self.threshold = nn.Parameter(torch.tensor(threshold))
            else:
                self.threshold = threshold
            if quality_function == 'relu':
                q = torch.relu
            elif quality_function == 'softplus':
                def q(x): return functional.softplus(x, beta=10)
            if attractive:
                self.q_func = lambda x: q(
                    (x - self.threshold) / max_dist) - 1.0 + bias
            else:
                self.q_func = lambda x: q(
                    1.0 - ((x - self.threshold) / max_dist)) + bias
        elif quality_function == 'sigmoid':
            warnings.warn(f"please use Relu (with soft for softplus) instead")
            if learn_threshold:
                self.threshold = nn.Parameter(torch.tensor(threshold))
            else:
                self.threshold = threshold
            if learn_slope:
                self.slope = nn.Parameter(torch.tensor(slope))
            else:
                self.slope = slope

            if attractive:
                self.q_func = lambda x: torch.sigmoid(
                    self.slope * (x - self.threshold)) + bias
            else:
                self.q_func = lambda x: 1 - \
                    torch.sigmoid(self.slope * (x - self.threshold)) + bias
        else:
            raise ValueError

        self.no_neighbors_value = 0.0

    @property
    def maximum_distance(self) -> float:
        return self.max_dist

    @property
    def name(self) -> str:
        return self._name

    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, interactions_mask: Tensor,
                global_max_dist: float, return_argmax: bool = False, gap_matrix=None, **kwargs) -> Tensor:
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]
        if self.use_gap:
            repulsive_prior = self.q_func(torch.relu(gap_matrix))
        else:
            repulsive_prior = self.q_func(distance_matrix)
        # repulsive_energy = interactions_mask * repulsive_prior
        # repulsive_energy = repulsive_energy * points_mask.view((n_sets, n_points, 1)) * context_points_mask.view(
        #     (n_sets, 1, n_others))
        # if self.attractive:
        #     repulsive_energy, argmax = torch.min(repulsive_energy, dim=2)
        # else:
        #     repulsive_energy, argmax = torch.max(repulsive_energy, dim=2)

        ib_points = interactions_mask
        if global_max_dist > self.max_dist:
            ib_points = ib_points & (distance_matrix <= self.max_dist)

        if self.attractive:  # min
            w = torch.where(ib_points, repulsive_prior,
                            torch.tensor([torch.inf], device=repulsive_prior.device, dtype=repulsive_prior.dtype))
            repulsive_energy, arg_reduce = torch.min(w, dim=2)
        else:  # max
            w = torch.where(ib_points, repulsive_prior,
                            torch.tensor([-torch.inf], device=repulsive_prior.device, dtype=repulsive_prior.dtype))
            repulsive_energy, arg_reduce = torch.max(w, dim=2)

        repulsive_energy = torch.nan_to_num(
            repulsive_energy, self.no_neighbors_value, self.no_neighbors_value, self.no_neighbors_value
        )
        if not return_argmax:
            return repulsive_energy
        return repulsive_energy, arg_reduce

    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor, distance_matrix: Tensor, **kwargs) -> Tensor:
        energy, argmax = self.forward(points, points_mask, context_points, context_points_mask, distance_matrix,
                                      return_argmax=True, **kwargs)
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]
        weights = torch.zeros(
            (n_sets, n_points, n_others), device=energy.device)
        weights[:, argmax] = 1.0
        return weights


class GaussianMixturePrior(PriorModule, Module):

    def __init__(self, target_value: Union[str, int], mu: Union[float, List[float]], sigma: Union[float, List[float]],
                 weight: Union[float, List[float]] = 1.0, v_min=0, v_max=1, prior_name: str = 'gaussianmixtureprior',
                 device: torch.device = 'cpu', learn_mu=False, learn_sigma=False, learn_weight=False,
                 true_gaussian: bool = False, **kwargs):
        super(GaussianMixturePrior, self).__init__()
        if type(mu) is float:
            mu = [mu]
        self.n_modes = len(mu)
        if type(sigma) is float:
            sigma = [sigma] * self.n_modes
        if type(weight) is float:
            weight = [weight] * self.n_modes
        self.true_gaussian = true_gaussian
        self.mu_t = torch.tensor(mu, device=device)
        self.sigma_t = torch.tensor(sigma, device=device)
        self.weight_t = torch.tensor(weight, device=device)

        if learn_mu:
            self.mu_t = nn.Parameter(self.mu_t)
        if learn_sigma:
            self.sigma_t = nn.Parameter(self.sigma_t)
        if learn_weight:
            self.weight_t = nn.Parameter(self.weight_t)

        self.v_min = v_min
        self.v_max = v_max

        if type(target_value) is int:
            self._extract_fn = lambda x: x[..., [2 + target_value]]
        elif target_value == 'ratio':
            self._extract_fn = lambda x: torch.nan_to_num(
                x[..., [2]] / x[..., [3]], nan=0.0)
        elif target_value == 'area':
            self._extract_fn = lambda x: x[..., [2]] * x[..., [3]]
        elif target_value == 'ratio-area':
            self._extract_fn = lambda x: torch.stack(
                (torch.nan_to_num(x[..., 2] / x[..., 3], nan=0.0), x[..., 2] * x[..., 3]), dim=-1)
        self.eps = torch.tensor(1e-8)

        if 'name' in kwargs:
            prior_name = kwargs['name']
        self._name = prior_name

    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor, distance_matrix: Tensor, **kwargs) -> Tensor:
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]
        return torch.zeros((n_sets, n_points, n_others), device=torch.device)

    def _gaussian_mixture(self, x: Tensor):

        g = self.weight_t * \
            torch.exp(-0.5 * torch.square((x - self.mu_t) / self.sigma_t))

        if not self.true_gaussian:
            return torch.sum(g, dim=-1)
        else:
            return -torch.log(torch.sum(g, dim=-1) + self.eps)

    def _remap(self, x: Tensor):
        return map_range(x, 0, 1, self.v_max, self.v_min)

    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, **kwargs) -> Tensor:
        values = self._extract_fn(points)
        values_prior = self._remap(self._gaussian_mixture(values))
        return values_prior * points_mask

    @property
    def name(self) -> str:
        return self._name

    @property
    def maximum_distance(self) -> float:
        return 0


class Noise(PriorModule, Module):

    def __init__(self, name='noise', **kwargs):
        super(Noise, self).__init__()
        self._name = name

    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor, distance_matrix: Tensor, **kwargs) -> Tensor:
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]
        return torch.zeros((n_sets, n_points, n_others), device=torch.device)

    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, **kwargs) -> Tensor:
        noise = torch.randn(size=points_mask.shape, device=points.device)
        return noise * points_mask

    @property
    def name(self) -> str:
        return self._name

    @property
    def maximum_distance(self) -> float:
        return 0.0


class Similarity(PriorModule, Module):

    def __init__(self, max_dist: float, reduce: str, mark_index: Union[int, List[int]],
                 mappings: List[ValueMapping], trainable: bool, bounded: bool, device: torch.device,
                 prior_name='similarity', no_neighbors_energy: float = 0, allow_neg_output: bool = False,
                 q_fun: str = 'linear', attention_weights: str = 'linear',
                 distance_in_attention: bool = False,
                 use_position_delta: bool = False, use_self_input: bool = False, use_distance: bool = False,
                 q_fun_bias=None, q_fun_slope=None,
                 mlp_hidden_dims: int = 4, mlp_layers=2, atn_mlp_hidden_dims: int = 4, atn_mlp_layers=2,
                 **kwargs):
        super(Similarity, self).__init__()
        self.use_position_delta = use_position_delta
        self.use_self_input = use_self_input
        self.use_distance = use_distance
        assert not (use_distance and distance_in_attention)
        self.max_dist = max_dist
        self.reduce = reduce
        self.no_neighbors_value = no_neighbors_energy
        if type(mark_index) is not list:
            mark_index = [mark_index]
        for m in mark_index:
            if m < 2:
                raise RuntimeError(
                    f"mark index {m} corresponds to a coordinate and not a mark (marks have index >=2)")
        self.mappings = [mappings[m - 2] for m in mark_index]
        if prior_name != 'similarity':
            self._name = prior_name + '_' + \
                '_'.join([f'{m.name}' for m in self.mappings])
        else:
            self._name = prior_name

        self._mark_ids = mark_index
        self._feature_ids = mark_index
        if self.use_position_delta:
            self._feature_ids = [0, 1] + self._feature_ids

        self.n_features = len(self._feature_ids)
        if self.use_self_input:
            self.n_features = self.n_features + len(self.mappings)
        if self.use_distance:
            self.n_features = self.n_features + 1

        self.features_range = [m.range for m in self.mappings]
        self.features_cyclic = [m.is_cyclic for m in self.mappings]
        if self.use_position_delta:
            self.features_range = [self.max_dist,
                                   self.max_dist] + self.features_range
            self.features_cyclic = [False, False] + self.features_cyclic
        if self.use_self_input:
            self.features_range = [
                m.range for m in self.mappings] + self.features_range
            self.features_cyclic = [
                m.is_cyclic for m in self.mappings] + self.features_cyclic
        if self.use_distance:
            self.features_range = [self.max_dist] + self.features_range
            self.features_cyclic = [False] + self.features_cyclic
        self.features_range = torch.tensor(self.features_range, device=device)
        self.features_cyclic = torch.tensor(
            self.features_cyclic, device=device)

        self.device = device

        ops = []
        if trainable:
            if q_fun == 'linear':
                ops.append(nn.Linear(self.n_features, 1, device=self.device))
            elif q_fun == 'MLP':
                mini_mlp = []
                for i in range(mlp_layers - 1):
                    input_dim = self.n_features if i == 0 else mlp_hidden_dims
                    mini_mlp = mini_mlp + [
                        nn.Linear(input_dim, mlp_hidden_dims,
                                  device=self.device),
                        nn.ReLU()
                    ]
                mini_mlp = mini_mlp + [
                    nn.Linear(mlp_hidden_dims, 1, device=self.device)
                ]
                ops = ops + mini_mlp
            else:
                raise ValueError
            if bounded:
                if not allow_neg_output:
                    ops.append(nn.Sigmoid())
                else:
                    ops.append(NegPosSigmoid())
        else:
            if q_fun == 'linear' and bounded and not allow_neg_output:
                assert q_fun_bias is not None and q_fun_slope is not None
                ops.append(Sigmoid(
                    trainable=False, device=self.device,
                    bias_value=q_fun_bias, slope_value=q_fun_slope
                ))
            elif q_fun == 'linear' and bounded and allow_neg_output:
                assert q_fun_bias is not None and q_fun_slope is not None
                ops.append(Scale(a=q_fun_slope, b=q_fun_bias))
                ops.append(NegPosSigmoid())

            elif q_fun == 'cosine':
                assert len(self.mappings) == 1
                ops.append(
                    Cosine(mapping=self.mappings[0])
                )
            else:
                raise NotImplementedError

        self.q_fun = nn.Sequential(*ops)

        self.dist_in_att = distance_in_attention
        if self.reduce == 'attention':
            self.attention = MaskedAttention(
                in_features=(
                    1 + self.n_features) if self.dist_in_att else self.n_features,
                weight_model=attention_weights,
                device=self.device,
                mlp_layers=atn_mlp_layers, mlp_hidden_dims=atn_mlp_hidden_dims
            )
        else:
            self.attention = None

    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, interactions_mask: Tensor, global_max_dist: float,
                return_weights: bool = False, **kwargs) -> Tensor:
        """

        :param return_weights:
        :type return_weights:
        :param points:
        :type points:
        :param points_mask:
        :type points_mask:
        :param context_points:
        :type context_points:
        :param context_points_mask:
        :type context_points_mask:
        :param distance_matrix: distance matrix, must be such that distance_matrix[points_mask] > max_dist and
        distance_matrix[:,context_points_mask] > max_dist
        :type distance_matrix:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]
        # if self.use_position default to computing the diff
        if 'diff_tensor' in kwargs and not self.use_position_delta:
            features_diff = kwargs['diff_tensor'][...,
                                                  [m for m in self._feature_ids]]
        else:
            points_features = points[..., self._feature_ids].reshape(
                (n_sets, n_points, 1, len(self._feature_ids)))
            others_features = context_points[..., self._feature_ids].reshape(
                (n_sets, 1, n_others, len(self._feature_ids)))
            features_diff = points_features - others_features
        features_diff = torch.abs(features_diff)
        features = features_diff
        if self.use_self_input:
            features = torch.concat(
                [points[..., self._mark_ids].reshape((n_sets, n_points, 1, len(self._mark_ids))).expand(
                    (n_sets, n_points, n_others, len(self._mark_ids))), features], dim=-1)
        if self.use_distance:
            features = torch.concat([distance_matrix.reshape(
                (n_sets, n_points, n_others, 1)), features], dim=-1)

        features_norm = features / \
            self.features_range.view((1, 1, 1, self.n_features))
        # if self.mapping.is_cyclic:
        #     features_norm = torch.remainder(features_norm, 1.0)
        #     features_norm = torch.minimum(features_norm, 1.0 - features_norm)
        features_norm_cyclic = torch.remainder(features_norm, 1.0)
        features_norm_cyclic = torch.minimum(
            features_norm_cyclic, 1.0 - features_norm_cyclic)
        features_norm = torch.where(
            self.features_cyclic, features_norm_cyclic, features_norm)

        if features_norm.shape[-1] != self.n_features:
            raise RuntimeError(f"features shape {features_norm.shape} does not match the expected number of dimentions "
                               f"on last axis {self.n_features}")

        similarity_energy = self.q_fun(features_norm).squeeze(dim=-1)
        assert len(similarity_energy.shape) == 3
        ib_points = interactions_mask
        if self.max_dist < global_max_dist:
            ib_points = ib_points & (distance_matrix <= self.max_dist)
        device = features_norm.device
        if self.reduce == 'min':
            w = torch.where(ib_points, similarity_energy,
                            torch.tensor([torch.inf], device=device))
            similarity_energy_reduced, weights = torch.min(w, dim=2)
        elif self.reduce == 'max':
            w = torch.where(ib_points, similarity_energy,
                            torch.tensor([-torch.inf], device=device))
            similarity_energy_reduced, weights = torch.max(w, dim=2)
        elif self.reduce == 'mean':
            w = torch.where(ib_points, similarity_energy,
                            torch.tensor([torch.nan], device=device))
            similarity_energy_reduced = torch.nanmean(w, dim=2)
            weights = ib_points
        elif self.reduce == 'attention':
            if self.dist_in_att:
                d = torch.nan_to_num(
                    distance_matrix, self.max_dist + 1, self.max_dist + 1, self.max_dist + 1)
                d = torch.clip(d, 0, self.max_dist + 1) / (self.max_dist + 1)
                atn_features = torch.concat(
                    [features_norm, d.unsqueeze(dim=-1)], dim=-1)
            else:
                atn_features = features_norm
            similarity_energy_reduced, weights = self.attention(
                values=similarity_energy, features=atn_features,
                mask=ib_points, reduce_dim=2, return_weights=True
            )
        else:
            raise ValueError(
                f"reduce={self.reduce} is not a valid option, choose either min, max, mean or attention")

        similarity_energy_reduced = torch.nan_to_num(
            similarity_energy_reduced, self.no_neighbors_value, self.no_neighbors_value, self.no_neighbors_value
        )
        if not return_weights:
            return similarity_energy_reduced
        return similarity_energy_reduced, weights

    def get_attention(self, points: Tensor, context_points: Tensor, distance_matrix: Tensor):
        if self.attention is None:
            print('this module has no attention')
            return None
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]
        points_features = points[..., self._feature_ids].reshape(
            (n_sets, n_points, 1, self.n_features))
        others_features = context_points[..., self._feature_ids].reshape(
            (n_sets, 1, n_others, self.n_features))
        features_diff = points_features - others_features
        features_diff = torch.abs(features_diff)
        features_diff_norm = features_diff / \
            self.features_range.view((1, 1, 1, self.n_features))
        features_diff_norm_cyclic = torch.remainder(features_diff_norm, 1.0)
        features_diff_norm_cyclic = torch.minimum(
            features_diff_norm_cyclic, 1.0 - features_diff_norm_cyclic)
        features_diff_norm = torch.where(
            self.features_cyclic, features_diff_norm_cyclic, features_diff_norm)

        similarity_energy = self.q_fun(features_diff_norm).squeeze(dim=-1)
        oob_points = (distance_matrix == 0) | (distance_matrix > self.max_dist)
        if self.dist_in_att:
            d = torch.nan_to_num(
                distance_matrix, self.max_dist + 1, self.max_dist + 1, self.max_dist + 1)
            d = torch.clip(d, 0, self.max_dist + 1) / (self.max_dist + 1)
            atn_features = torch.concat(
                [features_diff_norm, d.unsqueeze(dim=-1)], dim=-1)
        else:
            atn_features = features_diff_norm

        _, weights = self.attention.forward(values=similarity_energy, features=atn_features,
                                            mask=~oob_points, reduce_dim=2, return_weights=True)

        return weights

    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor, distance_matrix: Tensor, **kwargs) -> Tensor:
        energy, weights = self.forward(points, points_mask, context_points, context_points_mask, distance_matrix,
                                       return_weights=True, **kwargs)
        if self.reduce == 'mean':
            weights = weights / torch.sum(weights, dim=2, keepdim=True)
            weights = torch.nan_to_num(weights, 0)
            return weights
        elif self.reduce in ['min', 'max']:
            n_sets = points.shape[0]
            n_points = points.shape[1]
            n_others = context_points.shape[1]
            w = torch.zeros((n_sets, n_points, n_others), device=energy.device)
            for k in range(n_sets):
                for i in range(n_points):
                    w[k, i, weights[k, i]] = 1.0
            return w
        elif self.reduce == 'attention':
            return weights
        else:
            raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @property
    def maximum_distance(self) -> float:
        return self.max_dist


class AttentionSimilarity(PriorModule, Module):

    def __init__(self, prior_name: str, max_dist: float, device: torch.device, mappings: List[ValueMapping],
                 emb_dims: int, key_dim: int, bounded: bool, allow_neg_output: bool,
                 mlp_emb_hidden_dims: int, mlp_emb_layers: int, input_marks: bool, input_pos: bool,
                 normalize_features: bool,
                 position_embeddings: bool, position_embeddings_periods: list = None,
                 randomized_embeddings: bool = None,
                 **kwargs):
        super(AttentionSimilarity, self).__init__()
        self._name = prior_name
        self.max_dist = max_dist
        self.device = device
        self.mappings = mappings
        self.in_features = 0
        self.input_marks = input_marks
        self.input_pos = input_pos
        self.normalize_features = normalize_features
        self.position_embeddings = position_embeddings
        self.position_embeddings_periods = position_embeddings_periods
        self.randomized_embeddings = randomized_embeddings
        self._feature_ids = []
        if self.input_pos:
            raise RuntimeError("Use position embeddings instead !")
            # self.in_features += 2
            # self._feature_ids = self._feature_ids + [0, 1]
        if self.input_marks:
            self.in_features += len(self.mappings)
            self._feature_ids = self._feature_ids + \
                [2 + i for i in range(len(self.mappings))]
        if self.position_embeddings:
            assert self.position_embeddings_periods is not None
            assert self.randomized_embeddings is not None
            self.position_embedding_module = PositionEmbedding(
                periods=self.position_embeddings_periods,
                scale_factor=1, device=self.device
            )
            self.in_features += len(2 * self.position_embeddings_periods)
        else:
            self.position_embedding_module = None

        if normalize_features:
            # be carefull with the position embeddings
            raise NotImplementedError()  # todo

        self.attention = TrueAttention(
            in_features=self.in_features,
            out_features=1,
            device=self.device,
            emb_dims=emb_dims,
            key_dim=key_dim,
            mlp_emb_hidden_dims=mlp_emb_hidden_dims,
            mlp_emb_layers=mlp_emb_layers,
            bounded=bounded,
            allow_neg_output=allow_neg_output
        )

        self.features_range = [m.range for m in self.mappings]
        self.features_cyclic = [m.is_cyclic for m in self.mappings]

    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, interactions_mask: Tensor, global_max_dist: float,
                return_weights: bool = False, **kwargs) -> Tensor:
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]

        points_features = points[..., self._feature_ids].reshape(
            (n_sets, n_points, len(self._feature_ids)))
        if context_points is points:
            others_features = points_features
        else:
            others_features = context_points[..., self._feature_ids].reshape(
                (n_sets, n_others, len(self._feature_ids)))

        if self.position_embedding_module is not None:
            if self.randomized_embeddings:
                trl, rot = self.position_embedding_module.sample_trl_and_rot()
            else:
                trl, rot = None, None
            points_pos_emb = self.position_embedding_module.forward(
                points[..., :2], translation=trl, rotation=rot)
            points_features = torch.concat(
                (points_features, points_pos_emb), dim=-1)
            if context_points is points:
                others_features = points_features
            else:
                others_pos_emb = self.position_embedding_module.forward(context_points[..., :2], translation=trl,
                                                                        rotation=rot)
                others_features = torch.stack(
                    (others_features, others_pos_emb), dim=-1)

        # todo normalize ?

        similarity_energy_reduced, weights = self.attention(
            y=points_features,
            x=others_features,
            mask=interactions_mask,
            return_weights=True
        )  # should be of shape (n_sets,n_points,1)
        similarity_energy_reduced = torch.squeeze(
            similarity_energy_reduced, dim=-1)
        similarity_energy_reduced.masked_fill_(
            torch.isnan(similarity_energy_reduced), 0.0)
        similarity_energy_reduced = similarity_energy_reduced * points_mask

        if not return_weights:
            return similarity_energy_reduced
        return similarity_energy_reduced, weights

    @property
    def name(self) -> str:
        return self._name

    @property
    def maximum_distance(self) -> float:
        return self.max_dist

    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor, distance_matrix: Tensor, **kwargs) -> Tensor:
        _, weights = self.forward(points, points_mask, context_points, context_points_mask, distance_matrix,
                                  return_weights=True, **kwargs)
        return weights


class DiffAttentionSimilarity(PriorModule, Module):

    def __init__(self, prior_name: str, max_dist: float, device: torch.device,
                 resolution: float, mappings: List[ValueMapping],
                 emb_dim: int, bounded: bool, allow_neg_output: bool,
                 mlp_emb_hidden_dims: int, mlp_emb_layers: int,
                 local_polar_coordinates: bool,
                 **kwargs):
        super(DiffAttentionSimilarity, self).__init__()
        self._name = prior_name
        self.max_dist = max_dist
        self.device = device
        self.local_polar_coordinates = local_polar_coordinates
        self.resolution = resolution
        self.emb_dim = emb_dim
        self.device = device
        self.mappings = mappings

        self.n_features = 5
        self.scaling_tensor = torch.tensor([(resolution if i != ANGLE_MARK_INDEX else 1.0) for i in range(5)],
                                           device=self.device)  # do not scale angle

        mini_mlp = []
        input_dim = self.n_features
        for i in range(mlp_emb_layers - 1):
            mini_mlp = mini_mlp + [
                nn.Linear(input_dim, mlp_emb_hidden_dims, device=self.device),
                nn.ReLU()
            ]
            input_dim = mlp_emb_hidden_dims
        mini_mlp = mini_mlp + [
            nn.Linear(input_dim, emb_dim, device=self.device)
        ]
        self.emb_model = nn.Sequential(*mini_mlp)

        self.linear_w = nn.Linear(
            in_features=emb_dim, out_features=1, device=self.device)
        ops = [nn.Linear(in_features=emb_dim, out_features=1,
                         bias=True, device=device)]
        if bounded:
            if not allow_neg_output:
                ops.append(nn.Sigmoid())
            else:
                ops.append(NegPosSigmoid())
        self.linear_v = nn.Sequential(*ops)

    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, interactions_mask: Tensor, global_max_dist: float,
                return_weights: bool = False, **kwargs) -> Tensor:
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]

        if 'diff_tensor' in kwargs:
            diff_tensor = kwargs['diff_tensor']
        else:
            points_features = points.reshape(n_sets, n_points, 1, 5)
            others_features = context_points.reshape(n_sets, 1, n_others, 5)
            diff_tensor = points_features - others_features

        # scale tensor
        diff_tensor = diff_tensor * self.scaling_tensor

        if self.local_polar_coordinates:
            # sq_dist = diff_tensor[..., :2].square().sum(dim=-1, keepdim=True)
            # distance = torch.sqrt(sq_dist + (sq_dist == 0.0) * EPSILON)
            distance = torch.unsqueeze(distance_matrix, dim=-1)
            # WARNING: atan2(x,y) produces nan gradients if x=y=0, these should be masked out, however
            # https://discuss.pytorch.org/t/how-to-avoid-nan-output-from-atan2-during-backward-pass/176890
            num = diff_tensor[..., [1]]
            den = diff_tensor[..., [0]]
            nudge = (den == 0.0) * EPSILON
            rel_angle = torch.atan2(num, den + nudge)

            # offset rel angle by object angle
            rel_angle = points[..., ANGLE_MARK_INDEX].view(
                n_sets, n_points, 1, 1) - rel_angle
            diff_tensor = torch.cat([
                distance, rel_angle, diff_tensor[..., 2:]
            ], dim=-1)

        # (n_sets,n_points,n_others,emb_dim)
        diff_emb = self.emb_model(diff_tensor)

        reduce_dim = 2
        weights = masked_softmax(
            x=self.linear_w(diff_emb).squeeze(
                dim=-1),  # (n_sets,n_points,n_others)
            mask=interactions_mask,
            dim=reduce_dim
        )
        values = self.linear_v(diff_emb).squeeze(
            dim=-1)  # (n_sets,n_points,n_others)
        energy_per_point = torch.sum(
            weights * values, dim=reduce_dim) * points_mask
        if not return_weights:
            return energy_per_point
        return energy_per_point, weights

    @property
    def name(self) -> str:
        return self._name

    @property
    def maximum_distance(self) -> float:
        return self.max_dist

    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor, distance_matrix: Tensor, **kwargs) -> Tensor:
        energy, weights = self.forward(points, points_mask, context_points, context_points_mask, distance_matrix,
                                       return_weights=True, **kwargs)
        return weights


class ShapeOverlap(PriorModule, Module):

    def __init__(self, mappings: List[ValueMapping], device: torch.device, quality_function: str,
                 quality_function_args: Dict, shape: str,
                 prior_name='overlap',
                 intersect_method: str = None, reduce: str = 'max', max_dist: float = None,
                 distance_in_attention: bool = False, resolution=1.0, **kwargs):
        super(ShapeOverlap, self).__init__()

        self._name = prior_name

        self.reduce = reduce
        self.device = device
        self.dist_in_att = distance_in_attention
        self.resolution = resolution
        self.state_scale = torch.tensor([self.resolution if i != ANGLE_MARK_INDEX else 1.0 for i in range(5)],
                                        device=self.device)

        if shape == 'rectangle':
            self.intersect_module = RectangleIntersect(
                approx=intersect_method,
                norm_p=10
            )
            a, b = None, None
            for m in mappings:
                if m.name == 'length':
                    a = m.v_max
                elif m.name == 'width':
                    b = m.v_max
            if a is None or b is None:
                raise RuntimeError(
                    f'Mapping has no length or width mapping ? {mappings}')
            self.max_dist = np.sqrt(a ** 2 + b ** 2)
        elif shape == 'circle':
            assert mappings[0].name == 'radius'
            self.max_dist = mappings[0].v_max * 2
            self.intersect_module = CircleIntersect(
                intersect_method=intersect_method)
        else:
            raise ValueError

        if quality_function == 'none' or quality_function is None:
            warnings.warn(
                f"ShapeOverlap needs at least a simple relu since energies can be <0 and unbounded")
            self.q_fun = nn.ReLU()
        else:
            self.q_fun = getattr(quality_function_modules, quality_function)(device=self.device,
                                                                             **quality_function_args)
        if max_dist is not None:
            self.max_dist = min(max_dist, self.max_dist)

        if self.reduce == 'attention':
            self.attention = MaskedAttention(
                in_features=2 if self.dist_in_att else 1,
                device=self.device
            )
        else:
            self.attention = None

        self.no_neighbors_value = 0.0

    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, gap_matrix: Tensor, global_max_dist: float, interactions_mask: Tensor,
                return_weights: bool = False, **kwargs) -> Tensor:
        # n_sets = points.shape[0]
        # n_points = points.shape[1]
        # n_others = context_points.shape[1]
        if self.resolution != 1.0:
            points = points * self.state_scale
            context_points = context_points * self.state_scale
            distance_matrix = distance_matrix * self.resolution
            global_max_dist = global_max_dist * self.resolution
            max_dist_s = self.max_dist * self.resolution
            if gap_matrix is not None:
                gap_matrix = gap_matrix * self.resolution
        else:
            max_dist_s = self.max_dist

        if gap_matrix is None:
            overlap = self.intersect_module.forward(
                state=points,
                state_mask=points_mask,
                state_other=context_points,
                distance_matrix=distance_matrix)
            overlap = self.q_fun(overlap)
        else:
            overlap = self.q_fun(torch.relu(-gap_matrix))

        ib_points = interactions_mask
        if max_dist_s < global_max_dist:
            ib_points = ib_points & (distance_matrix <= max_dist_s)
        if self.reduce == 'min':
            w = torch.where(ib_points, overlap, torch.tensor(
                [torch.inf], device=self.device))
            overlap_energy_reduced, weights = torch.min(w, dim=2)
        elif self.reduce == 'max':
            w = torch.where(ib_points, overlap, torch.tensor(
                [-torch.inf], device=self.device, dtype=overlap.dtype))
            overlap_energy_reduced, weights = torch.max(w, dim=2)
        elif self.reduce == 'mean':
            w = torch.where(ib_points, overlap, torch.tensor(
                [torch.nan], device=self.device, dtype=overlap.dtype))
            overlap_energy_reduced = torch.nanmean(w, dim=2)
            weights = ib_points
        elif self.reduce == 'attention':
            if self.dist_in_att:
                d = torch.nan_to_num(
                    distance_matrix, max_dist_s + 1, max_dist_s + 1, max_dist_s + 1)
                d = torch.clip(d, 0, max_dist_s + 1) / (max_dist_s + 1)
                atn_features = torch.stack([d, overlap], dim=-1)
            else:
                atn_features = overlap.unsqueeze(dim=-1)
            overlap_energy_reduced, weights = self.attention(
                values=overlap, features=atn_features,
                mask=ib_points, reduce_dim=2, return_weights=True
            )
        else:
            raise ValueError(
                f"reduce={self.reduce} is not a valid option, choose either min, max, mean or attention")

        overlap_energy_reduced = torch.nan_to_num(
            overlap_energy_reduced, self.no_neighbors_value, self.no_neighbors_value, self.no_neighbors_value
        )
        if not return_weights:
            return overlap_energy_reduced
        return overlap_energy_reduced, weights

    @property
    def name(self) -> str:
        return self._name

    @property
    def maximum_distance(self) -> float:
        return self.max_dist

    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor, distance_matrix: Tensor, **kwargs) -> Tensor:
        if self.reduce == 'max':
            overlap_energy, argmax = self.forward(points, points_mask, context_points, context_points_mask,
                                                  distance_matrix,
                                                  return_weights=True, **kwargs)
            n_sets = points.shape[0]
            n_points = points.shape[1]
            n_others = context_points.shape[1]
            weights = torch.zeros(
                (n_sets, n_points, n_others), device=overlap_energy.device)
            weights[:, argmax] = 1.0
            return weights
        else:
            raise NotImplementedError("only reduce=max is implemented yet")


class ConstantNeighbor(PriorModule, Module):

    def __init__(self, max_dist: float, prior_name='constant_neighbor', value=1.0, legacy: bool = True, **kwargs):
        super(ConstantNeighbor, self).__init__()
        self.value = value
        self._name = prior_name
        self.max_dist = max_dist
        self.legacy = legacy

    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, gap_matrix: Tensor, interactions_mask: Tensor, global_max_dist: float,
                return_argmax: bool = False,
                **kwargs) -> Tensor:
        ib_points = interactions_mask
        if global_max_dist > self.max_dist:
            ib_points = ib_points & (distance_matrix <= self.max_dist)

        energy, arg_reduce = torch.max(ib_points, dim=2)
        energy = energy * self.value
        if not self.legacy:
            energy = 1 - energy

        if not return_argmax:
            return energy
        return energy, arg_reduce

    @property
    def name(self) -> str:
        return self._name

    @property
    def maximum_distance(self) -> float:
        return self.max_dist

    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor, distance_matrix: Tensor, **kwargs) -> Tensor:
        energy, argmax = self.forward(points, points_mask, context_points, context_points_mask, distance_matrix,
                                      return_argmax=True, **kwargs)
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]
        weights = torch.zeros(
            (n_sets, n_points, n_others), device=energy.device)
        weights[:, argmax] = 1.0
        return weights


class GenericLocal(PriorModule, Module):

    def __init__(self, mappings: List[ValueMapping], q_fun: str, features_index: List[int], device: torch.device,
                 bounded: bool = True, allow_neg_output: bool = False,
                 mlp_hidden_dims: int = 4, mlp_layers=2, prior_name='generic_local', resolution: float = None,
                 **kwargs):
        super(GenericLocal, self).__init__()
        self.device = device
        self.prior_name = prior_name
        self.resolution = resolution
        self.features_index = features_index
        if 0 in self.features_index or 1 in self.features_index:
            raise RuntimeError("wont handle position in this prior !")
        self.n_features = len(self.features_index)

        if self.resolution is None:
            # scale features to mappings
            self.mappings = [mappings[m - 2] for m in self.features_index]
            v_min = torch.tensor(
                [m.v_min for m in self.mappings], device=self.device)
            v_max = torch.tensor(
                [m.v_max for m in self.mappings], device=self.device)
            cyclic = torch.tensor(
                [m.is_cyclic for m in self.mappings], device=self.device)
            self.features_offset = v_min
            self.features_scale = 1 / (v_max - v_min)
        else:
            # scale features to resolution
            self.features_offset = 0.0
            self.features_scale = torch.tensor(
                [(self.resolution if i != ANGLE_MARK_INDEX else 1.0)
                 for i in self.features_index],
                device=self.device)
            # do not scale angles ! (the rest yes)

        ops = []
        if q_fun == 'linear':
            ops.append(nn.Linear(self.n_features, 1, device=self.device))
        elif q_fun == 'MLP':
            mini_mlp = []
            for i in range(mlp_layers - 1):
                input_dim = self.n_features if i == 0 else mlp_hidden_dims
                mini_mlp = mini_mlp + [
                    nn.Linear(input_dim, mlp_hidden_dims, device=self.device),
                    nn.ReLU()
                ]
            mini_mlp = mini_mlp + [
                nn.Linear(mlp_hidden_dims, 1, device=self.device)
            ]
            ops = ops + mini_mlp
        else:
            raise ValueError

        if bounded:
            if not allow_neg_output:
                ops.append(nn.Sigmoid())
            else:
                ops.append(NegPosSigmoid())

        self.q_fun = nn.Sequential(*ops)

    def _extract_features(self, points):
        feat = points[..., self.features_index]
        return (feat - self.features_offset) * self.features_scale

    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, **kwargs) -> Tensor:

        features = self._extract_features(points)
        values = self.q_fun(features).squeeze(dim=-1)
        return values * points_mask

    @property
    def name(self) -> str:
        return self.prior_name

    @property
    def maximum_distance(self) -> float:
        return 0.0

    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor, distance_matrix: Tensor, **kwargs) -> Tensor:
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]
        return torch.zeros((n_sets, n_points, n_others), device=torch.device)


class CircleExpand(PriorModule, Module):
    def __init__(self, in_range, out_range, prior_name='expand', **kwargs):
        super(CircleExpand, self).__init__()
        self._name = prior_name
        self.in_range = in_range
        self.out_range = out_range

    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor, distance_matrix: Tensor, **kwargs) -> Tensor:
        n_sets = points.shape[0]
        n_points = points.shape[1]
        n_others = context_points.shape[1]
        return torch.zeros((n_sets, n_points, n_others), device=torch.device)

    @property
    def maximum_distance(self) -> float:
        return 0

    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, **kwargs) -> Tensor:
        radius = points[:, :, 2]
        enr_prior = map_range(radius,
                              self.in_range[0], self.in_range[1], self.out_range[0], self.out_range[1],
                              clip=True)
        return enr_prior * points_mask

    @property
    def name(self) -> str:
        return self._name
