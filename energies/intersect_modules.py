import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Module, functional

from modules.ellipsis_overlap import ellipsis_overlap_jit, dividing_axis_gap

APPROX_LIST = ['radius', 'ellipsis', 'exact', 'dividing_axis']


class RectangleIntersect(Module):

    def __init__(self, approx: str, norm_p=None):
        super(RectangleIntersect, self).__init__()
        assert approx in APPROX_LIST
        self.approx = approx
        if self.approx == 'ellipsis':
            assert norm_p is not None
            self.norm_p = norm_p

        self.eps = torch.tensor(1e-5)

    def forward(self, state: Tensor, state_mask: Tensor, state_other: Tensor, distance_matrix: Tensor):
        n_sets = state.shape[0]
        n_points = state.shape[1]
        n_others = state_other.shape[1]

        if self.approx == 'ellipsis':
            intersections = ellipsis_overlap_jit(
                state_1=state, state_2=state_other, distance_matrix=distance_matrix, norm_p=self.norm_p, eps=self.eps
            )
            # intersections = torch.square(interstices)
            # assert not torch.any(torch.isnan(intersections))
        elif self.approx == 'dividing_axis':
            gaps = dividing_axis_gap(state, state_other, std_intersect=True)
            intersections = functional.relu(-gaps)

        elif self.approx == 'radius_legacy':
            points_radius = 0.5 * torch.minimum(
                state[:, :, 2], state[:, :, 3]
            ).view((n_sets, n_points, 1))
            # radius is half the min of width or length
            others_radius = 0.5 * torch.minimum(
                state_other[:, :, 2], state_other[:, :, 3]
            ).view((n_sets, 1, n_others))
            max_overlap = torch.clip(
                points_radius + others_radius, self.eps, torch.inf)
            # warning may be < 0
            intersections = 1 - (distance_matrix / max_overlap)
        elif self.approx == 'radius':
            points_radius = 0.5 * torch.minimum(
                state[:, :, 2], state[:, :, 3]
            ).view((n_sets, n_points, 1))
            # radius is half the min of width or length
            others_radius = 0.5 * torch.minimum(
                state_other[:, :, 2], state_other[:, :, 3]
            ).view((n_sets, 1, n_others))
            superposition = torch.minimum(points_radius, distance_matrix + others_radius) - \
                torch.maximum(-points_radius, distance_matrix - others_radius)
            max_superposition = torch.minimum(
                2 * others_radius, 2 * points_radius)
            intersections = torch.relu(superposition / max_superposition)
        elif self.approx == 'exact':
            raise NotImplementedError
            # corners = get_corners_torch(state)
            # intersections = torch.stack(
            #     [intersection_area(corners[i], corners[closest_k[i, j]])
            #      if distance_matrix[i, closest_k[i, j]] < self.max_dist
            #      else torch.tensor(0.0, device=state.device)
            #      for i in range(n_points)
            #      for j in range(min(n_points, self.k_neighbors))])
            #
            # intersections = intersections.reshape(
            #     (n_points, min(n_points, self.k_neighbors)))  # todo check if good reshape
        else:
            raise ValueError
        return intersections


class CircleIntersect(Module):

    def __init__(self, intersect_method: str = None):
        super(CircleIntersect, self).__init__()
        self.eps = torch.tensor(1e-5)
        self.intersect_method = intersect_method
        if self.intersect_method is None:
            self.intersect_method = 'approximate'

    def forward(self, state: Tensor, state_mask: Tensor, state_other: Tensor, distance_matrix: Tensor):
        n_sets = state.shape[0]
        n_points = state.shape[1]
        n_others = state_other.shape[1]

        points_radius = state[:, :, 2].view((n_sets, n_points, 1))
        others_radius = state_other[:, :, 2].view((n_sets, 1, n_others))
        # points_mask = state_mask[:, :].view((n_sets, n_points, 1))
        # others_mask = state_mask[:, :].view((n_sets, 1, n_others))
        # interactions_mask = points_mask * others_mask
        if self.intersect_method == 'legacy':
            max_overlap = torch.clip(
                points_radius + others_radius, self.eps, torch.inf)
            # warning may be < 0
            intersections = 1 - (distance_matrix / max_overlap)
            return intersections
        if self.intersect_method == 'approximate':
            # see https://gitlab.inria.fr/ayana-ads/lab_notebook/-/blob/0bb0c39abf469af8eade77b64d857c038b9a6fbe/Notebook/img/overlap_formula.png
            superposition = torch.minimum(points_radius, distance_matrix + others_radius) - \
                torch.maximum(-points_radius, distance_matrix - others_radius)
            max_superposition = torch.minimum(
                2 * others_radius, 2 * points_radius)
            max_superposition = torch.clip(
                max_superposition, self.eps, torch.inf)
            return torch.relu(superposition / max_superposition)
        elif self.intersect_method == 'exact':
            # see https://diego.assencio.com/?index=8d6ca3d82151bad815f78addf9b5c1c6
            raise NotImplementedError("this does not work yet")
            r1 = torch.clip(points_radius, self.eps, torch.inf)
            r2 = torch.clip(others_radius, self.eps, torch.inf)
            r1_sq = torch.square(r1)
            r2_sq = torch.square(r2)
            d = torch.clip(distance_matrix, self.eps, torch.inf)
            d1 = (r1_sq - r2_sq - torch.square(d)) / (2 * d)
            d2 = d - d1
            intersection = \
                r1_sq * torch.arccos(d1 / r1) + r2_sq * torch.arccos(d2 / r2) - d1 * torch.sqrt(
                    r1_sq - torch.square(d1)) - d2 * torch.sqrt(r2_sq - torch.square(d2))
            max_intersection = np.pi * torch.square(torch.minimum(r1, r2))
            return intersection / max_intersection
        else:
            raise ValueError(
                f"intersect_method={self.intersect_method} is not a correct value")
