import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import relu, softplus, leaky_relu

from base.images import map_range


def area_relu(max_area: float, min_area: float, bounded: bool):
    if bounded:
        def b(x):
            return torch.minimum(x, torch.tensor(1.0))
    else:
        def b(x):
            return x

    def fun(areas: Tensor) -> Tensor:
        return b(relu(areas - max_area) + relu(min_area - areas))

    return fun


def area_softplus(max_area: float, min_area: float, bounded: bool):
    if bounded:
        def b(x):
            return torch.minimum(x, torch.tensor(1.0))
    else:
        def b(x):
            return x

    def fun(areas: Tensor) -> Tensor:
        return b(softplus(areas - max_area) + softplus(min_area - areas))

    return fun


def area_sigmoid(max_area: float, min_area: float, slope: float):
    def fun(areas: Tensor) -> Tensor:
        return torch.sigmoid(slope * (areas - max_area)) + torch.sigmoid(slope * (min_area - areas))

    return fun


def area_normalised(max_area: float, min_area: float):
    def fun(areas: Tensor) -> Tensor:
        return (areas - min_area) / (max_area - min_area)

    return fun


def to_quarters(angles):
    angles = torch.minimum(angles, np.pi - angles)
    return torch.minimum(angles, np.pi / 2 - angles) * 2


def remap(x, v_min, v_max):
    return x * (v_max - v_min) + v_min


def align_neg_cos(align_90deg: bool, v_min=-1, v_max=1, offset=0.0):
    def fun(angles_dist: Tensor) -> Tensor:
        angles_dist = torch.remainder(angles_dist - offset, np.pi)
        if align_90deg:
            angles_dist = to_quarters(angles_dist)
        v = -torch.cos(2 * (angles_dist - offset)) * 0.5 + 0.5
        return remap(v, v_min, v_max)

    return fun


def align_none():
    def fun(angles_dist: Tensor) -> Tensor:
        return angles_dist

    return fun


def align_normalised(align_90deg: bool, v_min=-1, v_max=1):
    def fun(angles_dist: Tensor) -> Tensor:
        if align_90deg:
            angles_dist = to_quarters(angles_dist)
        a = angles_dist / np.pi
        v = 2 * torch.minimum(a, 1 - a)
        return remap(v, v_min, v_max)

    return fun


def align_softmax(thresh, slope, align_90deg: bool, v_min=-1, v_max=1, offset=0.0):
    def fun(angles_dist: Tensor) -> Tensor:
        angles_dist = torch.remainder(angles_dist - offset, np.pi)
        if align_90deg:
            angles_dist = to_quarters(angles_dist)
        angles_dist = torch.minimum(
            torch.abs(angles_dist), torch.abs(np.pi - angles_dist))
        v = torch.sigmoid(slope * (angles_dist - thresh))
        return remap(v, v_min, v_max)

    return fun


def align_gaussian(sigma: float, align_90deg: bool, v_min=-1, v_max=1):
    sigma = sigma / np.pi

    def fun(angles_dist: Tensor) -> Tensor:
        if align_90deg:
            angles_dist = to_quarters(angles_dist)
        x = angles_dist / np.pi
        x = torch.minimum(x, 1 - x)
        v = 1 - torch.exp(-0.5 * torch.square(x / sigma))
        return remap(v, v_min, v_max)

    return fun


def overlap_normalised():
    def fun(overlap: Tensor) -> Tensor:
        return relu(overlap)

    return fun


def overlap_margin(margin: float):
    def fun(overlap: Tensor) -> Tensor:
        return relu(overlap - margin) / (1 - margin)

    return fun


def overlap_softplus(margin: float, beta: float = 50):
    def fun(overlap: Tensor) -> Tensor:
        return softplus(overlap - margin, beta=beta) / (1 - margin)

    return fun


def repulsive_falloff(falloff: float, max_dist: float):
    b = 1 / np.power(1 + max_dist, falloff)

    def fun(distance: Tensor) -> Tensor:
        return torch.clip(map_range(1 / torch.pow(1 + distance, falloff), b, 1, 0, 1), 0, 1.0)

    return fun


def repulsive_linear(max_dist: float):
    def fun(distance: Tensor) -> Tensor:
        return map_range(distance, 0, max_dist, 1, 0, clip=True)

    return fun
