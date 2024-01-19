from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

import torch
from torch import Tensor

from base.mappings import ValueMapping


class LikelihoodModule(ABC):

    @abstractmethod
    def forward(self, points: Tensor, points_mask: Tensor, position_energy_map: Tensor,
                marks_energy_maps: List[Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def energy_maps_from_image(self, image, **kwargs) -> Tuple[Tensor, List[Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def densities_from_energy_maps(self, pos_energy_map, marks_energy_maps, **kwargs) -> Tuple[Tensor, List[Tensor]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def sub_energies(self) -> List[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def mappings(self) -> List[ValueMapping]:
        raise NotImplementedError

    @property
    def is_trainable(self) -> bool:
        return False


class PriorModule(ABC):
    @abstractmethod
    def forward(self, points: Tensor, points_mask: Tensor, context_points: Tensor, context_points_mask: Tensor,
                distance_matrix: Tensor, gap_matrix: Tensor, interactions_mask: Tensor, global_max_dist: float, **kwargs) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def maximum_distance(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def interaction_weight(self, points: Tensor, points_mask: Tensor, context_points: Tensor,
                           context_points_mask: Tensor,
                           distance_matrix: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError


class DummyPrior(PriorModule):

    def __init__(self, name='DummyPrior'):
        self._name = name

    def forward(self, points: Tensor, **kwargs) -> Tensor:
        n_sets = points.shape[0]
        n_points = points.shape[1]
        return torch.zeros((n_sets, n_points), device=points.device)

    @property
    def name(self) -> str:
        return self._name

    @property
    def maximum_distance(self) -> float:
        return 0
