from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from torch import Tensor

from samplers.mic_set_utils import state_to_context_cube
from samplers.types import EnergyFunc


class BaseEnergyModel(ABC):

    def __init__(self):
        super(BaseEnergyModel, self).__init__()

    @abstractmethod
    def forward(self, context_cube: Tensor, context_cube_mask: Tensor,
                position_energy_map: Tensor, marks_energy_maps: List[Tensor], compute_context: bool) -> Dict[
            str, Tensor]:
        """
                :param context_cube: tensor of shape (B,3,3,N,D)
                :param context_cube_mask: tensor of shape (B,3,3,N)
                :return: dict
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def max_interaction_distance(self) -> float:
        raise NotImplementedError

    @property
    def trainable_maps(self) -> bool:
        return False

    @abstractmethod
    def energy_maps_from_image(self, image: Tensor, large_image: bool, **kwargs):
        return NotImplementedError

    def forward_state(self, state: Tensor,
                      position_energy_map: Tensor, marks_energy_maps: List[Tensor], compute_context: bool) -> Dict[
            str, Tensor]:
        context_cube, context_cube_mask = state_to_context_cube(state)
        return self.forward(
            context_cube=context_cube, context_cube_mask=context_cube_mask,
            position_energy_map=position_energy_map, marks_energy_maps=marks_energy_maps,
            compute_context=compute_context
        )

    @property
    @abstractmethod
    def combination_module_weights(self) -> Dict[str, float]:
        raise NotImplementedError

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

    def energy_func_wrapper(self, image=None, position_energy_map: Tensor = None,
                            marks_energy_maps: List[Tensor] = None, compute_context: bool = True) -> EnergyFunc:
        if position_energy_map is None or marks_energy_maps is None:
            assert image is not None
            position_energy_map, marks_energy_maps = self.energy_maps_from_image(image, as_energies=True,
                                                                                 large_image=True)

        if len(position_energy_map.shape) == 3:
            position_energy_map = position_energy_map.unsqueeze(dim=0)
        if len(marks_energy_maps) > 0 and len(marks_energy_maps[0].shape) == 3:
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

    @property
    @abstractmethod
    def sub_energies(self) -> List[str]:
        raise NotImplementedError
