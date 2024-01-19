import logging
from abc import ABC, abstractmethod
from typing import List

import torch
from numpy.random import Generator
from torch import Tensor

from base.mappings import ValueMapping
from base.parse_config import ConfigParser
from energies.energy_from_maps import EnergyFromMaps
from samplers.rjmcmc import ParallelRJMCMC


class BasePerturbationMethod(ABC):

    @abstractmethod
    def apply(self, states: Tensor, rng: Generator, images: Tensor, **kwargs) -> List[Tensor]:
        raise NotImplementedError


class NormalPert(BasePerturbationMethod):

    def __init__(self, mappings: List[ValueMapping], sigma: float, **kwargs):
        super(NormalPert, self).__init__()
        self.mappings = mappings
        self.mark_min_bounds = [m.v_min for m in self.mappings]
        self.mark_max_bounds = [m.v_max for m in self.mappings]
        self.sigma = sigma

    def apply(self, states: Tensor, rng: Generator, images: Tensor, **kwargs) -> List[Tensor]:
        res = []
        for state, image in zip(states, images):
            h, w = image.shape[1:]
            min_bound = torch.tensor([0.0, 0.0] + self.mark_min_bounds)
            max_bound = torch.tensor([h, w] + self.mark_max_bounds)
            if len(state) > 0:
                res.append(torch.clip(
                    state +
                    torch.from_numpy(rng.normal(
                        scale=self.sigma, size=state.shape)),
                    min_bound, max_bound
                ))
            else:
                res.append(state)
        return res


class MCMCPert(BasePerturbationMethod):

    def __init__(self, model: EnergyFromMaps, mappings: List[ValueMapping], config: ConfigParser,
                 init_w_gt: float, fast_compute: bool, rjmcmc_params_override: dict, **kwargs):
        super(MCMCPert, self)
        self.model = model
        self.mappings = mappings
        self.config = config
        self.init_w_gt = init_w_gt
        self.fast_compute = fast_compute
        self.config_rjmcmc = self.config['RJMCMC_params'].copy()

        for k, v in rjmcmc_params_override.items():
            self.config_rjmcmc[k] = v

    def apply(self, states: Tensor, rng: Generator, images: Tensor, **kwargs) -> List[Tensor]:
        res = []
        pos_e_m = kwargs['pos_e_m']
        marks_e_m = kwargs['marks_e_m']

        self.model.eval()
        for i, s in enumerate(states):
            pos_e_map = pos_e_m[i]
            marks_e_map = [m[i] for m in marks_e_m]
            shape = pos_e_map.shape[1:]

            pos_density_map, marks_density_maps = self.model.densities_from_energy_maps(
                pos_e_map, marks_e_map)

            init_w_gt = rng.random() < self.init_w_gt

            mc = ParallelRJMCMC(
                support_shape=shape,
                device=self.model.device,
                max_interaction_distance=self.config['model']["maximum_distance"],
                rng=rng,
                energy_func=self.model.energy_func_wrapper(position_energy_map=pos_e_map,
                                                           marks_energy_maps=marks_e_map,
                                                           compute_context=not self.fast_compute),
                intensity_map=pos_density_map,
                mappings=self.mappings,
                marks_density_maps=marks_density_maps,
                init_state=states[i] if init_w_gt else None,
                **self.config_rjmcmc
            )

            mc.run(verbose=0)

            res.append(mc.current_state)

            logging.info(f"nb points {len(s)} -> {len(res[-1])}")

        return res
