import logging
import time

import numpy as np
import torch
from numpy.random import Generator

from base.images import map_range
from data.synthetic.pointssimulator import PointsSimulator
from energies.generic_energy_model import GenericEnergyModel
from samplers.rjmcmc import ParallelRJMCMC


class EnergySimulator(PointsSimulator):

    def __init__(self, config):
        self.config = config

        self.model = GenericEnergyModel(config)

    def make_points(self, rng: Generator, image) -> np.ndarray:
        shape = image.shape[:2]
        if len(image.shape) != 3:
            image = np.stack([image] * 3, axis=-1)
        pos_e_map, marks_e_map = self.model.energy_maps_from_image(
            torch.from_numpy(image).permute((2, 0, 1)).float(), as_energies=True, large_image=True)
        pos_density_map, marks_density_maps = self.model.densities_from_energy_maps(
            pos_e_map, marks_e_map)
        intensity_map = pos_density_map

        mc = ParallelRJMCMC(
            support_shape=shape,
            device=self.model.device,
            max_interaction_distance=self.model.max_interaction_distance,
            rng=rng,
            energy_func=self.model.energy_func_wrapper(position_energy_map=pos_e_map,
                                                       marks_energy_maps=marks_e_map,
                                                       compute_context=True),
            intensity_map=intensity_map,
            mappings=self.model.mappings,
            marks_density_maps=marks_density_maps,
            **self.config['RJMCMC_params']
        )

        start = time.perf_counter()
        mc.run(verbose=1)
        elapsed_time = time.perf_counter() - start

        logging.info(f"simulated config in {elapsed_time}")

        last_state = mc.current_state.cpu().numpy()

        return last_state
