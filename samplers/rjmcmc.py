import logging
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Dict, Union, List

import numpy as np
import torch
from numpy.random import Generator
from scipy.special import binom
from torch import Tensor
from tqdm import tqdm

from base.mappings import ValueMapping
from base.misc import append_lists_in_dict
from base.state_ops import clip_state_to_bounds, check_inbound_state
from base.timer import Timer
from samplers.mic_set_utils import slice_state_to_context_cubes, state_to_context_cube
from samplers.mic_sets import TorchTimeMicSets
from samplers.point_samplers import DensityPointSampler, LocalMaxPointSampler, BasePointSampler, SamplerMix
from samplers.types import EnergyFunc

FAST_DIFFUSION = True
ESTIMATE_U_PER_CELL = False  # because the estimation is wrong


class Kernel(Enum):
    BIRTH = 'birth'
    DEATH = 'death'
    TRANSFORM = 'transform'
    DIFFUSION = 'diffusion'


def legacy_config_translator(config: Dict):
    new_config = config.copy()
    if 'sampling_exponent' in new_config:
        v = new_config['sampling_exponent']
        new_config.pop('sampling_exponent')
        if 'sampling_temperature' in new_config:
            raise RuntimeError("cannot override existing key sampling_temperature")
        new_config['sampling_temperature'] = 1 / v
        logging.warning(
            f"replaced sampling_exponent={v} with sampling_temperature={new_config['sampling_temperature']}")
    if 'force_intensity' in new_config:
        new_config.pop('force_intensity')
        logging.warning(f"removed force_intensity={config['force_intensity']} from config")
    if 'intensity_multiplier' in new_config:
        new_config.pop('intensity_multiplier')
        logging.warning(f"removed intensity_multiplier={config['intensity_multiplier']} from config")
    return new_config


@dataclass
class ParallelRJMCMC:
    support_shape: Tuple[int, int]
    mappings: List[ValueMapping]
    max_interaction_distance: float
    max_move_distance: float
    steps_per_pixel: float
    rng: Generator
    energy_func: EnergyFunc
    device: torch.device
    start_temperature: Union[float, str]
    end_temperature: float
    transform_sigma: Union[np.ndarray, List[float]]
    kernel_weights: Dict[str, float]
    intensity: float  # pixel intensity
    scale_temperature: bool = False
    n_samples_autoscale: int = 100
    sampling_temperature: float = 1.0  # temperature is exponent^-1
    position_birth_density: Union[np.ndarray, Tensor] = None
    mark_birth_density: List[Union[np.ndarray, Tensor]] = None
    diffusion_dt: Union[float, List[float]] = None
    diffusion_decay: str = 'polynomial'
    diffusion_dt_gamma: float = None
    diffusion_dt_alpha: float = None
    diffusion_dt_final: Union[float, List[float]] = None
    init_state: Union[np.ndarray, Tensor, str] = None
    uniform_sampling: bool = False
    debug_mode: bool = False
    max_points_per_set: int = None
    max_points_per_pixel: float = None
    expit_sampling_density: bool = False
    local_max_sampler_weight: float = 0.0
    local_max_sampler_sampling_temperature: float = 2.0
    local_max_distance: int = 2
    local_max_thresh: float = 0.1
    sampler_density_w_marks: bool = False
    cell_pick_method: str = 'uniform'
    cell_pick_n: int = None

    def __post_init__(self):
        assert type(self.intensity) is float
        if self.sampler_density_w_marks:
            # when using marks in sampler density, the intensity should be specicified for a pixel in SxM, not S
            # Here I scale the intensity to match the density model
            # \lambda'= \frac{\lambda}{\prod_{m} N_c^m}
            self.intensity = self.intensity / np.prod([m.n_classes for m in self.mappings])
        self.n_marks = len(self.mappings)
        self.state_min_bound = torch.tensor(
            [0.0, 0.0] + [m.v_min for m in self.mappings],
            device=self.device)
        eps = 1e-3
        self.state_max_bound = torch.tensor(
            [self.support_shape[0] - eps, self.support_shape[1] - eps] + [m.v_max for m in self.mappings],
            device=self.device)
        self.state_cyclic = torch.tensor(
            [False, False] + [m.is_cyclic for m in self.mappings],
            device=self.device)
        assert len(self.transform_sigma) == 2 + self.n_marks

        self.mic_points = TorchTimeMicSets(
            support_shape=self.support_shape,
            spatial_interaction_distance=self.max_interaction_distance,
            move_distance=self.max_move_distance,
            max_points_per_cell=self.max_points_per_set,
            max_points_per_pixel=self.max_points_per_pixel,
            n_marks=self.n_marks,
            temporal=False,
            constant_cell_size=False,
            device=self.device
        )

        logging.info(f"cells avg. size: {self.mic_points.mean_cell_size}")
        logging.info(f"number of cells: {self.mic_points.nb_cells}")
        self.n_steps = self.mic_points.get_sim_steps(steps_per_pixel=self.steps_per_pixel)
        # if self.cell_subsampling_factor != 1.0:
        #     logging.warning(f"cell_subsampling_factor != 1 ({self.cell_subsampling_factor}) "
        #                     f"makes energy logging inaccurate")

        if self.uniform_sampling:
            self.position_birth_density = np.ones(self.support_shape) / np.prod(self.support_shape)
            self.mark_birth_density = [np.ones((m.n_classes,) + self.support_shape) / m.n_classes for m in
                                       self.mappings]
        else:
            assert self.position_birth_density is not None
            assert self.mark_birth_density is not None

        if type(self.position_birth_density) is Tensor:
            self.position_birth_density = self.position_birth_density.detach().cpu().numpy()
        self.position_birth_density = np.nan_to_num(self.position_birth_density, nan=0, posinf=0, neginf=0)

        if len(self.mark_birth_density) > 0 and type(self.mark_birth_density[0]) is Tensor:
            self.mark_birth_density = [m.detach().cpu().numpy() for m in self.mark_birth_density]

        self.position_birth_density = np.squeeze(self.position_birth_density)
        self.mark_birth_density = [np.squeeze(m) for m in self.mark_birth_density]
        assert len(self.position_birth_density.shape) == 2
        assert all([len(m.shape) == 3 for m in self.mark_birth_density])

        self.sampler: BasePointSampler
        dps = DensityPointSampler(
            bounds=self.mic_points.cells_bounds,
            position_density=self.position_birth_density,
            mark_densities=self.mark_birth_density,
            mappings=self.mappings,
            sampling_temperature=self.sampling_temperature,
            debug_mode=self.debug_mode,
            expit=self.expit_sampling_density,
            density_w_marks=self.sampler_density_w_marks
        )
        if self.local_max_sampler_weight > 0:
            lms = LocalMaxPointSampler(
                bounds=self.mic_points.cells_bounds,
                position_density=self.position_birth_density,
                mark_densities=self.mark_birth_density,
                mappings=self.mappings,
                sampling_temperature=self.local_max_sampler_sampling_temperature,
                debug_mode=self.debug_mode,
                n_proposals_per_cell=self.mic_points.max_pts,
                local_max_distance=self.local_max_distance,
                local_max_thresh=self.local_max_thresh,
                density_w_marks=self.sampler_density_w_marks
            )
            self.sampler = SamplerMix(
                samplers=[dps, lms], weights=[1 - self.local_max_sampler_weight, self.local_max_sampler_weight]
            )
        else:
            self.sampler = dps

        self.density_per_set = [
            np.sum(self.sampler.cell_density[self.mic_points.cells_indices_per_set[s]])
            for s in self.mic_points.available_sets
        ]
        self.density_per_set_norm = self.density_per_set / np.sum(self.density_per_set)

        logging.info(f"intensity over image (lambda*|S|): {self.intensity_over_image}")

        sample_energy_flag = (self.start_temperature == 'auto') or (type(self.diffusion_dt) is float) or (
            self.scale_temperature)
        sqrt_var_energies = None

        if sample_energy_flag:
            logging.debug("sampling random energies")
            # generating random configurations
            intensity = max(1, self.intensity_over_image)
            batch_size = 16
            energy_samples = []
            # energy_samples_per_point = []
            start = time.perf_counter()
            for _ in range(self.n_samples_autoscale):
                state = self.generate_random_config(
                    log=False, sampler=self.sampler, total_intensity=intensity
                )
                context_cube, context_cube_mask = slice_state_to_context_cubes(state,
                                                                               bounds=self.mic_points.cells_bounds)
                n_cells = len(context_cube)
                batches = np.array_split(np.arange(n_cells), np.ceil(n_cells / batch_size))
                for b in batches:
                    res = self.energy_func(
                        context_cube[b].to(self.device), context_cube_mask[b].to(self.device)
                    )
                    energy_samples.append(res['energy_per_subset'].detach().cpu().numpy())
                    # energy_samples_per_point.append(res['energy_per_point'][0].detach().cpu().numpy())

            energy_samples = np.concatenate(energy_samples, axis=0)
            if np.any(np.isnan(energy_samples)):
                logging.warning(f"found NaNs in energy samples {energy_samples[np.isnan(energy_samples)]}, supressing")
                energy_samples = energy_samples[~np.isnan(energy_samples)]
                if len(energy_samples) == 0:
                    energy_samples = np.ones(self.n_samples_autoscale)
            if np.any(energy_samples > 1e8):
                vmax = np.sqrt(np.finfo(energy_samples.dtype.type).max)
                logging.warning(
                    f"very high enrgy samples {np.max(energy_samples)}, clipping (absolute) to {vmax:.1e} !")
                energy_samples = np.clip(energy_samples, -vmax, vmax)
            sqrt_var_energies = np.clip(np.sqrt(np.var(energy_samples)), 1e-8, np.inf)
            assert not np.isnan(sqrt_var_energies)
            end = time.perf_counter()
            logging.info(f"Sampled energies of random configs ({len(energy_samples)}) in {end - start:.1e}s "
                         f"at intensity {intensity} with std deviation {sqrt_var_energies}")

        if self.scale_temperature:
            scale_t = 2 * sqrt_var_energies
            if type(self.start_temperature) is float:
                self.start_temperature = scale_t * self.start_temperature
            self.end_temperature = scale_t * self.end_temperature
            logging.info(f"scaled temperatures by {scale_t:.1e}")

        if self.start_temperature == 'auto':
            # T0 = 2 sqrt(var(U(T=infty,Y~random config)))
            self.start_temperature = 2 * sqrt_var_energies
            logging.info(f"auto start_temperature set at {self.start_temperature:.1e} "
                         f"(sqrt(var(U))={sqrt_var_energies:.1e})")
            assert not np.isnan(self.start_temperature)
        elif self.start_temperature == 'constant':
            self.start_temperature = self.end_temperature
        else:
            assert type(self.start_temperature) is float or type(self.start_temperature) is np.float64
        self.temperature = self.start_temperature
        self.alpha_temperature = np.power(self.end_temperature / self.start_temperature, 1 / self.n_steps)
        logging.info(f"set alpha = {self.alpha_temperature} : "
                     f"T0({self.start_temperature:.1e})->Tf({self.end_temperature:.1e}) in {self.n_steps} steps")
        self.time_log = []

        self.set_init_state(init_state=self.init_state)

        if ESTIMATE_U_PER_CELL:
            self.last_energy_per_cell = np.zeros(self.mic_points.nb_cells)  # keep last energy per subset type
        else:
            self.last_energy_per_cell = None
        kernel_names = [k.value for k in Kernel]
        assert all([v in kernel_names for v in self.kernel_weights])
        self._p_kernels = np.array([self.kernel_weights.get(k, 0.0) for k in kernel_names])
        self._p_kernels = self._p_kernels / np.sum(self._p_kernels)
        self._p_kernels_dict = {k: p for k, p in zip(Kernel, self._p_kernels)}
        # logging.info(f"kernel probabilities : {self._p_kernels_dict}")

        # find dt
        self.diffusion_dt_t: Union[Tensor, None]
        if self.diffusion_dt is not None:
            if type(self.diffusion_dt) is list:
                warnings.warn("diffusion_dt should be specified as a float, "
                              "otherwise it won´t be scaled properly to the energy amplitude")
                self.diffusion_dt_t = torch.tensor(np.array(self.diffusion_dt), device=self.device)
                self.diffusion_dt_final_t = None if self.diffusion_dt_final is None \
                    else torch.tensor(np.array(self.diffusion_dt_final), device=self.device)
            else:
                assert type(self.diffusion_dt) is float
                # use energy amplitude to estimate adequate dt
                amplitude = sqrt_var_energies + 1e-8
                dt_spatial = self.diffusion_dt / amplitude
                dt_marks = [self.diffusion_dt / amplitude * (m.range / m.n_classes) for m in self.mappings]
                self.diffusion_dt_t = torch.tensor(np.array(2 * [dt_spatial] + dt_marks), device=self.device)
                if self._p_kernels_dict[Kernel.DIFFUSION] != 0:
                    assert not torch.any(torch.isnan(self.diffusion_dt_t))

                if self.diffusion_dt_final is not None:
                    assert type(self.diffusion_dt_final) is float
                    dt_spatial = self.diffusion_dt_final / amplitude
                    dt_marks = [self.diffusion_dt_final / amplitude * (m.range / m.n_classes) for m in self.mappings]
                    self.diffusion_dt_final_t = torch.tensor(np.array(2 * [dt_spatial] + dt_marks), device=self.device)
                    assert not torch.any(torch.isnan(self.diffusion_dt_final_t))
                    add_log = f"and dt_final={self.diffusion_dt_final_t.cpu().numpy()} "
                else:
                    self.diffusion_dt_final_t = None
                    add_log = ''

                logging.info(f"from dt={self.diffusion_dt}(/px at unit amplitude) "
                             f"auto set dt={self.diffusion_dt_t.cpu().numpy()} " + add_log +
                             f"(avg amplitude: {amplitude:.1e})")

            if self.diffusion_dt_final_t is not None or self.diffusion_dt_gamma is not None:
                if self.diffusion_decay == 'polynomial':
                    assert self.diffusion_dt_final_t is not None and self.diffusion_dt_gamma is not None
                    assert torch.all(self.diffusion_dt_final_t < self.diffusion_dt_t)
                    self.diffusion_dt_b = self.n_steps / (torch.pow(self.diffusion_dt_final_t / self.diffusion_dt_t,
                                                                    -1 / self.diffusion_dt_gamma) - 1)
                    self.diffusion_dt_a = self.diffusion_dt_t * torch.pow(self.diffusion_dt_b, self.diffusion_dt_gamma)
                    logging.info(f"From diffusion initial step {self.diffusion_dt_t.cpu().numpy()}"
                                 f" and final step {self.diffusion_dt_final_t.cpu().numpy()}, found :"
                                 f" a={self.diffusion_dt_a.cpu().numpy()} and b={self.diffusion_dt_b.cpu().numpy()} ")
                elif self.diffusion_decay == 'exponential':
                    if self.diffusion_dt_alpha is None:
                        self.diffusion_dt_alpha: Tensor = torch.pow(self.diffusion_dt_final_t / self.diffusion_dt_t,
                                                                    1 / self.n_steps)
                        logging.info(f"From diffusion initial step {self.diffusion_dt_t.cpu().numpy()}"
                                     f" and final step {self.diffusion_dt_final_t.cpu().numpy()}, found :"
                                     f" alpha={self.diffusion_dt_alpha.cpu().numpy()}")
                    else:
                        self.diffusion_dt_alpha: Tensor = torch.tensor([self.diffusion_dt_alpha], device=self.device)
                else:
                    raise ValueError
            else:
                self.diffusion_dt_final_t, self.diffusion_dt_a, self.diffusion_dt_b = None, None, None
        else:
            if self._p_kernels_dict[Kernel.DIFFUSION] != 0:
                raise RuntimeError('cannot have p(Kernel.DIFFUSION)>0 and not specify diffusion_dt')
            self.diffusion_dt_t = None
        self.step_count = 0
        self.seq_step_count = 0
        self.eps = 1e-8

    @property
    def intensity_over_image(self) -> float:
        if self.sampler_density_w_marks:
            return self.intensity * np.prod(self.support_shape) * np.prod([m.n_classes for m in self.mappings])
        else:
            return self.intensity * np.prod(self.support_shape)

    def generate_random_config(self, total_intensity: float, sampler: DensityPointSampler = None, log=True):
        n_points = self.rng.poisson(total_intensity)
        # n_points = self.rng.poisson(len(s))
        if sampler is None:
            init_positions = self.rng.uniform((0, 0), self.support_shape, size=(n_points, 2))
            init_marks = self.rng.uniform(
                [m.v_min for m in self.mappings],
                [m.v_max for m in self.mappings],
                size=(n_points, len(self.mappings)))
            state = np.concatenate((init_positions, init_marks), axis=-1)
        else:
            cell_p = np.array(self.sampler.cell_density)
            cell_p = cell_p / np.sum(cell_p)
            points = []
            assert len(cell_p) == self.mic_points.nb_cells
            for _ in range(n_points):
                current_cell = self.rng.choice(range(self.mic_points.nb_cells), p=cell_p)
                p = self.sampler.sample(current_cells=np.array([current_cell]), rng=self.rng)[0]
                points.append(p)
            if len(points) > 0:
                state = np.stack(points, axis=0)
            else:
                state = np.empty((0, 2 + self.n_marks))
        if log:
            logging.debug(f'generated random init_state with {n_points} points for shape {self.support_shape}')
        return state

    def update_sim_params(self, start_temperature=None, end_temperature=None, steps_per_pixel=None):
        if start_temperature is not None:
            self.start_temperature = start_temperature
        if end_temperature is not None:
            self.end_temperature = end_temperature
        if steps_per_pixel is not None:
            self.steps_per_pixel = steps_per_pixel
            self.n_steps = self.mic_points.get_sim_steps(steps_per_pixel=self.steps_per_pixel)
        self.alpha_temperature = np.power(self.end_temperature / self.start_temperature, 1 / self.n_steps)

    def reset(self, init_state=None):
        self.mic_points = TorchTimeMicSets(
            support_shape=self.support_shape,
            spatial_interaction_distance=self.max_interaction_distance,
            move_distance=self.max_move_distance,
            max_points_per_cell=self.max_points_per_set,
            max_points_per_pixel=self.max_points_per_pixel,
            n_marks=self.n_marks,
            temporal=False,
            constant_cell_size=False,
            device=self.device
        )
        self.set_init_state(init_state)
        self.temperature = self.start_temperature
        self.step_count = 0
        self.seq_step_count = 0

    def set_init_state(self, init_state):
        if type(self.init_state) is Tensor:
            self.init_state = check_inbound_state(init_state.to(self.device), self.state_min_bound, self.state_max_bound,
                                                  cyclic=self.state_cyclic, clip_if_oob=True)
            self.init_state = self.init_state.cpu().numpy()
            if len(self.init_state) > 0:
                self.mic_points.add_points(points=self.init_state)
            logging.debug(f"setting init_state with {len(self.init_state)} point(s)")
        elif type(self.init_state) is np.ndarray:
            self.mic_points.add_points(points=self.init_state)
            logging.debug(f"setting init_state with {len(self.init_state)} point(s)")
        elif self.init_state is None or self.init_state == 'empty':
            logging.debug(f"setting init_state as empty")
        elif self.init_state == 'random_uniform':
            self.init_state = self.generate_random_config(total_intensity=self.intensity_over_image)
            self.mic_points.add_points(points=self.init_state)
            logging.debug(f"setting init_state as random uniform with {len(self.init_state)} points")
        elif self.init_state == 'random_sampler':
            self.init_state = self.generate_random_config(
                sampler=self.sampler, total_intensity=self.intensity_over_image)
            self.mic_points.add_points(points=self.init_state)
            logging.debug(f"setting init_state as random uniform with {len(self.init_state)} points")
        else:
            raise ValueError(f"non supported init_state type/value {type(init_state)} "
                             f"{init_state if type(init_state) is str else ''}")

    def pick_current_cells(self):
        if self.cell_pick_method == 'pick':
            assert self.cell_pick_n is not None
            n_picks = self.cell_pick_n
            available_sets = self.mic_points.available_sets
            set_index = self.rng.choice(len(available_sets), p=self.density_per_set_norm)
            set_class = available_sets[set_index]
            set_pick_p = self.density_per_set_norm[set_index]

            current_cells = self.mic_points.cells_indices_per_set[set_class]
            if n_picks < len(current_cells):
                cells_densities = self.sampler.cell_density[current_cells]
                cells_p = cells_densities / np.sum(cells_densities)
                picked_indices = self.rng.choice(len(current_cells), size=n_picks, replace=False, p=cells_p)
                current_cells = current_cells[picked_indices]
                cells_pp = cells_p[picked_indices]
                current_cells_pick_p = \
                    binom(len(current_cells), n_picks) * cells_pp * np.power(cells_pp, n_picks - 1) * set_pick_p
                # current_cells_pick_p = cells_p[picked_indices] * n_picks * set_pick_p
            else:
                pass
                # current_cells_pick_p = np.ones(len(current_cells)) * set_pick_p
        elif self.cell_pick_method == 'prune':
            assert self.cell_pick_n is not None
            n_picks = self.cell_pick_n
            available_sets = self.mic_points.available_sets
            set_index = self.rng.choice(len(available_sets), p=self.density_per_set_norm)
            set_class = available_sets[set_index]
            # set_pick_p = self.density_per_set_norm[set_index]

            current_cells = self.mic_points.cells_indices_per_set[set_class]
            cells_densities = self.sampler.cell_density[current_cells]
            # p_cells = cells_densities / np.max(cells_densities)  # so that we pick at least one
            p_cells = np.minimum(1.0, n_picks * cells_densities / np.sum(cells_densities))
            p_cells = p_cells / np.max(p_cells)
            n_cells = len(current_cells)
            picked_indices = np.arange(n_cells)[p_cells > self.rng.random(n_cells)]
            current_cells = current_cells[picked_indices]
            # current_cells_pick_p = p_cells[picked_indices] * set_pick_p
        elif self.cell_pick_method == 'uniform':
            set_class = self.rng.choice(self.mic_points.available_sets)
            current_cells = self.mic_points.cells_indices_per_set[set_class]
            if type(self.cell_pick_n) is int:
                if self.cell_pick_n < len(current_cells):
                    current_cells = self.rng.choice(current_cells, size=self.cell_pick_n, replace=False)
        else:
            raise ValueError

        return set_class, current_cells, None

    def step(self, log: bool = False, force_accept: bool = None):
        start = time.perf_counter()

        kernel = self.rng.choice(Kernel, p=self._p_kernels)
        set_class, current_cells, current_cells_pick_p = self.pick_current_cells()
        context_cube, context_cube_mask, _ = self.mic_points.points_and_context(
            cell_indicies=current_cells, reduce=True, keep_one=True)

        context_cube = clip_state_to_bounds(context_cube, self.state_min_bound, self.state_max_bound, self.state_cyclic)

        if kernel is not Kernel.DIFFUSION:
            res_0 = self.energy_func(
                context_cube.to(self.device), context_cube_mask.to(self.device)
            )
            energy_per_cell_0 = res_0['energy_per_subset']
            if self.last_energy_per_cell is not None:
                try:
                    energy_per_cell_0_inner = res_0['energy_per_subset_inner']
                except KeyError:
                    warnings.warn("energy_func does not return energy_per_subset_inner, "
                                  "using energy_per_subset instead")
                    energy_per_cell_0_inner = res_0['energy_per_subset']  # this is wrong if compute_context is true
                # self.last_energy[set_class] = float(torch.sum(energy_per_cell_0_inner).detach().cpu())
                self.last_energy_per_cell[current_cells] = energy_per_cell_0_inner.detach().cpu().numpy()
        else:
            energy_per_cell_0 = None

        with Timer() as timer:
            if kernel is Kernel.BIRTH:
                log_dict = self.kernel_birh(context_cube=context_cube, context_cube_mask=context_cube_mask,
                                            current_cells=current_cells, energy_per_cell_0=energy_per_cell_0,
                                            current_cells_pick_p=current_cells_pick_p, log=log,
                                            force_accept=force_accept)
            elif kernel is Kernel.DEATH:
                log_dict = self.kernel_death(context_cube=context_cube, context_cube_mask=context_cube_mask,
                                             current_cells=current_cells, energy_per_cell_0=energy_per_cell_0,
                                             current_cells_pick_p=current_cells_pick_p, log=log,
                                             force_accept=force_accept)
            elif kernel is Kernel.TRANSFORM:
                log_dict = self.kernel_transform(context_cube=context_cube, context_cube_mask=context_cube_mask,
                                                 current_cells=current_cells, energy_per_cell_0=energy_per_cell_0,
                                                 current_cells_pick_p=current_cells_pick_p, log=log,
                                                 force_accept=force_accept)
            elif kernel is Kernel.DIFFUSION:
                log_dict = self.kernel_diffusion(context_cube=context_cube, context_cube_mask=context_cube_mask,
                                                 current_cells=current_cells, set_class=set_class,
                                                 current_cells_pick_p=current_cells_pick_p, log=log)
            else:
                raise ValueError
        if log:
            log_dict['time'] = timer()

        self.time_log.append(time.perf_counter() - start)
        self.step_count += 1
        self.seq_step_count += len(current_cells)
        if log:
            return log_dict

    def kernel_birh(self, context_cube: Tensor, context_cube_mask: Tensor, current_cells: np.ndarray,
                    energy_per_cell_0: Dict, current_cells_pick_p: np.ndarray, log: bool, force_accept: bool):
        # todo work directly with non full cells to deal with acceptance issues on full cells
        n_points_init = len(self.mic_points)
        non_full_cells = ~context_cube_mask[:, 0, 0, -1].cpu().numpy().astype(bool)
        current_non_full_cells = current_cells[non_full_cells]
        # current_cells_pick_p = current_cells_pick_p[non_full_cells]
        if len(current_non_full_cells) > 0:
            nb_cells = len(current_non_full_cells)
            points_number = self.mic_points.cells_point_number[current_non_full_cells]
            new_points = self.sampler.sample(current_non_full_cells, self.rng, )

            new_points_t = torch.tensor(new_points, device=self.device).float()

            # assert torch.all(new_points_t <= self.state_max_bound)
            # assert torch.all(new_points_t >= self.state_min_bound)

            context_cube[non_full_cells, 0, 0, -1] = new_points_t
            context_cube_mask[non_full_cells, 0, 0, -1] = True

            res_1 = self.energy_func(
                context_cube[non_full_cells].to(self.device), context_cube_mask[non_full_cells].to(self.device)
            )
            energy_per_cell_1 = res_1['energy_per_subset']
            energy_delta = (energy_per_cell_1 - energy_per_cell_0[
                non_full_cells]).detach().cpu().numpy()  # todo switch to torch
            # Q(0->1)=d(p)/nu(S)
            points_density = self.sampler.get_point_density(current_non_full_cells, new_points, True)
            # points_density = np.clip(points_density, np.finfo(points_density.dtype).eps, np.inf)
            forward_log_density = np.log(points_density + self.eps) - np.log(self.intensity + self.eps)
            # Q(1->0)=1/(n+1)
            backward_log_density = -np.log(points_number + 1)
            # alpha(0->1) = exp(-(U(1)-U(0))) * Q(1->0) / Q(0->1)
            kernel_ratio = self._p_kernels_dict[Kernel.DEATH] / self._p_kernels_dict[Kernel.BIRTH]
            log_green_ratio = (-energy_delta / self.temperature) \
                              + backward_log_density \
                              - forward_log_density \
                              + np.log(kernel_ratio)
            if force_accept is None:
                accepted = np.log(self.rng.random(nb_cells)) < log_green_ratio
            else:
                accepted = np.full(log_green_ratio.shape, fill_value=force_accept)

            if self.debug_mode:
                # if np.any(energy_delta[accepted] > 0) and self.temperature < 1e-3:
                #     logging.info(f'accepted positive delta U at low temp ({self.temperature:.1e})!')

                if self.temperature < 1e-3:
                    added_points_e = res_1['energy_per_point'][:, -1].detach().cpu().numpy()
                    pos_delta_points = added_points_e > 0
                    if np.any(pos_delta_points[accepted]):
                        bad_points = new_points_t[pos_delta_points & accepted]
                        warnings.warn(f"added high energy points at low T ({self.temperature:.1e}):\n"
                                      # f"{bad_points}\n"
                                      # f"with energies:\n"
                                      f"{added_points_e[accepted]}")

            self.mic_points.add_points(
                points=new_points[accepted],
                cell_indices=current_non_full_cells[accepted]
            )

            assert n_points_init + np.sum(accepted) == len(self.mic_points)

            if log:
                return {
                    'kernel': Kernel.BIRTH,
                    'new_points': new_points,
                    'points_density': points_density,
                    'log_green_ratio': log_green_ratio,
                    'accepted': accepted,
                    'delta_energy': energy_delta,
                    'temperature': self.temperature * np.ones_like(accepted),
                    'current_cells': current_non_full_cells
                }
            else:
                return {}
        if log:
            return {
                'kernel': Kernel.BIRTH,
                'new_points': [],
                'points_density': [],
                'log_green_ratio': [],
                'accepted': [],
                'delta_energy': [],
                'temperature': [],
                'current_cells': current_non_full_cells
            }
        else:
            return {}

    def kernel_death(self, context_cube: Tensor, context_cube_mask: Tensor, current_cells: np.ndarray,
                     energy_per_cell_0: Dict, current_cells_pick_p: np.ndarray, log: bool, force_accept: bool):
        n_points_init = len(self.mic_points)
        non_empty_current_cells = np.intersect1d(self.mic_points.non_empty_cell_indices, current_cells)
        non_empty_current_cells_mask = np.isin(current_cells, non_empty_current_cells)
        nb_cells = len(non_empty_current_cells)
        # current_cells_pick_p = current_cells_pick_p[non_empty_current_cells_mask]
        if nb_cells > 0:
            removal_indices = self.rng.integers(
                0, self.mic_points.cells_point_number[non_empty_current_cells], size=nb_cells
            )
            context_cube_mask[non_empty_current_cells_mask, 0, 0, removal_indices] = False
            removed_points = context_cube[non_empty_current_cells_mask, 0, 0, removal_indices]

            res_1 = self.energy_func(
                context_cube.to(self.device), context_cube_mask.to(self.device)
            )
            energy_per_cell_1 = res_1['energy_per_subset']
            energy_delta = (
                    energy_per_cell_1[non_empty_current_cells_mask] -
                    energy_per_cell_0[non_empty_current_cells_mask]
            ).detach().cpu().numpy()

            # Q(0->1)=1/n
            forward_log_density = -np.log(self.mic_points.cells_point_number[non_empty_current_cells])
            # Q(1->0)=d(p)/nu(S)
            points_density = self.sampler.get_point_density(non_empty_current_cells, removed_points, True)
            points_density = np.clip(points_density, np.finfo(points_density.dtype).eps, np.inf)
            backward_log_density = np.log(points_density + self.eps) - np.log(self.intensity + self.eps)
            # alpha(0->1) = exp(-(U(1)-U(0))) * Q(1->0) / Q(0->1)
            kernel_ratio = self._p_kernels_dict[Kernel.BIRTH] / self._p_kernels_dict[Kernel.DEATH]
            log_green_ratio = (
                                      -energy_delta / self.temperature) + backward_log_density - forward_log_density + np.log(
                kernel_ratio)

            if force_accept is None:
                accepted = np.log(self.rng.random(nb_cells)) < log_green_ratio
            else:
                accepted = np.full(log_green_ratio.shape, fill_value=force_accept)

            if self.debug_mode:
                pass
                # if np.any(energy_delta[accepted] > 0) and self.temperature < 1e-3:
                #     logging.info(f'accepted positive delta U at low temp ({self.temperature:.1e})!')

            self.mic_points.remove_points(
                points_indices=removal_indices[accepted],
                cell_indices=non_empty_current_cells[accepted]
            )

            assert n_points_init - np.sum(accepted) == len(self.mic_points)

            if log:
                return {
                    'kernel': Kernel.DEATH,
                    'removed_points': removed_points.detach().cpu().numpy(),
                    'points_density': points_density,
                    'log_green_ratio': log_green_ratio,
                    'accepted': accepted,
                    'temperature': self.temperature * np.ones_like(accepted),
                    'delta_energy': energy_delta,
                    'current_cells': non_empty_current_cells
                }
            else:
                return {}
        if log:
            return {
                'kernel': Kernel.DEATH,
                'removed_points': [],
                'points_density': [],
                'log_green_ratio': [],
                'accepted': [],
                'temperature': [],
                'delta_energy': [],
                'current_cells': non_empty_current_cells
            }
        else:
            return {}

    def kernel_transform(self, context_cube: Tensor, context_cube_mask: Tensor, current_cells: np.ndarray,
                         energy_per_cell_0: Dict, current_cells_pick_p: np.ndarray, log: bool, force_accept: bool):
        non_empty_current_cells = np.intersect1d(self.mic_points.non_empty_cell_indices, current_cells)
        non_empty_current_cells_mask = np.isin(current_cells, non_empty_current_cells)
        nb_cells = len(non_empty_current_cells)
        if nb_cells > 0:

            if self.debug_mode:
                n_points_0 = self.mic_points.__len__()
            else:
                n_points_0 = None

            selected_points = self.rng.integers(
                0, self.mic_points.cells_point_number[non_empty_current_cells], size=nb_cells
            )
            pos_deltas = np.clip(
                self.rng.normal(0, self.transform_sigma[:2], size=(nb_cells, 2)),
                -self.max_move_distance, self.max_move_distance
            )
            marks_delta = self.rng.normal(0, self.transform_sigma[2:], size=(nb_cells, self.n_marks))

            delta = torch.concat([torch.from_numpy(pos_deltas).float(), torch.from_numpy(marks_delta).float()],
                                 dim=-1).to(self.device)  # todo move the whole tensor making to gpu from the beginning
            updated_points = context_cube[non_empty_current_cells_mask, 0, 0, selected_points]
            # keep points in bounds !
            updated_points = clip_state_to_bounds(updated_points + delta,
                                                  self.state_min_bound, self.state_max_bound, self.state_cyclic)

            context_cube[non_empty_current_cells_mask, 0, 0, selected_points] = updated_points

            res_1 = self.energy_func(
                context_cube.to(self.device), context_cube_mask.to(self.device)
            )
            energy_per_cell_1 = res_1['energy_per_subset']

            energy_delta = (
                    energy_per_cell_1[non_empty_current_cells_mask] -
                    energy_per_cell_0[non_empty_current_cells_mask]
            ).detach().cpu().numpy()  # todo switch to torch
            # alpha(0->1) = exp(-(U(1)-U(0))) * Q(1->0) / Q(0->1)
            # Q(1->0) = Q(0->1) because of normal distribution over deltas
            # kernel_ratio = 1.0 since transform is balanced by transform p_t / p_t = 1.0
            log_green_ratio = (-energy_delta / self.temperature)
            if force_accept is None:
                accepted = np.log(self.rng.random(nb_cells)) < log_green_ratio
            else:
                accepted = np.full(log_green_ratio.shape, fill_value=force_accept)

            if self.debug_mode:
                pass
                # if np.any(energy_delta[accepted] > 0) and self.temperature < 1e-3:
                #     logging.info(f'accepted positive delta U at low temp ({self.temperature:.1e})!')

            self.mic_points.update_points(
                points=updated_points[accepted],
                cell_indices=non_empty_current_cells[accepted],
                points_indices=selected_points[accepted]
            )

            if self.debug_mode:
                n_points_1 = self.mic_points.__len__()
                assert n_points_1 == n_points_0

            if log:
                return {
                    'kernel': Kernel.TRANSFORM,
                    'updated_points': updated_points.cpu().numpy(),
                    'delta': delta.squeeze().cpu().numpy(),
                    'log_green_ratio': log_green_ratio,
                    'accepted': accepted,
                    'temperature': self.temperature * np.ones_like(accepted),
                    'delta_energy': energy_delta,
                    'current_cells': non_empty_current_cells
                }
            else:
                return {}
        if log:
            return {
                'kernel': Kernel.TRANSFORM,
                'updated_points': [],
                'delta': [],
                'log_green_ratio': [],
                'accepted': [],
                'temperature': [],
                'delta_energy': [],
                'current_cells': non_empty_current_cells
            }
        else:
            return {}

    def kernel_diffusion(self, context_cube: Tensor, context_cube_mask: Tensor, current_cells: np.ndarray,
                         current_cells_pick_p: np.ndarray, set_class: int, log: bool):
        assert not context_cube.requires_grad
        context_cube.requires_grad = True
        nb_cells = len(current_cells)

        if self.debug_mode:
            n_points_0 = self.mic_points.__len__()
        else:
            n_points_0 = None

        res = self.energy_func(
            context_cube.to(self.device), context_cube_mask.to(self.device)
        )
        cells_energy = res['energy_per_subset']
        energy = torch.sum(cells_energy)
        energy.backward()

        if self.last_energy_per_cell is not None:
            try:
                energy_per_cell_inner = res['energy_per_subset_inner']
            except KeyError:
                warnings.warn("energy_func does not return energy_per_subset_inner, "
                              "using energy_per_subset instead")
                energy_per_cell_inner = res['energy_per_subset']  # this is wrong if compute_context is true
            self.last_energy_per_cell[current_cells] = energy_per_cell_inner.detach().cpu().numpy()

        mask = context_cube_mask[:, 0, 0]
        current_points = context_cube[:, 0, 0][mask]
        e_grad = context_cube.grad[:, 0, 0][mask]  # look at grad only in current sets

        sq2t = torch.sqrt(torch.tensor(2 * self.temperature, device=self.device))
        # pert = dt sqrt(2T) N(0,1)
        pert = sq2t * self.diffusion_dt_t * torch.randn(size=e_grad.shape, device=self.device)
        # pert = self.diffusion_dt_t * torch.randn(size=e_grad.shape)

        step_size = self.diffusion_dt_t
        # step_size = sq2t * self.diffusion_dt_t

        delta = - e_grad * step_size + pert
        delta[..., :2] = torch.clip(delta[..., :2], -self.max_move_distance, self.max_move_distance)
        updated_points = clip_state_to_bounds(current_points + delta,
                                              self.state_min_bound, self.state_max_bound, self.state_cyclic)
        if FAST_DIFFUSION:
            # clear current cells
            self.mic_points.clear_cells(current_cells)
            # add new poitns to cells (current and neighboring)
            self.mic_points.add_points(updated_points.detach())
        else:
            raise NotImplementedError
            # for i in range(nb_cells):  # todo move this to mic_sets class
            #     points = updated_points[i, context_cube_mask[i, 0, 0]].detach().view((-1, self.n_marks + 2))
            #     points_np = points.detach().cpu().numpy()
            #     new_cell_indices = self.mic_points._find_corresponding_cell(points_np[:, :2], None)
            #     cell_id = current_cells[i]
            #     oob = new_cell_indices != cell_id
            #     n_inbound = np.sum(~oob)
            #     self.mic_points.cells[cell_id, :n_inbound] = points[~oob]
            #     self.mic_points.masks[cell_id] = False
            #     self.mic_points.masks[cell_id, :n_inbound] = True
            #     self.mic_points.cells_point_number[cell_id] = n_inbound
            #     self.mic_points.add_points(points=points_np[oob], cell_indices=new_cell_indices[oob])
        if self.debug_mode:
            n_points_1 = self.mic_points.__len__()
            assert n_points_1 == n_points_0

        if log:
            return {
                'kernel': Kernel.DIFFUSION,
                'accepted': np.ones(len(current_cells), dtype=bool),
                'current_cells': current_cells,
                'updated_points': updated_points.detach().cpu().numpy(),
                'energy_per_cell': cells_energy.detach().cpu().numpy(),
                'delta': delta.detach().cpu().numpy(),
                'grad': e_grad.detach().cpu().numpy(),
                'state_mask': mask.detach().cpu().numpy(),
                'pert': pert.detach().cpu().numpy(),
                'dt': step_size.detach().cpu().numpy(),
                'temperature': self.temperature * np.ones(step_size.shape)
            }
        else:
            return {}

    def step_temperature(self):
        self.temperature *= self.alpha_temperature
        if self.diffusion_dt_gamma is not None or self.diffusion_dt_alpha is not None:
            if self.diffusion_decay == 'polynomial':
                self.diffusion_dt_t = self.diffusion_dt_a * \
                                      torch.pow(self.diffusion_dt_b + self.step_count, -self.diffusion_dt_gamma)
            elif self.diffusion_decay == 'exponential':
                self.diffusion_dt_t = self.diffusion_dt_t * self.diffusion_dt_alpha
            else:
                raise ValueError

    def __iter__(self):
        return self

    def __next__(self):
        return self.step()

    def safety_check(self, extensive: bool = False):
        # checks that empty cells have zero energy
        start = time.perf_counter()
        context_cube = torch.rand((4, 3, 3, 32, self.n_marks + 2), device=self.device)
        context_cube = context_cube * (self.state_max_bound - self.state_min_bound) + self.state_min_bound

        context_cube_mask = torch.zeros((4, 3, 3, 32)).bool()
        res = self.energy_func(
            context_cube.to(self.device), context_cube_mask.to(self.device)
        )
        assert res['total_energy'] == 0
        assert torch.all(res['energy_per_subset'] == 0)
        assert torch.all(res['energy_per_point'] == 0)

        if extensive:
            for _ in range(10):
                context_cube = torch.rand((4, 3, 3, 32, self.n_marks + 2), device=self.device)
                context_cube = context_cube * (self.state_max_bound - self.state_min_bound) + self.state_min_bound

                context_cube_mask = torch.rand((4, 3, 3, 32)) > 0.9
                res = self.energy_func(
                    context_cube.to(self.device), context_cube_mask.to(self.device)
                )
                energy_per_point = res['energy_per_point']
                assert torch.all((~context_cube_mask * energy_per_point.cpu().view(context_cube_mask.shape)) == 0)

        end = time.perf_counter()
        logging.info(f"safety check passed in {end - start:.1e}s")

    def run(self, verbose=1, log: bool = False, tqdm_kwargs=None, kernel_logs: bool = False, state_logs: bool = False,
            compute_total_energy: bool = False):
        if kernel_logs or state_logs or compute_total_energy:
            logging.warning(f"extensive logging activated({kernel_logs=}, {state_logs=}, {compute_total_energy=})"
                            f" might SLOW DOWN execution")
        self.safety_check(extensive=True)
        pbar = range(self.n_steps)
        energy_compute_cooldown = 0
        run_log = {}
        if verbose > 0:
            if tqdm_kwargs is None:
                tqdm_kwargs = {}
            if 'desc' not in tqdm_kwargs:
                tqdm_kwargs['desc'] = "RJMCMC"
            pbar = tqdm(pbar, **tqdm_kwargs)
        for n in pbar:
            k_log = self.step(log=kernel_logs)
            self.step_temperature()
            if verbose > 1 or log:
                if self.last_energy_per_cell is not None:
                    energy = np.sum(self.last_energy_per_cell)
                else:
                    energy = None
            else:
                energy = None

            if compute_total_energy and energy_compute_cooldown == 0:
                with Timer() as timer:
                    try:
                        context_cube, context_cube_mask = state_to_context_cube(self.mic_points.all_points)
                        res = self.energy_func(context_cube.to(self.device), context_cube_mask.to(self.device))
                        energy = float(res['total_energy'].detach().cpu())
                    except Exception as e:
                        logging.warning(f"compute total energy failed: {e}\n"
                                        f"trying again in 100 steps")
                        energy_compute_cooldown = 10
                        energy = None
                energy_compute_time = timer()
            else:
                energy_compute_time = None
                energy_compute_cooldown = max(0, energy_compute_cooldown - 1)

            if verbose > 1:
                pbar.set_postfix({
                    'T': self.temperature,
                    'N': self.mic_points.__len__(),
                    'E': energy,
                    'dt': np.mean(self.diffusion_dt_t.detach().cpu().numpy())
                })
            step_log = {}
            if log:
                step_log.update({
                    'temperature': self.temperature,
                    'dt': self.diffusion_dt_t.detach().cpu().numpy(),
                    'n_points': self.mic_points.__len__(),
                    'energy': energy,
                    'energy_compute_time': energy_compute_time
                })
            if kernel_logs:
                step_log['kernel_log'] = k_log
            if state_logs:
                step_log['state'] = self.mic_points.all_points

            if len(step_log) > 0:
                append_lists_in_dict(run_log, step_log)
        if log or state_logs or kernel_logs:
            return run_log

    @property
    def current_state(self):
        return self.mic_points.all_points
