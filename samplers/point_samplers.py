import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

import numpy as np
from numpy.random import Generator
from scipy.ndimage import gaussian_filter, gaussian_filter1d, distance_transform_edt
from skimage.segmentation import watershed
from torch import Tensor

from base.array_operations import safe_softmax
from base.mappings import ValueMapping
from base.state_ops import maps_local_max_state, maps_sample_state, clip_state_to_bounds_np
from samplers.mic_set_utils import pad_to_size, random_to_grid_samples, sample_marks_values
from samplers.mic_sets import cell_ids_from_coordinates

DENSITY_FUZZ = 1e-8


class BasePointSampler(ABC):

    @abstractmethod
    def sample(self, current_cells: np.ndarray, rng: Generator, return_density: bool = False) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def cell_density(self) -> Union[list, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def get_point_density(self, current_cells: np.ndarray, points: np.ndarray, cell_normalized: bool = False):
        raise NotImplementedError

    @abstractmethod
    def get_density(self, norm_per_cell: bool):
        raise NotImplementedError


class SamplerMix(BasePointSampler):

    def __init__(self, weights, samplers: List[BasePointSampler]):
        self.samplers = samplers
        weights = np.array(weights)
        self.weights = weights / np.sum(weights)

    def sample(self, current_cells: np.ndarray, rng: Generator, return_density: bool = False) -> np.ndarray:
        s_index = rng.choice(range(len(self.samplers)), p=self.weights)
        return self.samplers[s_index].sample(
            current_cells=current_cells, rng=rng, return_density=return_density
        )

    @property
    def cell_density(self) -> Union[list, np.ndarray]:
        return np.sum(np.stack([
            w * np.array(s.cell_density) for w, s in zip(self.weights, self.samplers)
        ], axis=0), axis=0)

    def get_point_density(self, current_cells: np.ndarray, points: np.ndarray, cell_normalized: bool = False):
        return np.sum(np.stack([
            w * s.get_point_density(
                current_cells=current_cells, points=points, cell_normalized=cell_normalized
            ) for w, s in zip(self.weights, self.samplers)
        ], axis=0), axis=0)

    def get_density(self, norm_per_cell: bool):
        return np.sum(np.stack([
            w * s.get_density(norm_per_cell=norm_per_cell) for w, s in zip(self.weights, self.samplers)
        ], axis=0), axis=0)


@dataclass
class DensityPointSampler(BasePointSampler):
    bounds: np.ndarray
    position_density: np.ndarray
    mappings: List[ValueMapping]
    mark_densities: List[np.ndarray]
    sampling_temperature: float
    debug_mode: bool = False
    expit: bool = False
    density_w_marks: bool = True

    def __post_init__(self):
        self.n_cells = len(self.bounds)
        shape = self.position_density.shape
        self.shape = shape
        assert len(shape) == 2
        grid = np.stack(np.mgrid[:shape[0], :shape[1]], axis=-1)

        self.position_density = self.position_density.astype(np.double)
        self.mark_densities = [m.astype(np.double)
                               for m in self.mark_densities]

        assert not np.any(np.isnan(self.position_density))
        assert all([not np.any(np.isnan(m)) for m in self.mark_densities])

        slice_size = np.max(np.diff(self.bounds, axis=1), axis=0)[0]

        eps = 1e-8

        if self.sampling_temperature != 1.0:
            # self.position_density = np.power(self.position_density, 1 / self.sampling_temperature)
            # if np.any(np.isnan(self.position_density)):
            #     raise RuntimeError(f"nan in position_density**(1/sampling_temperature), "
            #                        f"sampling_temperature might be too low {self.sampling_temperature})")
            # norm = np.sum(self.position_density)
            # if norm == 0.0:
            #     raise RuntimeError(f"temperature scaled position_density has sum zero : "
            #                        f"sampling_temperature (T={self.sampling_temperature}) might be too low")
            # self.position_density = self.position_density / norm
            position_potential = -np.log(self.position_density + eps)
            new_position_potential = position_potential / self.sampling_temperature
            assert not np.any(np.isnan(new_position_potential))
            self.position_density = safe_softmax(
                -new_position_potential, axis=(0, 1))
            assert not any([np.any(np.isnan(d)) for d in self.mark_densities])
            self.mark_densities = [safe_softmax(np.log(d + eps) / self.sampling_temperature, axis=0)
                                   for d in self.mark_densities]
        else:
            # ensure sum =1
            self.position_density = self.position_density / \
                np.sum(self.position_density, keepdims=True)
            self.mark_densities = [
                m / m.sum(axis=0, keepdims=True) for m in self.mark_densities]

        if self.expit:
            logging.info("using expit for point sampling")
            self.position_density = self.position_density / \
                (1 + self.position_density)
            self.mark_densities = [m / (1 + m) for m in self.mark_densities]
            self.position_density = self.position_density / \
                np.sum(self.position_density, keepdims=True)
            self.mark_densities = [
                m / m.sum(axis=0, keepdims=True) for m in self.mark_densities]

        # position_birth_density sums to 1
        s = self.position_density.sum()
        if not np.all(np.abs(1 - s) < DENSITY_FUZZ):
            raise RuntimeError(f"position_density does not sum to 1 over all pixels "
                               f"(sum={s}) type {self.position_density.dtype}")

        # mark_birth_densities sums to 1
        for i, d in enumerate(self.mark_densities):
            v = d.sum(axis=0)
            if not np.all(np.abs(1 - v) < DENSITY_FUZZ):
                raise RuntimeError(f"mark_densities for mark {self.mappings[i].name} does not sum to 1 over all bins "
                                   f"(sum={v[~(np.abs(1 - v) < 1e-6)]}) type {d.dtype}")

        if self.debug_mode:
            logging.info(
                f"birth position density set w/ px. avg.:{np.mean(self.position_density):.1e} | "
                f"tot: {np.sum(self.position_density):.2f}")
            for m, d in zip(self.mappings, self.mark_densities):
                logging.info(
                    f"birth {m.name} mark density set w/ value avg.:{np.mean(d):.1e} | "
                    f"avg tot per pixel: {np.sum(d, axis=0).mean():.2f}")

        # self.position_density = np.clip(self.position_density, np.finfo(self.position_density.dtype).eps, np.inf)

        self.position_density_per_cell_norm = np.empty_like(
            self.position_density)
        for tl, br in self.bounds:
            cell = self.position_density[tl[0]:br[0], tl[1]:br[1]]
            self.position_density_per_cell_norm[tl[0]                                                :br[0], tl[1]:br[1]] = cell / np.sum(cell)

        sliced_density = np.stack(
            [pad_to_size(self.position_density[tl[0]:br[0], tl[1]                         :br[1]], slice_size) for tl, br in self.bounds],
            axis=0)
        sliced_grid = np.stack([pad_to_size(grid[tl[0]:br[0], tl[1]:br[1]], slice_size) for tl, br in self.bounds],
                               axis=0)

        self.flat_density = sliced_density.reshape((self.n_cells, -1))
        self._cell_densities = np.sum(self.flat_density, axis=-1)
        norm = np.sum(self.flat_density, axis=-1, keepdims=True)
        if np.any(norm == 0.0):
            # raise RuntimeError(f"found zero norm density {norm}")
            self.flat_density = np.where(norm == 0.0, 1.0, self.flat_density)
            norm = np.sum(self.flat_density, axis=-1, keepdims=True)
            assert not np.any(norm == 0.0)
        flat_p_norm = self.flat_density / norm
        self.flat_p_cumdist: np.ndarray = np.cumsum(flat_p_norm, axis=-1)
        self.flat_grid: np.ndarray = sliced_grid.reshape((self.n_cells, -1, 2))

        self.n_marks = len(self.mappings)
        if self.n_marks > 0:
            self._marks_density_maps = np.stack(self.mark_densities, axis=0)

            assert self._marks_density_maps.shape[1] == self.mappings[0].n_classes

            self._marks_density_maps = self._marks_density_maps / np.sum(self._marks_density_maps, axis=1,
                                                                         keepdims=True)
            self._marks_cum_density_maps = np.moveaxis(
                np.cumsum(self._marks_density_maps, axis=1), 1, -1)
            self._mappings = np.stack(
                [m.feature_mapping for m in self.mappings])
        else:
            self._marks_cum_density_maps = np.array([])

        self.min_bounds = np.array(
            [0.0, 0.0] + [m.v_min for m in self.mappings])
        self.max_bounds = np.array(
            list(self.shape) + [m.v_max for m in self.mappings])

    def get_density(self, norm_per_cell: bool = True):
        if norm_per_cell:
            return self.position_density_per_cell_norm
        else:
            return self.position_density

    def sample(self, current_cells: np.ndarray, rng: Generator, **kwargs):
        n_cells: int = current_cells.shape[0]
        rd = rng.random((n_cells, 1))
        grid_samples, _ = random_to_grid_samples(
            rd=rd,
            flat_p_cumdist=self.flat_p_cumdist[current_cells],
            flat_grid=self.flat_grid[current_cells],
            flat_densities=self.flat_density[current_cells],
            n_cells=n_cells
        )
        # assert grid_samples < np.array(self.shape)
        grid_samples = grid_samples + rng.random(size=grid_samples.shape)
        if np.any(grid_samples < self.min_bounds[:2]) or np.any(grid_samples > self.max_bounds[:2]):
            oob_points = np.any(grid_samples < self.min_bounds[:2], axis=-1) | \
                np.any(grid_samples > self.max_bounds[:2], axis=-1)
            e = f"generated points out of range:\n" + \
                f"{grid_samples[oob_points]}\n" + \
                f"with bounds {self.min_bounds[:2]}-{self.max_bounds[:2]}\n" + \
                f">>> CLIPPING !"
            if self.debug_mode:
                raise RuntimeError(e)
            else:
                logging.error(e)

        grid_samples = np.clip(
            grid_samples, self.min_bounds[:2], self.max_bounds[:2])
        if self.n_marks > 0:
            rd_marks = rng.random((self.n_marks, n_cells))
            marks = sample_marks_values(
                marks_cum_density_map=self._marks_cum_density_maps,
                positions=np.clip(grid_samples.astype(
                    int), (0, 0), (self.shape[0] - 1, self.shape[1] - 1)),
                rd_values=rd_marks,
                mapping=self._mappings
            ).T
            points = np.concatenate((grid_samples, marks), axis=-1)
        else:
            points = grid_samples
        # marks = rng.uniform(self.marks_min_bound, self.marks_max_bound, size=(grid_samples.shape[0], self.n_marks))
        try:
            assert np.all(points <= self.max_bounds)
            assert np.all(points >= self.min_bounds)
        except AssertionError:
            logging.error(f"POINTS OUT OF BOUND"
                          f"\n{points=}"
                          f"\n{self.min_bounds=}"
                          f"\n{self.max_bounds=}")

        return points

    def get_point_density(self, current_cells: np.ndarray, points: Union[Tensor, np.ndarray],
                          cell_normalized: bool = False):

        if type(points) is Tensor:
            points = points.cpu().numpy()
        pos = np.clip(points[..., :2].astype(int), (0, 0),
                      (self.shape[0] - 1, self.shape[1] - 1))

        if cell_normalized:
            pos_d = self.position_density_per_cell_norm[pos[..., 0], pos[..., 1]]
        else:
            pos_d = self.position_density[pos[..., 0], pos[..., 1]]
        if self.density_w_marks:
            marks_classes = [m.value_to_class(
                points[..., 2 + i]) for i, m in enumerate(self.mappings)]
            mark_d = np.stack([md[mc, pos[..., 0], pos[..., 1]] for mc, md in zip(marks_classes, self.mark_densities)],
                              axis=0)
            return pos_d * np.prod(mark_d, axis=0)
        else:
            return pos_d

    @property
    def cell_density(self) -> Union[list, np.ndarray]:
        return self._cell_densities

    @property
    def total_density(self) -> float:
        return np.sum(self._cell_densities)


@dataclass
class LocalMaxPointSampler(BasePointSampler):
    bounds: np.ndarray
    position_density: np.ndarray
    mappings: List[ValueMapping]
    mark_densities: List[np.ndarray]
    sampling_temperature: float
    n_proposals_per_cell: int
    debug_mode: bool = False
    local_max_distance: int = 2
    local_max_thresh: float = 0.1
    density_w_marks: bool = False

    def __post_init__(self):
        self.n_cells = len(self.bounds)
        shape = self.position_density.shape
        self.shape = shape
        assert len(shape) == 2
        grid = np.stack(np.mgrid[:shape[0], :shape[1]], axis=-1)

        self.position_density = self.position_density.astype(np.double)
        self.mark_densities = [m.astype(np.double)
                               for m in self.mark_densities]

        assert not np.any(np.isnan(self.position_density))
        assert all([not np.any(np.isnan(m)) for m in self.mark_densities])

        slice_size = np.max(np.diff(self.bounds, axis=1), axis=0)[0]

        eps = 1e-8

        # ensure max position =1
        self.position_heatmap = self.position_density / \
            np.max(self.position_density, keepdims=True)

        proposal_states = maps_local_max_state(
            position_map=self.position_heatmap, mark_maps=self.mark_densities,
            mappings=self.mappings, local_max_distance=2, local_max_thresh=0.1
        )
        xy = proposal_states[:, [0, 1]].astype(int)
        # slice proposals into cells
        proposals_cell_id = cell_ids_from_coordinates(
            coordinates=xy, bounds=self.bounds
        )
        n_cells = len(self.bounds)
        state_size = 2 + len(self.mappings)
        self.proposals_per_cell = np.empty(
            (n_cells, self.n_proposals_per_cell, state_size))
        rng = np.random.default_rng(0)
        for cell_id in range(len(self.bounds)):
            points = proposal_states[cell_id == proposals_cell_id]

            if len(points) == 0:
                tl, br = self.bounds[cell_id]
                cell_density = self.position_density[tl[0]:br[0], tl[1]:br[1]]
                points = maps_sample_state(
                    position_density=cell_density,
                    mark_maps=[m[:, tl[0]:br[0], tl[1]:br[1]]
                               for m in self.mark_densities],
                    mappings=self.mappings,
                    n_points=self.n_proposals_per_cell, rng=rng,
                    argmax_marks=True
                )
                points[:, :2] = points[:, :2] + tl
            elif len(points) < self.n_proposals_per_cell:
                xy = points[:, [0, 1]].astype(int)
                scores = self.position_heatmap[xy[:, 0], xy[:, 1]]
                arg_sort = np.argsort(-scores)
                points = np.resize(
                    points[arg_sort], (self.n_proposals_per_cell, state_size))
            elif len(points) > self.n_proposals_per_cell:
                xy = points[:, [0, 1]].astype(int)
                scores = self.position_heatmap[xy[:, 0], xy[:, 1]]
                arg_sort = np.argsort(scores)
                points = points[arg_sort][-self.n_proposals_per_cell:]
            assert len(points) == self.n_proposals_per_cell

            self.proposals_per_cell[cell_id] = points

        self.sigma_pos = np.sqrt(self.sampling_temperature * 0.5)
        self.sigma_marks = [self.sigma_pos *
                            (m.range / m.n_classes) for m in self.mappings]
        self.sigma_state = np.array(
            [self.sigma_pos, self.sigma_pos] + self.sigma_marks)

        # remake position density
        self.position_density = np.zeros(shape)
        xy = np.concatenate([p[:, :2].astype(int)
                            for p in self.proposals_per_cell], axis=0)
        for x, y in zip(xy[:, 0], xy[:, 1]):
            self.position_density[x, y] += 1
        # self.position_density[xy[:, 0], xy[:, 1]] = 1
        self.position_density = gaussian_filter(
            self.position_density, sigma=self.sigma_pos)

        # remake marks densities
        if self.density_w_marks:
            # extract density for each sampled points
            n_points = len(xy)
            h, w = self.position_density.shape

            WATERSHED_APPROX = True
            begin = time.perf_counter()
            for k, mapping in enumerate(self.mappings):
                mark_classes = np.concatenate(
                    [mapping.value_to_class(p[:, 2 + k]).astype(int) for p in self.proposals_per_cell], axis=0)
                d_per_point = np.zeros((n_points, mapping.n_classes))
                d_per_point[np.arange(n_points), mark_classes] = 1
                d_per_point = gaussian_filter1d(
                    d_per_point, sigma=self.sigma_marks[k], axis=-1)

                if WATERSHED_APPROX:
                    seed = np.zeros((h, w), dtype=int)
                    for j, (x, y) in enumerate(zip(xy[:, 0], xy[:, 1])):
                        try:
                            seed[x, y] = k + 1
                        except IndexError as e:
                            pass
                    distance = distance_transform_edt(1 - (seed > 0))
                    voronoi = watershed(distance, seed) - 1
                    mark_density_map = d_per_point[voronoi]
                else:
                    mark_density_map = np.zeros(
                        self.position_density.shape + (mapping.n_classes,))
                    # todo make this FASTER
                    for j, (x, y) in enumerate(zip(xy[:, 0], xy[:, 1])):
                        one_point_map = np.zeros_like(self.position_density)
                        one_point_map[x, y] = 1
                        one_point_map = gaussian_filter(
                            one_point_map, sigma=self.sigma_pos)
                        mark_density_map += one_point_map.reshape((h, w, 1)) * d_per_point[j].reshape(
                            (1, 1, mapping.n_classes))
            end = time.perf_counter()
            logging.info(f"build mark density maps in {end - begin:.1e}s")
        # self.mark_densities = [  ]

        self.position_density = self.position_density / \
            np.sum(self.position_density, keepdims=True)
        self.mark_densities = [
            m / m.sum(axis=0, keepdims=True) for m in self.mark_densities]

        # position_birth_density sums to 1
        s = self.position_density.sum()
        if not np.all(np.abs(1 - s) < DENSITY_FUZZ):
            raise RuntimeError(f"position_density does not sum to 1 over all pixels "
                               f"(sum={s}) type {self.position_density.dtype}")

        # mark_birth_densities sums to 1
        for i, d in enumerate(self.mark_densities):
            v = d.sum(axis=0)
            if not np.all(np.abs(1 - v) < DENSITY_FUZZ):
                raise RuntimeError(f"mark_densities for mark {self.mappings[i].name} does not sum to 1 over all bins "
                                   f"(sum={v[~(np.abs(1 - v) < 1e-6)]}) type {d.dtype}")

        if self.debug_mode:
            logging.info(
                f"birth position density set w/ px. avg.:{np.mean(self.position_density):.1e} | "
                f"tot: {np.sum(self.position_density):.2f}")
            for m, d in zip(self.mappings, self.mark_densities):
                logging.info(
                    f"birth {m.name} mark density set w/ value avg.:{np.mean(d):.1e} | "
                    f"avg tot per pixel: {np.sum(d, axis=0).mean():.2f}")

        # self.position_density = np.clip(self.position_density, np.finfo(self.position_density.dtype).eps, np.inf)

        self.position_density_per_cell_norm = np.empty_like(
            self.position_density)
        for tl, br in self.bounds:
            cell = self.position_density[tl[0]:br[0], tl[1]:br[1]]
            self.position_density_per_cell_norm[tl[0]                                                :br[0], tl[1]:br[1]] = cell / np.sum(cell)

        sliced_density = np.stack(
            [pad_to_size(self.position_density[tl[0]:br[0], tl[1]                         :br[1]], slice_size) for tl, br in self.bounds],
            axis=0)
        sliced_grid = np.stack([pad_to_size(grid[tl[0]:br[0], tl[1]:br[1]], slice_size) for tl, br in self.bounds],
                               axis=0)

        self.flat_density = sliced_density.reshape((self.n_cells, -1))
        self._cell_densities = np.sum(self.flat_density, axis=-1)
        norm = np.sum(self.flat_density, axis=-1, keepdims=True)
        if np.any(norm == 0.0):
            # raise RuntimeError(f"found zero norm density {norm}")
            self.flat_density = np.where(norm == 0.0, 1.0, self.flat_density)
            norm = np.sum(self.flat_density, axis=-1, keepdims=True)
            assert not np.any(norm == 0.0)
        flat_p_norm = self.flat_density / norm
        self.flat_p_cumdist: np.ndarray = np.cumsum(flat_p_norm, axis=-1)
        self.flat_grid: np.ndarray = sliced_grid.reshape((self.n_cells, -1, 2))

        self.n_marks = len(self.mappings)
        if self.n_marks > 0:
            self._marks_density_maps = np.stack(self.mark_densities, axis=0)

            assert self._marks_density_maps.shape[1] == self.mappings[0].n_classes

            self._marks_density_maps = self._marks_density_maps / np.sum(self._marks_density_maps, axis=1,
                                                                         keepdims=True)
            self._marks_cum_density_maps = np.moveaxis(
                np.cumsum(self._marks_density_maps, axis=1), 1, -1)
            self._mappings = np.stack(
                [m.feature_mapping for m in self.mappings])
        else:
            self._marks_cum_density_maps = np.array([])

        self.state_min = np.array(
            [0.0, 0.0] + [m.v_min for m in self.mappings])
        self.state_max = np.array(
            list(self.shape) + [m.v_max for m in self.mappings])
        self.state_cyclic = np.array(
            [False, False] + [m.is_cyclic for m in self.mappings])

    def get_density(self, norm_per_cell: bool = True):
        if norm_per_cell:
            return self.position_density_per_cell_norm
        else:
            return self.position_density

    def sample(self, current_cells: np.ndarray, rng: Generator, **kwargs):
        n_cells: int = current_cells.shape[0]
        rd = rng.integers(low=0, high=self.n_proposals_per_cell, size=n_cells)

        points = self.proposals_per_cell[current_cells][np.arange(
            n_cells), rd].astype(float)

        points = points + rng.normal(size=points.shape) * self.sigma_state

        points[:, :2] = np.clip(
            points[:, :2], self.bounds[current_cells, 0], self.bounds[current_cells, 1] - 1e-8)
        points = clip_state_to_bounds_np(
            state=points, min_bound=self.state_min, max_bound=self.state_max, cyclic=self.state_cyclic
        )
        # marks = rng.uniform(self.marks_min_bound, self.marks_max_bound, size=(grid_samples.shape[0], self.n_marks))
        try:
            assert np.all(points <= self.state_max)
            assert np.all(points >= self.state_min)
        except AssertionError:
            logging.error(f"POINTS OUT OF BOUND"
                          f"\n{points=}"
                          f"\n{self.state_min=}"
                          f"\n{self.state_max=}")

        return points

    def get_point_density(self, current_cells: np.ndarray, points: Union[Tensor, np.ndarray],
                          cell_normalized: bool = False):

        if type(points) is Tensor:
            points = points.cpu().numpy()
        pos = np.clip(points[..., :2].astype(int), (0, 0),
                      (self.shape[0] - 1, self.shape[1] - 1))

        if cell_normalized:
            pos_d = self.position_density_per_cell_norm[pos[..., 0], pos[..., 1]]
        else:
            pos_d = self.position_density[pos[..., 0], pos[..., 1]]
        if self.density_w_marks:
            marks_classes = [m.value_to_class(
                points[..., 2 + i]) for i, m in enumerate(self.mappings)]
            mark_d = np.stack([md[mc, pos[..., 0], pos[..., 1]] for mc, md in zip(marks_classes, self.mark_densities)],
                              axis=0)
            return pos_d * np.prod(mark_d, axis=0)
        else:
            return pos_d

    @property
    def cell_density(self) -> Union[list, np.ndarray]:
        return self._cell_densities

    @property
    def total_density(self) -> float:
        return np.sum(self._cell_densities)
