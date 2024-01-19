import logging
import warnings
from typing import Tuple, Union, Iterable
import numpy as np
import torch
from matplotlib import patches
from torch import Tensor
import matplotlib.pyplot as plt


def show_one_mic_cell(set_bounds, ax: plt.Axes, **rect_kwargs):
    tl, br = np.flip(set_bounds, axis=-1)
    width, height = br[0] - tl[0], br[1] - tl[1]
    rect = patches.Rectangle(tl, width, height, **rect_kwargs)
    ax.add_patch(rect)


def cell_ids_from_coordinates(coordinates: np.ndarray, bounds: np.ndarray, times=None, cells_frame=None,
                              temporal=False):
    coordinates = coordinates.reshape((-1, 1, 2))
    bounds = bounds.reshape((1, -1, 2, 2))
    if temporal:
        assert times is not None
        times = times.reshape((-1, 1))
        k = np.where(
            np.all(coordinates < bounds[:, :, 1], axis=-1) &
            np.all(coordinates >= bounds[:, :, 0], axis=-1) &
            (times == cells_frame.reshape((1, -1)))
        )[1]
    else:
        k = np.where(
            np.all(coordinates < bounds[:, :, 1], axis=-1) &
            np.all(coordinates >= bounds[:, :, 0], axis=-1)
        )[1]
    if len(k) != len(coordinates):
        logging.error(f"some points are out of bound in \n{coordinates}")
        raise RuntimeError("Point out of bound")
    return k


def make_cell_bounds(cell_size: int, support_shape: Tuple[int, int], n_frames: int, temporal: bool,
                     constant_cell_size: bool, return_ndivs: bool = False):
    n_x = int(np.ceil(support_shape[0] / cell_size))
    n_y = int(np.ceil(support_shape[1] / cell_size))
    if not constant_cell_size:
        x_divs = np.linspace(0, support_shape[0], num=n_x + 1, dtype=int)
        y_divs = np.linspace(0, support_shape[1], num=n_y + 1, dtype=int)
    else:
        x_divs = np.arange(n_x + 1) * cell_size
        y_divs = np.arange(n_y + 1) * cell_size

        x_pad = x_divs[-1] - support_shape[0]
        y_pad = y_divs[-1] - support_shape[1]
        if x_pad != 0 or y_pad != 0:
            logging.warning(
                f"image of shape {support_shape} cannot be divided into size {cell_size} patches: padded by ({x_pad},{y_pad})")

    n_cells_per_frame = n_x * n_y
    if temporal:
        n_cells = n_cells_per_frame * n_frames
    else:
        n_cells = n_cells_per_frame

    cells_bounds = np.empty((n_cells, 2, 2), dtype=int)
    k = 0
    for t in range(n_frames):
        for i in range(n_x):
            for j in range(n_y):
                tl_corner = (x_divs[i], y_divs[j])
                br_corner = (x_divs[i + 1], y_divs[j + 1])
                cells_bounds[k] = np.array([tl_corner, br_corner])
                k += 1
    if return_ndivs:
        return cells_bounds, n_x, n_y
    return cells_bounds


class TorchTimeMicSets:
    FAST_ADD = True

    def __init__(self, support_shape: Tuple[int, int],
                 spatial_interaction_distance: float, move_distance: float,
                 constant_cell_size: bool, device: torch.device,
                 n_marks: int, temporal: bool, temporal_interaction_distance: int = 0, n_frames: int = 1,
                 max_points_per_cell: int = None, max_points_per_pixel: float = None):

        self.n_dims = n_marks + 2

        cell_size = 2 * spatial_interaction_distance + 2 * move_distance
        logging.info(f"{cell_size=}")
        self.temporal = temporal
        if not self.temporal:
            assert n_frames == 1
            assert temporal_interaction_distance == 0
        self.n_frames = n_frames
        self.time_interval = 2 * temporal_interaction_distance + 1
        self.time_span = 2 * self.time_interval - 1
        self.support_shape = support_shape
        self.device = device
        # self.n_x = int(np.ceil(support_shape[0] / cell_size))
        # self.n_y = int(np.ceil(support_shape[1] / cell_size))
        # if not constant_cell_size:
        #     x_divs = np.linspace(0, support_shape[0], num=self.n_x + 1, dtype=int)
        #     y_divs = np.linspace(0, support_shape[1], num=self.n_y + 1, dtype=int)
        # else:
        #     x_divs = np.arange(self.n_x + 1) * cell_size
        #     y_divs = np.arange(self.n_y + 1) * cell_size
        #
        #     x_pad = x_divs[-1] - support_shape[0]
        #     y_pad = y_divs[-1] - support_shape[1]
        #     if x_pad != 0 or y_pad != 0:
        #         logging.warning(
        #             f"image of shape {support_shape} cannot be divided into size {cell_size} patches: padded by ({x_pad},{y_pad})")

        self.cells_bounds, self.n_x, self.n_y = make_cell_bounds(cell_size=cell_size, support_shape=self.support_shape,
                                                                 n_frames=self.n_frames, temporal=self.temporal,
                                                                 constant_cell_size=constant_cell_size,
                                                                 return_ndivs=True)
        self.n_cells_per_frame = self.n_x * self.n_y
        n_cells = len(self.cells_bounds)
        if max_points_per_cell is not None:
            self.max_pts = max_points_per_cell
        elif max_points_per_pixel is not None:
            cell_area = cell_size * cell_size
            self.max_pts = max(0, int(cell_area * max_points_per_pixel))
            logging.info(f"set max_points_per_cell to {self.max_pts} "
                         f"(=cell_area({cell_area})*max_points_per_pixel({max_points_per_pixel}))")
        else:
            raise RuntimeError(
                "either max_points_per_set or max_points_per_pixel should be set")

        # self.n_cells_per_frame = self.n_x * self.n_y
        if self.temporal:
            # n_cells = self.n_cells_per_frame * self.n_frames
            n_mic_sets = min(2, self.n_x) * \
                min(2, self.n_y) * self.time_interval
            self.iterations_multiplier = n_mic_sets / \
                (self.n_cells_per_frame * self.n_frames)
        else:
            # n_cells = self.n_cells_per_frame
            n_mic_sets = min(2, self.n_x) * min(2, self.n_y)
            self.iterations_multiplier = n_mic_sets / self.n_cells_per_frame

        self.cells = torch.zeros((n_cells + 1, self.max_pts, self.n_dims),
                                 device=self.device)  # shape (n_cells,max_points,n_dims)
        self.masks = torch.zeros((n_cells + 1, self.max_pts),
                                 device=self.device).bool()  # shape (time_steps,n_cells,max_points)
        # adding one extra empty set to hand out when querying oob sets
        # shape (time_steps,n_cells))
        self.cells_point_number = np.zeros(n_cells, dtype=int)
        # self.cells_bounds = np.empty((n_cells, 2, 2),
        #                              dtype=int)  # shape (n_cells,2,2) each is [[tl_x,tl_y],[br_x,br_y]]
        self.cells_frame = np.empty(n_cells, dtype=int)
        self.cells_set = np.empty(n_cells, dtype=int)  # shape (n_cells,)
        self.n_sets = 4 * self.time_interval
        cells_indices_per_set = {c: [] for c in range(self.n_sets)}
        k = 0
        for t in range(self.n_frames):
            for i in range(self.n_x):
                for j in range(self.n_y):
                    # tl_corner = (x_divs[i], y_divs[j])
                    # br_corner = (x_divs[i + 1], y_divs[j + 1])
                    cell_set = (i % 2) + 2 * (j % 2) + 4 * \
                        (t % self.time_interval)
                    cells_indices_per_set[cell_set].append(k)
                    self.cells_set[k] = cell_set
                    # self.cells_bounds[k] = np.array([tl_corner, br_corner])
                    self.cells_frame[k] = t
                    k += 1

        self.cells_indices_per_set = {k: np.array(
            v) for k, v in cells_indices_per_set.items()}
        self.nb_cells = n_cells
        self.cells_area = np.prod(
            np.diff(self.cells_bounds, axis=1)[:, 0], axis=-1)
        self.available_sets = [
            k for k, v in self.cells_indices_per_set.items() if len(v) > 0]

    def get_sim_steps(self, steps_per_pixel, cell_subsampling_fac: float = 1.0):
        param_space_multiplier = np.prod(self.support_shape) * self.n_frames
        n_steps_unscaled = steps_per_pixel * param_space_multiplier
        if cell_subsampling_fac == 1.0:
            n_steps = max(
                1, int(self.iterations_multiplier * n_steps_unscaled))
            logging.info(f"{steps_per_pixel:.2f} steps/px "
                         f"-> {n_steps_unscaled} unique steps (x{param_space_multiplier:.1e})"
                         f"-> {n_steps} parallel steps (x{self.iterations_multiplier:.1e})")
        else:
            n_steps_nss = self.iterations_multiplier * n_steps_unscaled
            n_steps = max(1, int(n_steps_nss / cell_subsampling_fac))
            logging.info(f"{steps_per_pixel:.2f} steps/px "
                         f"-> {n_steps_unscaled} unique steps (x{param_space_multiplier:.1e})"
                         f"-> {n_steps_nss} parallel steps (x{self.iterations_multiplier:.1e})"
                         f"-> {n_steps} subsampled parallel steps (x{1 / cell_subsampling_fac:.1e})")
        return n_steps

    @property
    def mean_cell_size(self):
        size_x, size_y = np.mean(
            np.diff(self.cells_bounds, axis=1)[:, 0], axis=0)
        return size_x, size_y

    def _find_corresponding_cell(self, coordinates: Tensor, times=None):

        return cell_ids_from_coordinates(
            coordinates=coordinates, bounds=self.cells_bounds, times=times, cells_frame=self.cells_frame,
            temporal=self.temporal
        )

    def display_mic_set(self, ax: plt.Axes, frame: int = None, cmap=None, label=False, **kwargs):
        sets_colors = plt.get_cmap(cmap) if cmap is not None else None
        if self.temporal:
            assert frame is not None
        else:
            frame = 0
        for i, (mic_set, bounds, f) in enumerate(zip(self.cells_set, self.cells_bounds, self.cells_frame)):
            if f == frame:
                if sets_colors is not None:
                    kwargs = {**kwargs, 'color': sets_colors(mic_set)}
                show_one_mic_cell(bounds, ax, **kwargs)
                if label:
                    tl, br = np.flip(bounds, axis=-1)
                    if sets_colors is not None:
                        color = sets_colors(mic_set)
                    else:
                        color = 'black'
                    ax.text(tl[0], tl[1], s=f"S{mic_set}", color=color, va='top', alpha=0.5, size='x-small',
                            clip_on=True)

    def clear_cells(self, cell_indices: np.ndarray):
        self.masks[cell_indices] = False
        self.cells_point_number[cell_indices] = 0

    def add_one_point(self, point: np.ndarray, time=None):
        if self.temporal:
            assert time is not None
        k = self._find_corresponding_cell(point[:2], times=time)[0]
        new_index = self.cells_point_number[k]
        if new_index >= self.max_pts:
            warnings.warn(
                f"no room left for {point} in cell {k} at {self.cells_bounds[k]}: adding SKIPPED")
            return
        self.cells[k, new_index] = torch.tensor(point)
        self.masks[k, new_index] = True
        self.cells_point_number[k] += 1

    def add_points(self, points: Union[Tensor, np.ndarray], cell_indices: np.ndarray = None, times: np.ndarray = None,
                   raise_full_cell_error: bool = False):
        # points of shape (N,D)
        if type(points) is Tensor:
            coordinates = points[:, :2].cpu().numpy()
            points = points.float()
            points = points.to(self.device)
        else:
            coordinates = points[:, :2]
            points = torch.tensor(points, device=self.device).float()

        if self.temporal:
            assert times is not None
        if len(points) == 0:
            return
        if cell_indices is None:
            cell_indices = self._find_corresponding_cell(
                coordinates, times=times)
        unique_cells, cells_counts_inv, cells_counts = np.unique(
            cell_indices, return_counts=True, return_inverse=True)
        if np.all(cells_counts <= 1):  # if all points are in unique cells
            new_index = self.cells_point_number[cell_indices]
            if np.any(new_index >= self.max_pts):
                full_cells = np.argwhere(new_index >= self.max_pts)[:, 0]
                err_txt = f"no room left for point(s): in cells {cell_indices[full_cells]}: SKIPPING !"
                if raise_full_cell_error:
                    raise RuntimeError(err_txt)
                else:
                    warnings.warn(err_txt)
                    non_full_cells = np.argwhere(
                        new_index < self.max_pts)[:, 0]
                    new_index = new_index[non_full_cells]
                    points = points[non_full_cells]
                    cell_indices = cell_indices[non_full_cells]

            self.cells[cell_indices, new_index] = points
            self.masks[cell_indices, new_index] = True
            self.cells_point_number[cell_indices] += 1
        elif self.FAST_ADD:
            arg = np.argsort(cell_indices)
            cell_indices_srt = cell_indices[arg]
            points_srt = points[arg]
            cells_counts_inv_srt = cells_counts_inv[arg]
            # assert unique_cells is sorted
            assert not np.any(np.diff(unique_cells) < 0)
            cells_counts_cum = np.cumsum(cells_counts)
            ramp = np.arange(len(points)) - np.pad(cells_counts_cum,
                                                   (1, 0))[cells_counts_inv_srt]

            next_index_per_unique_cell = self.cells_point_number[unique_cells]
            next_index_per_point = ramp + \
                next_index_per_unique_cell[cells_counts_inv_srt]

            oob = next_index_per_point >= self.max_pts
            if not np.any(oob):  # all points fit
                self.cells[cell_indices_srt, next_index_per_point] = points_srt
                self.masks[cell_indices_srt, next_index_per_point] = True
                self.cells_point_number[unique_cells] += cells_counts
            else:
                err_txt = f"no room left for {np.sum(oob)} point(s): in some mic cells : skipping those !"
                if raise_full_cell_error:
                    raise RuntimeError(err_txt)
                else:
                    warnings.warn(err_txt)
                    ib = ~oob
                    self.cells[cell_indices_srt[ib],
                               next_index_per_point[ib]] = points_srt[ib]
                    self.masks[cell_indices_srt[ib],
                               next_index_per_point[ib]] = True

                    u, c = np.unique(cell_indices_srt[ib], return_counts=True)
                    self.cells_point_number[u] += c
        else:
            logging.info(
                "Adding more than one point per cell, falling back to loop")
            if self.temporal:
                for p, t in zip(points, times):
                    self.add_one_point(p, t)
            else:
                for p in points:
                    self.add_one_point(p)

    def update_points(self, points: Tensor, cell_indices: np.ndarray, points_indices: np.ndarray, times=None):
        # if all points are in unique sets
        if np.all(np.unique(cell_indices, return_counts=True)[1] <= 1):
            new_cell_indices = self._find_corresponding_cell(
                points[:, :2].cpu().numpy(), times)
            oob = new_cell_indices != cell_indices
            self.cells[cell_indices[~oob], points_indices[~oob]] = points[~oob]
            self.remove_points(
                points_indices=points_indices[oob], cell_indices=cell_indices[oob])
            self.add_points(points=points[oob].cpu(
            ).numpy(), cell_indices=new_cell_indices[oob])
        else:
            logging.info(
                "Updating more than one point per cell, falling back to loop")
            raise NotImplementedError

    def remove_one_point(self, point_index, cell_index):
        # move last point to the location of the point to remove
        self.cells[cell_index, point_index] = \
            self.cells[cell_index, self.cells_point_number[cell_index] - 1]
        self.masks[cell_index, self.cells_point_number[cell_index] - 1] = False
        # decrease point number
        self.cells_point_number[cell_index] -= 1

    def remove_points(self, points_indices, cell_indices):
        if type(points_indices) is str and points_indices == 'all':
            self.masks[cell_indices] = False
            self.cells_point_number[cell_indices] = 0
            return
        if np.all(np.unique(cell_indices, return_counts=True)[1] <= 1):
            self.cells[cell_indices, points_indices] = \
                self.cells[cell_indices,
                           self.cells_point_number[cell_indices] - 1]
            self.masks[cell_indices,
                       self.cells_point_number[cell_indices] - 1] = False
            self.cells_point_number[cell_indices] -= 1
        else:
            for i_p, i_s, t in zip(points_indices, cell_indices):
                self.remove_one_point(i_p, i_s)

    def __len__(self):
        return np.sum(self.cells_point_number)

    @property
    def all_points(self):
        res = [
            torch.concat([
                cell[:n_points] for cell, n_points in
                zip(self.cells[t * self.n_cells_per_frame:(t + 1) * self.n_cells_per_frame],
                    self.cells_point_number[t * self.n_cells_per_frame:(t + 1) * self.n_cells_per_frame])
            ], dim=0)
            for t in range(self.n_frames)
        ]
        if self.temporal:
            return res
        else:
            return res[0]

    @property
    def non_empty_cell_indices(self):
        return np.argwhere(self.cells_point_number > 0)[:, 0]

    def points_and_context(self, mic_set: Union[str, int] = None, cell_indicies: Iterable = None,
                           reduce: bool = False, keep_one: bool = True):
        """
        returns a context cube for every cell of the set
        :param mic_set:
        :param reduce: if true reduce P to max number of points in all cells of set
        :param keep_one: if true P = (max number of points in all cells of set) + 1
        :return:
            two tensors of shape (N,3,3,3,P,D) and (N,3,3,3,P) with N the number of cells,
            P the max number of points, and D = 2+number of marks
            Tensor1[:,1,1,1] are the current cells and Tensor2[:,1,1,1] their masks
            Tensor3 is a matrix of times per cell
            if self.temporal is False, then returns tensors of shape (N,3,3,P,D) and (N,3,3,P)
            dimensions mean (nb of cells,(time),i,j,points nb, (state size))

        """
        assert mic_set is None or cell_indicies is None  # cannot specify both

        if cell_indicies is not None:
            indices = np.array(cell_indicies)
        else:
            if mic_set == 'all':
                indices = np.arange(self.nb_cells)
            elif type(mic_set) is not str:
                indices = self.cells_indices_per_set[mic_set]
            else:
                raise TypeError
        if reduce:
            max_nb_points = min(
                np.max(self.cells_point_number[indices]) + keep_one, self.max_pts)
        else:
            max_nb_points = self.max_pts
        # offsets = np.array([-self.n_y, -1, 0, 1, self.n_y])
        if self.temporal:
            t = indices // (self.n_y * self.n_x)
            i, j = (indices // self.n_y) % self.n_x, indices % self.n_y
            # set offsets
            offsets_t = np.roll(np.arange(-(self.time_interval - 1),
                                self.time_interval), -(self.time_interval - 1))
            offsets_i = np.roll(np.array([-1, 0, 1]), -1)
            offsets_j = np.roll(np.array([-1, 0, 1]), -1)
            ott, oii, ojj = np.meshgrid(
                offsets_t, offsets_i, offsets_j, indexing='ij')

            # offset indices
            new_t = t.reshape((-1, 1, 1, 1)) + ott
            new_i = i.reshape((-1, 1, 1, 1)) + oii
            new_j = j.reshape((-1, 1, 1, 1)) + ojj
            # check boundaries
            oob_t = (new_t < 0) | (new_t >= self.n_frames)
            oob_i = (new_i < 0) | (new_i >= self.n_x)
            oob_j = (new_j < 0) | (new_j >= self.n_x)
            oob = oob_t | oob_i | oob_j

            context_cube_indices = new_i * self.n_y + \
                new_j + new_t * self.n_cells_per_frame
            times = new_t
        else:
            i, j = (indices // self.n_y) % self.n_x, indices % self.n_y
            # set offsets
            offsets_i = np.roll(np.array([-1, 0, 1]), -1)
            offsets_j = np.roll(np.array([-1, 0, 1]), -1)
            oii, ojj = np.meshgrid(offsets_i, offsets_j, indexing='ij')

            # offset indices
            new_i = i.reshape((-1, 1, 1)) + oii
            new_j = j.reshape((-1, 1, 1)) + ojj
            # check boundaries
            oob_i = (new_i < 0) | (new_i >= self.n_x)
            oob_j = (new_j < 0) | (new_j >= self.n_y)
            oob = oob_i | oob_j

            context_cube_indices = new_i * self.n_y + new_j
            times = None
        # set oob indices to empty cell
        context_cube_indices[oob] = self.nb_cells

        assert np.all(context_cube_indices[:, 0, 0] == indices)

        context_cube = self.cells[context_cube_indices, :max_nb_points]
        context_cube_mask = self.masks[context_cube_indices, :max_nb_points]

        if self.temporal and self.time_span != context_cube.shape[1]:
            warnings.warn(f"time span is {self.time_span} "
                          f"but context cube has temporal dimension {context_cube.shape[1]}")

        return context_cube, context_cube_mask, times

    def check_if_all_inbound(self):
        for i in range(self.nb_cells):
            tl, br = self.cells_bounds[i]
            coordinates = self.cells[i][self.masks[i]][:, :2].numpy()
            inbound = np.all(coordinates < br, axis=-
                             1) & np.all(coordinates > tl, axis=-1)
            assert np.all(inbound)

    def get_cell_neighbors(self, cell_indices: Union[np.ndarray, int]):
        single_output = False
        if type(cell_indices) is int:
            cell_indices = np.array([cell_indices])
            single_output = True
        if self.temporal:
            raise NotImplementedError
        else:
            i, j = (cell_indices // self.n_y) % self.n_x, cell_indices % self.n_y
            # set offsets
            offsets_i = np.roll(np.array([-1, 0, 1]), -1)
            offsets_j = np.roll(np.array([-1, 0, 1]), -1)
            oii, ojj = np.meshgrid(offsets_i, offsets_j, indexing='ij')

            # offset indices
            new_i = i.reshape((-1, 1, 1)) + oii
            new_j = j.reshape((-1, 1, 1)) + ojj
            # check boundaries
            oob_i = (new_i < 0) | (new_i >= self.n_x)
            oob_j = (new_j < 0) | (new_j >= self.n_y)
            oob = oob_i | oob_j

            neighbor_indices = new_i * self.n_y + new_j
            res = [neighbor_indices[i, ~oob[i]]
                   for i in range(len(cell_indices))]
            if single_output:
                return res[0]
            else:
                return res
