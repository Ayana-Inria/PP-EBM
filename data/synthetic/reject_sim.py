import logging
from typing import Tuple, Callable

import numpy as np
from numpy.random import Generator
from tqdm import tqdm

from base.sampler2d import sample_point_2d
from data.synthetic import reject_functions
from data.synthetic.pointssimulator import PointsSimulator
from data.synthetic.reject_functions import composite_reject, radius_reject


def mpp_reject_sim(rng: Generator, shape: Tuple[int, int], marks_min, marks_max, n_points: int,
                   reject_fn: Callable[[np.ndarray, np.ndarray], bool] = None, rejects_limit=1e4,
                   error_on_rejects_limit=True, intensity_map=None,
                   verbose: int = 1):
    state_lb = (0, 0) + tuple(marks_min)
    state_hb = shape + tuple(marks_max)
    state_size = 2 + len(marks_min)
    if intensity_map is None:
        state = rng.uniform(state_lb, state_hb, size=(n_points, state_size))
    else:
        marks = rng.uniform(
            state_lb[2:], state_hb[2:], size=(n_points, state_size - 2))
        coordinates = sample_point_2d(
            img_shape=intensity_map.shape, size=n_points, density=intensity_map, rng=rng)
        state = np.concatenate((coordinates, marks), axis=-1)

    if reject_fn is not None:
        if verbose > 0:
            pbar = tqdm(range(n_points), desc='Filtering points')
        else:
            pbar = range(n_points)
        for i in pbar:
            reject_flag = True
            n = 0
            while reject_flag:
                if n > rejects_limit:
                    if error_on_rejects_limit:
                        raise RuntimeError("too many rejects, aborting")
                    else:
                        logging.warning("too many rejects, aborting")
                        return state[:i]
                s1 = state[i]
                reject_flag = False
                for j in range(i):
                    if reject_fn(s1, state[j]):  # point rejected
                        reject_flag = True
                        break
                if reject_flag:
                    if intensity_map is None:
                        state[i] = rng.uniform(
                            state_lb, state_hb, size=state_size)
                    else:
                        state[i] = np.concatenate((
                            sample_point_2d(
                                img_shape=intensity_map.shape, size=1, density=intensity_map, rng=rng)[0],
                            rng.uniform(
                                state_lb[2:], state_hb[2:], size=(state_size - 2))
                        ), axis=-1)
                    n += 1
                if verbose > 0:
                    pbar.set_postfix({'reject': f"{n / rejects_limit:.2%}"})
    return state


class RejectSimulator(PointsSimulator):

    def __init__(self, config):
        self.config = config

        if 'reject_functions' in self.config:
            reject_fn = []
            for k, v in config['reject_functions'].items():
                reject_fn.append(getattr(reject_functions, k)(**v))
            if len(reject_fn) == 1:
                self.reject_fn = reject_fn[0]
            else:
                self.reject_fn = composite_reject(reject_fn)
        else:
            self.reject_fn = radius_reject(margin=0)

    def make_points(self, rng: Generator, image) -> np.ndarray:

        if self.config['n_points_max'] > self.config['n_points_min']:
            n_points = rng.integers(
                self.config['n_points_min'], self.config['n_points_max'])
        else:
            n_points = 0

        shape = image.shape[:2]

        image_as_intensity = self.config.get('image_as_intensity', False)
        if image_as_intensity:
            transfer_mode = self.config.get('image_to_intensity', None)
            if transfer_mode is None:
                intensity_map = image
            elif transfer_mode == 'invert':
                intensity_map = 1 - image
            else:
                raise ValueError
        else:
            intensity_map = None

        return mpp_reject_sim(
            rng=rng,
            shape=shape,
            marks_min=self.config['marks_min'],
            marks_max=self.config['marks_max'],
            n_points=n_points,
            reject_fn=self.reject_fn,
            verbose=0,
            intensity_map=intensity_map
        )
