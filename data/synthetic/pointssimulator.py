from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator


class PointsSimulator(ABC):

    @abstractmethod
    def make_points(self, rng: Generator, image) -> np.ndarray:
        raise NotImplementedError
