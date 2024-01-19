import logging
from typing import Union, Dict

from numpy.random import Generator
from torch import Tensor


class StateMemory:

    def __init__(self, memory_size: int, memory_proba: float):
        self.memory_size = memory_size
        self.memory_proba = memory_proba
        self._mem = {}
        self._mem_meta = {}
        self._mem_iter = {}

    def clear(self):
        self._mem = {}
        self._mem_meta = {}
        self._mem_iter = {}

    def __setstate__(self, state):
        for k, v in state.items():
            self.__setattr__(k, v)

        if '_mem_meta' not in state:
            logging.warning(f"loading StateMemory with missing attribute _mem_meta "
                            f"completing with empty dicts")
            self._mem_meta = {}
            for image_id in self._mem.keys():
                self._mem_meta[image_id] = [None] * self.memory_size
                for i in range(0, self.image_current_memory(image_id)):
                    self._mem_meta[image_id][i] = {}

    def append(self, image_id: int, state: Tensor, metadata: Dict = None):
        if metadata is None:
            metadata = {}
        if image_id not in self._mem:
            self._mem[image_id] = [None] * self.memory_size
            self._mem_meta[image_id] = [None] * self.memory_size
            self._mem_iter[image_id] = 0
        add_index = self._mem_iter[image_id] % self.memory_size
        self._mem[image_id][add_index] = state.cpu()
        self._mem_meta[image_id][add_index] = metadata
        self._mem_iter[image_id] += 1

    def last_append_index(self, image_id: int):
        if image_id not in self._mem:
            return None
        return (self._mem_iter[image_id] - 1) % self.memory_size

    def select_rd(self, rng: Generator, image_id, force_select: bool = False) -> Union[None, Tensor]:
        if image_id not in self._mem:
            return None, None
        elif (not force_select) and rng.random() > self.memory_proba:
            logging.info("memory selected None")
            return None, None
        else:
            n_items = self.image_current_memory(image_id)
            rd_id = rng.integers(n_items)
            logging.info(f"memory selected {rd_id}")
            return self._mem[image_id][rd_id], self._mem_meta[image_id][rd_id]

    def image_current_memory(self, image_id: int):
        return min(self.memory_size, self._mem_iter[image_id])

    @property
    def current_memory_per_image(self):
        return {k: self.image_current_memory(k) for k, v in self._mem.items()}

    def get_memory(self, image_id: int, memory_id: int, get_meta: bool = True):
        assert self._mem_iter[image_id] > memory_id
        assert memory_id < self.memory_size
        if get_meta:
            return self._mem[image_id][memory_id], self._mem_meta[image_id][memory_id]
        return self._mem[image_id][memory_id]
