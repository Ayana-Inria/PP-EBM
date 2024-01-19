import numpy as np
import torch
from numpy.random import Generator
from torch import Tensor


def rot90_state(state: Tensor, size: int, k: int, shape: str):
    if k == 0 or len(state) == 0:
        return state
    else:
        new_state = state.clone()
        new_state[..., 0] = size - 1 - state[..., 1]
        new_state[..., 1] = state[..., 0]
        if shape == 'rectangle':
            new_state[..., 4] = (new_state[..., 4] + np.pi / 2) % np.pi
        elif shape == 'circle':
            pass
        else:
            raise ValueError
        return rot90_state(new_state, size, k - 1, shape=shape)


class BasicAugmenter:

    def __init__(self, p_rotation: float, p_flip: float, shape: str, **kwargs):
        self.p_rotation = [1 - p_rotation] + [p_rotation / 4] * 4
        self.p_flip = [1 - p_flip] + [p_flip / 2] * 2
        self.shape = shape

    def draw(self, rng: Generator):
        transform_dict = {
            'rotation': int(rng.choice(range(5), p=self.p_rotation)),
            'flip': int(rng.choice(range(3), p=self.p_flip))
        }
        return transform_dict

    def transform_image(self, transform_dict, image: np.ndarray):
        rotation = transform_dict['rotation']
        flip = transform_dict['flip']
        if rotation > 0:
            image = np.rot90(image, k=rotation)
        if flip > 0:
            image = np.flip(image, axis=flip - 1)
        return image

    def transform_image_t(self, transform_dict, image: Tensor):
        assert len(image.shape) == 3  # (C,H,W)
        rotation = transform_dict['rotation']
        flip = transform_dict['flip']
        if rotation > 0:
            image = torch.rot90(image, k=rotation, dims=[1, 2])
        if flip > 0:
            image = torch.flip(image, dims=[flip])
        return image

    # @staticmethod
    # def get_reverse_transform(transform_dict):
    #     return {
    #         'rotation': (-transform_dict['rotation']) % 4,
    #         'flip': transform_dict['flip']
    #     }

    def transform_state(self, transform_dict, state: Tensor, image_size: int):
        rotation = transform_dict['rotation']
        flip = transform_dict['flip']
        state = state.clone()
        if len(state) == 0:
            return state
        if rotation > 0:
            state = rot90_state(state, size=image_size,
                                k=rotation, shape=self.shape)

        if flip > 0:
            new_state = state.detach()
            if flip == 1:
                new_state[..., 0] = image_size - 1 - state[..., 0]
                if self.shape == 'rectangle':
                    new_state[..., 4] = torch.remainder(
                        np.pi - new_state[..., 4], np.pi)
            else:
                new_state[..., 1] = image_size - 1 - state[..., 1]
                if self.shape == 'rectangle':
                    new_state[...,
                              4] = torch.remainder(-new_state[..., 4], np.pi)

            state = new_state

        return state

    def reverse_transform_state(self, transform_dict, state: Tensor, image_size: int):
        # return self.transform_state(
        #     transform_dict=self.get_reverse_transform(transform_dict),
        #     state=state,
        #     image_size=image_size
        # )
        rotation = (-transform_dict['rotation']) % 4
        flip = transform_dict['flip']
        state = state.clone()
        if len(state) == 0:
            return state

        if flip > 0:
            new_state = state.detach()
            if flip == 1:
                new_state[..., 0] = image_size - 1 - state[..., 0]
                if self.shape == 'rectangle':
                    new_state[..., 4] = torch.remainder(
                        np.pi - new_state[..., 4], np.pi)
            else:
                new_state[..., 1] = image_size - 1 - state[..., 1]
                if self.shape == 'rectangle':
                    new_state[...,
                              4] = torch.remainder(-new_state[..., 4], np.pi)

            state = new_state

        if rotation > 0:
            state = rot90_state(state, size=image_size,
                                k=rotation, shape=self.shape)

        return state
