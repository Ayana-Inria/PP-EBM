import logging
from typing import Union, Dict

import numpy as np
import torch
from numpy.random import Generator
from scipy import ndimage
from scipy.ndimage import gaussian_filter, filters
from torch import Tensor
from torch.nn import functional


def extract_patch(image: np.ndarray, center_anchor: np.ndarray, patch_size: int, pad_value=0.0,
                  allow_padding: bool = True, minimal_padding: bool = False):
    assert center_anchor.shape == (2,)
    tl_anchor = center_anchor - patch_size // 2
    shape = np.array(image.shape[:2])
    if minimal_padding:
        assert allow_padding
        if np.any(patch_size > shape):
            # padding will happen, ensure minimum amount of it by centering image
            center_anchor = shape // 2
            tl_anchor = center_anchor - patch_size // 2
    centers_offset = np.zeros((2,), dtype=int)

    if allow_padding:
        if tl_anchor[0] < 0 or tl_anchor[0] + patch_size >= shape[0]:
            image = np.pad(image, ((patch_size // 2, patch_size // 2), (0, 0), (0, 0)), 'constant',
                           constant_values=pad_value)

            centers_offset[0] = patch_size // 2

            tl_anchor[0] = tl_anchor[0] + patch_size // 2
        if tl_anchor[1] < 0 or tl_anchor[1] + patch_size >= shape[1]:
            image = np.pad(image, ((0, 0), (patch_size // 2, patch_size // 2), (0, 0)), 'constant',
                           constant_values=pad_value)
            centers_offset[1] = patch_size // 2
            tl_anchor[1] = tl_anchor[1] + patch_size // 2
    else:
        tl_anchor = np.clip(tl_anchor, (0, 0), np.array(shape) - patch_size)

    s = np.s_[tl_anchor[0]:tl_anchor[0] + patch_size,
              tl_anchor[1]:tl_anchor[1] + patch_size]
    patch = image[s]
    return patch, tl_anchor, centers_offset


def patchify_big_image(image: Tensor, margin=0, patch_size=512):
    assert len(image.shape) == 3
    shape = image.shape

    v_remainder = (patch_size - 2 * margin) - \
        (shape[1] % (patch_size - 2 * margin))
    h_remainder = (patch_size - 2 * margin) - \
        (shape[2] % (patch_size - 2 * margin))

    if margin > 0:
        image = functional.pad(
            image, (margin, margin + h_remainder, margin, margin + v_remainder), mode='replicate')

    shape = image.shape

    nx = int(np.ceil((shape[1] - 2 * margin) / (patch_size - 2 * margin)))
    ny = int(np.ceil((shape[2] - 2 * margin) / (patch_size - 2 * margin)))
    logging.info(f"splitting into {nx}x{ny} patches")

    xx = np.linspace(0, shape[1] - patch_size, num=nx, dtype=int)
    yy = np.linspace(0, shape[2] - patch_size, num=ny, dtype=int)
    patches = []
    for x in xx:
        for y in yy:
            s = np.s_[x:x + patch_size, y:y + patch_size]
            patch = image[:, s[0], s[1]]
            if patch.shape[1] < patch_size:
                patch = functional.pad(
                    patch, (0, 0, 0, patch_size - patch.shape[1]))
            if patch.shape[2] < patch_size:
                patch = functional.pad(patch, (0, patch_size - patch.shape[2]))
            patches.append(patch)

    patches = torch.stack(patches, dim=0)

    return patches, image, (xx, yy)


def unpatchify_big_image(patches, original_size, margin, cuts, keep_batch_dim=False):
    assert len(original_size) == 3
    xcuts, ycuts = cuts
    patch_size = patches.shape[2]
    n_channels = patches.shape[1]
    big_image = torch.empty(
        (n_channels,
         len(xcuts) * (patch_size - 2 * margin) + 2 * margin + patch_size,
         len(ycuts) * (patch_size - 2 * margin) + 2 * margin + patch_size))
    k = 0
    for x in xcuts:
        for y in ycuts:
            if margin != 0:
                big_image[:, x + margin:x + patch_size - margin, y + margin:y + patch_size - margin] = \
                    patches[k, :, margin:-margin, margin:-margin]
            else:
                big_image[:, x:x + patch_size, y:y + patch_size] = patches[k]
            k += 1
    if margin != 0:
        big_image = big_image[:, margin:-margin, margin:-margin]
    big_image = big_image[:, :original_size[1], :original_size[2]]
    if keep_batch_dim:
        return big_image.unsqueeze(dim=0)
    return big_image


def map_range(arr: Union[np.ndarray, Tensor], min_in, max_in, min_out, max_out, clip=False):
    res = (arr - min_in) / (max_in - min_in) * (max_out - min_out) + min_out
    if clip:
        a, b = sorted((min_out, max_out))
        if type(arr) is Tensor:
            return torch.clip(res, a, b)
        elif type(arr) is np.ndarray:
            return np.clip(res, a, b)
        else:
            raise TypeError
    else:
        return res


def map_range_auto(arr: Union[np.ndarray, Tensor], min_out, max_out):
    if arr.size > 0:
        return map_range(arr, arr.min(), arr.max(), min_out, max_out)
    else:
        return arr


def deteriorate_image(image: np.ndarray, blur_sigma: float, noise_factor: float, rng: Generator = None,
                      noise_type: str = 'uniform', repeat=None):
    if rng is None:
        rng = np.random.default_rng()

    if blur_sigma != 0.0:
        new_image = gaussian_filter(
            image, sigma=blur_sigma
        )
    else:
        new_image = image.copy()
    if noise_type == 'uniform':
        new_image = new_image * (1 - noise_factor) + \
            rng.random(image.shape) * noise_factor
    elif noise_type == 'gaussian':
        new_image = new_image + noise_factor * rng.normal(size=image.shape)
        new_image = map_range_auto(new_image, 0.0, 1.0)
    else:
        raise ValueError

    if repeat is None:
        return new_image
    if repeat == 1:
        return [new_image]
    else:
        return [new_image] + deteriorate_image(new_image, blur_sigma, noise_factor, rng=rng, repeat=repeat - 1)


def pad_to_valid_size(image: Tensor, valid_size_divider: int):
    assert len(image.shape) == 4
    # functional.pad takes padding args in reverse dim order
    shape = np.array(image.shape[2:])
    mul = shape / valid_size_divider
    pad = np.ceil(mul).astype(int) * valid_size_divider - shape
    # assert np.all(np.remainder(shape + pad,valid_size_divider) == 0.0)
    padded_image = functional.pad(image, (0, pad[1], 0, pad[0]))
    assert np.all(np.remainder(
        padded_image.shape[2:], valid_size_divider) == 0.0)
    return padded_image


def unpad_to_size(output: Dict[str, Tensor], original_size: tuple):
    return {
        k: v[:, :, :original_size[0], :original_size[1]] for k, v in output.items()
    }
