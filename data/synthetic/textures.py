import sys
from typing import Tuple, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import Generator

from base.images import map_range, map_range_auto


def checker(rng: Generator, shape: Tuple[int, int], n_divs: Union[int, Tuple], v_min=0, v_max=1):
    image = np.ones(shape) * v_min
    offset = rng.random() > 0.5
    if type(n_divs) is int:
        n_divs = (n_divs, n_divs)
    xx = np.linspace(0, shape[0], n_divs[0] + 1, dtype=int)
    yy = np.linspace(0, shape[1], n_divs[1] + 1, dtype=int)
    for i in range(n_divs[0]):
        for j in range(n_divs[1]):
            if (i + j + offset) % 2 == 0:
                image[xx[i]:xx[i + 1], yy[j]:yy[j + 1]] = v_max
    return image


def noise(rng: Generator, shape: Tuple[int, int], levels):
    im_all = []
    weights = []
    for i in levels:
        im = rng.random((np.array(shape) / i).astype(int))
        im_rs = cv2.resize(im, dsize=shape, interpolation=cv2.INTER_CUBIC)
        ksize = (i // 2) * 2 + 1
        im_rs = cv2.GaussianBlur(im_rs, ksize=(ksize, ksize), sigmaX=2 * i)
        w = i
        im_all.append(w * im_rs)
        weights.append(w)

    res = np.sum(im_all, axis=0) / np.sum(weights)
    v_min = np.min(res)
    v_max = np.max(res)
    return map_range(res, v_min, v_max, 0, 1)


def color_noise(rng: Generator, shape: Tuple[int, int], levels):
    rgb_stack = [
        noise(rng, shape, levels) for _ in range(3)
    ]
    return np.stack(rgb_stack, axis=-1)


def dots(rng: Generator, shape: Tuple[int, int], density_min, density_max, bg_col=1, dot_col=0, sigma=1.0):
    image = np.ones(shape) * bg_col
    n_points = int(np.prod(shape) * rng.uniform(density_min, density_max))
    points = rng.uniform((0, 0), shape, (n_points, 2)).astype(int)
    image[points[:, 0], points[:, 1]] = dot_col
    image = cv2.GaussianBlur(image, ksize=(9, 9), sigmaX=sigma)
    return image


def random_rect(rng: Generator, shape: Tuple[int, int], intensity: float, rgb: bool, size: float,
                size_sigma: float, threshold: float = None, fg_color=None):
    image = np.zeros(shape + (3,))
    n_rect = int(intensity * np.prod(shape))
    for i in range(n_rect):
        # x1, x2 = rng.integers(0, shape[0], size=2)
        # x1, x2 = sorted((x1, x2))
        # y1, y2 = rng.integers(0, shape[1], size=2)
        # y1, y2 = sorted((y1, y2))
        xy = rng.integers(np.zeros(2) - size, np.array(shape) + size, size=2)
        hw = np.clip(rng.normal((size, size), size_sigma, size=2), 0, np.inf)

        x2, y2 = np.clip((xy + hw / 2).astype(int), (0, 0), np.array(shape))
        x1, y1 = np.clip((xy - hw / 2).astype(int), (0, 0), np.array(shape))
        if fg_color is not None:
            v = rng.uniform(-1, 1, size=1)[0]
            v = np.array(fg_color) * v
        elif rgb:
            v = rng.uniform(-1, 1, size=3)
        else:
            v = rng.uniform(-1, 1, size=1)[0]
            v = [v] * 3
        image[x1:x2, y1:y2] += v

    image = image + 0.0001 * rng.normal(size=image.shape)

    image = map_range(image, np.min(image, axis=(0, 1)),
                      np.max(image, axis=(0, 1)), 0, 1, )

    if threshold is not None:
        image = (image > threshold).astype(float)

    return image


def simple_test():
    rng = np.random.default_rng(1)
    # image = checker(rng, (128, 256), (4, 16))
    # image = noise(rng, (256, 256), levels=[32, 64, 128])
    # image = dots(rng, (256, 256), density_min=0.01, density_max=0.02, sigma=2.0)
    image = random_rect(rng, (256, 256), intensity=0.001, threshold=None, rgb=True, size=64, size_sigma=32,
                        fg_color=[1, 0, 0])
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    print(sys.path)
    simple_test()
