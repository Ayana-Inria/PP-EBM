import numpy as np
from skimage.draw import draw

from display.draw_on_img import draw_shapes_on_img


def points(image: np.ndarray, state: np.ndarray, color=None, radius=1, alpha=1.0):
    if color is None:
        color = [1.0, 0, 0]
    color = np.array(color)
    shape = image.shape[:2]
    for p in state:
        ii, jj = draw.disk(center=p[[0, 1]].astype(
            int), radius=radius, shape=shape)
        image[ii, jj] = (1 - alpha) * image[ii, jj] + alpha * color

    return image


def rectangle(image: np.ndarray, state: np.ndarray, color=None):
    if color is None:
        color = [1.0, 0, 0]

    return draw_shapes_on_img(image, state, draw_method='rectangle', color=color)
