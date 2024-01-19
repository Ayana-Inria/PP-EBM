from typing import Union

import numpy as np
from matplotlib.colors import ColorConverter
from skimage import draw


def plot_stuff(arr: np.ndarray, h: int, w: int, pad_value=0.0, support_value=1.0, color='tab:blue'):
    image = np.zeros((h, w, 3))

    if not len(arr.shape) == 2:
        arr = np.expand_dims(arr, 0)
        color = [color]
    n_pts = arr.shape[1]
    n_series = arr.shape[0]

    pad = 2
    left_limit, right_limit = pad, w - pad - 1

    ticks = np.linspace(left_limit, right_limit, n_pts, dtype=int)

    image[-pad - 1, pad:-pad] = support_value
    image[-pad, ticks] = support_value

    vmax = np.max(arr)
    vmin = np.min(arr)

    plot_up = pad
    plot_down = h - 2 *pad - 1
    plot_height = plot_down - plot_up

    arr_norm = (arr - vmin) / (vmax - vmin)
    h_plot = ((1 - arr_norm) * plot_height).astype(int) + plot_up

    for j in range(n_series):
        c = ColorConverter.to_rgb(color[j])
        for i in range(n_pts - 1):
            v0 = h_plot[j, i]
            v1 = h_plot[j, i + 1]
            t0 = ticks[i]
            t1 = ticks[i + 1]
            coor = draw.line(v0, t0, v1, t1)
            image[coor] = c

    return image


def hist_image(size: int, distribution: np.ndarray, vmax: Union[float, str] = 1, vmin=0, pad_value=0.0,
               support_value=1.0,
               plot_color=0.5, plot_cmap=None, gt=None, gt_color=(0, 0.5, 0)):
    """

    :param size: size of image
    :param distribution: hist to show
    :param vmax:
    :param vmin:
    :param pad_value:
    :param support_value:
    :param plot_color:
    :return:
    """
    if vmax == 'auto':
        vmax = np.max(distribution)
    if vmin == 'auto':
        vmin = np.min(distribution)
    distribution = np.clip(distribution, vmin, vmax)

    plot = np.full((size, size, 3), pad_value)
    d_size = distribution.shape[0]
    assert size >= d_size
    bar_width = size // d_size
    support_len = d_size * bar_width
    pad_left_support = (size - support_len) // 2
    # plot[-1, pad_left_support:pad_left_support + support_len] = support_value
    bar_range = size - 1

    bar_height = (bar_range * (distribution - vmin) / (vmax - vmin)).astype(int)
    norm_value = bar_height / np.sum(bar_height)

    for k, h in enumerate(bar_height):
        if gt is not None and k == gt:
            sv = gt_color
        else:
            sv = support_value
        plot[-1:, pad_left_support + k * bar_width] = sv

        if plot_cmap is not None:
            plot_color = plot_cmap(norm_value[k])[:3]

        plot[bar_range - h:-2, pad_left_support + k * bar_width] = plot_color

    return plot


def multi_hist_image(size: int, distribution: np.ndarray, vmax: Union[float, str] = 1, vmin=0, pad_value=0.0,
                     support_value=1.0,
                     plot_color=0.5, plot_cmap=None, gt=None, gt_color=(0, 0.5, 0), min_plot_size=5):
    if vmax == 'auto':
        vmax = np.max(distribution)
    if vmin == 'auto':
        vmin = np.min(distribution)
    distribution = np.clip(distribution, vmin, vmax)

    plot = np.full((size, size, 3), pad_value)

    n_dist = len(distribution)
    plot_height = size // n_dist
    if plot_height < min_plot_size:
        n_dist = size // min_plot_size
        plot_height = size // n_dist
        distribution = distribution[:n_dist]

    for i, d in enumerate(distribution):

        v_offset = i * plot_height

        d_size = d.shape[0]
        assert size >= d_size
        bar_width = size // d_size
        support_len = d_size * bar_width
        pad_left_support = (size - support_len) // 2
        # plot[-1, pad_left_support:pad_left_support + support_len] = support_value
        bar_range = plot_height - 4

        bar_height = np.ceil((bar_range * (d - vmin) / (vmax - vmin))).astype(int)
        assert np.all(bar_height < plot_height)
        norm_value = bar_height / np.sum(bar_height)

        for k, h in enumerate(bar_height):
            if gt is not None and k == gt[i]:
                sv = gt_color
            else:
                sv = support_value
            plot[size - v_offset - 1, pad_left_support + k * bar_width:pad_left_support + (k + 1) * bar_width - 1] = sv

            if plot_cmap is not None:
                plot_color = plot_cmap(norm_value[k])[:3]

            plot[size - v_offset - 2 - h: size - v_offset - 2,
            pad_left_support + k * bar_width:pad_left_support + (k + 1) * bar_width - 1] = plot_color

    return plot


def distrib_pixel(size: int, distributions: np.ndarray, vmax=1, vmin=0, pad_value=0.0, support_value=1.0, cmap=None):
    distributions = np.clip(distributions, vmin, vmax)
    plot = np.full((size, size, 3), pad_value)
    n_dist = distributions.shape[0]
    d_size = distributions.shape[1]
    assert size >= d_size
    bar_width = size // d_size
    support_len = d_size * bar_width
    pad_left_support = (size - support_len) // 2
    # plot[-1, pad_left_support:pad_left_support + support_len] = support_value

    values = (distributions - vmin) / (vmax - vmin)

    height_per_d = (size - 3) // n_dist

    for k in range(d_size):
        plot[0, pad_left_support + k * bar_width] = support_value
        plot[height_per_d * n_dist + 2, pad_left_support + k * bar_width] = support_value

        for d in range(n_dist):
            if cmap is not None:
                v = cmap(values[d, k])[:3]
            else:
                v = values[d, k]
            plot[2 + d * height_per_d:2 + (d + 1) * height_per_d - 1, pad_left_support + k * bar_width] = v

    return plot