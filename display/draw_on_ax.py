import logging
from typing import Union

import numpy as np
from matplotlib import pyplot as plt, patches
from torch import Tensor

from base.geometry import rect_to_poly


def rectangle_patch(s, **kwargs):
    poly = rect_to_poly(s[[0, 1]], s[2], s[3], s[4])
    return patches.Polygon(poly[:, [1, 0]], **kwargs)


def poly_patch(s, **kwargs):
    assert s.shape == (4, 2)
    return patches.Polygon(s[:, [1, 0]], **kwargs)


def circle_patch(s, **kwargs):
    return patches.Circle(s[[1, 0]], radius=s[2], **kwargs)


def point_patch(s, **kwargs):
    return patches.Circle(s[[1, 0]], **kwargs)


def draw_shapes_on_ax(ax: plt.Axes, states: Union[np.ndarray, Tensor], colors: np.ndarray = None,
                      labels: list = None, draw_method='rectangle', alpha_fill: float = 0.0, **kwargs):
    if type(states) is Tensor:
        states = states.detach().cpu().numpy()

    if draw_method == 'rectangle':
        draw_fun = rectangle_patch
        if len(states) > 0 and len(states[0]) < 5:
            raise RuntimeError(f"state is not big enough to encode a rectangle ({len(states[0])}<5), "
                               f"choose the correct draw_method")
    elif draw_method == 'poly':
        draw_fun = poly_patch
    elif draw_method == 'circle':
        draw_fun = circle_patch
    elif draw_method == 'point':
        draw_fun = point_patch
    else:
        raise ValueError

    for i, s in enumerate(states):

        local_kwargs = kwargs.copy()
        if colors is not None:
            local_kwargs['color'] = colors[i]

        if alpha_fill > 0.0:
            # local_kwargs['fill'] = True
            color = tuple(local_kwargs['color'])
            if len(color) == 4:
                color = color[:3]
            elif len(color) == 3:
                pass
            else:
                raise ValueError
            local_kwargs['fill'] = True
            local_kwargs['facecolor'] = color + (alpha_fill,)
            local_kwargs.pop('color')
            local_kwargs['edgecolor'] = color + (1.0,)

        ax.add_patch(draw_fun(s, **local_kwargs))
        if labels is not None:
            t_kwargs = {}
            if 'color' in local_kwargs:
                t_kwargs['color'] = local_kwargs['color']
            if draw_method == 'poly':
                text_pos = np.min(s[:, 0], axis=0), np.max(s[:, 1], axis=0)
            else:
                text_pos = s
            ax.text(text_pos[1], text_pos[0],
                    s=labels[i], clip_on=True, **t_kwargs)
