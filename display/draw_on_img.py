from typing import Union, Callable

import cv2
import numpy as np
from torch import Tensor

from base.geometry import rect_to_poly


def cv_draw_rectangle(img, s, fill: bool = False, dilation: int = 0, **kwargs):
    pts = rect_to_poly(center=(s[0], s[1]), long=s[3],
                       short=s[2], angle=s[4], dilation=dilation)
    pts.reshape((-1, 1, 2))
    pts = pts.astype(np.int32)
    pts = np.flip(pts, axis=-1)
    pts = np.array([pts])
    if not fill:
        cv2.polylines(img=img, pts=pts, isClosed=True, **kwargs)
    else:
        cv2.fillPoly(img=img, pts=pts, **kwargs)


def cv_draw_circle(img, s, fill=False, **kwargs):
    center = s[[1, 0]].astype(int)
    radius = int(s[2])
    if fill:
        kwargs['thickness'] = - 1
    cv2.circle(img, center, radius, **kwargs)


def cv_draw_point(img, s, **kwargs):
    center = s[[1, 0]].astype(int)
    radius = 4
    cv2.circle(img, center, radius, **kwargs)


def draw_shapes_on_img(image: np.ndarray, states: Union[np.ndarray, Tensor], draw_method='rectangle',
                       colormap_func: Callable = None,
                       **kwargs):
    if type(states) is Tensor:
        states = states.detach().cpu().numpy()

    image = image.copy()

    if draw_method == 'rectangle':
        draw_fun = cv_draw_rectangle
        if len(states) > 0 and len(states[0]) < 5:
            raise RuntimeError(f"state is not big enough to encode a rectangle ({len(states[0])}<5), "
                               f"choose the correct draw_method")
    elif draw_method == 'circle':
        draw_fun = cv_draw_circle
    elif draw_method == 'point':
        draw_fun = cv_draw_point
    else:
        raise ValueError

    for i, s in enumerate(states):

        local_kwargs = kwargs.copy()
        if colormap_func is not None:
            local_kwargs['color'] = colormap_func(i)

        draw_fun(image, s, **local_kwargs)

    return image
