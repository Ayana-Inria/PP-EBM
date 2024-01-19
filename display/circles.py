from typing import Union, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from torch import Tensor


def draw_circles_on_ax(ax: plt.Axes, states: Union[np.ndarray, Tensor], colors: np.ndarray = None,
                       labels: list = None, **kwargs):
    if type(states) is Tensor:
        states = states.detach().cpu().numpy()

    for i, s in enumerate(states):

        local_kwargs = kwargs.copy()
        if colors is not None:
            local_kwargs['color'] = colors[i]
        cir = patches.Circle(s[[1, 0]], radius=s[2], **local_kwargs)
        ax.add_patch(cir)
        if labels is not None:
            t_kwargs = {}
            if 'color' in local_kwargs:
                t_kwargs['color'] = local_kwargs['color']
            ax.text(s[1], s[0], s=labels[i], **t_kwargs)
