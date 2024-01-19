import numpy as np
from matplotlib import colors, pyplot as plt

INRIA_COLORS = {
    'inria:red': (230, 51, 18),
    'inria:gray-blue': (56, 66, 87),
    'inria:white': (255, 255, 255),
    'inria:orange': (240, 126, 38),
    'inria:lilac': (266, 0, 79),
    'inria:blue': (20, 136, 202),
    'inria:green': (149, 193, 31),
    'inria:yellow': (255, 205, 28),
    'inria:mauve': (101, 97, 169),
    'inria:light-blue': (137, 204, 202),
    'inria:light-green': (199, 214, 79)
}


def colormaker(color):
    if type(color) is str:
        if 'inria:' in color:
            return tuple(np.array(INRIA_COLORS[color]) / 255) + (1.0,)
        else:
            return colors.to_rgba(color)
    elif type(color) is dict:
        if 'cmap' in color:
            assert 'value' in color
            return plt.get_cmap(color['cmap'])(color['value'])
    elif type(color) in (np.ndarray, list):
        color = np.array(color)
        if np.max(color) > 1.0:
            color = color / 255.0
        if len(color) == 3:
            return tuple(color) + (1.0,)
        elif len(color) == 4:
            return tuple(color)
        else:
            raise RuntimeError(
                f"bad color size, expected 3 or 4: color={color}")
    else:
        raise TypeError
