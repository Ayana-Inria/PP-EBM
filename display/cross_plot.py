import logging
from typing import Callable, List

import numpy as np
import torch
from matplotlib import pyplot as plt


def show_energy_iso(ax: plt.Axes, i, j, function, n_dim, return_contourf=False, **contour_kwargs):
    """
    Displays a heatmap/contour of a decision function that take n_dim-dimentional vectors as input,
    for a i,j show a heatmap of the deicision function in function of i and j with other parameters fixed
    Parameters
    ----------
    ax : the pyplot axis to display the contour
    i : id of first param
    j : id of second param
    function : decision function (for instance a smv.decision_function)
    n_dim : vector dimension of the input of the decision function
    """
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    x = np.linspace(min_x, max_x, 100)
    y = np.linspace(min_y, max_y, 100)
    xx, yy = np.meshgrid(x, y)
    arr = np.zeros(xx.shape + (n_dim,))
    arr[:, :, j] = xx
    arr[:, :, i] = yy
    shape = arr.shape[:2]
    Z = -function(torch.from_numpy(arr.reshape((-1, n_dim)))
                  ).detach().cpu().numpy().reshape(shape)
    im = ax.contourf(xx, yy, -Z, levels=10, alpha=0.6,
                     cmap='coolwarm', **contour_kwargs)
    if return_contourf:
        return im


def cross_plot(vectors: np.ndarray, dim_names: List[str], colors, labels=None, label_names: List[str] = None,
               decision_function: Callable = None):
    n_dims = vectors.shape[1]
    n_rows, n_cols = n_dims, n_dims
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(
        n_cols * 3, n_rows * 3), squeeze=False)

    if labels is None:
        labels = np.zeros(vectors.__len__(), dtype=int)
        label_names = []
    assert len(vectors) == len(labels)
    color_per_point = [colors[l] for l in labels]

    for i in range(n_dims):
        for j in range(n_dims):
            ax: plt.Axes = axs[i, j]
            if j != i:
                ax.scatter(vectors[:, j], vectors[:, i],
                           color=color_per_point, zorder=10, s=1.0)
                if decision_function is not None:
                    show_energy_iso(
                        ax, i, j, decision_function, n_dims, zorder=0)
            else:
                try:
                    ax.hist([vectors[labels == c, i] for c in np.sort(np.unique(labels))],
                            density=True, label=label_names,
                            color=colors)
                    ax.legend()
                except ValueError as e:
                    logging.warning(f"unable to display hist because {e}")

            if i == (n_dims - 1):
                ax.set_xlabel(dim_names[j])
            if j == 0:
                ax.set_ylabel(dim_names[i])

    return fig
