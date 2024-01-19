import logging
from typing import Union, Tuple

import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import draw


def rotation_matrix(alpha) -> np.ndarray:
    cos, sin = np.cos(alpha), np.sin(alpha)
    return np.array([[cos, -sin], [sin, cos]])


def rect_to_poly(center: Union[Tuple[int, int], np.ndarray], short: float, long: float, angle: float,
                 dilation: int = 0) -> np.ndarray:
    """
    converts rectangle parameters to polygon point coordinates
    Parameters
    ----------
    center : cneter coodinates
    short : length
    long : width
    angle : angle
    dilation : dilation of the shape

    Returns
    -------
    array of coordinates of shape (4,2)

    """
    # centered non rotated coordinates
    poly_coord = np.array([[short / 2 + dilation, long / 2 + dilation],
                           [short / 2 + dilation, - long / 2 - dilation],
                           [-short / 2 - dilation, - long / 2 - dilation],
                           [- short / 2 - dilation, long / 2 + dilation]])
    rot_matrix = rotation_matrix(angle).T
    try:
        rotated = np.matmul(poly_coord, rot_matrix)
        return rotated + center
    except Exception as e:
        print(f"rect_to_poly failed with {e}")
        print(f"{center=},{short=},{long=},{angle=}")
        print(f"{poly_coord=}")
        print(f"{rot_matrix=}")
        raise e


def poly_to_rect(poly: np.ndarray):
    assert poly.shape == (4, 2)
    norm_axis_1 = np.mean(
        [np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3])])
    norm_axis_2 = np.mean(
        [np.linalg.norm(poly[1] - poly[2]), np.linalg.norm(poly[3] - poly[0])])

    if norm_axis_1 < norm_axis_2:
        a, b = norm_axis_1, norm_axis_2
        axis_vector = np.mean([poly[2], poly[1]], axis=0) - \
            np.mean([poly[0], poly[3]], axis=0)
    else:
        a, b = norm_axis_2, norm_axis_1
        axis_vector = np.mean([poly[1], poly[0]], axis=0) - \
            np.mean([poly[3], poly[2]], axis=0)

    angle = np.arctan2(axis_vector[1], axis_vector[0]) % np.pi

    return a, b, angle


def poly_to_rect_array(poly: np.ndarray):
    assert poly.shape[1:] == (4, 2)
    if len(poly) == 0:
        return np.empty((0, 3))

    # norm_axis_1 = np.mean([np.linalg.norm(poly[:, 0] - poly[:, 1], axis=-1),
    #                        np.linalg.norm(poly[:, 2] - poly[:, 3])], axis=-1)
    # norm_axis_2 = np.mean([np.linalg.norm(poly[:, 1] - poly[:, 2], axis=-1),
    #                        np.linalg.norm(poly[:, 3] - poly[:, 0])], axis=-1)

    norm_axis_1 = np.mean(np.linalg.norm(
        poly[:, [0, 2]] - poly[:, [1, 3]], axis=-1), axis=-1)
    norm_axis_2 = np.mean(np.linalg.norm(
        poly[:, [1, 3]] - poly[:, [2, 0]], axis=-1), axis=-1)

    a = np.minimum(norm_axis_1, norm_axis_2)
    b = np.maximum(norm_axis_1, norm_axis_2)
    axis_vector = np.where(np.expand_dims(norm_axis_1 < norm_axis_2, axis=-1),
                           np.mean(poly[:, [1, 2]], axis=1) -
                           np.mean(poly[:, [0, 3]], axis=1),
                           np.mean(poly[:, [1, 0]], axis=1) -
                           np.mean(poly[:, [3, 2]], axis=1)
                           )

    angle = np.arctan2(axis_vector[:, 1], axis_vector[:, 0]) % np.pi

    return np.stack([a, b, angle], axis=-1)


def n_poly_to_rect(poly: np.ndarray, q=0.0):
    # https://stackoverflow.com/questions/70893304/finding-a-minimum-area-rectangle-of-a-convex-polygon
    assert len(poly.shape) == 2
    assert poly.shape[1] == 2
    if np.any(np.all(poly[0, :] == poly[:, :], axis=0)):
        logging.warning(
            f"found object with same coordinates on one axis: {poly}")
        return None
    convex = ConvexHull(poly)
    convex_points = poly[convex.vertices]
    convex_points = np.concatenate(
        [convex_points, convex_points[[-1]]], axis=0)
    diff = np.diff(convex_points, axis=0)
    angles = np.arctan2(diff[:, 0], diff[:, 1])
    n = len(poly)
    poly = poly.reshape((1, -1, 2))
    angles = angles.reshape((-1, 1))
    new_x = np.cos(angles) * poly[..., 0] - np.sin(angles) * poly[..., 1]
    new_y = np.sin(angles) * poly[..., 0] + np.cos(angles) * poly[..., 1]

    sqx = np.quantile(new_x, q=[q, 1 - q], axis=-1)
    sx = sqx[1] - sqx[0]
    sqy = np.quantile(new_y, q=[q, 1 - q], axis=-1)
    sy = sqy[1] - sqy[0]
    # sx = np.max(new_x ,axis=-1) - np.min(new_x, axis=-1)
    # sy = np.max(new_y, axis=-1) - np.min(new_y, axis=-1)
    area = sx * sy
    argmin_area = np.argmin(area)
    alpha = angles[argmin_area, 0]
    sx = sx[argmin_area]
    sy = sy[argmin_area]
    if sx < sy:
        a, b = sx, sy
        alpha = alpha + np.pi / 2
    else:
        a, b = sy, sx
    alpha = alpha % np.pi
    # x, y = np.mean(poly[0], axis=0)

    xx, yy = draw.polygon(poly[0, :, 0], poly[0, :, 1])
    x = np.mean(xx)
    y = np.mean(yy)
    if not a <= b:
        raise RuntimeError(f"{a=} should be lower than {b=} !")
    return y, x, a, b, alpha
