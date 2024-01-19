import numpy as np
import torch
from skimage.draw import draw
from torch import Tensor


def square_coordinates(res: int):
    """
    returns coordinates for res**2 samples in a square from [-0.5,-0.5] to [0.5,0.5]
    :param res: resolution
    :return:
    """
    res = res - 1
    samples_x, samples_y = np.array(draw.rectangle((0, 0), (res, res)))
    samples = np.stack([samples_x.ravel(), samples_y.ravel()], axis=-1)
    samples = samples / res - 0.5
    return torch.tensor(samples)


def square_perimeter_coordinates(res: int, dilation: int = 1):
    res = res - 1
    samples_base_x, samples_base_y = np.array(
        draw.rectangle((0, 0), (res, res)))
    samples_dil_x, samples_dil_y = np.array(draw.rectangle(
        (-dilation, -dilation), (res + dilation, res + dilation)))

    samples_base = np.stack(
        [samples_base_x.ravel(), samples_base_y.ravel()], axis=-1)
    samples_dil = np.stack(
        [samples_dil_x.ravel(), samples_dil_y.ravel()], axis=-1)
    samples = np.array(list(set(map(tuple, samples_dil)) -
                       set(map(tuple, samples_base))))
    samples = samples / res - 0.5
    return torch.tensor(samples)


def make_affine2(dx, dy, sx, sy, theta):
    # https://en.wikipedia.org/wiki/Affine_transformation

    # mat = torch.tensor([
    #     [sx * torch.cos(theta), -sx * torch.sin(theta), sx * dx],
    #     [sy * torch.sin(theta), sy * torch.cos(theta), sy * dy],
    # ], requires_grad=True)

    mat = torch.stack([
        torch.stack([sx * torch.cos(theta), -sy *
                    torch.sin(theta), dx], dim=-1),
        torch.stack([sx * torch.sin(theta), sy *
                    torch.cos(theta), dy], dim=-1),
        torch.tensor([0, 0, 1], device=dx.device).repeat(sx.shape + (1,))
    ], dim=-2)

    return mat


def affine_from_state(point: Tensor):
    # point as [x,y,sx,sy,theta]
    return make_affine2(
        dx=point[..., 0],
        dy=point[..., 1],
        sx=point[..., 2],
        sy=point[..., 3],
        theta=point[..., 4]
    )


def affine_from_point(point: Tensor):
    # point as [x,y,sx,sy,theta]
    return make_affine2(
        dx=point[0],
        dy=point[1],
        sx=point[2],
        sy=point[3],
        theta=point[4]
    )


def contrast_measure(fill, outer, dim=1):
    mean_fill = torch.mean(fill, dim=dim)
    mean_outer = torch.mean(outer, dim=dim)
    var_fill = torch.var(fill, dim=dim)
    var_outer = torch.var(outer, dim=dim)

    return torch.sqrt((var_outer + var_fill) / (torch.square(mean_outer - mean_fill) + 1e-16))
