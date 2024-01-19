import torch
from torch import Tensor
from torch.nn.functional import grid_sample

from base.mappings import ValueMapping


def interpolate_position(positions: Tensor, image: Tensor, mode: str = 'bilinear') -> Tensor:
    """

    :param mode: nearest, bilinear or bicubic
    :type mode: str
    :param positions: tensor of coordinates (B,K,P,2)
    :type positions: Tensor
    :param image: image tensor (B,C,H,W)
    :type image: Tensor
    :return: tensor of values at positions (B,K,P,C)
    :rtype: Tensor
    """
    assert len(image.shape) == 4
    assert len(positions.shape) == 4
    h, w = image.shape[2:]
    samples = torch.stack([
        positions[..., 1] / (w - 1),
        positions[..., 0] / (h - 1)
    ], dim=-1) * 2 - 1
    return grid_sample(image, samples, mode=mode, align_corners=True, padding_mode='border')


def interpolate_marks(positions: Tensor, marks: Tensor, image: Tensor, mapping: ValueMapping,
                      mode: str = 'bilinear', mask=None) -> Tensor:
    """

    :param positions: tensor of coordinates (B,L,K,P,2)
    :type positions: Tensor
    :param marks: tensor of marks values (B,L,K,P)
    :type marks: Tensor
    :param image: image tensor (B,C,D,H,W)
    :type image: Tensor
    :param mapping: mapping of the current mark
    :type mapping: ValueMapping
    :return: tensor of shape (B,C,L,K,P)
    :rtype:
    """

    assert len(image.shape) == 5
    assert len(positions.shape) == 5
    assert len(marks.shape) == 4
    assert image.shape[2] == mapping.n_classes
    if mask is not None:
        assert torch.all(marks[mask] <= mapping.v_max)
        assert torch.all(marks[mask] >= mapping.v_min)

    h, w = image.shape[3:]
    samples = torch.stack([
        positions[..., 1] / (w - 1),
        positions[..., 0] / (h - 1),
        (marks - mapping.v_min) / (mapping.v_max - mapping.v_min),
    ], dim=-1) * 2 - 1
    return grid_sample(image, samples, mode=mode, padding_mode='border', align_corners=True)


if __name__ == '__main__':
    import numpy as np

    n_c = 8
    shape = (16, 16)
    n_pt = 32
    dist = np.arange(n_c)
    image = np.empty(shape + (n_c,))
    image[:, :] = dist
    # image[:8] = image[:8] + 100
    image[8:] = -1
    image[:, 8:] = -1

    rng = np.random.default_rng(0)

    # limit = shape
    limit = (7, 7)
    positions = rng.uniform((0, 0), limit, size=(n_pt, 2))

    print(f"{dist=}")
    print(f"{positions=}")

    mapping = ValueMapping('test', n_c, v_min=0, v_max=1)

    # marks = rng.uniform(0, 1, size=(n_pt))
    marks = np.linspace(0, 1, n_pt)
    # marks = mapping.feature_mapping
    print(f"{marks=}")

    res = interpolate_marks(
        positions=torch.from_numpy(positions).view((1, 1, 1, n_pt, 2)),
        marks=torch.from_numpy(marks).view((1, 1, 1, n_pt)),
        mapping=mapping,
        image=torch.from_numpy(image).permute(
            (2, 0, 1)).view((1, 1, n_c, shape[0], shape[1]))
    ).view((n_pt))

    dists_interp = interpolate_position(
        positions=torch.from_numpy(positions).view((1, 1, n_pt, 2)),
        image=torch.from_numpy(image).permute(
            (2, 0, 1)).view((1, n_c, shape[0], shape[1]))
    ).view((n_c, n_pt)).T

    print(f"{dists_interp[0]=}")

    print(f"{res=}")

    res_np = res.squeeze().numpy()
    assert np.all(res_np >= 0)
    assert np.all(np.diff(res_np) > 0)
