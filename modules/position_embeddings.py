import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Module


class PositionEmbedding(Module):

    def __init__(self, periods: list, scale_factor: float, device: torch.device):
        super(PositionEmbedding, self).__init__()
        self.device = device
        self.scale = scale_factor
        self.d = len(periods)
        self.max_translate = max(periods)

        self.w = self.scale * \
            (2 * np.pi / torch.tensor(periods, device=self.device)).view((1, self.d))

    def sample_trl_and_rot(self):
        trl = self.max_translate * (2 * torch.rand(size=(1, 2)) - 1)
        phi = torch.rand(size=(1,)) * np.pi * 2
        s = torch.sin(phi)
        c = torch.cos(phi)
        rot = torch.stack([torch.stack([c, -s]),
                           torch.stack([s, c])]).view((2, 2))

        return trl, rot

    def forward(self, x: Tensor, translation, rotation) -> Tensor:
        """
        :param x: tensor of coordinates of shape (...,2)
        :return: tensor of embeddings of shape (...,2D) where D is the number of periods
        :rtype:
        """
        # shape = x.shape
        if translation is not None:
            x = x + translation
        if rotation is not None:
            x = torch.matmul(torch.unsqueeze(x, dim=-2),
                             rotation).squeeze(dim=-2)

        sin_x = torch.sin(self.w * x[..., [0]])
        sin_y = torch.sin(self.w * x[..., [1]])

        return torch.concat((sin_x, sin_y), dim=-1)
