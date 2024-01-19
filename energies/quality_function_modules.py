import numpy as np
import torch
from torch.nn import Module
from torch import nn, Tensor

from base.mappings import ValueMapping
from energies.quality_functions import remap


class Relu(Module):
    def __init__(self, trainable: bool, device: torch.device, bias_value: float = None, soft: bool = False,
                 soft_beta: float = 10.0, positive_bias: bool = False):
        super(Relu, self).__init__()
        self.device = device

        if bias_value is not None:
            self._bias = torch.tensor([bias_value], device=self.device)
        else:
            self._bias = torch.zeros((1,), device=self.device)
        if trainable:
            self._bias = nn.Parameter(self._bias)

        self.positive_bias = positive_bias

        if not soft:
            self.non_linear = nn.ReLU()
        else:
            self.non_linear = nn.Softplus(beta=soft_beta)

    @property
    def bias(self) -> Tensor:
        if self.positive_bias:
            return torch.square(self._bias)
        else:
            return self._bias

    def forward(self, x: Tensor):
        return self.non_linear(x - self.bias)


class Sigmoid(Module):
    def __init__(self, trainable: bool, device: torch.device, bias_value=None, slope_value=None):
        # slope value ~ 10, bias_value ~0.25
        super(Sigmoid, self).__init__()

        self.linear = None
        self.bias_value = None
        self.slope_value = None
        if trainable:
            self.linear = nn.Linear(
                in_features=1, out_features=1, device=device)
        else:
            if bias_value is None or slope_value is None:
                raise ValueError(
                    "if trainable is False must specify bias_value and slope_value")
            self.bias_value = torch.tensor(bias_value, device=device)
            self.slope_value = torch.tensor(slope_value, device=device)

    def forward(self, x: Tensor):
        if self.linear is not None:
            return torch.sigmoid(self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1))
        else:
            return torch.sigmoid(self.slope_value * (x - self.bias_value))


class Cosine(Module):
    def __init__(self, mapping: ValueMapping):
        # slope value ~ 10, bias_value ~0.25
        super(Cosine, self).__init__()

        self.v_min = mapping.v_min
        self.v_max = mapping.v_max

    def forward(self, x: Tensor):
        return -torch.cos((x - self.v_min) / (self.v_max - self.v_min) * np.pi) * 0.5 + 0.5
