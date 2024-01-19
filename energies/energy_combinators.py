import warnings
from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from torch import Tensor, nn
from torch.nn import Module

from modules.custom import SigmoidEnergy, LinearEnergy, PositiveLinear, NegPosSigmoid, NormalisedLinear, \
    FullNormalisedLinear, NegSoftplus, LinearWDeltas
from modules.mlp import MLP


class BaseEnergyCombinator(Module, ABC):

    def __init__(self, sub_energies_names: List[str], device: torch.device):
        super(BaseEnergyCombinator, self).__init__()
        self.sub_energies_names = sub_energies_names
        self.device = device

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> Dict[str, float]:
        raise NotImplementedError


class ManualEnergyCombinator(BaseEnergyCombinator):

    def __init__(self, sub_energies_names: List[str], device: torch.device,
                 weights: Dict[str, float], sigmoid: bool, thresholds: Dict[str, float] = None, bias: float = 0):
        super(ManualEnergyCombinator, self).__init__(
            sub_energies_names, device)

        self.weights = torch.tensor(
            [weights[k] for k in sub_energies_names], device=self.device)
        if thresholds is not None:
            self.thresholds = torch.tensor(
                [thresholds[k] for k in sub_energies_names], device=self.device)
        else:
            self.thresholds = torch.tensor(
                [0 for _ in sub_energies_names], device=self.device)
        self.sigmoid = sigmoid
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        res = torch.sum((x - self.thresholds) *
                        self.weights, dim=-1) + self.bias
        if self.sigmoid:
            return 2 * torch.sigmoid(res) - 1
        else:
            return res

    def describe(self) -> Dict[str, float]:
        return {
            **{f'weight_{k}': float(v.detach().cpu()) for k, v in
               zip(self.sub_energies_names, self.weights)},
            **{f'thresh_{k}': float(v.detach().cpu()) for k, v in
               zip(self.sub_energies_names, self.thresholds)}
        }


class LinearCombinator(BaseEnergyCombinator):
    def __init__(self, sub_energies_names: List[str], device: torch.device, bias: bool = True, init_ones: bool = True,
                 weights_constraint: str = 'none',
                 positive_weights: bool = None, normalised_weights: bool = None, sigmoid: bool = False,
                 softplus: bool = False, use_deltas: bool = False):
        super(LinearCombinator, self).__init__(
            sub_energies_names=sub_energies_names, device=device
        )
        self.use_bias = bias
        assert not (sigmoid and softplus)

        if positive_weights or normalised_weights:
            warnings.warn("positive_weights and normalised_weights are deprecated, please use weights_constraint",
                          DeprecationWarning, stacklevel=2)
            assert not (positive_weights and normalised_weights)

            if positive_weights:
                weights_constraint = 'positive'
            elif normalised_weights:
                weights_constraint = 'normalized'

        if weights_constraint is None:
            weights_constraint = 'none'

        self.use_deltas = use_deltas
        if not self.use_deltas:

            if weights_constraint == 'positive':
                ops = [PositiveLinear(in_features=len(
                    self.sub_energies_names), out_features=1, bias=self.use_bias)]
            elif weights_constraint == 'normalized':
                ops = [NormalisedLinear(in_features=len(
                    self.sub_energies_names), out_features=1, bias=self.use_bias)]
            elif weights_constraint == 'full_normalized':
                ops = [FullNormalisedLinear(in_features=len(
                    self.sub_energies_names), out_features=1, bias=self.use_bias)]
            elif weights_constraint == 'none':
                ops = [nn.Linear(in_features=len(
                    self.sub_energies_names), out_features=1, bias=self.use_bias)]
            else:
                raise ValueError
             # todo that is a very bad name ! it just sums stuff
        else:
            if weights_constraint == 'none':
                ops = [LinearWDeltas(in_features=len(
                    self.sub_energies_names), out_features=1, delta=True)]
            else:
                raise NotImplementedError

        ops.append(LinearEnergy())

        if sigmoid:
            ops.append(NegPosSigmoid())
        if softplus:
            ops.append(NegSoftplus(device=self.device, learn_bias=True))

        self.model = nn.Sequential(*ops)
        if init_ones:
            self.model[0].weight.data.fill_(1.0)
            self.model[0].bias.data.fill_(0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)

    def describe(self) -> Dict[str, float]:
        if not self.use_deltas:
            return {
                **{
                    f'weight_{k}': float(v.detach().cpu()) for k, v in
                    zip(self.sub_energies_names, self.model[0].weight[0])
                },
                'bias': float(self.model[0].bias[0].detach().cpu())
            }
        else:
            return {
                **{
                    f'weight_{k}': float(v.detach().cpu()) for k, v in
                    zip(self.sub_energies_names, self.model[0].weight[0])
                },
                **{
                    f'delta_{k}': float(v.detach().cpu()) for k, v in
                    zip(self.sub_energies_names, self.model[0].delta)
                }
            }

    @property
    def weights(self) -> Tensor:
        return self.model[0].weight

    @property
    def bias(self) -> Tensor:
        return self.model[0].bias


class LogisticCombinator(BaseEnergyCombinator):

    def __init__(self, sub_energies_names: List[str], device: torch.device, init_ones: bool = True):
        super(LogisticCombinator, self).__init__(
            sub_energies_names=sub_energies_names, device=device
        )

        self.model = nn.Sequential(
            nn.Linear(in_features=len(self.sub_energies_names),
                      out_features=1, bias=True),
            SigmoidEnergy()
        )
        if init_ones:
            self.model[0].weight.data.fill_(1.0)
            self.model[0].bias.data.fill_(0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)

    def describe(self) -> Dict[str, float]:
        return {
            f'weight_{k}': float(v.detach().cpu()) for k, v in
            zip(self.sub_energies_names, self.model[0].weight[0])
        }


class MLPCombinator(BaseEnergyCombinator):

    def __init__(self, sub_energies_names: List[str], hidden_dims: int, layers: int, device: torch.device,
                 sigmoid: bool = True):
        super(MLPCombinator, self).__init__(
            sub_energies_names=sub_energies_names, device=device
        )

        self.model = nn.Sequential(
            MLP(in_features=len(sub_energies_names), out_features=1,
                hidden_features=hidden_dims, hidden_layers=layers),
            SigmoidEnergy() if sigmoid else LinearEnergy()
        )
        self.sub_energies = sub_energies_names

    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)

    def describe(self) -> Dict[str, float]:
        n = len(self.sub_energies_names)
        input_vector = torch.zeros((1, n, n), device=self.device)
        input_vector[0, range(n), range(n)] = 1.0
        pseudo_weights = self.forward(input_vector)[0]
        return {
            f'weight_{k}': float(v.detach().cpu()) for k, v in
            zip(self.sub_energies_names, pseudo_weights)
        }
