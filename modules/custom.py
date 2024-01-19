import logging
import math
import warnings
from typing import List

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Module, Parameter, init, functional


class Scale(Module):
    def __init__(self, a: float, b: float = 0.0):
        super(Scale, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x: Tensor):
        return self.a * x + self.b


class SigmoidEnergy(Module):

    def __init__(self):
        super(SigmoidEnergy, self).__init__()

    def forward(self, x: Tensor):
        return 2 * torch.sigmoid(torch.sum(x, dim=-1)) - 1


class NegPosSigmoid(Module):

    def __init__(self):
        super(NegPosSigmoid, self).__init__()

    def forward(self, x: Tensor):
        return 2 * torch.sigmoid(x) - 1


class LinearEnergy(Module):
    def __init__(self):
        super(LinearEnergy, self).__init__()

    def forward(self, x: Tensor):
        return torch.sum(x, dim=-1)


class NegSoftplus(Module):
    def __init__(self, device: torch.device, learn_bias: bool = False):
        super(NegSoftplus, self).__init__()
        if learn_bias:
            self.bias = nn.Parameter(torch.zeros(1, device=device))
        else:
            self.bias = None

    def forward(self, x: Tensor):
        if self.bias is None:
            return functional.softplus(-x)
        else:
            return functional.softplus(-x + self.bias.to(x.device))


class NegLogSofmax(Module):
    def __init__(self, dim):
        super(NegLogSofmax, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        # = -x + log(sum_j(exp(x_j)))
        # see https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html?highlight=logsoftmax
        return - x + torch.logsumexp(x, dim=self.dim, keepdim=True)


def masked_softmax(x: Tensor, mask: Tensor, dim: int, eps=1e-5):
    maxes = torch.max(x, dim=dim, keepdim=True)[0]
    exp = torch.exp(x - maxes)
    masked_exp = exp * mask.float()
    masked_sum = masked_exp.sum(dim, keepdims=True) + eps
    res = masked_exp / masked_sum
    # if torch.any(torch.isnan(res)):
    #     print(f"{x=}")
    #     print(f"{exp=}")
    #     print(f"{mask.float()=}")
    #     print(f"{masked_exp=}")
    #     print(f"{res=}")
    #     print(f"{masked_sum=}")
    return res


def dot_product_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor, return_weight: bool = False):
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
    # attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
    assert attn_mask.dtype == torch.bool
    if attn_mask.shape[-1] != key.shape[-2]:
        raise RuntimeError(
            f"key ({key.shape}) and mask ({attn_mask.shape}) shape doe no match")
    if attn_mask.shape[-2] != query.shape[-2]:
        raise RuntimeError(
            f"query ({query.shape}) and mask ({attn_mask.shape}) shape doe no match")
    attn_mask = attn_mask.masked_fill(~attn_mask, -float('inf'))
    try:
        attn_weight = torch.softmax(
            (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1)
    except RuntimeError as e:
        raise RuntimeError(f"Failed with {e}\n"
                           f"{query.shape=}, {key.shape=}, {value.shape=}, {attn_mask.shape=}"
                           f"{query.size(-1)=}")
    # attn_weight = torch.dropout(attn_weight, dropout_p)
    if return_weight:
        return attn_weight @ value, attn_weight
    return attn_weight @ value


class MaskedAttention(Module):
    def __init__(self, in_features: int, device: torch.device, weight_model: str = 'linear',
                 mlp_hidden_dims: int = 4, mlp_layers: int = 2):
        super(MaskedAttention, self).__init__()
        self.in_feat = in_features
        if weight_model == 'linear':
            self.linear_w = nn.Linear(
                in_features=in_features, out_features=1, bias=False, device=device)
        elif weight_model == 'MLP':
            mini_mlp = []
            last_dim = in_features
            for i in range(mlp_layers - 1):
                mini_mlp = mini_mlp + [
                    nn.Linear(last_dim, mlp_hidden_dims, device=device),
                    nn.ReLU()
                ]
                last_dim = mlp_hidden_dims
            mini_mlp = mini_mlp + [
                nn.Linear(last_dim, 1, device=device)
            ]
            self.linear_w = nn.Sequential(*mini_mlp)
        else:
            raise ValueError

    def forward(self, values: Tensor, features: Tensor, mask: Tensor, reduce_dim: int, return_weights: bool = False):
        assert features.shape[-1] == self.in_feat
        weights = masked_softmax(
            x=self.linear_w(features).squeeze(dim=-1),
            mask=mask,
            dim=reduce_dim
        )
        # assert not torch.any(torch.isnan(weights))
        res = torch.sum(weights * values, dim=reduce_dim)
        if not return_weights:
            return res
        return res, weights


class TrueAttention(Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device, emb_dims: int, key_dim: int,
                 bounded: bool, allow_neg_output: bool,
                 mlp_emb_hidden_dims: int = 4,
                 mlp_emb_layers: int = 2, share_query_key_linear: bool = False):
        super(TrueAttention, self).__init__()
        self.in_feat = in_features
        self.emb_dims = emb_dims
        self.key_dim = key_dim

        mini_mlp = []
        last_dim = in_features
        for i in range(mlp_emb_layers - 1):
            mini_mlp = mini_mlp + [
                nn.Linear(last_dim, mlp_emb_hidden_dims, device=device),
                nn.ReLU()
            ]
            last_dim = mlp_emb_hidden_dims
        mini_mlp = mini_mlp + [
            nn.Linear(last_dim, emb_dims, device=device)
        ]
        self.emb_model = nn.Sequential(*mini_mlp)

        self.linear_k = nn.Linear(
            in_features=emb_dims, out_features=key_dim, bias=True, device=device)
        if not share_query_key_linear:
            self.linear_q = nn.Linear(
                in_features=emb_dims, out_features=key_dim, bias=True, device=device)
        else:
            self.linear_q = self.linear_k

        ops = [nn.Linear(in_features=emb_dims,
                         out_features=out_features, bias=True, device=device)]
        if bounded:
            if not allow_neg_output:
                ops.append(nn.Sigmoid())
            else:
                ops.append(NegPosSigmoid())

        self.linear_v = nn.Sequential(*ops)

    def forward(self, x: Tensor, mask: Tensor, y=None, return_weights: bool = False):

        x_emb = self.emb_model(x)
        if y is None:
            y = x
            y_emb = x_emb
        else:
            y_emb = self.emb_model(y)
        mask_shape = mask.shape
        if mask_shape[-2] != y_emb.shape[-2] or mask_shape[-1] != x_emb.shape[-2]:
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
            raise RuntimeError(f"mask shape does not match the inputs shape "
                               f"x ({x_emb.shape}) and y (or x) ({y_emb.shape}) with mask ({mask_shape})")

        res, weights = dot_product_attention(
            query=self.linear_q(y_emb),
            key=self.linear_k(x_emb),
            value=self.linear_v(x_emb),
            attn_mask=mask,
            return_weight=True
        )
        if not return_weights:
            return res
        return res, weights


# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class PositiveLinear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    _weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._weight = Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self._weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self._weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> Tensor:
        return torch.abs(self._weight)

    def forward(self, input: Tensor) -> Tensor:
        return functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class LinearWDeltas(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, delta: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearWDeltas, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        if delta:
            self.delta = Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('delta', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.delta is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.delta, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return functional.linear(input, self.weight - self.delta)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, delta={}'.format(
            self.in_features, self.out_features, self.delta is not None
        )


class NormalisedLinear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    _weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NormalisedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._weight = Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self._weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self._weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> Tensor:
        return self._weight / torch.linalg.norm(self._weight, ord=2, dim=1, keepdim=True)

    def forward(self, input: Tensor) -> Tensor:
        return functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class FullNormalisedLinear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    _weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FullNormalisedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._weight = Parameter(torch.empty(
            (out_features, in_features + bias), **factory_kwargs))
        self.reset_parameters()
        self._bias = bias

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self._weight, a=math.sqrt(5))

    @property
    def weight(self) -> Tensor:
        res = self._weight / \
            torch.linalg.norm(self._weight, ord=2, dim=1, keepdim=True)
        if self._bias:
            return res[:, :-1]
        else:
            return res

    @property
    def bias(self) -> Tensor:
        res = self._weight / \
            torch.linalg.norm(self._weight, ord=2, dim=1, keepdim=True)
        if self._bias:
            return res[:, [-1]]
        else:
            return None

    def forward(self, input: Tensor) -> Tensor:
        w = self._weight / \
            torch.linalg.norm(self._weight, ord=2, dim=1, keepdim=True)
        if self._bias:
            return functional.linear(input, w[:, :-1], w[:, [-1]])
        else:
            return functional.linear(input, self.weight)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ScaleModule(Module):

    def __init__(self, v_min, v_max, is_cyclic: List[bool], device: torch.device):
        super(ScaleModule, self).__init__()
        self.v_min = v_min.to(device) if type(
            v_min) is Tensor else torch.tensor(v_min).to(device)
        self.v_max = v_max.to(device) if type(
            v_max) is Tensor else torch.tensor(v_max).to(device)

        self.v_min = self.v_min.view((1, -1, 1, 1))
        self.v_max = self.v_max.view((1, -1, 1, 1))

        self.range = self.v_max - self.v_min
        if any(is_cyclic):
            logging.warning("cyclic not take into account yet")
        self.is_cyclic = is_cyclic

    def forward(self, vec: Tensor):
        vec_cos = torch.cos(vec * np.pi)
        return (vec_cos + 1) / 2 * self.range + self.v_min
