from torch import nn, Tensor
from torch.nn import Module
import torch


class MLP(Module):

    def __init__(self, in_features: int, out_features: int, hidden_features: int, hidden_layers: int,
                 device: torch.device):
        super(MLP, self).__init__()

        layers = [
            nn.Linear(in_features=in_features,
                      out_features=hidden_features, device=device),
            nn.ReLU(),
        ]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(in_features=hidden_features,
                          out_features=hidden_features, device=device))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features=hidden_features,
                      out_features=out_features, device=device))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)
