from typing import Dict

import numpy as np
from torch import Tensor
from torch.nn import Module

from base.base_model import BaseModel


class DummyMPPDataModel(Module, BaseModel):
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return {
            'center_heatmap': x
        }

    def make_figures(self, epoch: int, inputs, output, labels, loss_dict) -> np.ndarray:
        pass
