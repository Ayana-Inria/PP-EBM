from typing import Callable, Optional, Dict

from torch import Tensor

EnergyFunc = Callable[[Tensor, Tensor], Dict[str, Tensor]]
