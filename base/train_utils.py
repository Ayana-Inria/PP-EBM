from typing import Dict, List

import numpy as np
from torch import Tensor

MetricsDict = Dict[str, List[float]]


def update_metrics(loss_dict: Dict[str, Tensor], metrics: MetricsDict) -> MetricsDict:
    if metrics is None:
        metrics = {k: [v.detach().cpu().numpy()] for k, v in loss_dict.items()}
    else:
        for k in loss_dict:
            metrics[k].append(loss_dict[k].detach().cpu().numpy())
    return metrics


def print_metrics(epoch: int, train_metrics: MetricsDict, val_metrics: MetricsDict):
    print(f"[{epoch:04}] Train ", end='')
    for k, v in train_metrics.items():
        print(f"{k}: {np.mean(v):.3f} ", end='')
    print(f"| Eval ", end='')
    for k, v in val_metrics.items():
        print(f"{k}: {np.mean(v):.3f} ", end='')
    print('')
