import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Union, Any, TypeVar
import numpy as np
from matplotlib import pyplot as plt

DictOfLists = Dict[str, List]


def append_lists_in_dict(dict_to_update: DictOfLists, new_values: Dict[str, Union[float, str]], add_keys=True,
                         prefix: str = None):
    for k, v in new_values.items():
        if prefix is not None:
            k = prefix + k
        if k in dict_to_update:
            dict_to_update[k].append(v)
        elif add_keys:
            dict_to_update[k] = [v]
        else:
            raise KeyError


def reduce_dict(input_dict):
    new_dict = {}
    for k, v in input_dict.items():
        new_dict[k] = np.mean(v)
    return new_dict


def timestamp() -> str:
    return datetime.now().strftime("%y%m%d-%H%M%S")


def running_avg(vec, window):
    v = np.pad(vec, (window, 0), mode='edge')
    cumsumvec = np.cumsum(v)
    smooth_vec = (cumsumvec[window:] - cumsumvec[:-window]) / window
    return smooth_vec


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


def save_figure(save_path: str, name: str, tight: bool = False, **kwargs):
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_file = os.path.join(save_path, name)
    if tight:
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0, **kwargs)
    else:
        plt.savefig(save_file, **kwargs)
    plt.close('all')
    logging.info(f"saved figure at {save_file}")


KeyType = TypeVar('KeyType')


def deep_update(mapping: Dict[KeyType, Any], *updating_mappings: Dict[KeyType, Any]) -> Dict[KeyType, Any]:
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping
