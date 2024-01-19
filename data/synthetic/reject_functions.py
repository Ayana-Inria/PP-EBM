from typing import List, Callable

import numpy as np


def distance_reject(distance_threshold: float):
    def reject_fn(p1, p2):
        return np.linalg.norm(p1[:2] - p2[:2]) < distance_threshold

    return reject_fn


def radius_reject(margin=0):
    def reject_fn(p1, p2):
        return np.linalg.norm(p1[:2] - p2[:2]) - p1[2] - p2[2] - margin < 0

    return reject_fn


def composite_reject(reject_fns: List[Callable]):
    def reject_fn(p1, p2):
        return any([r(p1, p2) for r in reject_fns])

    return reject_fn
