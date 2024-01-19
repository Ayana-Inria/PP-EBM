import torch
from torch import Tensor


# @torch.jit.script
def compute_distances_in_sets(points: Tensor, points_mask: Tensor, others: Tensor, others_mask: Tensor,
                              maximum_dist: float):
    n_points = points.shape[1]
    n_others = others.shape[1]
    n_sets = points.shape[0]
    points_pos = points[..., :2]
    points_pos = points_pos.reshape((n_sets, n_points, 1, 2))
    other_pos = others[..., :2]
    other_pos = other_pos.reshape((n_sets, 1, n_others, 2))

    self_dist_mask = torch.zeros(
        (n_sets, n_points, n_others), device=points.device, dtype=torch.bool)
    self_dist_mask[torch.all(points_pos == other_pos, dim=-1)] = True

    dist = torch.sqrt(
        torch.sum(torch.square(points_pos - other_pos), dim=-1) + 1e-8
    )

    points_mask = ~points_mask.reshape((n_sets, n_points, 1))
    others_mask = ~others_mask.reshape((n_sets, 1, n_others))

    dist_mask = others_mask | points_mask | self_dist_mask

    dist = dist + dist_mask * (maximum_dist + 1)
    #
    # dist = torch.nan_to_num(dist, maximum_dist + 1, maximum_dist + 1, maximum_dist + 1)

    return dist


# @torch.jit.script
def compute_distances_and_marks_diffs_in_sets(points: Tensor, points_mask: Tensor, others: Tensor, others_mask: Tensor,
                                              maximum_dist: float):
    n_points = points.shape[1]
    n_others = others.shape[1]
    n_sets = points.shape[0]
    state_dim = points.shape[-1]

    diff_tensor = points.reshape(
        (n_sets, n_points, 1, state_dim)) - others.reshape((n_sets, 1, n_others, state_dim))

    diff_pos = diff_tensor[..., :2]

    # self_dist_mask = torch.zeros((n_sets, n_points, n_others), device=points.device, dtype=torch.bool)
    # self_dist_mask[torch.all(diff_pos == 0.0, dim=-1)] = True

    dist = torch.sqrt(
        torch.sum(torch.square(diff_pos), dim=-1) + 1e-8
    )

    points_mask = ~points_mask.reshape((n_sets, n_points, 1))
    others_mask = ~others_mask.reshape((n_sets, 1, n_others))

    dist_mask = others_mask | points_mask | torch.all(diff_pos == 0.0, dim=-1)

    dist = dist + dist_mask * (maximum_dist + 1)
    return dist, diff_tensor
