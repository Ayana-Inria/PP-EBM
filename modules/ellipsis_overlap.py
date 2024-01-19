import numpy as np
import torch
from torch.nn import Module, functional


# @torch.jit.script
def ellipsis_interstices_jit(state_1, state_2, distance_matrix, norm_p: float, eps: float):
    n_sets = state_1.shape[0]
    n_points = state_1.shape[1]
    n_others = state_2.shape[1]

    radius_short = (state_1[:, :, 2] / 2).view((n_sets, n_points, 1))
    radius_long = (state_1[:, :, 3] / 2).view((n_sets, n_points, 1))
    positions = state_1[:, :, :2]

    radius_short_other = (state_2[:, :, 2] / 2).view((n_sets, 1, n_others))
    radius_long_other = (state_2[:, :, 3] / 2).view((n_sets, 1, n_others))
    positions_other = state_2[:, :, :2]

    vectors = positions_other.view(
        (n_sets, 1, n_others, 2)) - positions.view((n_sets, n_points, 1, 2))
    vectors = vectors + \
        torch.all(vectors == 0, dim=-
                  1).view((n_sets, n_points, n_others, 1)) * 128
    alpha = torch.atan2(vectors[..., 1], vectors[..., 0]
                        ) * torch.any(vectors != 0, dim=-1)

    pointing_1 = state_2[:, :, 4].view((n_sets, 1, n_others)) - alpha
    pointing_2 = state_1[:, :, 4].view(
        (n_sets, n_points, 1)) - (alpha + 3.141592653589793)

    radius_1 = (radius_long_other * radius_short_other) / (torch.pow(
        torch.pow(torch.cos(pointing_1) * radius_long_other, norm_p) +
        torch.pow(torch.abs(torch.sin(pointing_1))
                  * radius_short_other, norm_p),
        1 / norm_p
    ) + eps)
    radius_2 = (radius_long * radius_short) / (torch.pow(
        torch.pow(torch.cos(pointing_2) * radius_long, norm_p) +
        torch.pow(torch.sin(pointing_2) * radius_short, norm_p),
        1 / norm_p
    ) + eps)

    radius_1 = torch.clip(radius_1, eps, torch.inf)
    radius_2 = torch.clip(radius_2, eps, torch.inf)

    return distance_matrix - radius_1 - radius_2


# @torch.jit.script
def ellipsis_overlap_jit(state_1, state_2, distance_matrix, norm_p: float, eps: float):
    n_sets = state_1.shape[0]
    n_points = state_1.shape[1]
    n_others = state_2.shape[1]

    radius_short = (state_1[:, :, 2] / 2).view((n_sets, n_points, 1))
    radius_long = (state_1[:, :, 3] / 2).view((n_sets, n_points, 1))
    positions = state_1[:, :, :2]

    radius_short_other = (state_2[:, :, 2] / 2).view((n_sets, 1, n_others))
    radius_long_other = (state_2[:, :, 3] / 2).view((n_sets, 1, n_others))
    positions_other = state_2[:, :, :2]

    vectors = positions_other.view(
        (n_sets, 1, n_others, 2)) - positions.view((n_sets, n_points, 1, 2))
    vectors = vectors + \
        torch.all(vectors == 0, dim=-
                  1).view((n_sets, n_points, n_others, 1)) * 128
    alpha = torch.atan2(vectors[..., 1], vectors[..., 0]
                        ) * torch.any(vectors != 0, dim=-1)

    pointing_1 = state_2[:, :, 4].view((n_sets, 1, n_others)) - alpha
    pointing_2 = state_1[:, :, 4].view(
        (n_sets, n_points, 1)) - (alpha + 3.141592653589793)

    radius_1 = (radius_long_other * radius_short_other) / (torch.pow(
        torch.pow(torch.cos(pointing_1) * radius_long_other, norm_p) +
        torch.pow(torch.abs(torch.sin(pointing_1))
                  * radius_short_other, norm_p),
        1 / norm_p
    ) + eps)
    radius_2 = (radius_long * radius_short) / (torch.pow(
        torch.pow(torch.cos(pointing_2) * radius_long, norm_p) +
        torch.pow(torch.sin(pointing_2) * radius_short, norm_p),
        1 / norm_p
    ) + eps)

    radius_1 = torch.clip(radius_1, eps, torch.inf)
    radius_2 = torch.clip(radius_2, eps, torch.inf)

    interstices = distance_matrix - radius_1 - radius_2

    # keep only neg interstices (ie overlaps)
    interstices = functional.relu(-interstices)
    return interstices / (radius_1 + radius_2)


# @torch.jit.script
def dividing_axis_gap(state_1, state_2, std_intersect: bool):
    n_sets = state_1.shape[0]
    n_1 = state_1.shape[1]
    n_2 = state_2.shape[1]
    # state_dim = state_1.shape[-1]

    offsets = torch.tensor([0.0, 1.5707963267948966],
                           device=state_1.device).view((1, 1, 2))

    proj_vec_1 = torch.view_as_real(torch.polar(
        abs=torch.ones((n_sets, n_1, 2), device=state_1.device),
        angle=state_1[:, :, [4]] - offsets))  # (n_sets,n_1,2,2)
    proj_vec_2 = torch.view_as_real(torch.polar(
        abs=torch.ones((n_sets, n_2, 2), device=state_2.device),
        angle=state_2[:, :, [4]] - offsets))  # (n_sets,n_2,2,2)

    # proj_vectors = torch.concat(
    #     [
    #         torch.broadcast_to(proj_vec_1.unsqueeze(dim=2), (n_sets, n_1, n_2, 2, 2)),
    #         torch.broadcast_to(proj_vec_2.unsqueeze(dim=1), (n_sets, n_1, n_2, 2, 2))  #
    #     ], dim=3
    # ) # 4 projection axis, 2 per rectangle

    # states to rectangle coordinates
    coord_idx = [[2, 3], [2, 3], [2, 3], [2, 3]]
    coord_mul = 0.5 * torch.tensor([[1, 1], [1, -1], [-1, -1], [-1, 1]], device=state_1.device
                                   ).unsqueeze(dim=0).unsqueeze(dim=0)
    rect_local_coord_1 = state_1[:, :, coord_idx] * coord_mul
    rect_local_coord_2 = state_2[:, :, coord_idx] * coord_mul

    rect_coord_1 = torch.sum(
        proj_vec_1.unsqueeze(dim=2) * rect_local_coord_1.unsqueeze(dim=-1),
        dim=-2
    ) + state_1[..., :2].unsqueeze(dim=2)

    rect_coord_2 = torch.sum(
        proj_vec_2.unsqueeze(dim=2) * rect_local_coord_2.unsqueeze(dim=-1),
        dim=-2
    ) + state_2[..., :2].unsqueeze(dim=2)

    # project 1 to axes from 1
    # # maybe there is a shortcut ?

    rect_coord_1 = rect_coord_1.unsqueeze(dim=2)
    rect_coord_2 = rect_coord_2.unsqueeze(dim=2)
    proj_vec_1 = proj_vec_1.unsqueeze(dim=3)
    proj_vec_2 = proj_vec_2.unsqueeze(dim=3)

    s1_proj_1 = torch.sum(
        rect_coord_1 * proj_vec_1,
        dim=-1
    ).unsqueeze(dim=2)
    s1_proj_2 = torch.sum(
        rect_coord_1.unsqueeze(dim=2) * proj_vec_2.unsqueeze(dim=1),
        dim=-1
    )
    s2_proj_2 = torch.sum(
        rect_coord_2 * proj_vec_2,
        dim=-1
    ).unsqueeze(dim=1)
    s2_proj_1 = torch.sum(
        rect_coord_2.unsqueeze(dim=1) * proj_vec_1.unsqueeze(dim=2),
        dim=-1
    )

    s1_proj_1 = torch.broadcast_to(s1_proj_1, s2_proj_1.shape)
    s2_proj_2 = torch.broadcast_to(s2_proj_2, s1_proj_2.shape)

    proj_1 = torch.stack([
        torch.min(s1_proj_1, dim=-1)[0], torch.max(s1_proj_1, dim=-1)[0],
        torch.min(s2_proj_1, dim=-1)[0], torch.max(s2_proj_1, dim=-1)[0]
    ], dim=-1)
    proj_2 = torch.stack([
        torch.min(s1_proj_2, dim=-1)[0], torch.max(s1_proj_2, dim=-1)[0],
        torch.min(s2_proj_2, dim=-1)[0], torch.max(s2_proj_2, dim=-1)[0]
    ], dim=-1)

    union_1 = torch.max(proj_1[:, :, :, :, [1, 3]], dim=-1)[0] - \
        torch.min(proj_1[:, :, :, :, [0, 2]], dim=-1)[0]
    union_2 = torch.max(proj_2[:, :, :, :, [1, 3]], dim=-1)[0] - \
        torch.min(proj_2[:, :, :, :, [0, 2]], dim=-1)[0]
    inter_1 = torch.max(proj_1[:, :, :, :, [0, 2]], dim=-1)[0] - \
        torch.min(proj_1[:, :, :, :, [1, 3]], dim=-1)[0]
    inter_2 = torch.max(proj_2[:, :, :, :, [0, 2]], dim=-1)[0] - \
        torch.min(proj_2[:, :, :, :, [1, 3]], dim=-1)[0]

    # union_1 = proj_1[:, :, :, :, 3] - proj_1[:, :, :, :, 0]
    # inter_1 = proj_1[:, :, :, :, 2] - proj_1[:, :, :, :, 1]
    # union_2 = proj_2[:, :, :, :, 3] - proj_2[:, :, :, :, 0]
    # inter_2 = proj_2[:, :, :, :, 2] - proj_2[:, :, :, :, 1]

    if std_intersect:
        gap_proj_1 = torch.where(inter_1 > 0, inter_1,
                                 inter_1 / (union_1 + 1e-12))
        gap_proj_2 = torch.where(inter_2 > 0, inter_2,
                                 inter_2 / (union_2 + 1e-12))
    else:
        gap_proj_1 = inter_1
        gap_proj_2 = inter_2

    gap = torch.maximum(
        torch.max(gap_proj_1, dim=-1)[0],
        torch.max(gap_proj_2, dim=-1)[0]
    )

    return gap


class EllipsisOverlap(Module):

    def __init__(self, norm_p: float):
        super(EllipsisOverlap, self).__init__()
        self.norm_p = norm_p
        self.pi = np.pi
        self.eps = 1e-5

    def forward(self, state_1, state_2, distance_matrix):
        return ellipsis_interstices_jit(
            state_1=state_1, state_2=state_2, distance_matrix=distance_matrix, norm_p=self.norm_p, eps=self.eps
        )
