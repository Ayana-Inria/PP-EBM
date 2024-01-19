from abc import ABC, abstractmethod
from typing import List

import torch
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy, cross_entropy


class BaseCriterion(ABC):

    @abstractmethod
    def forward(self, output, target):
        raise NotImplementedError

    @property
    @abstractmethod
    def maps_for_loss(self) -> List[str]:
        raise NotImplementedError


class PosAndShapeCriterion(BaseCriterion, Module):

    def __init__(self, vector_distance_to_center: float, blob_sigma: float, learn_masked_vector: bool,
                 heatmap_loss: bool = True):
        super(PosAndShapeCriterion, self).__init__()

        self.vector_distance_to_center = vector_distance_to_center
        self.blob_sigma = blob_sigma
        self.learn_masked_vector = learn_masked_vector
        self.heatmap_loss = heatmap_loss

        self.eps = 1e-5

    @property
    def maps_for_loss(self) -> List[str]:
        return ['vector', 'distance', 'object_mask', 'marks']

    def forward(self, output, target):
        # output_mask = output['mask']
        # output_vec = output['vector']
        # center_heatmap = output['']

        target_mask = (target['distance'] <
                       self.vector_distance_to_center).float()

        mask_sig = output['mask']
        if self.learn_masked_vector:
            vector_loss = torch.sum(torch.square(
                output['vector'] - target['vector']), dim=1) * target_mask
            vector_loss = torch.sum(vector_loss, dim=(
                1, 2)) / torch.sum(target_mask, dim=(1, 2))
            vector_loss = torch.mean(vector_loss)
        else:
            vector_loss = torch.sum(torch.square(
                (output['vector'] * mask_sig) - target['vector']), dim=1)
            vector_loss = torch.mean(vector_loss)

        mask_loss = binary_cross_entropy(
            mask_sig.squeeze(dim=1), target_mask, reduction='none')
        mask_loss = torch.mean(mask_loss)

        if self.heatmap_loss:
            heatmap_sig = torch.sigmoid(
                output['center_heatmap']).squeeze(dim=1)
            target_heatmap = torch.exp(- 0.5 * torch.square(
                target['distance']) / (self.blob_sigma ** 2))

            beta = 1 - torch.sum(target_heatmap) / torch.numel(target_heatmap)
            heatmap_loss = -beta * target_heatmap * torch.log(heatmap_sig + self.eps) \
                           - (1 - beta) * (1 - target_heatmap) * \
                torch.log(1 - heatmap_sig + self.eps)
            heatmap_loss = torch.mean(heatmap_loss)
        else:
            heatmap_loss = torch.zeros(1, device=mask_sig.device)

        marks = [k for k in output.keys() if 'mark_' in k]
        marks_losses = {}
        sum_marks_losses = 0
        for m in marks:
            mark_loss = cross_entropy(
                input=output[m], target=target[m], reduction='none')
            mark_loss = mark_loss * target['object_mask']
            mark_loss = torch.sum(mark_loss, dim=(1, 2))
            mark_loss = torch.mean(mark_loss)
            marks_losses[m] = mark_loss
            sum_marks_losses = sum_marks_losses + mark_loss

        return {
            'vector': vector_loss,
            'mask': mask_loss,
            'heatmap': heatmap_loss,
            **marks_losses,
            'loss': vector_loss + mask_loss + heatmap_loss + sum_marks_losses
        }


class DummyLoss(Module):

    def __init__(self):
        super(DummyLoss, self).__init__()

    def forward(self, output, target):
        pass
