import logging
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from base.images import patchify_big_image, unpatchify_big_image


class BaseModel(ABC):

    @abstractmethod
    def make_figures(self, epoch: int, inputs, output, labels, loss_dict) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def infer_on_large_image(self, x: Tensor, margin: int = 34, patch_size: int = 128, batch_size: int = 4, verbose=0,
                             return_split_info: bool = False, restore_batch_dim: bool = True, **kwargs):
        self.eval()
        assert 2 * margin < patch_size
        original_size = x.shape
        patches, padded_image, cuts = patchify_big_image(
            image=x, margin=margin, patch_size=patch_size)
        # logging.info(f"splitting image into {patches.shape[0]} patches")
        redo = True
        while redo:
            try:
                inferred_patches = {}
                batches = np.array_split(np.arange(len(patches)), int(
                    np.ceil(len(patches) / batch_size)))
                if verbose > 0:
                    batches = tqdm(batches, desc='inferring on batches')
                for b in batches:
                    res = {k: v.cpu() for k, v in self.forward(
                        patches[b].to(self.device)).items()}
                    for k, v in res.items():
                        if k in inferred_patches:
                            inferred_patches[k].append(v)
                        else:
                            inferred_patches[k] = [v]
                redo = False
            except RuntimeError as e:
                if batch_size == 1:
                    raise e
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
                logging.error(
                    f"memory error, too big batch or patch size : retry with {batch_size=}")

        inferred_patches = {k: torch.concat(
            v, dim=0) for k, v in inferred_patches.items()}
        inferred_maps = {}
        for k, v in inferred_patches.items():
            inferred_maps[k] = unpatchify_big_image(
                v, original_size=original_size, margin=margin, cuts=cuts)
            if restore_batch_dim:
                inferred_maps[k] = torch.unsqueeze(inferred_maps[k], dim=0)
        if return_split_info:
            split_info = {
                'patches': patches, 'padded_image': padded_image, 'cuts': cuts, 'margin': margin,
                'patch_size': patch_size, 'inferred_patches': inferred_patches, 'batch_size': batch_size
            }
            return inferred_maps, split_info
        return inferred_maps
