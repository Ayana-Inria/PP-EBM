import logging
from typing import Dict

import numpy as np
import torch
import yaml
from skimage.feature import peak_local_max
from torch import Tensor
from torch.nn import Module

from base.base_model import BaseModel
from base.data import get_model_config_by_name
from base.mappings import mappings_from_config
from base.misc import append_lists_in_dict
from base.parse_config import ConfigParser
from base.state_ops import maps_local_max_state
from energies.sub_energies_modules import likelihood
from energies.sub_energies_modules.base import LikelihoodModule
from energies.sub_energies_modules.likelihood import CNNEnergies
from models.mpp_data_net.model import MPPDataModel


class NaiveCnn(BaseModel, Module):
    def __init__(self, config: ConfigParser, device=None):
        super(NaiveCnn, self).__init__()
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        logging.info(f"device is {self.device}")
        self.config = config
        mpp_data_net = config['model']['mpp_data_net']
        map_model_config = get_model_config_by_name(mpp_data_net)
        with open(map_model_config, 'r') as f:
            map_model_config = yaml.load(f, Loader=yaml.SafeLoader)
        map_model_config = ConfigParser(
            map_model_config, 'MPPDataModel', load_model=True, resume=True)
        self.maps_module: MPPDataModel = MPPDataModel(map_model_config)
        checkpoint = torch.load(map_model_config.resume, )
        self.maps_module.load_state_dict(checkpoint['state_dict'])

        self.device = self.maps_module.device
        logging.debug(
            f"loaded MPPDataModel with config :\n{repr(map_model_config)}")
        self._mappings = mappings_from_config(self.maps_module.config)

    def make_figures(self, epoch: int, inputs, output, labels, loss_dict) -> np.ndarray:
        pass

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        pass

    def infer_on_image(self, images: Tensor, lm_distance=5, lm_thresh=0.5, large_image: bool = False):
        if len(images.shape) == 3:
            images = torch.unsqueeze(images, dim=0)

        if large_image:
            if images.shape[0] > 1:

                raise RuntimeError(f"energy_maps_from_image with large_image=True only works on one image, "
                                   f"image shape is {images.shape}")
            output = self.maps_module.infer_on_large_image(
                images[0].to(self.device), margin=68 // 2, patch_size=128)
            # output = {k: v.unsqueeze(dim=0) for k, v in output.items()}
        else:
            if len(images.shape) == 3:
                images = images.unsqueeze(dim=0)
            output = self.maps_module.forward(images)

        # output = self.maps_module.forward(images)

        center_heatmap = torch.sigmoid(output['center_heatmap'])
        # marks_maps = [torch.softmax(output[f'mark_{m.name}'], dim=1).detach().cpu().numpy() for m in self._mappings]
        marks_maps = [output[f'mark_{m.name}'].detach(
        ).cpu().numpy() for m in self._mappings]

        results = {}

        for i in range(len(images)):
            # local maxima
            heatmap = center_heatmap[i].detach().cpu().numpy()
            # xy = peak_local_max(heatmap[0], min_distance=lm_distance, threshold_abs=lm_thresh)
            #
            # marks_per_point = [
            #     mapping.class_to_value(np.argmax(mm[i, :, xy[:,0], xy[:,1]], axis=1))
            #     for mm, mapping in zip(marks_maps, self._mappings)
            # ]
            #
            # proposed_state = np.concatenate([xy,np.stack(marks_per_point,axis=-1)],axis=-1)
            proposed_state = maps_local_max_state(
                position_map=heatmap[0],
                mark_maps=[m[i] for m in marks_maps],
                mappings=self._mappings,
                local_max_thresh=lm_thresh, local_max_distance=lm_distance
            )
            proposal_scores = heatmap[0, proposed_state[:, 0].astype(
                int), proposed_state[:, 1].astype(int)]

            append_lists_in_dict(results, {
                'heatmap': heatmap,
                'proposals': proposed_state,
                'scores': proposal_scores
            })

        return results
