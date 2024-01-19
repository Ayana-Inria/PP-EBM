import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.random import Generator
from scipy.ndimage import distance_transform_edt
from skimage import draw
from skimage.segmentation import watershed
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.base_model import BaseModel
from base.data import get_dataset_base_path
from base.geometry import rect_to_poly
from base.mappings import mappings_from_config, values_to_class_id
from base.misc import timestamp
from base.parse_config import ConfigParser
from base.train_utils import update_metrics, print_metrics
from base.trainer import BaseTrainer
from data.augmentation import DataAugment
from data.image_dataset import ImageDataset, LabelProcessor
from data.patch_making import make_patch_dataset


@dataclass
class PatchLabelProcessor(LabelProcessor):

    def __init__(self, config, maps_to_compute: List[str], mode: str, rng: Generator):
        self.rng = rng
        self.config = config
        self.maps_to_compute = maps_to_compute
        self.mode = mode
        if 'marks' in maps_to_compute:
            self.mappings = mappings_from_config(self.config)

    def process(self, patch: np.ndarray, centers: np.ndarray, params: np.ndarray, idx: int) -> Tuple[
            Tensor, Dict[str, Tensor]]:
        label_dict = {}
        center_bin_map = np.zeros(patch.shape[:2], dtype=bool)
        centers = centers.astype(int)
        for k, c in enumerate(centers):
            try:
                center_bin_map[c[0], c[1]] = 1
            except IndexError as e:
                logging.info(
                    f"point ({c}) out of bounds in patch of shape {patch.shape}: {e}")

        distance = distance_transform_edt(1 - center_bin_map)
        if 'distance' in self.maps_to_compute:
            label_dict['distance'] = torch.tensor(distance)

        if 'vector' in self.maps_to_compute:
            vector_seed = np.zeros(patch.shape[:2], dtype=int)
            for k, c in enumerate(centers):
                try:
                    vector_seed[c[0], c[1]] = k + 1
                except IndexError as e:
                    pass
            all_centers = np.array(centers)
            vector_map = watershed(distance, vector_seed) - 1
            # vector_map = all_centers[vector_map.ravel()].reshape(patch.shape[:2]+(2,))
            if len(centers) == 0:
                pointy_map = np.zeros(patch.shape[:2] + (2,))
            else:
                vector_map = all_centers[vector_map]

                coor_map = np.stack(
                    np.mgrid[:patch.shape[0], :patch.shape[1]], axis=-1)

                pointy_map = vector_map - coor_map
            norm = np.linalg.norm(pointy_map, axis=-1) + 1e-8

            pointy_map = pointy_map / np.stack((norm, norm), axis=-1)
            pointy_map[np.isnan(pointy_map)] = 0

            label_dict['vector'] = torch.tensor(
                pointy_map, dtype=torch.float).permute((2, 0, 1))

        if 'object_mask' in self.maps_to_compute or 'marks' in self.maps_to_compute:
            if len(params) > 0:
                classes = values_to_class_id(
                    params, self.mappings, as_tensor=False)
            else:
                classes = [[], [], []]
            if self.mode == 'train' and len(classes) > 0:
                for i in range(3):
                    pert = self.rng.normal(
                        scale=self.config['data_loader']['mark_class_sigma'], size=len(
                            params)
                    ).astype(int)
                    if self.mappings[i].is_cyclic:
                        classes[i] = (
                            classes[i] + pert) % self.mappings[i].n_classes
                    else:
                        classes[i] = np.clip(
                            classes[i] + pert, 0, self.mappings[i].n_classes - 1)

            value_class_map = [
                np.zeros(patch.shape[:2], dtype=int) for _ in range(3)]
            loss_mask = np.zeros(patch.shape[:2], dtype=bool)
            for k, (c, p) in enumerate(zip(centers, params)):
                a, b, w = p
                object_mask = draw.polygon2mask(
                    patch.shape[:2],
                    rect_to_poly(c, a, b, w))
                loss_mask += object_mask
                for i in range(3):
                    value_class_map[i][object_mask] = classes[i][k]
            if len(centers) == 0:
                loss_mask = np.zeros(loss_mask.shape)
            else:
                s = np.sum(loss_mask)
                if s != 0:
                    loss_mask = loss_mask / s

            label_dict['object_mask'] = loss_mask
            label_dict['mark_width'] = value_class_map[0]
            label_dict['mark_length'] = value_class_map[1]
            label_dict['mark_angle'] = value_class_map[2]

        patch = torch.tensor(patch).permute((2, 0, 1))
        return patch, label_dict


class EnergyMapTrainer(BaseTrainer):

    def __init__(self, model: Union[Module, BaseModel], criterion, optimizer, config: ConfigParser,
                 rng: Generator, force_dataset: str = None, scheduler=None):
        super().__init__(model, criterion, optimizer, config, scheduler)
        self.config = config
        self.rng = rng
        self.dataset = self.config['data_loader']["dataset"] if force_dataset is None else force_dataset
        self.error_update_interval = self.config['data_loader'].get(
            "error_update_interval")
        self.temp_dataset = 'temp_' + self.config['name'] + '_' + timestamp()
        self.error_densities = None
        self.n_epochs = self.config['trainer']['n_epochs']
        self.batch_size = self.config['trainer']['batch_size']
        self.figure_interval = self.config['trainer']['figure_interval']

        reuse_data = False
        collate_fn = None

        if not reuse_data:
            make_patch_dataset(new_dataset=self.temp_dataset,
                               source_dataset=self.dataset,
                               config=self.config.config,
                               make_val=True,
                               rng=self.rng)
        self.dataset_update_interval = self.config["data_loader"]['dataset_update_interval']

        augmenter = DataAugment(
            rng=self.rng,
            dataset=self.dataset, subset='train',
            **self.config['data_loader'].get('augment_params')
        ) if 'augment_params' in self.config['data_loader'] else None

        self.data_train = ImageDataset(
            dataset=self.temp_dataset,
            subset='train',
            rgb=True,
            rng=self.rng,
            augmenter=augmenter,
            label_processor=PatchLabelProcessor(
                config=self.config, mode='train',
                maps_to_compute=self.criterion.maps_for_loss, rng=rng
            )
        )

        self.data_val = ImageDataset(
            dataset=self.temp_dataset,
            subset='val',
            rgb=True,
            rng=self.rng,
            label_processor=PatchLabelProcessor(
                config=self.config, mode='val',
                maps_to_compute=self.criterion.maps_for_loss, rng=rng
            )
        )

        self.train_loader = DataLoader(self.data_train, batch_size=self.batch_size, num_workers=8, prefetch_factor=16,
                                       collate_fn=collate_fn)
        self.val_loader = DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=8,
                                     prefetch_factor=16, collate_fn=collate_fn)
        self.figure_loader = DataLoader(
            self.data_val, batch_size=8, shuffle=True, collate_fn=collate_fn)

        self.images_figs, self.label_figs = self.figure_loader.__iter__().next()

    def train_epoch(self):
        self.model.train()
        all_metrics = None
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model.forward(data.to(self.model.device))
            loss_dict = self.criterion.forward(
                output, {k: v.to(self.model.device) for k, v in target.items()})
            loss_dict['loss'].backward()
            self.optimizer.step()
            all_metrics = update_metrics(loss_dict, all_metrics)
        return all_metrics

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        all_metrics = None
        for batch_idx, (data, target) in enumerate(self.val_loader):
            output = self.model.forward(data.to(self.model.device))
            loss_dict = self.criterion.forward(
                output, {k: v.to(self.model.device) for k, v in target.items()})
            all_metrics = update_metrics(loss_dict, all_metrics)
        return all_metrics

    @torch.no_grad()
    def make_figures(self, epoch: int):
        self.model.eval()
        output = self.model.forward(self.images_figs.to(self.model.device))
        loss_dict = self.criterion.forward(
            output, {k: v.to(self.model.device) for k, v in self.label_figs.items()})
        if callable(getattr(self.model, 'make_figures')):
            image = self.model.make_figures(epoch=epoch, inputs=self.images_figs, output=output, labels=self.label_figs,
                                            loss_dict=loss_dict)
            image_path = os.path.join(
                self.config.save_dir, f'res_{epoch:04}.png')
            plt.imsave(image_path, image)
            shutil.copy(image_path, os.path.join(
                self.config.save_dir, f'res_last.png'))

    def make_plots(self):
        with open(os.path.join(self.config.save_dir, 'log.json'), "r") as f:
            log = json.load(f)

        keys = log.keys()
        epochs = log['epoch']
        sub_losses_keys = list(set(
            [re.match(r'(?:(?:train)|(?:val))_(.*)', k).group(1) for k in keys
             if ('train_' in k or 'val_' in k)]))
        sub_losses_keys.remove('loss')
        sub_losses_keys.sort()
        sub_losses_keys = ['loss'] + sub_losses_keys
        n_rows, n_cols = 1, len(sub_losses_keys)
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

        cmap = plt.get_cmap('tab10')

        for i, k in enumerate(sub_losses_keys):
            train_key = f'train_{k}'
            axs[i].plot(epochs, log[train_key], color=cmap(i), label=train_key)
            val_key = f'val_{k}'
            axs[i].plot(epochs, log[val_key], color=cmap(i),
                        label=val_key, ls='--', alpha=0.65)
            axs[i].set_title(k)
        fig.tight_layout()
        plt.savefig(os.path.join(self.config.save_dir,
                    'loss_plot.png'), bbox_inches='tight')
        plt.close('all')

    def train(self):
        for epoch in range(self.start_epoch, self.n_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.val_epoch()
            print_metrics(epoch, train_metrics, val_metrics)
            self.config.logger.update_train_val(
                epoch, train_metrics, val_metrics)
            if self.scheduler is not None:
                self.scheduler.step()
            self.make_plots()

            if epoch % self.figure_interval == 0 or epoch == self.n_epochs - 1:
                self.make_figures(epoch)

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

            rescale_fac = 1 / 8
            if epoch % self.dataset_update_interval == 0 and epoch != 0:
                if self.error_update_interval is not None and epoch % self.error_update_interval == 0:
                    print('computing errors')
                    self.error_densities = self.model.compute_errors(
                        rescale_fac=rescale_fac
                    )
                # error_maps = self.compute_errors(os.path.join(self.data_path, 'train'))
                logging.info("remaking patch dataset")
                make_patch_dataset(new_dataset=self.temp_dataset,
                                   source_dataset=self.dataset,
                                   config=self.config.config,
                                   make_val=False,
                                   sampling_densities=self.error_densities,
                                   densities_rescale_fac=rescale_fac,
                                   d_sampler_weight=1 / 2,
                                   rng=self.rng)
                self.data_train.update_files()
                logging.info("patch dataset done, resuming !")

        self.make_figures(self.n_epochs)
        self._save_checkpoint(epoch=self.n_epochs)
        print("Saved model")
        self.clean()
        print("cleared temp files")

    def clean(self):
        shutil.rmtree(os.path.join(get_dataset_base_path(), self.temp_dataset))

    def eval(self):
        raise NotImplementedError

    def infer(self, overwrite_results: bool, draw: bool):
        raise NotImplementedError
