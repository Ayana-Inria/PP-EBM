import logging
import os
import pickle
import re
import shutil
import sys
import time
import traceback
import warnings
from typing import Tuple, Dict, Union, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from numpy.random import Generator
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from base.data import get_inference_path, fetch_data_paths
from base.files import make_if_not_exist
from base.geometry import rect_to_poly
from base.mappings import ValueMapping
from base.misc import timestamp, append_lists_in_dict
from base.parse_config import ConfigParser
from base.state_ops import clip_state_to_bounds, check_inbound_state
from base.timer import Timer
from base.trainer import BaseTrainer
from data.augmentation import DataAugment
from data.image_dataset import ImageDataset, LabelProcessor
from data.patch_making import make_patch_dataset
from data.state_transforms import BasicAugmenter
from display.draw_on_img import draw_shapes_on_img
from display.mpp_model_results import make_figure, analyse_results
from energies.base import BaseEnergyModel
from energies.generic_energy_model import GenericEnergyModel
from metrics.dota_results_translator import DOTAResultsTranslator
from metrics.papangelou import compute_papangelou_scoring, papangelou_score_scale
from samplers.rjmcmc import ParallelRJMCMC
from trainers import mpp_norm
from trainers.mpp_norm import EnergyReg, Void
from trainers.mpp_trainer_memory import StateMemory


def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())


yaml.add_representer(np.ndarray, ndarray_representer)


class MPPLabelProcessor(LabelProcessor):

    def process(self, patch: np.ndarray, centers: np.ndarray, params: np.ndarray, idx: int) -> Tuple[
            Tensor, Dict[str, Union[Tensor, List[Tensor]]]]:
        label = {
            'gt_state': torch.from_numpy(np.concatenate((centers, params), axis=-1)),
            'id': idx
        }
        patch_t = torch.from_numpy(patch).permute((2, 0, 1))
        return patch_t, label


def my_collate(batch):
    data = torch.stack([item[0] for item in batch], dim=0)
    keys = batch[0][1].keys()
    target = {k: [item[1][k] for item in batch] for k in keys}
    return data, target


def add_noise_to_state(state: Tensor, sigma: Tensor, min_bound: Tensor, max_bound: Tensor, cyclic: Tensor):
    state_w_noise = state + \
        torch.randn(size=state.shape, device=state.device) * \
        sigma.to(state.device)
    return clip_state_to_bounds(state_w_noise, min_bound, max_bound, cyclic)


class MPPTrainer(BaseTrainer):
    USE_DEFAULT_PARAMS = True

    def __init__(self, model: BaseEnergyModel, criterion, optimizer, config: ConfigParser,
                 rng: Generator, force_dataset: str = None, scheduler=None):
        super(MPPTrainer, self).__init__(model=model, criterion=criterion, optimizer=optimizer, config=config,
                                         scheduler=scheduler)

        self.config = config
        self.rng = rng
        self.dataset = self.config['data_loader']["dataset"] if force_dataset is None else force_dataset
        self.error_update_interval = self.config['data_loader'].get(
            "error_update_interval")
        self.temp_dataset = 'temp_' + self.config['name'] + '_' + timestamp()
        self.error_densities = None
        self.n_epochs = self.config['trainer']['n_epochs']
        self.reduce_mode = self.config['trainer']['reduce']
        self.batch_size = self.config['trainer']['batch_size']
        self.figure_interval = self.config['trainer']['figure_interval']

        self.model: Union[Module, BaseEnergyModel] = model
        self.mappings: List[ValueMapping] = self.model.mappings

        self.fast_energy_delta_compute = self.config['trainer']['fast_energy_delta_compute']
        if self.fast_energy_delta_compute:
            logging.warning("fast_energy_delta_compute set to True : energy is not computed on context, "
                            "thus may lead to inaccuracies when computing energy deltas")

        self.gt_as_init = self.config['trainer'].get('gt_as_init', 0)
        self.memorise_mc_metadata = self.config['trainer'].get(
            'memorise_mc_metadata', True)
        self.memory_size = self.config['trainer'].get('memory_size', None)
        self.memory: Union[StateMemory, None]
        self.train_memory_save_path = os.path.join(
            self.config.save_dir, 'state_train_memory_checkpoint.pkl')
        self.val_memory_save_path = os.path.join(
            self.config.save_dir, 'state_val_memory_checkpoint.pkl')

        if self.memory_size is not None and self.memory_size > 0:
            if self.gt_as_init > 0:
                # raise RuntimeError("cannot use BOTH gt_as_init and non-zero memory: pick one !")
                logging.info(f"using both gt_as_init and memory: "
                             f"will first pick from memory w/ proba memory_proba, "
                             f"otherwise pick from gt w/ proba gt_as_init, "
                             f"otherwise init randomly")
            if os.path.exists(self.train_memory_save_path):
                logging.info(
                    f"loading previous state memeory from {self.train_memory_save_path}")
                with open(self.train_memory_save_path, 'rb') as f:
                    self.train_memory: StateMemory = pickle.load(f)
            else:
                self.train_memory = StateMemory(
                    memory_size=self.memory_size,
                    memory_proba=self.config['trainer']['memory_proba']
                )
            if os.path.exists(self.val_memory_save_path):
                logging.info(
                    f"loading previous state memeory from {self.val_memory_save_path}")
                with open(self.val_memory_save_path, 'rb') as f:
                    self.val_memory: StateMemory = pickle.load(f)
            else:
                self.val_memory = StateMemory(
                    memory_size=self.memory_size,
                    memory_proba=self.config['trainer']['memory_proba']
                )
        else:
            self.train_memory = None
            self.val_memory = None

        random_configurations_sampling = self.config['trainer'].get(
            'random_configurations_sampling', 'uniform')
        if random_configurations_sampling == 'uniform':
            self.rd_init_config = 'random_uniform'
        elif random_configurations_sampling == 'sampler':
            self.rd_init_config = 'random_sampler'
        else:
            raise ValueError(f'{random_configurations_sampling=} not valid')

        regul_method = self.config['trainer'].get('regul_method', 'Void')
        self.regularizer: EnergyReg = getattr(mpp_norm, regul_method)(
            **self.config['trainer'].get('regul_args', {}))
        logging.info(f"using regularization {self.regularizer}")

        self.gt_sigma = self.config['trainer'].get('gt_noise_sigma', None)
        if self.gt_sigma is not None:
            self.gt_sigma[2:] = [s * m.range / m.n_classes for s,
                                 m in zip(self.gt_sigma, self.mappings)]
            self.gt_sigma = torch.tensor(self.gt_sigma)

        self.mark_min_bound = [m.v_min for m in self.mappings]
        self.mark_max_bound = [m.v_max for m in self.mappings]
        self.mark_cyclic = [m.is_cyclic for m in self.mappings]

        self.config_rjmcmc = self.config['RJMCMC_params'].copy()
        rjmcmc_params_override = self.config['trainer'].get(
            'rjmcmc_params_override', {})
        for k, v in rjmcmc_params_override.items():
            self.config_rjmcmc[k] = v

        if type(self.config_rjmcmc['end_temperature']) is list:
            assert all(
                [type(t) is float for t in self.config_rjmcmc['end_temperature']])

        self.train_log_path = os.path.join(
            self.config.save_dir, 'train_log.yaml')

    def _init_data(self):

        self.temp_dataset = 'temp_' + self.config['name'] + '_' + timestamp()
        make_patch_dataset(
            new_dataset=self.temp_dataset,
            source_dataset=self.dataset,
            config=self.config.config,
            make_val=True,
            rng=self.rng
        )
        self.dataset_update_interval = self.config["data_loader"]['dataset_update_interval']

        self.augmentation = self.config["data_loader"].get(
            'augmentation', None)
        if self.augmentation is not None:
            # the rotation/flip/scale of gt is not translated in state memories
            assert 'static' in self.augmentation

            augmenter = DataAugment(
                rng=self.rng, dataset=self.dataset, subset='train', hist_match_images=True, aug_level=self.augmentation
            )
        else:
            augmenter = None

        self.data_train = ImageDataset(
            dataset=self.temp_dataset,
            subset='train',
            rgb=True,
            rng=self.rng,
            augmenter=augmenter,
            label_processor=MPPLabelProcessor()
        )
        self.train_memory.clear()

        self.data_val = ImageDataset(
            dataset=self.temp_dataset,
            subset='val',
            rgb=True,
            rng=self.rng,
            label_processor=MPPLabelProcessor()
        )
        self.val_memory.clear()

        self.use_training_sampler = 'sampler' in self.config['data_loader']

        if self.use_training_sampler:
            n_samples = self.config['data_loader']['sampler']['n_samples']
            self._training_samples = self.rng.choice(
                np.arange(len(self.data_train)), size=n_samples, replace=False)
            train_loader_params = {
                'sampler': SubsetRandomSampler(indices=self._training_samples)
            }
        else:
            train_loader_params = {'shuffle': True}
            self._training_samples = None

        self.train_loader = DataLoader(self.data_train, batch_size=self.batch_size, num_workers=8, prefetch_factor=16,
                                       collate_fn=my_collate, **train_loader_params)
        self.val_loader = DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=8,
                                     collate_fn=my_collate)

        self.figure_loader = DataLoader(self.data_val, batch_size=4, shuffle=True,
                                        collate_fn=my_collate)
        self.images_figs, self.label_figs = self.figure_loader.__iter__().next()

        self.aug = None
        if self.augmentation is not None:
            self.aug = BasicAugmenter(p_rotation=0.5, p_flip=0.5,
                                      shape=self.config['trainer'].get('draw_method', 'rectangle'))

    def update_data(self):
        logging.info("remaking patch dataset")
        make_patch_dataset(new_dataset=self.temp_dataset,
                           source_dataset=self.dataset,
                           config=self.config.config,
                           rng=self.rng,
                           make_val=False
                           )
        self.data_train.update_files()
        if self.train_memory is not None:
            self.train_memory.clear()
        logging.info("patch dataset done, resuming !")

    def _sample_contrastive(self, gt_states, pos_e_m, marks_e_m, image_ids, aug_ds: List, image_size: int,
                            memory: StateMemory):
        neg_states = []
        pos_states = []
        pert_states_meta = []
        temperatures = []
        self.model.eval()
        for i, gt in enumerate(gt_states):
            pos_e_map = pos_e_m[i]
            marks_e_map = [m[i] for m in marks_e_m]
            shape = pos_e_map.shape[1:]

            min_bound = torch.tensor(
                [0, 0] + self.mark_min_bound, device=gt.device)
            max_bound = torch.tensor(
                [shape[0], shape[1]] + self.mark_max_bound, device=gt.device)
            cyclic = torch.tensor(
                [False, False] + [m.is_cyclic for m in self.mappings], device=gt.device)

            gt = check_inbound_state(
                gt, min_bound, max_bound, cyclic, clip_if_oob=True)

            pos_density_map, marks_density_maps = self.model.densities_from_energy_maps(
                pos_e_map, marks_e_map)

            init_state, init_meta = None, {}
            init_n_step = 0
            if memory is not None:
                # may return init_state=None with proba 1-self.memory.memory_proba
                init_state, init_meta = memory.select_rd(
                    rng=self.rng, image_id=image_ids[i])
                if init_state is not None:
                    if self.memorise_mc_metadata:
                        if len(init_meta) == 0:
                            raise RuntimeError(
                                f"expected memory w/ metadata, got dict {init_meta}")
                        init_n_step = init_meta['total_n_steps']
                    if aug_ds is not None:
                        init_state = self.aug.transform_state(
                            aug_ds[i], init_state, image_size)
                        init_state = check_inbound_state(
                            init_state, min_bound, max_bound, cyclic, clip_if_oob=True)

            if init_state is None:  # either memory is None OR memory.select_rd returned None
                if self.gt_as_init > 0:
                    if self.gt_as_init > self.rng.random():  # init_state= gt, w/ proba self.gt_as_init
                        init_state = gt
                    else:
                        init_state = self.rd_init_config
                else:
                    init_state = self.rd_init_config

            c = self.config_rjmcmc.copy()
            if self.memorise_mc_metadata and init_meta is not None:  # use MC metadata to setup RJMCMC parameters
                if init_meta['start_temperature'] == 'constant':
                    logging.info(f"using memorised temperatures {init_meta}")
                    c['start_temperature'] = init_meta['start_temperature']
                    c['end_temperature'] = init_meta['end_temperature']
                else:
                    raise NotImplementedError(
                        "only supports constant temperature MC for now")  # todo

            # choose running temp in list
            elif type(self.config_rjmcmc['end_temperature']) is list:
                c['end_temperature'] = self.rng.choice(
                    self.config_rjmcmc['end_temperature'])
                logging.info(f"choose end_temperature {c['end_temperature']}")

            if self.config['trainer'].get('match_intensity', 'True'):
                gt_intensity = float(len(gt))
                logging.info(f"matching intensity with gt ({gt_intensity})")
                c['intensity'] = float(gt_intensity / np.prod(shape))
            elif 'intensity' in c:
                pass
            else:
                raise NotImplementedError(
                    "model inferred intensity is not implemented")
                # logging.info(f"intensity set to {image_intensity} (whole image)")

            pert_mc = ParallelRJMCMC(
                support_shape=shape,
                device=self.model.device,
                max_interaction_distance=self.model.max_interaction_distance,
                rng=self.rng,
                energy_func=self.model.energy_func_wrapper(position_energy_map=pos_e_map,
                                                           marks_energy_maps=marks_e_map,
                                                           compute_context=not self.fast_energy_delta_compute),
                position_birth_density=pos_density_map,
                mark_birth_density=marks_density_maps,
                mappings=self.mappings,
                init_state=init_state,
                debug_mode=self.config.debug,
                **c
            )
            start = time.perf_counter()
            pert_mc.run(verbose=0)
            elapsed_time = time.perf_counter() - start
            logging.info(f"sampled contrastive sample in {elapsed_time:.1e}s")
            last_state = pert_mc.current_state.detach()
            neg_states.append(last_state)
            temperatures.append(c['end_temperature'])
            meta = {
                'start_temperature': c['start_temperature'],
                'end_temperature': c['end_temperature'],
                'new_n_steps': pert_mc.n_steps,
                'total_n_steps': init_n_step + pert_mc.n_steps
            }
            if aug_ds is not None:
                meta = {**meta, 'augmentations': aug_ds[i]}
            if memory is not None:
                if aug_ds is not None:
                    upright_state = self.aug.reverse_transform_state(
                        aug_ds[i], last_state, image_size)
                else:
                    upright_state = last_state
                if self.memorise_mc_metadata:
                    memory.append(
                        image_id=image_ids[i], state=upright_state, metadata=meta)
                else:
                    memory.append(
                        image_id=image_ids[i], state=upright_state, metadata={})
            pert_states_meta.append(meta)

            if self.gt_sigma is not None:

                state_cyclic = torch.tensor([False, False] + self.mark_cyclic)
                if len(gt) > 0:
                    gt_w_noise = add_noise_to_state(
                        state=gt, sigma=self.gt_sigma, min_bound=min_bound, max_bound=max_bound,
                        cyclic=state_cyclic
                    )
                else:
                    gt_w_noise = gt

                pos_states.append(gt_w_noise)
            else:
                pos_states.append(gt)

        return pos_states, neg_states

    def _epoch(self, log, pbar, epoch: int, training: bool):

        epoch_log = {}
        if not training:
            self.model.eval()
            mode = 'val'
            loader = self.val_loader
            memory = self.val_memory
        else:
            mode = 'train'
            loader = self.train_loader
            memory = self.train_memory

        for batch_id, (images, labels) in enumerate(loader):
            timings = {'epoch': epoch,
                       'batch_progress': batch_id / loader.__len__()}
            gt_states = labels['gt_state']
            image_ids = labels['id']
            shape = images.shape[2:]
            image_size = shape[0]

            # image augmentation
            if training:
                with Timer() as timer:
                    if self.aug is not None:
                        assert image_size == shape[1]
                        aug_ds = [self.aug.draw(self.rng) for _ in images]
                        images = torch.stack([self.aug.transform_image_t(
                            d, i) for d, i in zip(aug_ds, images)], dim=0)
                        gt_states = [self.aug.transform_state(
                            d, s, image_size) for d, s in zip(aug_ds, gt_states)]
                    else:
                        aug_ds = None
                timings['augmentation'] = timer()
            else:
                aug_ds = None

            # compute energy maps
            with Timer() as timer:
                images = images.to(self.model.device)
                pos_e_m, marks_e_m = self.model.energy_maps_from_image(
                    images,
                    as_energies=True)
            timings['cnn_inference_1'] = timer()

            # Sample contrastive states
            with Timer() as timer:
                pos_states, neg_states = self._sample_contrastive(
                    gt_states=gt_states, pos_e_m=pos_e_m, marks_e_m=marks_e_m,
                    image_ids=image_ids, image_size=image_size, aug_ds=aug_ds,
                    memory=memory
                )
            timings['sampling_contrastive'] = timer()

            # has to be here to not accumulate grads from simulating the RJMCMC
            self.optimizer.zero_grad()
            if training:
                self.model.train()
            # compute energy maps (if grad on CNN)
            if training:
                with Timer() as timer:
                    if self.model.trainable_maps:
                        pos_e_m, marks_e_m = self.model.energy_maps_from_image(
                            images,
                            as_energies=True, requires_grad=True
                        )
                timings['cnn_inference_2'] = timer()

            # compute pos energies
            with Timer() as timer:
                pos_res_dict = [
                    self.model.forward_state(
                        s.to(self.model.device), pos_e_m[[i]], [
                            m[[i]] for m in marks_e_m],
                        compute_context=False) for i, s in enumerate(pos_states)
                ]
            timings['compute_pos'] = timer()

            # compute neg energies
            with Timer() as timer:
                neg_res_dict = [
                    self.model.forward_state(
                        s.to(self.model.device), pos_e_m[[i]], [
                            m[[i]] for m in marks_e_m],
                        compute_context=False) for i, s in enumerate(neg_states)
                ]
            timings['compute_neg'] = timer()

            if self.config['trainer'].get('scale_loss_w_temperature', False):
                raise NotImplementedError

            # compute norm
            with Timer() as timer:
                norms_per_patch = []
                norms_log_per_patch = {}
                for pos_d, neg_d in zip(pos_res_dict, neg_res_dict):
                    pos_point_energies = torch.ravel(pos_d['energy_per_point'])
                    neg_point_energies = torch.ravel(neg_d['energy_per_point'])
                    if self.regularizer.requires_sub_energies:
                        pos_sub_energies = {k: torch.ravel(pos_d[k])
                                            for k in self.model.sub_energies
                                            }
                        neg_sub_energies = {k: torch.ravel(neg_d[k])
                                            for k in self.model.sub_energies
                                            }
                    else:
                        pos_sub_energies, neg_sub_energies = None, None

                    patch_norm, patch_log_reg = self.regularizer.forward(
                        pos_energies=pos_point_energies, neg_energies=neg_point_energies,
                        model=self.model,
                        pos_sub_energies=pos_sub_energies, neg_sub_energies=neg_sub_energies
                    )

                    norms_per_patch.append(patch_norm)
                    append_lists_in_dict(norms_log_per_patch, patch_log_reg)
                norms_per_patch = torch.stack(norms_per_patch)
                log_reg = {k: float(np.sum(v))
                           for k, v in norms_log_per_patch.items()}
            timings['compute_norm'] = timer()

            with Timer() as timer:
                pos_config_energies = torch.stack(
                    [torch.sum(d['energy_per_point']) for d in pos_res_dict])
                neg_config_energies = torch.stack(
                    [torch.sum(d['energy_per_point']) for d in neg_res_dict])
                assert self.reduce_mode == 'sum'
                loss_per_patch = pos_config_energies - neg_config_energies + norms_per_patch
                loss = loss_per_patch.mean()
                if training:
                    loss.backward()
                    self.optimizer.step()
            timings['loss_backward'] = timer()

            # log everything
            with Timer() as timer:
                loss_float = float(loss.detach().cpu())
                norm_float = float(norms_per_patch.mean())
                e_pos = float(pos_config_energies.mean().detach().cpu())
                e_neg = float(neg_config_energies.mean().detach().cpu())
                n_pos = [len(s) for s in pos_states]
                n_neg = [len(s) for s in neg_states]
                e_pos_avg = float(
                    torch.concat([torch.ravel(d['energy_per_point']) for d in pos_res_dict],
                                 dim=0).mean().detach().cpu()
                )
                e_neg_avg = float(
                    torch.concat([torch.ravel(d['energy_per_point']) for d in neg_res_dict],
                                 dim=0).mean().detach().cpu()
                )
                weights_dict = self.model.combination_module_weights

                if self.memorise_mc_metadata:
                    temperatures = [
                        memory._mem_meta[img_id][memory.last_append_index(
                            img_id)]['end_temperature']
                        for img_id in image_ids]
                else:
                    temperatures = [None for _ in image_ids]

                log_dict = {
                    'epoch': epoch,
                    'batch_progress': batch_id / loader.__len__(),
                    'U(Y+)': e_pos, 'avg(U(y+))': e_pos_avg,
                    'U(Y-)': e_neg, 'avg(U(y-))': e_neg_avg,
                    'Reg': norm_float,
                    'Loss': loss_float,
                    **weights_dict,
                    **log_reg,
                    'patches': [int(img_id) for img_id in image_ids],
                    'patch_loss': loss_per_patch.detach().cpu().numpy(),
                    'patch_norms': norms_per_patch.detach().cpu().numpy(),
                    'patch_U(Y+)': pos_config_energies.detach().cpu().numpy(),
                    'patch_U(Y-)': neg_config_energies.detach().cpu().numpy(),
                    'patch_|Y+|': n_pos,
                    'patch_|Y-|': n_neg,
                    'patch_temp': temperatures
                }

                postfix_prefix = f"{mode}/"
                postfix_dict = {
                    postfix_prefix + 'batch': f"{batch_id / loader.__len__():.1%}",
                    postfix_prefix + 'U+': e_pos,
                    postfix_prefix + 'U-': e_neg,
                    postfix_prefix + 'L': loss_float
                }
                if training and (self.scheduler is not None):
                    logging.info(f'LR: {self.scheduler.get_last_lr()}')
                    lr = float(self.scheduler.get_last_lr()[0])

                    postfix_dict[postfix_prefix + 'lr'] = lr
                    log_dict['lr'] = lr
                if self.regularizer is not Void:
                    postfix_dict[postfix_prefix + 'Reg'] = norm_float

                pbar.set_postfix(postfix_dict)

                # logging.info(f"weights: {weights_dict}")

                append_lists_in_dict(epoch_log, log_dict)
                append_lists_in_dict(log, log_dict, prefix=f'{mode}/batch/')

            timings['logging'] = timer()

            with Timer() as timer:
                self._plot_log(
                    log=log
                )
            timings['plotting_log'] = timer()

            append_lists_in_dict(log, timings, prefix=f'timings/{mode}/batch/')

        timings = {'epoch': epoch}
        with Timer() as timer:
            if epoch % self.figure_interval == 0 or epoch == self.n_epochs - 1:
                try:
                    max_fig = 4
                    save_file = os.path.join(
                        self.config.save_dir, f'res_{epoch:04}_{mode}.png')
                    make_figure(
                        model=self.model,
                        images=images[:max_fig],
                        save_file=save_file,
                        states=neg_states[:max_fig], gt_states=pos_states[:max_fig],
                        pos_e_maps=[pos_e_m[[i]]
                                    for i in range(len(images[:max_fig]))],
                        marks_e_maps=[[marks_e_m[j][[i]] for j in range(len(marks_e_m))] for i in
                                      range(len(images[:max_fig]))],
                        draw_method=self.config['trainer'].get(
                            'draw_method', 'rectangle')
                    )
                    shutil.copy(save_file, os.path.join(
                        self.config.save_dir, f'res_last_{mode}.png'))
                except Exception as e:
                    logging.error(
                        f'make_figure failed with error:\n{e}\n{traceback.format_exc()}')
        timings['make_figure'] = timer()

        append_lists_in_dict(log, timings, prefix=f'timings/{mode}/')

        # reduce batch logs
        reduce_keys = ['U(Y+)', 'avg(U(y+))', 'U(Y-)', 'avg(U(y-))',
                       'Loss', 'Reg'] + [k for k in log_reg.keys()]
        append_lists_in_dict(log, {
            'epoch': epoch,
            **{k: float(np.mean(epoch_log[k])) for k in reduce_keys}
        }, prefix=f"{mode}/")

        return log

    def train(self):
        plt.ioff()
        self._init_data()
        try:
            make_figure(
                model=self.model,
                save_file=os.path.join(self.config.save_dir, f'res_gt.png'),
                images=self.images_figs,
                states=self.label_figs['gt_state'],
                gt_states=self.label_figs['gt_state'],
                draw_method=self.config['trainer'].get(
                    'draw_method', 'rectangle')
            )
        except Exception as e:
            logging.error(f'make_figure failed with error:\n{e} :\n'
                          f'{traceback.format_exc()}')

        if self.start_epoch != 0:
            logging.info(f"resuming trainign from {self.start_epoch}")

            if os.path.exists(self.train_log_path):
                copy_path = os.path.join(os.path.split(self.train_log_path)[
                                         0], f'train_log-{timestamp()}.yaml')
                shutil.copy(self.train_log_path, copy_path)
                with open(self.train_log_path, 'r') as f:
                    log = yaml.load(f, Loader=yaml.Loader)
                # prune to start epoch
                for prefix in ['train/', 'train/batch/', 'val/', 'val/batch/']:
                    n_keep = np.sum(
                        np.array(log[prefix + 'epoch']) < self.start_epoch)
                    for k in log.keys():
                        if prefix in k:
                            log[k] = log[k][:n_keep]
            else:
                log = {}
        else:
            log = {}

        log['dataset_path'] = log.get('dataset_path', []) + [self.temp_dataset]
        log['start_epoch'] = log.get('start_epoch', []) + [self.start_epoch]

        pbar = tqdm(range(self.start_epoch, self.n_epochs), file=sys.stdout)
        for epoch in pbar:
            log = self._epoch(
                log=log, pbar=pbar, epoch=epoch, training=True
            )
            log = self._epoch(
                log=log, pbar=pbar, epoch=epoch, training=False
            )
            with open(self.train_log_path, 'w') as f:
                yaml.dump(log, f, sort_keys=False, default_flow_style=None)

            if self.use_training_sampler:
                # renew a portion of the samples
                n_samples = self.config['data_loader']['sampler']['n_samples']
                renew_rate = self.config['data_loader']['sampler']['samples_renew_rate']
                assert len(self._training_samples) == n_samples
                n_new = int(n_samples * renew_rate)
                old_samples = self.rng.choice(
                    self._training_samples, replace=False, size=n_samples - n_new)
                new_samples = self.rng.choice(np.setdiff1d(np.arange(len(self.data_train)), old_samples), replace=False,
                                              size=n_new)
                self._training_samples = np.concatenate(
                    (old_samples, new_samples), axis=0)
                assert len(self._training_samples) == n_samples
                self.train_loader.sampler.indices = self._training_samples

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
                with open(self.train_memory_save_path, 'wb') as f:
                    pickle.dump(self.train_memory, f)
                with open(self.val_memory_save_path, 'wb') as f:
                    pickle.dump(self.val_memory, f)

            if epoch % self.dataset_update_interval == 0 and epoch != 0:
                self.update_data()

            if self.scheduler is not None:
                self.scheduler.step()

        self._save_checkpoint(self.n_epochs - 1)

    def _plot_log(self, log):
        fig = plot_log(log)
        fig.tight_layout()
        plt.savefig(os.path.join(self.config.save_dir,
                    f'weights_log.png'), bbox_inches='tight')
        plt.close('all')

    def infer(self, overwrite_results: bool, draw: bool, ignore_errors: bool = False):
        plt.ioff()
        subset = 'val'
        if ignore_errors:
            warnings.warn(
                f"with {ignore_errors=} you chose to IGNORE ERRORS: use it at your own risks !")

        draw_method = self.config['trainer'].get('draw_method', 'rectangle')

        results_dir = get_inference_path(
            model_name=os.path.split(self.config.save_dir)[1], dataset=self.dataset, subset=subset)
        make_if_not_exist(results_dir, recursive=True)

        with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config.config, f, sort_keys=False)

        dota_translator = DOTAResultsTranslator(
            self.dataset, subset, results_dir, det_type='obb', all_classes=['vehicle'])

        id_re = re.compile(r'([0-9]+).*.png')
        paths_dict = fetch_data_paths(self.dataset, subset=subset)
        image_paths = paths_dict['images']
        annot_paths = paths_dict['annotations']
        meta_paths = paths_dict['metadata']
        # SUBSET_DEBUG = [683,117]
        # SUBSET_DEBUG = [683]

        rjmcmc_params = self.config['RJMCMC_params'].copy()
        # if 'intensity' not in rjmcmc_params:
        #     rjmcmc_params['intensity'] = 0.01
        infer_params_path = os.path.join(
            self.config.save_dir, 'infer_params.yaml')
        if os.path.exists(infer_params_path):
            logging.info(f'Found infer params in model in {infer_params_path}')
            with open(infer_params_path, 'r') as f:
                infer_params = yaml.load(f, Loader=yaml.SafeLoader)

            rjmcmc_params_override = infer_params['RJMCMC_params']
            for k, w in rjmcmc_params_override.items():
                rjmcmc_params[k] = w
        elif self.USE_DEFAULT_PARAMS:
            with open('model_configs/infer_params.yaml', 'r') as f:
                infer_params = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            infer_params = {}

        energy_infer_args = infer_params.get('energy_infer_args', {})

        with open(os.path.join(results_dir, 'rjmcmc_params.yaml'), 'w') as f:
            yaml.dump(rjmcmc_params, f)
        with open(os.path.join(results_dir, 'infer_params.yaml'), 'w') as f:
            yaml.dump(infer_params, f)

        def patch_iterator(s): return zip(
            tqdm(image_paths,
                 desc=f'{s} on {self.dataset}/{subset}', file=sys.stdout),
            annot_paths,
            meta_paths)

        # infer results
        for pf, af, mf in patch_iterator('inferring'):
            loc_rjmcmc_args = rjmcmc_params.copy()
            patch_id = int(id_re.match(os.path.split(pf)[1]).group(1))

            logging.info(f"loading patch {patch_id}")

            image = plt.imread(pf)[..., :3]

            results_pickle = os.path.join(
                results_dir, f'{patch_id:04}_results.pkl')
            try:
                results_dict = {}
                # load if exists
                if os.path.exists(results_pickle) and not overwrite_results:
                    with open(results_pickle, 'rb') as f:
                        results_dict = pickle.load(f)
                    mc = None
                else:  # if not exist, compute
                    image_t = torch.from_numpy(image).permute(
                        (2, 0, 1)).unsqueeze(dim=0).to(self.model.device)
                    shape = image.shape[:2]
                    with torch.no_grad():
                        pos_e_map, marks_e_map = self.model.energy_maps_from_image(
                            image_t, **energy_infer_args)
                        pos_e_map = pos_e_map[0]  # batch item 0, (1,H,W)
                        marks_e_map = [m[0]
                                       for m in marks_e_map]  # list[(NC,H,W)]
                    pos_density_map, marks_density_maps = self.model.densities_from_energy_maps(
                        pos_e_map, marks_e_map)
                    torch.cuda.empty_cache()
                    shape = image.shape[:2]
                    assert pos_density_map.shape[1:] == shape
                    assert all([m.shape[1:] == shape for m in marks_e_map])

                    bound_min = self.model.bound_min
                    bound_max = self.model.bound_max
                    bound_max[0] = image.shape[0]
                    bound_max[1] = image.shape[1]
                    bound_cyclic = self.model.cyclic

                    if loc_rjmcmc_args.get('init_state') == 'cnn':
                        assert type(self.model) is GenericEnergyModel
                        self.model: GenericEnergyModel
                        cnn_params = infer_params['cnn_inference_args']
                        out = self.model.infer_from_cnn(
                            images=image_t, **energy_infer_args, **cnn_params
                        )
                        init_state = torch.from_numpy(
                            out['state'][0]).to(self.model.device)
                        init_state = check_inbound_state(
                            init_state, bound_min, bound_max, bound_cyclic, clip_if_oob=True)
                        logging.info(
                            f"CNN backbone yiels init state of size {init_state.shape}")
                        loc_rjmcmc_args['init_state'] = init_state

                    mc = ParallelRJMCMC(
                        support_shape=shape,
                        device=self.model.device,
                        max_interaction_distance=self.model.max_interaction_distance,
                        rng=self.rng,
                        energy_func=self.model.energy_func_wrapper(position_energy_map=pos_e_map,
                                                                   marks_energy_maps=marks_e_map,
                                                                   compute_context=True),
                        position_birth_density=pos_density_map,
                        mark_birth_density=marks_density_maps,
                        mappings=self.mappings,
                        **loc_rjmcmc_args
                    )
                    start = time.perf_counter()
                    mc.run(verbose=1, log=False)
                    elapsed_time = time.perf_counter() - start

                    last_state = mc.current_state.detach()
                    results_dict.update({
                        'state': last_state.cpu().detach(),
                        'elapsed_time': elapsed_time,
                        'n_objects': len(last_state),
                        'image_shape': image.shape[:2],
                        'rjmcmc_nb_steps': mc.n_steps,
                        'rjmcmc_nb_cells': mc.mic_points.nb_cells,
                        'rjmcmc_nb_seq_steps': mc.seq_step_count
                    })

                # ensure last state inbound before score compute
                last_state = results_dict['state'].to(self.model.device)
                bound_min = self.model.bound_min
                bound_max = self.model.bound_max
                bound_max[0] = image.shape[0]
                bound_max[1] = image.shape[1]
                bound_cyclic = self.model.cyclic
                last_state = check_inbound_state(
                    last_state, bound_min, bound_max, bound_cyclic, clip_if_oob=True)

                # check if score already computed
                if 'log_papangelou' in results_dict and not overwrite_results:
                    log_papangelou = results_dict['log_papangelou']
                else:  # compute if needed
                    image_t = torch.from_numpy(image).permute(
                        (2, 0, 1)).unsqueeze(dim=0).to(self.model.device)
                    with torch.no_grad():
                        pos_e_map, marks_e_map = self.model.energy_maps_from_image(
                            image_t, **energy_infer_args)
                        pos_e_map = pos_e_map[0]  # batch item 0, (1,H,W)
                        marks_e_map = [m[0]
                                       for m in marks_e_map]  # list[(NC,H,W)]
                    torch.cuda.empty_cache()

                    log_papangelou = compute_papangelou_scoring(
                        states=last_state,
                        model=self.model,
                        verbose=1,
                        log_values=True,
                        pos_e_m=pos_e_map,
                        marks_e_m=marks_e_map,
                        default_to_simple=False,
                        use_buffer=True
                    )
                    results_dict['log_papangelou'] = log_papangelou

                with open(results_pickle, 'wb') as f:
                    pickle.dump(results_dict, f)

            except Exception as e:
                torch.cuda.empty_cache()
                trace = traceback.format_exc()
                logging.error(f"FAILED inference with:\n{trace}")
                with open(os.path.join(results_dir, f'{patch_id:04}_inference_error.txt'), 'w') as f:
                    print(trace, file=f)
                if ignore_errors:
                    continue
                else:
                    raise e

            with open(af, 'rb') as f:
                labels_dict = pickle.load(f)

            centers = labels_dict['centers']
            params = labels_dict['parameters']
            gt_state = np.array([list(c) + list(p)
                                for c, p in zip(centers, params)])

            DRAW_SETS = True
            if draw:
                try:  # try drawing
                    image_d = image.copy()
                    if DRAW_SETS and mc is not None:
                        rect_image = np.zeros_like(image_d)
                        msb = mc.mic_points.cells_bounds
                        msc = mc.mic_points.cells_set
                        cmap = plt.get_cmap('Set1')
                        for (tl, br), c in zip(msb, msc):
                            color = cmap(c)[:3]
                            tl_y, tl_x = tl
                            br_y, br_x = br
                            rect_image = cv2.rectangle(
                                rect_image, (tl_x, tl_y), (br_x - 1, br_y - 1), color, 1)
                        mask = np.any(rect_image != 0, axis=-1)
                        alpha = 0.4
                        image_d[mask] = (1 - alpha) * \
                            image_d[mask] + alpha * rect_image[mask]

                    image_w_pred = draw_shapes_on_img(
                        image=image_d, states=last_state, color=(0, 1.0, 0), draw_method=draw_method
                    )
                    plt.imsave(os.path.join(
                        results_dir, f'{patch_id:04}_results.png'), image_w_pred)

                    if self.config['trainer'].get('analysis_at_inference', False):
                        image_t = torch.from_numpy(image).permute(
                            (2, 0, 1)).unsqueeze(dim=0).to(self.model.device)
                        with torch.no_grad():
                            pos_e_map, marks_e_map = self.model.energy_maps_from_image(
                                image_t, **energy_infer_args)
                            pos_e_map = pos_e_map[0]  # batch item 0, (1,H,W)
                            marks_e_map = [m[0]
                                           for m in marks_e_map]  # list[(NC,H,W)]
                        torch.cuda.empty_cache()

                        plt.ioff()
                        anchors = analyse_results(
                            model=self.model, rng=self.rng,
                            image=image,
                            state=last_state,
                            gt_state=gt_state,
                            pos_e_map=pos_e_map, marks_e_map=marks_e_map,
                            save_file=os.path.join(
                                results_dir, f'{patch_id:04}_results_details.png'),
                            draw_method=draw_method
                        )

                        analyse_results(
                            model=self.model, rng=self.rng,
                            image=image,
                            state=gt_state,
                            gt_state=gt_state,
                            pos_e_map=pos_e_map, marks_e_map=marks_e_map,
                            save_file=os.path.join(
                                results_dir, f'{patch_id:04}_gt_details.png'),
                            anchors=anchors,
                            draw_method=draw_method
                        )
                except Exception as e:
                    trace = traceback.format_exc()
                    logging.error(f"FAILED making figures with:\n{trace}")
                    with open(os.path.join(results_dir, f'{patch_id:04}_display_error.txt'), 'w') as f:
                        print(trace, file=f)

            logging.info(f'frame {patch_id:04} done !')
            torch.cuda.empty_cache()

        # aggregate scores
        all_papangelous = []
        for pf, af, mf in patch_iterator("aggregating scores"):
            patch_id = int(id_re.match(os.path.split(pf)[1]).group(1))
            results_pickle = os.path.join(
                results_dir, f'{patch_id:04}_results.pkl')
            with open(results_pickle, 'rb') as f:
                results_dict = pickle.load(f)
            all_papangelous.append(results_dict['log_papangelou'])

        # scale scores
        logging.info('hist-eq papangelou scores')
        # all_scores = papangelou_score_eq(all_papangelous, uniform_bins=False)
        all_scores = papangelou_score_scale(all_papangelous, log=False)

        # save scores
        for k, (pf, af, mf) in enumerate(patch_iterator("saving scores")):
            patch_id = int(id_re.match(os.path.split(pf)[1]).group(1))
            with open(af, 'rb') as f:
                labels_dict = pickle.load(f)

            centers = labels_dict['centers']
            params = labels_dict['parameters']
            difficult = labels_dict['difficult']
            categories = labels_dict['categories']
            gt_state = np.array([list(c) + list(p)
                                for c, p in zip(centers, params)])

            results_pickle = os.path.join(
                results_dir, f'{patch_id:04}_results.pkl')
            with open(results_pickle, 'rb') as f:
                results_dict = pickle.load(f)

            pred_scores = all_scores[k]
            last_state = results_dict['state'].cpu().numpy()
            results_dict['score'] = pred_scores

            with open(results_pickle, 'wb') as f:
                pickle.dump(results_dict, f)

            if draw_method == 'rectangle':
                gt_as_poly = np.array(
                    [rect_to_poly(c, short=p[0], long=p[1], angle=p[2]) for c, p in zip(centers, params)])

                dota_translator.add_gt(
                    image_id=patch_id,
                    polygons=gt_as_poly,
                    difficulty=[d or c == 'large-vehicle' for d, c in
                                zip(difficult, categories)],
                    categories=['vehicle' for _ in gt_as_poly])

                detection_as_poly = np.array(
                    [rect_to_poly(s[[0, 1]], short=s[2], long=s[3], angle=s[4]) for s in last_state])

                dota_translator.add_detections(
                    image_id=patch_id,
                    scores=pred_scores,
                    polygons=detection_as_poly,
                    flip_coor=True,
                    class_names=['vehicle' for _ in pred_scores])
            else:
                raise NotImplementedError

        dota_translator.save()
        print('saved dota translation')

    def eval(self):
        from metrics.dota_eval import dota_eval
        dota_eval(
            model_dir=self.config.save_dir,
            dataset=self.dataset,
            subset='val',
            det_type='obb',
            legacy_mode=False
        )


def plot_log(log, disp_lr: bool = True, disp_card: bool = False):
    weights_k = [k for k in log.keys() if 'train/batch/weight_' in k]
    delta_k = [k for k in log.keys() if 'train/batch/delta_' in k]

    # backward_compat
    if 'train/batch/train/lr' in log:
        lr_key = 'train/batch/train/lr'
    elif 'train/batch/lr' in log:
        lr_key = 'train/batch/lr'
    else:
        lr_key = None

    n_plots = 3
    if lr_key and disp_lr:
        n_plots += 1
    if disp_card and 'train/batch/patch_|Y+|' in log:
        n_plots += 1
    if len(delta_k) > 0:
        n_plots += 1
    n_cols = min(4, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(
        n_cols * 6, n_rows * 4), squeeze=False)
    xx_train_batch = np.array(
        log['train/batch/epoch']) + np.array(log['train/batch/batch_progress'])
    if 'val/epoch' in log:
        xx_val = np.array(log['val/epoch']) + 1  # val computed at end of epoch
    else:
        xx_val = None
    if 'train/epoch' in log:
        xx_train = np.array(log['train/epoch']) + 1
    else:
        xx_train = None

    axs = axs.ravel()
    ax_iter = 0

    # plot learning rate
    if lr_key is not None and disp_lr:
        ax: plt.Axes = axs[ax_iter]
        ax.plot(xx_train_batch, log[lr_key], label='learning rate')
        ax.set_yscale("log")
        ax.legend()
        ax_iter += 1

    # plot params
    ax: plt.Axes = axs[ax_iter]
    cmap = plt.get_cmap('gist_ncar')
    plot_keys = weights_k + (['train/batch/bias']
                             if 'train/batch/bias' in log.keys() else [])
    for i, k in enumerate(plot_keys):
        ax.plot(xx_train_batch, log[k], label=k.split(
            '/')[-1], color=cmap(i / (len(plot_keys) - 1)))
    ax.legend()
    ax_iter += 1

    # plot deltas
    if len(delta_k) > 0:
        ax: plt.Axes = axs[ax_iter]
        cmap = plt.get_cmap('Paired')
        for i, kd in enumerate(delta_k):
            ax.plot(xx_train_batch, log[kd], label=kd.split(
                '/')[-1], color=cmap(i / len(plot_keys)))
        ax.legend()
        ax_iter += 1

    line_kwargs = {'zorder': 2}
    dot_kwargs = {'zorder': 1, 'marker': '.', 'linestyle': '', 'markeredgewidth': 1.0, 'markersize': 3.0,
                  'alpha': 0.5}

    # plot loss
    ax: plt.Axes = axs[ax_iter]
    display_keys = ['U(Y+)', 'U(Y-)', 'Loss', 'Reg']
    lim_1, lim_2 = np.inf, -np.inf
    cmap = plt.get_cmap('Paired')
    for ki, k in enumerate(display_keys):
        values = log['train/batch/' + k]
        ax.plot(xx_train_batch, values, color=cmap(2 * ki), **dot_kwargs)
        k_epoch = 'train/' + k
        if k_epoch in log:
            ax.plot(
                xx_train, log[k_epoch], label=f'train/{k}', color=cmap(2 * ki), **line_kwargs)
        q1, q2 = np.quantile(values, [0.1, 0.9])
        lim_1, lim_2 = min(lim_1, q1), max(lim_2, q2)

        k_val = 'val/' + k
        if k_val in log:
            values = log[k_val]
            ax.plot(xx_val, values,
                    label=f'val/{k}', color=cmap(2 * ki + 1), **line_kwargs)
            q1, q2 = np.quantile(values, [0.1, 0.9])
            lim_1, lim_2 = min(lim_1, q1), max(lim_2, q2)
    r = lim_2 - lim_1
    ax.set_ylim(lim_1 - 0.005 * r, lim_2 + 0.005 * r)
    ax.legend()
    ax_iter += 1

    # plot norm energies
    ax: plt.Axes = axs[ax_iter]
    display_keys = ['avg(U(y+))', 'avg(U(y-))']
    lim_1, lim_2 = np.inf, -np.inf
    cmap = plt.get_cmap('Paired')
    for ki, k in enumerate(display_keys):
        values = log['train/batch/' + k]
        ax.plot(xx_train_batch, values, color=cmap(2 * ki), **dot_kwargs)
        k_epoch = 'train/' + k
        if k_epoch in log:
            ax.plot(
                xx_train, log[k_epoch], label=f'train/{k}', color=cmap(2 * ki), **line_kwargs)
        q1, q2 = np.quantile(values, [0.1, 0.9])
        lim_1, lim_2 = min(lim_1, q1), max(lim_2, q2)

        k_val = 'val/' + k
        if k_val in log:
            values = log[k_val]
            ax.plot(xx_val, values,
                    label=f'val/{k}', color=cmap(2 * ki + 1), **line_kwargs)
            q1, q2 = np.quantile(values, [0.1, 0.9])
            lim_1, lim_2 = min(lim_1, q1), max(lim_2, q2)
    r = lim_2 - lim_1
    ax.set_ylim(lim_1 - 0.005 * r, lim_2 + 0.005 * r)
    ax.legend()
    ax_iter += 1

    if disp_card and 'train/batch/patch_|Y+|' in log:
        ax: plt.Axes = axs[ax_iter]
        display_keys = ['patch_|Y+|', 'patch_|Y-|']
        lim_1, lim_2 = np.inf, -np.inf
        cmap = plt.get_cmap('Paired')
        # smoothing = 16
        for ki, k in enumerate(display_keys):
            values = np.array(log['train/batch/' + k]).mean(axis=-1)
            values_epoch = epochs_avg(log['train/batch/epoch'], values)
            ax.plot(xx_train_batch, values, color=cmap(2 * ki), **dot_kwargs)
            ax.plot(xx_train, values_epoch, label=f'train/{k}', color=cmap(2 * ki),
                    **line_kwargs)
            q1, q2 = np.quantile(values, [0.1, 0.9])
            lim_1, lim_2 = min(lim_1, q1), max(lim_2, q2)

            if ('val/batch/' + k) in log:
                values = np.array(log['val/batch/' + k]).mean(axis=-1)
                values_epoch = epochs_avg(log['val/batch/epoch'], values)
                ax.plot(xx_val, values_epoch,
                        label=f'val/{k}', color=cmap(2 * ki + 1), **line_kwargs)
                q1, q2 = np.quantile(values, [0.1, 0.9])
                lim_1, lim_2 = min(lim_1, q1), max(lim_2, q2)

        r = lim_2 - lim_1
        ax.set_ylim(lim_1 - 0.005 * r, lim_2 + 0.005 * r)
        ax.legend()
        ax_iter += 1

    while ax_iter < (n_rows * n_cols):
        ax: plt.Axes = axs[ax_iter]
        ax.clear()
        ax.axis('off')
        ax_iter += 1

    return fig


def epochs_avg(epoch_log, value_log):
    return np.array([np.mean(v) for v in np.split(value_log, np.unique(epoch_log, return_index=True)[1][1:])])
