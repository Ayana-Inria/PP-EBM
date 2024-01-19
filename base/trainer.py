import glob
import logging
import os
import re
import traceback
from abc import ABC, abstractmethod
from typing import Union

import torch
from torch.nn import Module

from base.base_model import BaseModel
from modules.losses import BaseCriterion


class BaseTrainer(ABC):

    def __init__(self, model, criterion, optimizer, config, scheduler=None):
        self.config = config

        self.model: Union[BaseModel, Module] = model
        self.criterion: BaseCriterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        config_trainer = config['trainer']
        self.epochs = config_trainer['n_epochs']
        self.save_period = config_trainer['save_interval']
        self.monitor = config_trainer.get('monitor', 'off')

        self.start_epoch = 0

        self.checkpoint_dir = config.save_dir

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'monitor_best': self.mnt_best,
            'config': self.config
        }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        filename = str(os.path.join(self.checkpoint_dir,
                       f'checkpoint_{epoch:04}.pth'))
        torch.save(state, filename)
        logging.info("Saving checkpoint: {} ...".format(filename))

        # do some cleanup
        try:
            prune_checkpoints(self.checkpoint_dir)
        except Exception as e:
            logging.error(
                f"failed pruning checkpoints with error {e}: \n {traceback.print_exc()}")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        logging.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, )
        self.start_epoch = checkpoint['epoch'] + 1
        # self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            logging.warning("Warning: Architecture configuration given in config file is different from that of "
                            "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer'] != self.config['optimizer']:
            logging.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                            "Optimizer parameters not being resumed.")
        else:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError as e:
                logging.error(f"optimizer loading FAILED because {e}")

        if self.scheduler is not None:
            if checkpoint['config']['scheduler'] != self.config['scheduler']:
                logging.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
            else:
                self.scheduler.load_state_dict(checkpoint['scheduler'])

        logging.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def infer(self, overwrite_results: bool, draw: bool):
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        raise NotImplementedError


def prune_checkpoints(base_path, interval=50):
    all_checkpoints = glob.glob(os.path.join(base_path, 'checkpoint_*.pth'))
    all_checkpoints.sort()
    checkpoints_ids = [int(re.match(r'checkpoint_([0-9]{4})\.pth', os.path.split(p)[1]).group(1)) for p in
                       all_checkpoints]
    if len(all_checkpoints) == 0:
        return

    keepers = [all_checkpoints[-1]]
    if len(all_checkpoints) > 1:
        keepers.append(all_checkpoints[0])
        last_id = checkpoints_ids[0]

        for c, i in zip(all_checkpoints[1:], checkpoints_ids[1:]):
            if i - last_id >= interval:
                keepers.append(c)
                last_id = i

    for c in all_checkpoints[:-1]:
        if c not in keepers:
            os.remove(c)
