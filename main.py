import argparse
import logging
import os.path
import sys
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch.optim
import yaml
from torch.nn import Module

from base.base_model import BaseModel
from base.data import resolve_model_config_path
from base.parse_config import ConfigParser, setup_logging
from base.trainer import BaseTrainer
from modules import losses


def main():
    plt.ioff()
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model', help='model to use')
    parser.add_argument(
        '-d', '--dataset', help='dataset to use, defaults to the one specified in config')
    parser.add_argument('-p', '--procedure', help='procedure to execute')
    parser.add_argument('-c', '--config',
                        help='model config file, can pass model name if model already is in models folder')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='if set then overwrites existing model')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='resumes training from checkpoint')
    parser.add_argument('--init_if_none', action='store_true',
                        help='if no checkpoints create new model instead')
    parser.add_argument('--no_draw', action='store_true',
                        help='on infer supresses drawing figures')
    parser.add_argument('--debug', action='store_true',
                        help='turns on debug mode')
    parser.add_argument('--ignore_errors', action='store_true',
                        help='tries to ignore errors as mush as possible (this is bad)')
    args = parser.parse_args()

    if args.debug:
        logging.warning("Debug mode activated, may be slower")
        np.seterr(all='raise')
        torch.autograd.set_detect_anomaly(True)

    procedure = args.procedure
    dataset = args.dataset
    overwrite_model = args.overwrite and procedure == 'train'
    overwrite_results = args.overwrite and procedure != 'train'
    infer_draw = not args.no_draw
    train_flag = args.procedure == 'train'
    load_flag = args.resume or args.procedure not in ['train', 'data_preview']

    config_file = resolve_model_config_path(args.config)
    print(f"loading config from {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if procedure == 'translate_dota':
        raise NotImplementedError

    config = ConfigParser(
        config=config, model_type=config['arch'], load_model=load_flag, overwrite=overwrite_model, resume=args.resume,
        debug_mode=args.debug, init_if_none=args.init_if_none
    )
    setup_logging(config.save_dir, level=logging.INFO)

    # if config['name'] != os.path.split(config_file)[-1].split('.')[0]:
    #     raise RuntimeError(" name in config does not correspond to config file name")

    model: Union[BaseModel, Module]

    if config['arch'] == 'MPPDataModel':
        from models.mpp_data_net.model import MPPDataModel
        model = MPPDataModel(config)
    elif config['arch'] == 'MPPwMaps':
        from energies.energy_from_maps import EnergyFromMaps
        model = EnergyFromMaps(config)
    elif config['arch'] == 'MPPEnergy':
        from energies.generic_energy_model import GenericEnergyModel
        model = GenericEnergyModel(config)
    elif config['arch'] == 'NaiveCnn':
        from models.naive_cnn import NaiveCnn
        model = NaiveCnn(config)
    else:
        raise ValueError

    rng = np.random.default_rng(0)
    torch.random.manual_seed(42)
    criterion = getattr(losses, config.config.get('loss', 'DummyLoss'))(
        **config.config.get('loss_params', {}))
    optimizer = getattr(torch.optim, config['optimizer'])(
        params=model.parameters(), **config['optimizer_params'])
    if 'scheduler' in config.config:
        scheduler = getattr(torch.optim.lr_scheduler, config['scheduler'])(optimizer=optimizer,
                                                                           **config['scheduler_params'])
    else:
        scheduler = None

    trainer: BaseTrainer
    if config['trainer']['type'] == 'EnergyMapTrainer':
        from trainers.energy_map_trainer import EnergyMapTrainer
        trainer = EnergyMapTrainer(config=config, model=model, criterion=criterion, optimizer=optimizer, rng=rng,
                                   scheduler=scheduler, force_dataset=dataset)
    elif config['trainer']['type'] == 'MPPTrainer':
        from trainers.mpp_trainer import MPPTrainer
        trainer = MPPTrainer(config=config, model=model, criterion=criterion, optimizer=optimizer, rng=rng,
                             scheduler=scheduler, force_dataset=dataset)
    elif config['trainer']['type'] == 'NaiveCNNTrainer':
        from trainers.naive_model_trainer import NaiveCNNTrainer
        trainer = NaiveCNNTrainer(config=config, model=model, criterion=criterion, optimizer=optimizer, rng=rng,
                                  scheduler=scheduler, force_dataset=dataset)
    else:
        raise ValueError

    infer_kwargs = {'overwrite_results': overwrite_results,
                    'draw': infer_draw, 'ignore_errors': args.ignore_errors}

    if procedure == 'train':
        logging.info("training starts now")
        trainer.train()
    elif procedure == 'infer':
        logging.info('infering on dataset')
        trainer.infer(**infer_kwargs)
    elif procedure == 'eval':
        logging.info('evaluating metrics')
        trainer.eval()
    elif procedure == 'infereval':
        logging.info('infering on dataset')
        trainer.infer(**infer_kwargs)
        print('evaluating metrics')
        trainer.eval()
    else:
        raise ValueError

    print('done !')


if __name__ == '__main__':
    main()
