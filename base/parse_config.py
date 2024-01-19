import glob
import logging
import os
import shutil
import sys
from typing import Dict, Any

import matplotlib.pyplot as plt
import yaml

from base.data import get_model_base_path
from base.files import make_if_not_exist
from base.logger import Logger
from base.misc import timestamp


class MaxLevelFilter(logging.Filter):
    '''Filters (lets through) all messages with level < LEVEL'''

    def __init__(self, level):
        super(MaxLevelFilter, self).__init__()
        self.level = level

    def filter(self, record):
        # "<" instead of "<=": since logger.setLevel is inclusive, this should be exclusive
        return record.levelno < self.level


def setup_logging(save_path: str, level=logging.INFO):
    # print(f'setting log level {level}')
    # logging.basicConfig(
    #     format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    #     datefmt='%Y-%m-%d:%H:%M:%S',
    #     level=level,
    #     force=True,
    #     stream=sys.stdout
    # )
    # root_logger = logging.getLogger()

    # log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s")
    log_path = os.path.join(save_path, f'log_{timestamp()}.log')
    print(f'saving logs to {log_path}')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(log_formatter)
    # root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    # console_handler.setFormatter(log_formatter)
    # root_logger.addHandler(console_handler)

    logging.basicConfig(
        format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO,
        force=True,
        handlers=[
            file_handler, console_handler
        ]
    )


class ConfigParser:
    def __init__(self, config: Dict[str, Any], model_type: str, load_model=False, overwrite=False,
                 resume: bool = False, debug_mode: bool = False, init_if_none: bool = False, save_path=None):
        """

        :return: config dict, save path
        """

        # get paths

        plt.ioff()
        self.debug = debug_mode

        if save_path is None:
            base_path_model = get_model_base_path()
            save_path = os.path.join(
                base_path_model, model_type, config['name'])

        if os.path.exists(save_path):
            if not load_model:
                if not overwrite:
                    logging.error(f'found model in {save_path}')
                    raise FileExistsError
                else:
                    logging.info(f'found model in {save_path}, writing over')
                    shutil.rmtree(save_path)
                    make_if_not_exist(save_path, recursive=True)
        else:
            make_if_not_exist(save_path, recursive=True)

        local_config_file = os.path.join(save_path, 'config.yaml')
        if not os.path.exists(local_config_file):
            with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        log_file = os.path.join(save_path, 'log.json')
        if os.path.exists(log_file) and load_model:
            logger = Logger.load(log_file)
        else:
            logger = Logger(save_dir=save_path)

        if not load_model:
            logger.clear()

        self.logger = logger
        self._config = config
        self._save_dir = save_path
        if resume or load_model:
            checkpoints = glob.glob(os.path.join(
                save_path, 'checkpoint_*.pth'))
            if len(checkpoints) == 0:

                if init_if_none:
                    logging.warning(
                        'no checkpoints found: however, init_if_none allows to init new model')
                    self.resume = None
                else:
                    logging.error("no checkpoints found")
                    raise FileNotFoundError
                # self.resume = None
            else:
                checkpoints.sort()
                self.resume = checkpoints[-1]
        else:
            self.resume = None

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def __contains__(self, item):
        return item in self.config

    def __repr__(self):
        return repr(self.config)

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir
