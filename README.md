# Combining Convolutional Neural Networks and Point Process for object detection


## Installation

### Conda env
- setup [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment with `conda env create -f env.yml`

### Paths setup
- setup models and data paths with [paths_config.yaml](paths_config.yaml).


### DOTA metrics
- to compute metrics install [dota devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) in `data/` (see installation for more info)
```
cd data/
git clone https://github.com/CAPTAIN-WHU/DOTA_devkit
cd DOTA_devkit/
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```
- configure `paths_configs.json` as needed
- conda env is provided `env.yml`, setup using `conda env create -f env.yml`


## Repo structure

- [saved_models](saved_models): saved models that can be used for inference (use setup in [paths_config.yaml](paths_config.yaml))
- [env.yml](env.yml): file defining the conda environment suitable to run the code
- [model_configs](model_configs): config files for models


## External resources
- [Minitel font](https://github.com/Zigazou/Minitel-Canvas) (CC0 license)

## Todo
- [ ] cleanup code
- [ ] finish documentation
