# Combining Convolutional Neural Networks and Point Process for object detection

## Todo
- [ ] cleanup useless code elements
- [ ] write documentation
- [ ] provide pretrained model files
- [ ] provide model config files

## Installation
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



## External resources
- [Minitel font](https://github.com/Zigazou/Minitel-Canvas) (CC0 license)