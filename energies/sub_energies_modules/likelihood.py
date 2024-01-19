import logging
import warnings
from contextlib import nullcontext
from typing import List, Dict, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import Module, functional

from base.data import get_model_config_by_name
from base.images import map_range, patchify_big_image, unpatchify_big_image, pad_to_valid_size, unpad_to_size
from base.mappings import mappings_from_config, ValueMapping
from base.parse_config import ConfigParser
from energies.contrast_utils import affine_from_state, contrast_measure
from energies.sub_energies_modules.base import LikelihoodModule
from modules.custom import ScaleModule, NegSoftplus, NegLogSofmax
from modules.interpolation import interpolate_position, interpolate_marks
from modules.torch_div import Divergence
from modules.unet.unet import Unet


class ImageEnergies(LikelihoodModule, Module):

    def __init__(self, in_range, out_range, image_to_maps_args: Dict = None, config=None):
        super(LikelihoodModule, self).__init__()
        self.in_range = in_range
        self.out_range = out_range
        self.image_to_maps_args = {} if image_to_maps_args is None else image_to_maps_args

        if 'marks_classes' not in config['model']:
            config['model']['marks_classes'] = 4
            if len(config['model']['marks']) > 0:
                assert 'marks_classes' in config['model']
        self._mappings = mappings_from_config(config)

    def forward(self, points: Tensor, points_mask: Tensor, position_energy_map: Tensor, **kwargs) -> Dict[str, Tensor]:
        n_sets = points.shape[0]
        n_points = points.shape[1]
        h, w = position_energy_map.shape[2:]

        positions = points[..., :2].view((1, n_sets, n_points, 2))
        position_energies = interpolate_position(
            positions=positions,
            image=position_energy_map
        ).view((n_sets, n_points))
        position_energies = position_energies * points_mask
        position_energies = position_energies.view((n_sets, n_points))

        return {
            'position': position_energies
        }

    def energy_maps_from_image(self, image: Tensor, **kwargs) -> Tuple[Tensor, List[Tensor]]:
        if len(image.shape) == 3:
            # add batch size 1
            image = torch.unsqueeze(image, dim=0)
        assert len(image.shape) == 4
        h, w = image.shape[2:]
        n_images = image.shape[0]

        if image.shape[1] == 3:  # RGB image
            channel = self.image_to_maps_args.get('use_channel', None)
            if channel is not None:
                image = image[:, channel]
            else:
                image = torch.mean(image, dim=1, keepdim=True)
        elif image.shape[1] == 1:  # GS image
            pass
        else:
            raise ValueError

        energy_map = map_range(
            image, self.in_range[0], self.in_range[1], self.out_range[0], self.out_range[1])

        marks_energy_map = [
            torch.zeros((n_images, m.n_classes, h, w), device=energy_map.device) for m in self._mappings
        ]
        # marks_energy_map = []

        return energy_map, marks_energy_map

    @property
    def sub_energies(self) -> List[str]:
        return ['position']

    @property
    def mappings(self) -> List[ValueMapping]:
        return self._mappings

    def densities_from_energy_maps(self, pos_energy_map, marks_energy_maps, **kwargs) -> Tuple[Tensor, List[Tensor]]:
        # todo do smth smarter
        density_map = map_range(
            pos_energy_map, self.out_range[0], self.out_range[1], 1, 0)
        mdm = [torch.ones_like(mem) for mem in marks_energy_maps]
        # mdm = []
        return density_map, mdm


class CNNEnergies(LikelihoodModule, Module):

    def __init__(self, mpp_data_net: str, freeze=True, position_softplus=False, mark_logsoftmax=False,
                 softplus_bias: bool = False, config=None):
        from models.mpp_data_net.model import MPPDataModel
        super(CNNEnergies, self).__init__()
        self.mpp_data_net = mpp_data_net
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

        self._sub_energies = ['position'] + [m.name for m in self._mappings]

        self.is_frozen = True

        if softplus_bias:
            assert position_softplus
            self.is_frozen = False

        self.position_softplus = None
        if position_softplus:
            self.position_softplus = NegSoftplus(
                device=self.device, learn_bias=softplus_bias)
        self.mark_logsoftmax = None
        if mark_logsoftmax:
            self.mark_logsoftmax = NegLogSofmax(dim=1)

        if type(freeze) is bool and freeze:
            self.maps_module.training = False
            for param in self.maps_module.parameters():
                param.requires_grad = False
        elif type(freeze) in [list, str, bool]:
            if type(freeze) is str:
                freeze = [freeze]
            elif type(freeze) is bool:
                assert not freeze
                freeze = []
            logging.info(f"freezing layers {freeze}")
            self.is_frozen = False
            self.maps_module.training = True
            for layer in freeze:
                for param in self.maps_module.__getattr__(layer).parameters():
                    param.requires_grad = False
            linear_on_heatmap = not self.maps_module.config['model'].get(
                'skip_linear', False)
            if not linear_on_heatmap:
                self.maps_module.div_linear_layer = nn.Sequential(
                    Divergence(div_channels=[
                               0, 1], mask_channel=2, sigmoid_on_mask=True),
                    nn.Conv2d(in_channels=1, out_channels=4,
                              kernel_size=(1, 1)),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=4, out_channels=1,
                              kernel_size=(1, 1))
                ).to(self.device)
        else:
            raise ValueError(f"freeze={freeze} is not a supported value")

    def reload_maps_module(self):
        warnings.warn("reloading maps module, it might break stuff")
        map_model_config = get_model_config_by_name(self.mpp_data_net)
        with open(map_model_config, 'r') as f:
            map_model_config = yaml.load(f, Loader=yaml.SafeLoader)
        map_model_config = ConfigParser(
            map_model_config, 'MPPDataModel', load_model=True, resume=True)
        from models.mpp_data_net.model import MPPDataModel
        self.maps_module: MPPDataModel = MPPDataModel(map_model_config)
        checkpoint = torch.load(map_model_config.resume, )
        self.maps_module.load_state_dict(checkpoint['state_dict'])

    def energy_maps_from_image(self, image: Tensor, requires_grad: bool = False,
                               pad_to_size: bool = False, large_image: bool = False, cut_and_stitch: bool = False,
                               **kwargs) -> Tuple[Tensor, List[Tensor]]:
        if len(image.shape) != 4:
            raise RuntimeError(
                f"Expected image(s) for shape (B,3,H,W) and got {tuple(image.shape)}")

        if not self.is_frozen:
            logging.debug(f"computing maps with requires_grad={requires_grad}")
            if not requires_grad:
                self.maps_module.eval()
        else:
            if requires_grad:
                logging.warning(
                    f"requires_grad={requires_grad} but maps module is frozen, setting back to False")
                requires_grad = False
        split_info = {}
        with nullcontext() if requires_grad else torch.no_grad():
            if large_image:
                raise DeprecationWarning(
                    "large_image is no longer supported, use cut_and_stitch instead")
            elif cut_and_stitch:
                if len(image.shape) != 4 or image.shape[0] != 1:
                    raise RuntimeError(f"energy_maps_from_image with large_image only works on one image, "
                                       f"image shape is {image.shape}")

                output = self.maps_module.infer_on_large_image(image[0].to(self.device),
                                                               restore_batch_dim=True, **kwargs)
            elif pad_to_size:
                output = self.maps_module.forward(pad_to_valid_size(image, 64))
                output = unpad_to_size(output, image.shape[2:])
            else:
                output = self.maps_module.forward(image)

            if self.position_softplus is not None:
                pos_energy_map = self.position_softplus(
                    output['center_heatmap'])
            else:
                pos_energy_map = -output['center_heatmap']

            if self.mark_logsoftmax is None:
                marks_energy_maps = [-output[f'mark_{m.name}']
                                     for m in self._mappings]
            else:
                marks_energy_maps = [self.mark_logsoftmax(
                    output[f'mark_{m.name}']) for m in self._mappings]

            return pos_energy_map, marks_energy_maps

    def densities_from_energy_maps(self, pos_energy_map, marks_energy_maps, energy_weights: Dict = None, **kwargs):

        if len(pos_energy_map.shape) == 3:  # dim (1,H,W)
            pos_energy_map = torch.unsqueeze(pos_energy_map, dim=0)
            squeeze_flag = True
        elif len(pos_energy_map.shape) == 4:  # dim (B,1,H,W)
            squeeze_flag = False
        else:
            raise RuntimeError(f'invalid shape for pos_energy_map {pos_energy_map.shape}, '
                               f'expected shape like (1,H,W) or (B,1,H,W)')

        if "weight_position" in energy_weights:
            position_density = torch.exp(-pos_energy_map *
                                         energy_weights[f"weight_position"])
        else:
            # todo deal with that case
            raise RuntimeError(
                "energy_weights does not contain weight_position key")
        position_density = position_density / \
            torch.sum(position_density, dim=(2, 3), keepdim=True)
        if squeeze_flag:
            position_density = position_density[0]

        marks_densities = []
        for mapping, mem in zip(self.mappings, marks_energy_maps):
            if len(mem.shape) == 3:  # dim (N_classes,H,W)
                mem = torch.unsqueeze(mem, dim=0)
                squeeze_flag = True
            elif len(pos_energy_map.shape) == 4:  # dim (B,N_classes,H,W)
                squeeze_flag = False
            else:
                raise RuntimeError(f'invalid shape for marks_energy_maps {mem.shape}, '
                                   f'expected shape like (N_classes,H,W) or (B,N_classes,H,W)')

            weight = energy_weights[f"weight_{mapping.name}"]
            mdm = torch.exp(-weight * mem)
            mdm = mdm / torch.sum(mdm, dim=1, keepdim=True)
            if torch.any(torch.isnan(mdm)):
                for k in range(len(mdm)):
                    if torch.any(torch.isnan(mdm[k])):
                        logging.warning(
                            f'found NaN in mark_density[{k}], replacing with uniform density')
                        mdm[k] = torch.ones_like(mdm[k]) / mapping.n_classes
            if squeeze_flag:
                marks_densities.append(mdm[0])
            else:
                marks_densities.append(mdm)

        return position_density, marks_densities

    def forward(self, points: Tensor, points_mask: Tensor, position_energy_map: Tensor,
                marks_energy_maps: List[Tensor]) -> Dict[str, Tensor]:
        n_sets = points.shape[0]
        n_points = points.shape[1]
        h, w = position_energy_map.shape[2:]

        positions = points[..., :2].view((1, n_sets, n_points, 2))
        position_energies = interpolate_position(
            positions=positions,
            image=position_energy_map
        ).view((n_sets, n_points))
        position_energies = position_energies * points_mask
        position_energies = position_energies.view((n_sets, n_points))

        res = {
            'position': position_energies
        }

        for i, (mark_energy_map, mapping) in enumerate(zip(marks_energy_maps, self.mappings)):
            mark_energies = interpolate_marks(
                positions=positions.view((1, 1, n_sets, n_points, 2)),
                marks=points[..., 2 + i].view((1, 1, n_sets, n_points)),
                image=mark_energy_map.view((1, 1, mapping.n_classes, h, w)),
                mapping=mapping
            ).view((n_sets, n_points))
            res[mapping.name] = mark_energies * points_mask

        return res

    @property
    def sub_energies(self) -> List[str]:
        return self._sub_energies

    @property
    def mappings(self) -> List[ValueMapping]:
        return self._mappings

    @property
    def is_trainable(self) -> bool:
        return not self.is_frozen


class InlineCNNEnergy(LikelihoodModule, Module):

    def __init__(self, config, bounded_position_energy: bool = True, bounded_mark_energy: bool = True,
                 hidden_dims: List[int] = None, last_unet_dim: int = None, position_term: bool = True):
        super(InlineCNNEnergy, self).__init__()
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device is {self.device}")

        self._mappings = mappings_from_config(self.config)
        self.n_marks = len(self._mappings)

        self.bounded_position_energy = bounded_position_energy
        self.bounded_mark_energy = bounded_mark_energy

        self.alternative_mode = False

        self.hidden_dims = hidden_dims
        if hidden_dims is None:
            hidden_dims = [8, 16]

        self.backbone = Unet(
            hidden_dims=hidden_dims,
            in_channels=3,
            last_channels=last_unet_dim,
            device=self.device
        ).to(self.device)

        self.compute_position_term = position_term
        assert position_term or (self.n_marks > 0)
        if self.compute_position_term:
            position_mid_channels = 2
            self.position_layer = nn.Sequential(
                nn.Conv2d(in_channels=self.backbone.out_channels, out_channels=position_mid_channels,
                          kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=position_mid_channels,
                          out_channels=1, kernel_size=(1, 1))
            ).to(self.device)
        else:
            self.position_layer = None

        self.marks_layers = nn.ModuleList()
        for m in self.mappings:
            self.marks_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.backbone.out_channels, out_channels=m.n_classes,
                              kernel_size=(1, 1)),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=m.n_classes, out_channels=m.n_classes,
                              kernel_size=(1, 1)),
                ).to(self.device)
            )

    def forward(self, points: Tensor, points_mask: Tensor, position_energy_map: Tensor,
                marks_energy_maps: List[Tensor]) -> Dict[str, Tensor]:
        # TODO MAKE REGRESSIVE ?

        n_sets = points.shape[0]
        n_points = points.shape[1]
        h, w = position_energy_map.shape[2:]
        res = {}
        positions = points[..., :2].view((1, n_sets, n_points, 2))
        if self.compute_position_term:
            position_energies = interpolate_position(
                positions=positions,
                image=position_energy_map
            ).view((n_sets, n_points))
            position_energies = position_energies * points_mask
            position_energies = position_energies.view((n_sets, n_points))

            res['position'] = position_energies
        else:
            pass  # there is nothing to compute here

        for i, (mark_energy_map, mapping) in enumerate(zip(marks_energy_maps, self.mappings)):
            mark_energies = interpolate_marks(
                positions=positions.view((1, 1, n_sets, n_points, 2)),
                marks=points[..., 2 + i].view((1, 1, n_sets, n_points)),
                image=mark_energy_map.view((1, 1, mapping.n_classes, h, w)),
                mapping=mapping
            ).view((n_sets, n_points))
            res[mapping.name] = mark_energies * points_mask

        return res

    def energy_maps_from_image(self, image, position_map_from_marks: bool = None, energy_weights: Dict = None,
                               center_position_map: bool = True, requires_grad: bool = False, large_image: bool = False,
                               **kwargs) -> Tuple[Tensor, List[Tensor]]:
        logging.debug(f"computing maps with requires_grad={requires_grad}")
        if large_image and requires_grad:
            raise RuntimeError(
                "Cannot have both large_image and requires_grad")

        if position_map_from_marks is None:
            position_map_from_marks = not self.compute_position_term
        assert position_map_from_marks or (self.position_layer is not None)

        if large_image:
            patch_size = 256
            margin = [14, 32, 68, 140][len(self.hidden_dims)] // 2
            assert 2 * margin < patch_size
            original_size = image.shape[1:]
            assert len(image) == 1  # only one image to patchify
            patches, padded_image, cuts = patchify_big_image(
                image=image[0], margin=margin, patch_size=patch_size)
            logging.info(f"cut big image into {len(patches)} patches")
            image = patches

        assert type(image) is Tensor
        assert len(image.shape) == 4  # (B,C,H,W)

        with nullcontext() if requires_grad else torch.no_grad():
            encoding = self.backbone(image)
            marks_maps = []
            for layer in self.marks_layers:
                m = layer(encoding)
                if self.bounded_mark_energy:
                    m = 2 * torch.sigmoid(m) - 1
                marks_maps.append(m)

            if position_map_from_marks:
                assert energy_weights is not None
                weights = torch.tensor(
                    [energy_weights[f'weight_{m.name}']
                        for m in self.mappings],
                    device=encoding.device)
                weights_v = weights.view((len(marks_maps), 1, 1, 1, 1))
                # dim is now (n_marks,B,C,H,W) C = n classes mark
                marks_maps_concat = torch.concat(marks_maps, dim=0)
                pos_map = torch.sum(
                    - torch.log(  # LogSumExp
                        torch.sum(  # sum over the value classes
                            torch.exp(-weights_v * marks_maps_concat),
                            dim=2, keepdim=True
                        )
                    ), dim=0
                )  # dim should be (B,1,H,W)
                if center_position_map:
                    # the zero point is at n_mark * lob(n_classes)
                    bias = self.n_marks * np.log(self.mappings[0].n_classes)
                    pos_map = pos_map + bias
                    # vmax =-vmin is sum on marks of theta_mark
                    pos_map = pos_map / torch.sum(torch.abs(weights))
            else:
                pos_map = self.position_layer(encoding)
                if self.bounded_position_energy:
                    pos_map = 2 * torch.sigmoid(pos_map) - 1

        if large_image:
            new_pos_map = unpatchify_big_image(patches=pos_map, original_size=original_size, margin=margin, cuts=cuts,
                                               keep_batch_dim=True)
            new_mark_maps = [
                unpatchify_big_image(patches=m, original_size=original_size, margin=margin, cuts=cuts,
                                     keep_batch_dim=True)
                for m in marks_maps
            ]
            pos_map = new_pos_map
            marks_maps = new_mark_maps

        return pos_map, marks_maps

    def densities_from_energy_maps(self, pos_energy_map, marks_energy_maps, energy_weights: Dict = None) -> Tuple[
            Tensor, List[Tensor]]:
        # todo beware of intensity set from density

        if len(pos_energy_map.shape) == 3:  # dim (1,H,W)
            pos_energy_map = torch.unsqueeze(pos_energy_map, dim=0)
            squeeze_flag = True
        elif len(pos_energy_map.shape) == 4:  # dim (B,1,H,W)
            squeeze_flag = False
        else:
            raise RuntimeError(f'invalid shape for pos_energy_map {pos_energy_map.shape}, '
                               f'expected shape like (1,H,W) or (B,1,H,W)')

        h, w = pos_energy_map.shape[2:]

        position_density = torch.exp(-pos_energy_map)
        position_density = position_density / \
            torch.sum(position_density, dim=(2, 3), keepdim=True)
        if torch.any(torch.isnan(position_density)):
            for k in range(len(position_density)):
                if torch.any(torch.isnan(position_density[k])):
                    logging.warning(
                        f'found NaN in position_density[{k}], replacing with uniform density')
                    position_density[k] = torch.ones_like(
                        position_density[k]) / (h * w)

        if squeeze_flag:
            position_density = position_density[0]

        marks_densities = []
        for mapping, mem in zip(self.mappings, marks_energy_maps):
            if len(mem.shape) == 3:  # dim (N_classes,H,W)
                mem = torch.unsqueeze(mem, dim=0)
                squeeze_flag = True
            elif len(pos_energy_map.shape) == 4:  # dim (B,N_classes,H,W)
                squeeze_flag = False
            else:
                raise RuntimeError(f'invalid shape for marks_energy_maps {mem.shape}, '
                                   f'expected shape like (N_classes,H,W) or (B,N_classes,H,W)')

            if not self.compute_position_term:
                weight = energy_weights[f"weight_{mapping.name}"]
                mdm = torch.exp(-weight * mem)
            else:
                mdm = torch.exp(-mem)
            mdm = mdm / torch.sum(mdm, dim=1, keepdim=True)
            if torch.any(torch.isnan(mdm)):
                for k in range(len(mdm)):
                    if torch.any(torch.isnan(mdm[k])):
                        logging.warning(
                            f'found NaN in mark_density[{k}], replacing with uniform density')
                        mdm[k] = torch.ones_like(mdm[k]) / mapping.n_classes
            marks_densities.append(mdm)

        return position_density, marks_densities

    @property
    def sub_energies(self) -> List[str]:
        if self.compute_position_term:
            return ['position'] + [m.name for m in self._mappings]
        else:
            return [m.name for m in self._mappings]

    @property
    def mappings(self) -> List[ValueMapping]:
        return self._mappings

    @property
    def is_trainable(self) -> bool:
        return True


class DummyLikelihood(LikelihoodModule, Module):

    def __init__(self, config: Dict, values: Dict[str, float], compute_position_term: bool, compute_mark_terms: bool,
                 mappings=None):
        super(LikelihoodModule, self).__init__()
        self.values = values
        if mappings is None:
            self._mappings = mappings_from_config(config)
        else:
            self._mappings = mappings
        self.compute_position_term = compute_position_term
        self.compute_mark_terms = compute_mark_terms
        assert self.compute_mark_terms or self.compute_position_term
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if not self.compute_mark_terms:
            for m in self._mappings:
                self.values[m.name] = 0.0

    def forward(self, points: Tensor, points_mask: Tensor, position_energy_map: Tensor,
                marks_energy_maps: List[Tensor]) -> Dict[str, Tensor]:
        n_sets = points.shape[0]
        n_points = points.shape[1]
        h, w = position_energy_map.shape[2:]
        res = {}
        positions = points[..., :2].view((1, n_sets, n_points, 2))
        if self.compute_position_term:
            position_energies = interpolate_position(
                positions=positions,
                image=position_energy_map
            ).view((n_sets, n_points))
            position_energies = position_energies * points_mask
            position_energies = position_energies.view((n_sets, n_points))

            res['position'] = position_energies
        else:
            pass  # there is nothing to compute here

        if self.compute_mark_terms:
            for i, (mark_energy_map, mapping) in enumerate(zip(marks_energy_maps, self.mappings)):
                mark_energies = interpolate_marks(
                    positions=positions.view((1, 1, n_sets, n_points, 2)),
                    marks=points[..., 2 + i].view((1, 1, n_sets, n_points)),
                    image=mark_energy_map.view(
                        (1, 1, mapping.n_classes, h, w)),
                    mapping=mapping
                ).view((n_sets, n_points))
                res[mapping.name] = mark_energies * points_mask

        return res

    def energy_maps_from_image(self, image, position_map_from_marks: bool = None, **kwargs) -> Tuple[
            Tensor, List[Tensor]]:
        shape = image.shape[2:]
        batch_size = image.shape[0]

        pos_map = torch.ones((batch_size, 1) + shape,
                             device=self.device) * self.values['position']
        marks_maps = [torch.ones((batch_size, m.n_classes) + shape, device=self.device) * self.values[m.name]
                      for m in self._mappings]

        return pos_map, marks_maps

    def densities_from_energy_maps(self, pos_energy_map, marks_energy_maps, energy_weights: Dict = None, **kwargs) -> \
            Tuple[Tensor, List[Tensor]]:

        if len(pos_energy_map.shape) == 4:
            pos_energy_map = pos_energy_map[0]  # dim (1,H,W)
        position_density = torch.exp(-pos_energy_map)
        position_density = position_density / \
            torch.sum(position_density, dim=(1, 2), keepdim=True)

        marks_densities = []
        for mapping, mem in zip(self.mappings, marks_energy_maps):
            assert len(mem.shape) == 3  # dim (N_classes,H,W)
            if not self.compute_position_term:
                weight = energy_weights[f"weight_{mapping.name}"]
                mdm = torch.exp(-weight * mem)
            else:
                mdm = torch.exp(-mem)
            mdm = mdm / torch.sum(mdm, dim=0, keepdim=True)
            marks_densities.append(mdm)

        return position_density, marks_densities

    @property
    def sub_energies(self) -> List[str]:
        keys = []
        if self.compute_position_term:
            keys = keys + ['position']
        if self.compute_mark_terms:
            keys = keys + [m.name for m in self._mappings]
        return keys

    @property
    def mappings(self) -> List[ValueMapping]:
        return self._mappings


class InlineCNNEnergyReg(LikelihoodModule, Module):

    def __init__(self, config, bounded_position_energy: bool = True, bounded_mark_energy: bool = True,
                 hidden_dims: List[int] = None, learn_distance: bool = False, position_term: bool = True):
        super(InlineCNNEnergyReg, self).__init__()
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device is {self.device}")

        self._mappings = mappings_from_config(self.config)
        self.n_marks = len(self._mappings)

        self.bounded_position_energy = bounded_position_energy
        self.bounded_mark_energy = bounded_mark_energy

        self.hidden_dims = hidden_dims
        if hidden_dims is None:
            hidden_dims = [8, 16]

        self.backbone = Unet(
            hidden_dims=hidden_dims,
            in_channels=3,
            device=self.device
        ).to(self.device)

        position_mid_channels = 2
        self.position_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.backbone.out_channels, out_channels=position_mid_channels,
                      kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=position_mid_channels,
                      out_channels=1, kernel_size=(1, 1))
        ).to(self.device)

        marks_mid_channels = self.n_marks
        self.marks_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.backbone.out_channels, out_channels=marks_mid_channels,
                      kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=marks_mid_channels, out_channels=self.n_marks,
                      kernel_size=(1, 1)),
            ScaleModule(v_min=[m.v_min for m in self._mappings],
                        v_max=[m.v_max for m in self._mappings],
                        is_cyclic=[m.is_cyclic for m in self._mappings], device=self.device)
        ).to(self.device)

        self.compute_position_term = position_term
        if not self.compute_position_term and not learn_distance:
            raise RuntimeError(
                "position_term=False and learn_distance=False are not supported simultaneously")
        if not self.compute_position_term:
            raise NotImplementedError("look, it is not implemented yet...")

        if not learn_distance:
            self.distance_function = None
        else:
            self.distance_function = nn.Sequential(
                nn.Linear(in_features=1, out_features=1, device=self.device),
                nn.Sigmoid()
            )

        self.scale_tensor = torch.tensor(
            np.array([m.range for m in self.mappings]), device=self.device)

    def forward(self, points: Tensor, points_mask: Tensor, position_energy_map: Tensor,
                marks_energy_maps: List[Tensor]) -> Dict[str, Tensor]:
        # TODO MAKE REGRESSIVE ?

        n_sets = points.shape[0]
        n_points = points.shape[1]
        h, w = position_energy_map.shape[2:]
        res = {}
        positions = points[..., :2].view((1, n_sets, n_points, 2))

        position_energies = interpolate_position(
            positions=positions,
            image=position_energy_map
        ).view((n_sets, n_points))
        position_energies = position_energies * points_mask
        position_energies = position_energies.view((n_sets, n_points))

        res['position'] = position_energies

        # for i, (mark_energy_map, mapping) in enumerate(zip(marks_energy_maps, self.mappings)):
        #     mark_energies = interpolate_marks(
        #         positions=positions.view((1, 1, n_sets, n_points, 2)),
        #         marks=points[..., 2 + i].view((1, 1, n_sets, n_points)),
        #         image=mark_energy_map.view((1, 1, mapping.n_classes, h, w)),
        #         mapping=mapping
        #     ).view((n_sets, n_points))
        #     res[mapping.name] = mark_energies * points_mask

        mark_vector = marks_energy_maps[-1]

        mark_vector_at_positions = interpolate_position(
            positions=positions,
            image=mark_vector
        ).view((n_sets, n_points, self.n_marks))

        mark_values = points[..., 2:]
        marks_energies = self._diff_to_energy(
            values=mark_values, estimates=mark_vector_at_positions, marks_dim=2)

        for i, m in enumerate(self.mappings):
            res[m.name] = marks_energies[..., i]

        return res

    def _diff_to_energy(self, values: Tensor, estimates: Tensor, marks_dim: int):
        # diff_tensor = torch.abs((values - estimates)).float()
        # diff_scaled = diff_tensor / torch.tensor(np.array([m.range for m in self.mappings]), device=diff_tensor.device)

        # estimates_t = torch.ones_like(estimates) * 8
        diff_tensor = torch.abs((values - estimates)).float()
        n_dims = len(diff_tensor.shape)
        scale_brdcst = [1] * n_dims
        scale_brdcst[marks_dim] = self.n_marks
        diff_scaled = torch.square(
            diff_tensor / self.scale_tensor.view(tuple(scale_brdcst)))
        if self.distance_function is None:
            return diff_scaled
        else:
            return self.distance_function(diff_scaled.unsqueeze(dim=-1)).squeeze(dim=-1)

    def energy_maps_from_image(self, image, energy_weights: Dict = None,
                               **kwargs) -> Tuple[
            Tensor, List[Tensor]]:
        requires_grad: bool = kwargs.get('requires_grad', False)
        logging.debug(f"computing maps with requires_grad={requires_grad}")
        large_image: bool = kwargs.get('large_image', False)
        if large_image and requires_grad:
            raise RuntimeError(
                "Cannot have both large_image and requires_grad")

        # if position_map_from_marks is None:
        #     position_map_from_marks = not self.compute_position_term
        # assert position_map_from_marks or (self.position_layer is not None)

        if large_image:
            patch_size = 256
            margin = [14, 32, 68, 140][len(self.hidden_dims)] // 2
            assert 2 * margin < patch_size
            original_size = image.shape[1:]
            assert len(image) == 1  # only one image to patchify
            patches, padded_image, cuts = patchify_big_image(
                image=image[0], margin=margin, patch_size=patch_size)
            logging.info(f"cut big image into {len(patches)} patches")
            image = patches

        assert type(image) is Tensor
        assert len(image.shape) == 4  # (B,C,H,W)

        with nullcontext() if requires_grad else torch.no_grad():
            encoding = self.backbone(image)
            marks_vector = self.marks_layer(encoding)

            pos_map = self.position_layer(encoding)
            if self.bounded_position_energy:
                pos_map = 2 * torch.sigmoid(pos_map) - 1

        if large_image:
            new_pos_map = unpatchify_big_image(
                patches=pos_map, original_size=original_size, margin=margin, cuts=cuts, keep_batch_dim=True)
            new_marks_vector = unpatchify_big_image(
                patches=marks_vector, original_size=original_size, margin=margin, cuts=cuts, keep_batch_dim=True)

            pos_map = new_pos_map
            marks_vector = new_marks_vector

        mark_space_tensor = torch.tensor(np.array([m.feature_mapping for m in self.mappings]),
                                         device=marks_vector.device)

        h, w = marks_vector.shape[2:]
        n_batch = marks_vector.shape[0]
        n_classes = mark_space_tensor.shape[1]

        marks_energies = self._diff_to_energy(
            values=mark_space_tensor.view(1, self.n_marks, n_classes, 1, 1),
            estimates=marks_vector.view(n_batch, self.n_marks, 1, h, w),
            marks_dim=1
        )

        marks_maps = [marks_energies[:, i]
                      for i in range(self.n_marks)] + [marks_vector]
        #  + [marks_vector,variance] is a sneak ugly way to pass this tensor to the forward method w/o having
        #  to rewrite the whole code

        return pos_map, marks_maps

    def densities_from_energy_maps(self, pos_energy_map, marks_energy_maps, energy_weights: Dict = None) -> Tuple[
            Tensor, List[Tensor]]:
        # todo beware of intensity set from density

        assert len(pos_energy_map.shape) == 3  # dim (1,H,W)
        weight = energy_weights[f"weight_position"]
        position_density = torch.exp(-weight * pos_energy_map)
        position_density = position_density / \
            torch.sum(position_density, dim=(1, 2), keepdim=True)

        marks_densities = []
        for mapping, mem in zip(self.mappings, marks_energy_maps):
            assert len(mem.shape) == 3  # dim (N_classes,H,W)
            weight = energy_weights[f"weight_{mapping.name}"]
            mdm = torch.exp(-weight * mem)
            mdm = mdm / torch.sum(mdm, dim=0, keepdim=True)
            marks_densities.append(mdm)

        return position_density, marks_densities

    @property
    def sub_energies(self) -> List[str]:
        return ['position'] + [m.name for m in self._mappings]

    @property
    def mappings(self) -> List[ValueMapping]:
        return self._mappings

    @property
    def is_trainable(self) -> bool:
        return True


class ContrastEnergy(LikelihoodModule, Module):

    def __init__(self, shape: str, mode: str, resolution: int, dilation: int, config):
        super(ContrastEnergy, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        assert mode in ['contrast', 'gradient']
        self.mode = mode

        if shape == 'rectangle':
            from energies.contrast_utils import square_coordinates
            from energies.contrast_utils import square_perimeter_coordinates
            self.stencil_in = square_coordinates(resolution).float()
            self.stencil_out = square_perimeter_coordinates(
                resolution, dilation=dilation).float()
        else:
            raise ValueError

        self.stencil_in = torch.concat(
            (self.stencil_in, torch.ones((len(self.stencil_in), 1))), dim=-1).to(self.device)
        self.stencil_out = torch.concat(
            (self.stencil_out, torch.ones((len(self.stencil_out), 1))), dim=-1).to(self.device)

        self._mappings = mappings_from_config(config)

    def forward(self, points: Tensor, points_mask: Tensor, position_energy_map: Tensor,
                marks_energy_maps: List[Tensor]) -> Dict[str, Tensor]:
        if self.mode == 'contrast':
            # position_energy_map should be the grayscale image
            affine_matrices = affine_from_state(points)

            in_coordinates = torch.matmul(
                affine_matrices, self.stencil_in.T).permute(0, 1, 3, 2)[..., :2]
            out_coordinates = torch.matmul(
                affine_matrices, self.stencil_out.T).permute(0, 1, 3, 2)[..., :2]

            n_sets = points.shape[0]
            n_points = points.shape[1]
            n_in = self.stencil_in.shape[0]
            n_out = self.stencil_out.shape[0]

            in_values = interpolate_position(
                image=position_energy_map,
                positions=in_coordinates.view(1, n_sets * n_points, n_in, 2)
            ).view(n_sets, n_points, n_in)  # (B,C,P,N)
            out_values = interpolate_position(
                image=position_energy_map,
                positions=out_coordinates.view(1, n_sets * n_points, n_out, 2)
            ).view(n_sets, n_points, n_out)  # (B,C,P,N)

            contrast = contrast_measure(in_values, out_values, dim=-1)

            return {
                'contrast': contrast * points_mask
            }

        elif self.mode == 'gradient':
            raise NotImplementedError  # todo
            # position_energy_map should be the iamge gradient
        else:
            raise ValueError

    def energy_maps_from_image(self, image, **kwargs) -> Tuple[Tensor, List[Tensor]]:
        if self.mode == 'contrast':
            # position_energy_map should be the grayscale image
            return torch.mean(image, dim=1, keepdim=True), []

        elif self.mode == 'gradient':
            raise NotImplementedError  # todo
            # position_energy_map should be the image gradient
        else:
            raise ValueError

    def densities_from_energy_maps(self, pos_energy_map, marks_energy_maps, **kwargs) -> Tuple[Tensor, List[Tensor]]:
        if self.mode == 'contrast':
            pass
        elif self.mode == 'gradient':
            pass
        else:
            raise ValueError
        assert len(pos_energy_map.shape) == 3
        position_density = torch.ones_like(pos_energy_map)
        position_density = position_density / \
            torch.sum(position_density, dim=(1, 2), keepdim=True)
        return position_density, []

    @property
    def sub_energies(self) -> List[str]:
        return ['contrast']

    @property
    def mappings(self) -> List[ValueMapping]:
        return self._mappings


@property
def mappings(self) -> List[ValueMapping]:
    pass
