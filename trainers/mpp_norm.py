import itertools
import sys
from abc import abstractmethod, ABC
from typing import List, Tuple, Dict, Union

import torch
from torch import Tensor
from torch.nn import functional

from energies.base import BaseEnergyModel
from energies.energy_combinators import LinearCombinator
from energies.generic_energy_model import GenericEnergyModel
from modules.custom import LinearWDeltas


class EnergyReg(ABC):

    @property
    @abstractmethod
    def log_keys(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, pos_energies: Tensor, neg_energies: Tensor, pos_sub_energies: Dict[str, Tensor],
                neg_sub_energies: Dict[str, Tensor],
                model: BaseEnergyModel) -> Tuple[Tensor, Dict]:
        """

        :param pos_energies:
        :type pos_energies:
        :param neg_energies:
        :type neg_energies:
        :param model:
        :type model:
        :return: (norm, log update)
        :rtype:
        """
        raise NotImplementedError

    @property
    def requires_sub_energies(self) -> bool:
        return False


class Void(EnergyReg):

    @property
    def log_keys(self) -> List[str]:
        return []

    def forward(self, pos_energies: Tensor, neg_energies: Tensor, model: BaseEnergyModel, **kwargs) -> Tuple[
            Tensor, Dict]:
        return torch.zeros((1,), device=pos_energies.device), {}


class Composite(EnergyReg):

    def __init__(self, **kwargs):
        self.sub_reg: List[EnergyReg] = [
            getattr(sys.modules[__name__], name)(**args) for name, args in kwargs.items()
        ]

    @property
    def log_keys(self) -> List[str]:
        return list(itertools.chain(*[r.log_keys for r in self.sub_reg]))

    def forward(self, pos_energies: Tensor, neg_energies: Tensor, pos_sub_energies: Dict[str, Tensor],
                neg_sub_energies: Dict[str, Tensor],
                model: BaseEnergyModel) -> Tuple[Tensor, Dict]:
        norms = []
        logs = {}
        for reg in self.sub_reg:
            norm, log = reg.forward(
                pos_energies=pos_energies,
                neg_energies=neg_energies,
                pos_sub_energies=pos_sub_energies,
                neg_sub_energies=neg_sub_energies,
                model=model)
            norms.append(norm)
            logs.update(log)
        return torch.sum(torch.stack(norms)), logs

    @property
    def requires_sub_energies(self) -> bool:
        return any([r.requires_sub_energies for r in self.sub_reg])


class NormEnergy(EnergyReg):

    def __init__(self, weight: float):
        self.weight_l2norm = weight

    def forward(self, pos_energies: Tensor, neg_energies: Tensor, model: BaseEnergyModel, **kwargs) -> Tuple[
            Tensor, Dict]:
        if self.weight_l2norm != 0.0:
            norm_pos = torch.mean(torch.square(pos_energies)) \
                if len(pos_energies) > 0 else torch.zeros([], device=pos_energies.device)
            norm_neg = torch.mean(torch.square(neg_energies)) \
                if len(neg_energies) > 0 else torch.zeros([], device=neg_energies.device)

        else:
            norm_pos = torch.zeros([], device=pos_energies.device)
            norm_neg = torch.zeros([], device=neg_energies.device)

        norm = self.weight_l2norm * (norm_pos + norm_neg)

        e_pos_norm = self.weight_l2norm * norm_pos
        e_neg_norm = self.weight_l2norm * norm_neg

        log_dict = {
            'weighted_e_pos_norm':
                e_pos_norm if type(e_pos_norm) is float else float(
                    e_pos_norm.cpu().detach().numpy()),
            'weighted_e_neg_norm':
                e_neg_norm if type(e_neg_norm) is float else float(
                    e_neg_norm.cpu().detach().numpy())
        }

        return norm, log_dict

    @property
    def log_keys(self) -> List[str]:
        return ['weighted_e_pos_norm', 'weighted_e_neg_norm']


class NormWeights(EnergyReg):
    def __init__(self, weight: float):
        self.weight = weight

    def forward(self, pos_energies: Tensor, neg_energies: Tensor, model: BaseEnergyModel, **kwargs) -> Tuple[
            Tensor, Dict]:
        comb_module = model.energy_combination_module
        if isinstance(comb_module, LinearCombinator):
            comb_module: LinearCombinator

            norm = torch.sum(torch.square(comb_module.weights)) + \
                torch.sum(torch.square(comb_module.bias))

            norm = norm / (len(model.sub_energies) + 1)
            norm = norm * self.weight
            log_dict = {
                'weights_l2_norm': float(norm.detach().cpu())
            }
        else:
            raise NotImplementedError

        return norm, log_dict

    @property
    def log_keys(self) -> List[str]:
        return ['weights_l2_norm']


class NormSubEnergies(EnergyReg):

    def __init__(self, weight: float, sub_energies: Union[str, List[str]]):
        self.weight = weight
        if sub_energies == 'all':
            self.all_sub_energies = True
            self.sub_energies = None
        else:
            if type(sub_energies) is not list:
                sub_energies = [sub_energies]
            self.all_sub_energies = False
            self.sub_energies = sub_energies

    @property
    def requires_sub_energies(self) -> bool:
        return True

    @property
    def log_keys(self) -> List[str]:
        return ['pos_sub_e_norm', 'neg_sub_e_norm']

    def forward(self, pos_energies: Tensor, neg_energies: Tensor, pos_sub_energies, neg_sub_energies,
                model: BaseEnergyModel) -> Tuple[Tensor, Dict]:

        if self.all_sub_energies:
            energy_keys = model.sub_energies
        else:
            energy_keys = self.sub_energies

        if self.weight != 0.0:
            norm_pos = torch.stack(
                [torch.mean(torch.square(pos_sub_energies[k])) if len(pos_sub_energies[k]) > 0 else torch.zeros([])
                 for k in energy_keys]
            ).mean()
            norm_neg = torch.stack(
                [torch.mean(torch.square(neg_sub_energies[k])) if len(neg_sub_energies[k]) > 0 else torch.zeros([])
                 for k in energy_keys]
            ).mean()
        else:
            norm_pos = torch.zeros([], device=pos_energies.device)
            norm_neg = torch.zeros([], device=neg_energies.device)

        norm = self.weight * (norm_pos + norm_neg)

        e_pos_norm = self.weight * norm_pos
        e_neg_norm = self.weight * norm_neg

        log_dict = {
            'pos_sub_e_norm':
                e_pos_norm if type(e_pos_norm) is float else float(
                    e_pos_norm.cpu().detach().numpy()),
            'neg_sub_e_norm':
                e_neg_norm if type(e_neg_norm) is float else float(
                    e_neg_norm.cpu().detach().numpy())
        }

        return norm, log_dict


class HingeSubEnergies(EnergyReg):

    def __init__(self, weight: float, square: bool):
        super(HingeSubEnergies, self).__init__()
        self.weight = weight
        self.square = square

    @property
    def requires_sub_energies(self) -> bool:
        return True

    @property
    def log_keys(self) -> List[str]:
        return ['norm_hinge_pos', 'norm_hinge_neg']

    def forward(self, pos_energies: Tensor, neg_energies: Tensor, pos_sub_energies: Dict[str, Tensor],
                neg_sub_energies: Dict[str, Tensor], model: BaseEnergyModel) -> Tuple[Tensor, Dict]:
        model: GenericEnergyModel
        assert model.energy_combination_module.use_deltas
        weight_model: LinearWDeltas = model.energy_combination_module.model[0]
        weighted_enr_pos = weight_model(torch.stack(
            list(pos_sub_energies.values()), dim=-1))
        weighted_enr_neg = weight_model(torch.stack(
            list(neg_sub_energies.values()), dim=-1))

        norm_hinge_pos = functional.relu(weighted_enr_pos)
        norm_hinge_neg = functional.relu(-weighted_enr_neg)
        if self.square:
            norm_hinge_pos = torch.square(norm_hinge_pos)
            norm_hinge_neg = torch.square(norm_hinge_neg)
        norm_hinge_pos = self.weight * torch.mean(norm_hinge_pos)
        norm_hinge_neg = self.weight * torch.mean(norm_hinge_neg)
        norm = norm_hinge_pos + norm_hinge_neg
        log_dict = {
            'norm_hinge_pos':
                norm_hinge_pos if type(norm_hinge_pos) is float else float(
                    norm_hinge_pos.cpu().detach().numpy()),
            'norm_hinge_neg':
                norm_hinge_neg if type(norm_hinge_neg) is float else float(
                    norm_hinge_neg.cpu().detach().numpy())
        }

        return norm, log_dict
