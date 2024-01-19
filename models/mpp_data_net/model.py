import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Module

from base.base_model import BaseModel
from base.mappings import mappings_from_config
from base.misc import timestamp
from display.light_display.image_stack import make_image_from_bunch, add_top_legend, add_header
from display.light_display.plots import multi_hist_image
from modules.torch_div import Divergence, torch_divergence, NegativeModule
from modules.unet.unet import Unet


class MPPDataModel(Module, BaseModel):

    def __init__(self, config):
        super(MPPDataModel, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device is {self.device}")

        self.config = config

        self.backbone = Unet(
            hidden_dims=self.config['model']['hidden_dims'],
            in_channels=3,
            device=self.device
        ).to(self.device)
        self.mappings = mappings_from_config(self.config)
        self.n_marks = len(self.mappings)

        encoding_process_depth = config['model'].get(
            'encoding_process_depth', 0)
        # use_3d_conv = config['model'].get('use_3d_conv', False)
        # if use_3d_conv:
        #     if encoding_process_depth < 1:
        #         logging.warning(f"cannot use Conv3d if encoding_process_depth={encoding_process_depth}<1")

        if encoding_process_depth == 0:  # backward compatibility
            self.vec_field_and_mask_layer = nn.Conv2d(
                in_channels=self.backbone.out_channels,
                out_channels=3,
                kernel_size=(1, 1),
                device=self.device,
            )
            self.marks_layers = nn.ModuleList()
            for _ in self.mappings:
                self.marks_layers.append(
                    nn.Conv2d(
                        in_channels=self.backbone.out_channels,
                        out_channels=self.config['model']['marks_classes'],
                        kernel_size=(1, 1), device=self.device
                    ).to(self.device)
                )
        else:
            hidden_dim = 8
            ops = [nn.Conv2d(
                in_channels=self.backbone.out_channels,
                out_channels=hidden_dim,
                kernel_size=(1, 1),
                device=self.device
            ), nn.ReLU()]
            for _ in range(encoding_process_depth - 1):
                ops = ops + [nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=(1, 1),
                    device=self.device
                ),
                    nn.ReLU()]
            ops.append(nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=3,
                kernel_size=(1, 1),
                device=self.device
            ))

            self.vec_field_and_mask_layer = nn.Sequential(*ops)

            self.marks_layers = nn.ModuleList()
            for _ in self.mappings:
                hidden_dim_m = self.config['model']['marks_classes']
                ops_m = [nn.Conv2d(
                    in_channels=self.backbone.out_channels,
                    out_channels=hidden_dim_m,
                    kernel_size=(1, 1),
                    device=self.device
                ), nn.ReLU()]
                for _ in range(encoding_process_depth - 1):
                    ops_m = ops_m + [nn.Conv2d(
                        in_channels=hidden_dim_m,
                        out_channels=hidden_dim_m,
                        kernel_size=(1, 1),
                        device=self.device
                    ), nn.ReLU()]
                ops_m.append(nn.Conv2d(
                    in_channels=hidden_dim_m,
                    out_channels=self.config['model']['marks_classes'],
                    kernel_size=(1, 1),
                    device=self.device
                ))
                self.marks_layers.append(nn.Sequential(*ops_m))

        if self.config['model'].get('skip_linear', False):
            self.div_linear_layer = nn.Sequential(
                Divergence(div_channels=[0, 1],
                           mask_channel=2, sigmoid_on_mask=True),
                NegativeModule()
            ).to(self.device)
        else:
            self.div_linear_layer = nn.Sequential(
                Divergence(div_channels=[0, 1],
                           mask_channel=2, sigmoid_on_mask=True),
                nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1, 1))
            ).to(self.device)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        backbone_res = self.backbone(x)
        # positional
        vector_field_and_mask = self.vec_field_and_mask_layer(backbone_res)
        center_heatmap = self.div_linear_layer(vector_field_and_mask)
        # marks
        marks_outputs = {
            f"mark_{m.name}": l(backbone_res) for m, l in zip(self.mappings, self.marks_layers)
        }

        return {
            'vector': vector_field_and_mask[:, [0, 1]],
            'mask': torch.sigmoid(vector_field_and_mask[:, [2]]),
            'center_heatmap': center_heatmap,
            **marks_outputs
        }

    def make_figures(self, epoch: int, inputs, output, labels, loss_dict) -> np.ndarray:
        patches = inputs.cpu().permute((0, 2, 3, 1)).numpy()

        n_images = len(patches)
        patch_size = patches.shape[1]

        bin_cmap = 'viridis'
        energy_cmap = 'coolwarm'

        # GT

        vec_mask = labels['distance'].cpu().numpy(
        ) < self.config['loss_params']['vector_distance_to_center']

        def display_normals(gt: bool):
            if gt:
                vec = labels['vector'].permute((0, 2, 3, 1)).cpu(
                ).numpy() * np.expand_dims(vec_mask, axis=-1)
            else:
                vec = output['vector'].permute((0, 2, 3, 1)).cpu().numpy()
            vec = np.concatenate([vec, np.ones(vec.shape[:3] + (1,))], axis=-1)
            vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
            vec = vec[..., [1, 0, 2]] * 0.5 + 0.5
            vec[..., 0] = 1 - vec[..., 0]
            return vec

        gt_vector = display_normals(True)

        gt_mask = plt.get_cmap(bin_cmap)(vec_mask.astype(float))[..., :3]

        gt_heatmap = torch.exp(
            - 0.5 * torch.square(labels['distance']) / (self.config['loss_params']['blob_sigma'] ** 2))
        gt_heatmap = gt_heatmap.cpu().numpy()
        gt_heatmap_col = plt.get_cmap(energy_cmap)(1 - gt_heatmap)[..., :3]

        def process_mark_gt(gt: bool, mark_str: str, colormap: str, use_mask: bool, fill_value=(0, 0, 0)):
            if gt:
                mark_map = labels[f'mark_{mark_str}'].cpu().numpy()
                mark_argmax = mark_map
            else:
                mark_map = output[f'mark_{mark_str}'].permute(
                    (0, 2, 3, 1)).cpu().numpy()
                mark_argmax = np.argmax(mark_map, axis=-1)
            n_classes = self.config['model']['marks_classes']
            mark_argmax_col = plt.get_cmap(colormap)(
                mark_argmax / n_classes)[..., :3]
            if use_mask:
                mask_mark = labels['object_mask'].cpu().numpy()
                mark_argmax_col[mask_mark == 0] = np.array(fill_value)
            return mark_argmax_col

        gt_mark_width = process_mark_gt(True, 'width', 'rainbow', True)
        gt_mark_length = process_mark_gt(True, 'length', 'rainbow', True)
        gt_mark_angle = process_mark_gt(True, 'angle', 'twilight', True)

        mark_width = process_mark_gt(False, 'width', 'rainbow', False)
        mark_length = process_mark_gt(False, 'length', 'rainbow', False)
        mark_angle = process_mark_gt(False, 'angle', 'twilight', False)

        def show_dists(mark_str: str):
            mark_map = torch.softmax(output[f'mark_{mark_str}'], dim=1).permute(
                (0, 2, 3, 1)).cpu().numpy()
            mark_map_gt = labels[f'mark_{mark_str}'].cpu().numpy()

            centers_bin = labels['distance'].cpu().numpy() == 0
            hist_images = []
            for i in range(n_images):
                centers = np.array(np.where(centers_bin[i])).T
                if len(centers) > 0:
                    n_dist = 4
                    dists = mark_map[i, centers[:n_dist, 0],
                                     centers[:n_dist, 1]]
                    gts = mark_map_gt[i, centers[:n_dist, 0],
                                      centers[:n_dist, 1]]
                    hist_images.append(
                        multi_hist_image(size=patch_size, distribution=dists, plot_cmap=plt.get_cmap('plasma'),
                                         gt=gts, vmax='auto', vmin=0)
                    )
                else:
                    hist_images.append(np.zeros((patch_size, patch_size, 3)))

            return np.stack(hist_images, axis=0)

            # return multi_hist_image()

        mark_width_hists = show_dists('width')
        mark_length_hists = show_dists('length')
        mark_angle_hists = show_dists('angle')

        # inference

        vector = display_normals(False)

        # div = torch_divergence(labels['vector'], indexing='ij').cpu().numpy().squeeze()
        div = torch_divergence(
            output['vector'], indexing='ij').cpu().numpy().squeeze()
        # div = torch_divergence(output['vector'][:, [0, 1]], indexing='ij').cpu().numpy().squeeze()
        div = div * output['mask'].cpu().numpy().squeeze()
        div = (div / np.max(np.abs(div), axis=(1, 2), keepdims=True))
        div = plt.get_cmap('coolwarm')(div * 0.5 + 0.5)[..., :3]

        mask = output['mask'].cpu().numpy().squeeze()
        mask_col = plt.get_cmap(bin_cmap)(mask)[..., :3]

        heatmap = torch.sigmoid(
            output['center_heatmap']).squeeze(dim=1).cpu().numpy()
        heatmap_col = plt.get_cmap(energy_cmap)(1 - heatmap)[..., :3]

        images_and_legends = [
            ("patches", patches),
            ("GT vector", gt_vector), ('GT mask',
                                       gt_mask), ("GT heatmap", gt_heatmap_col),
            ("GT mark width", gt_mark_width), ("GT mark length",
                                               gt_mark_length), ("GT mark angle", gt_mark_angle),
            ("vector", vector), ('div*mask', div), ('mask',
                                                    mask_col), ("heatmap", heatmap_col),
            ("mark width", mark_width), ("mark length",
                                         mark_length), ("mark angle", mark_angle),
            ("width dist.", mark_width_hists), ("length dist.",
                                                mark_length_hists), ("angle dist.", mark_angle_hists)
        ]

        legends = [il[0] for il in images_and_legends]
        images = np.stack([il[1] for il in images_and_legends], axis=1)
        big_image = make_image_from_bunch(images)
        big_image, _ = add_top_legend(big_image, legends)
        big_image, _ = add_header(big_image, f"epoch: {epoch:04} | " +
                                  ' | '.join(
                                      [f"{k}: {float(v.detach().cpu().numpy()):.4f}" for k, v in loss_dict.items()]) +
                                  f' | {timestamp()}')

        return big_image
