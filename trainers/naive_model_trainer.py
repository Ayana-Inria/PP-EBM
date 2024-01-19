import logging
import os
import pickle
import re
import sys
import traceback

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from numpy.random import Generator
from tqdm import tqdm

from base.data import get_inference_path, fetch_data_paths
from base.files import make_if_not_exist
from base.geometry import rect_to_poly
from base.parse_config import ConfigParser
from base.trainer import BaseTrainer
from display.draw_on_img import draw_shapes_on_img
from metrics.dota_results_translator import DOTAResultsTranslator
from models.naive_cnn import NaiveCnn


class NaiveCNNTrainer(BaseTrainer):

    def __init__(self, model: NaiveCnn, criterion, optimizer, config: ConfigParser, rng: Generator,
                 force_dataset: str = None, scheduler=None, ):
        super().__init__(model, criterion, optimizer, config, scheduler)

        self.config = config
        self.rng = rng
        self.dataset = self.config['data_loader']["dataset"] if force_dataset is None else force_dataset

    def train(self):
        raise NotImplementedError

    def infer(self, overwrite_results: bool, draw: bool, ignore_errors: bool = False):

        if ignore_errors:
            raise NotImplementedError()

        plt.ioff()
        subset = 'val'

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

        lm_dist = self.config['inference']['localmax_distance']
        lm_thresh = self.config['inference']['localmax_thresh']

        for pf, af, mf in zip(tqdm(image_paths, desc=f'inferring on {self.dataset}/{subset}', file=sys.stdout),
                              annot_paths,
                              meta_paths):
            patch_id = int(id_re.match(os.path.split(pf)[1]).group(1))

            # if patch_id not in SUBSET_DEBUG:
            #     continue

            logging.info(f"loading patch {patch_id}")
            results_pickle = os.path.join(
                results_dir, f'{patch_id:04}_results.pkl')

            image = plt.imread(pf)[..., :3]

            if os.path.exists(results_pickle) and not overwrite_results:
                with open(results_pickle, 'rb') as f:
                    results_dict = pickle.load(f)
                proposals = results_dict['state']
                scores = results_dict['score']
            else:
                image_t = torch.from_numpy(image).permute(
                    (2, 0, 1)).to(self.model.device)
                shape = image.shape[:2]

                output = self.model.infer_on_image(
                    image_t, lm_distance=lm_dist, lm_thresh=lm_thresh, large_image=True)

                proposals = output['proposals'][0]
                scores = output['scores'][0]

                results_dict = {
                    'state': proposals,
                    'score': scores
                }

                with open(os.path.join(results_dir, f'{patch_id:04}_results.pkl'), 'wb') as f:
                    pickle.dump(results_dict, f)

            with open(af, 'rb') as f:
                labels_dict = pickle.load(f)

            centers = labels_dict['centers']
            params = labels_dict['parameters']
            difficult = labels_dict['difficult']
            categories = labels_dict['categories']
            gt_state = np.array([list(c) + list(p)
                                for c, p in zip(centers, params)])
            draw_method = self.config['trainer'].get(
                'draw_method', 'rectangle')
            if draw_method == 'rectangle':
                gt_as_poly = np.array(
                    [rect_to_poly(c, short=p[0], long=p[1], angle=p[2]) for c, p in zip(centers, params)])

                dota_translator.add_gt(
                    image_id=patch_id,
                    polygons=gt_as_poly,
                    difficulty=[d or c == 'large-vehicle' for d, c in
                                zip(difficult, categories)],
                    categories=['vehicle' for _ in gt_as_poly])

                # pred_scores = np.ones(len(last_state))

                detection_as_poly = np.array(
                    [rect_to_poly(s[[0, 1]], short=s[2], long=s[3], angle=s[4]) for s in proposals])

                dota_translator.add_detections(
                    image_id=patch_id,
                    scores=scores,
                    polygons=detection_as_poly,
                    flip_coor=True,
                    class_names=['vehicle' for _ in scores])
            else:
                raise NotImplementedError
            if draw:
                try:  # try drawing
                    image_d = image.copy()
                    image_w_pred = draw_shapes_on_img(
                        image=image_d, states=proposals, color=(0, 1.0, 0), draw_method=draw_method
                    )
                    plt.imsave(os.path.join(
                        results_dir, f'{patch_id:04}_results.png'), image_w_pred)

                except Exception as e:
                    trace = traceback.format_exc()
                    logging.error(f"FAILED making figures with:\n{trace}")
                    with open(os.path.join(results_dir, f'{patch_id:04}_display_error.txt'), 'w') as f:
                        print(trace, file=f)

            logging.info(f'frame {patch_id:04} done !')

        dota_translator.save()
        print('saved dota translation')

    def eval(self):
        from metrics.dota_eval import dota_eval
        dota_eval(
            model_dir=self.config.save_dir,
            dataset=self.dataset,
            subset='val',
            det_type='obb'
        )
