import os
import pickle

import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt

from metrics.papangelou import efficient_papangelou_scoring
from tools.common_tools import load_mpp_model


def main():
    with open('sub_papangelou_score.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    energy_infer_args = {"cut_and_stitch": True}

    for model_name, model_args in config['models'].items():
        print(f"{model_name:=^30}")
        print(f"loading model !")
        model = load_mpp_model(os.path.join(config['models_path'], model_name))
        for dataset in config['datasets']:
            print(f"{dataset:-^30}")
            for i, p in enumerate(config['patches']):
                print(f"p{p:04} ({i + 1}/{len(config['patches'])})")
                save_file = os.path.join(config['inference_path'], dataset, 'val', model_name,
                                         f"{p:04}_results_details.pkl")
                if os.path.exists(save_file) and not config['overwrite']:
                    print("file exists: skipping !")
                else:
                    with open(os.path.join(config['inference_path'], dataset, 'val', model_name, f"{p:04}_results.pkl"),
                              'rb') as f:
                        res = pickle.load(f)

                    image = plt.imread(os.path.join(config['data_path'], dataset, 'val', 'images', f"{p:04}.png"))[...,
                            :3]
                    state = res['state']

                    image_t = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(dim=0)
                    support_shape = image.shape[:2]
                    pos_e_map, marks_e_map = model.energy_maps_from_image(image_t.to(model.device), **energy_infer_args)

                    states_scores, state_energies = efficient_papangelou_scoring(
                        states=state,
                        model=model,
                        log_values=True,
                        verbose=1,
                        image=image_t,
                        pos_e_m=pos_e_map,
                        marks_e_m=marks_e_map,
                        return_sub_energies=True,
                        use_buffer=config['use_buffer']
                    )

                    res = {
                        'state': state,
                        'score': np.log(states_scores),
                        'log_papangelou': states_scores,
                        'energies_deltas': state_energies,
                        'energies': model.sub_energies,
                        'weights': model.combination_module_weights
                    }
                    with open(save_file, 'wb') as f:
                        pickle.dump(res, f)

        del model


if __name__ == '__main__':
    main()
