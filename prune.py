import argparse
import json
import os
import pandas as pd

import torch
import numpy as np

from datasets import load_class_names, separate_class, prepare_loader
from models import construct_model
from test import test_v1, test_v2


def prune(model, pruning_perc):
    # https://github.com/zepx/pytorch-weight-prune
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    for p in model.parameters():
        if len(p.data.size()) != 1:
            mask = p.data.abs() > threshold
            p.data.mul_(mask.float())


def load_weight(model, path, device):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = json.load(open(args.config))

    exp_dir = os.path.dirname(args.config)
    modelpath = exp_dir + '/best.pth'

    class_names = load_class_names()
    num_classes = len(class_names)
    v2_info = separate_class(class_names)
    num_makes = len(v2_info['make'].unique())
    num_types = len(v2_info['model_type'].unique())
    train_loader, test_loader = prepare_loader(config)
    model = construct_model(config, num_classes, num_makes, num_types)

    def _prune(model, rate, save=True):
        print(f'Pruning rate: {rate:.2f}')
        load_weight(model, modelpath, device)
        model = model.to(device)

        if config['version'] == 1:
            test_fn = test_v1
        else:
            test_fn = test_v2

        prune(model, rate)

        res = test_fn(model, test_loader, device, config)

        if args.savefn is not None and save:
            savefndir = os.path.dirname(args.savefn)
            os.makedirs(savefndir, exist_ok=True)

            torch.save(model.state_dict(), args.savefn)

        return res

    hist = []

    if args.prune_all:
        for rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            hist.append(_prune(model, rate * 100, save=False))
    else:
        hist.append(_prune(model, args.prune_rate * 100))

    hist = pd.DataFrame(hist)
    hist.to_csv(exp_dir + '/prune.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pruning script')

    parser.add_argument('--config', required=True,
                        help='path to config file')
    parser.add_argument('--prune-rate', type=float, default=0.1,
                        help='pruning rate from 0~1')
    parser.add_argument('--prune-all', action='store_true', default=False,
                        help='whether to prune from 0.1 to 0.9')
    parser.add_argument('--savefn',
                        help='save file name, if provided, will save file')

    args = parser.parse_args()

    main(args)
