import argparse
import json
from datasets import load_class_names, separate_class
from models import construct_model


def main(args):
    config = json.load(open(args.config))

    class_names = load_class_names()
    num_classes = len(class_names)
    v2_info = separate_class(class_names)
    num_makes = len(v2_info['make'].unique())
    num_types = len(v2_info['model_type'].unique())

    model = construct_model(config, num_classes, num_makes, num_types)
    count = 0
    for p in list(model.parameters()) + list(model.buffers()):
        count += p.data.view(-1).size(0)

    print(f'Number of parameters for {config["arch"]}: {count}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='count model parameter')

    parser.add_argument('--config', required=True,
                        help='path to config file')

    args = parser.parse_args()

    main(args)
