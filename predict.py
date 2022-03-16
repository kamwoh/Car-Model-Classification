import argparse
import torch
import json
import os
from datasets import load_class_names, separate_class
from models import construct_model
from test import load_weight
from PIL import Image
from torchvision import transforms


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = json.load(open(args.config))
    config['imgsize'] = (args.imgsize, args.imgsize)

    exp_dir = os.path.dirname(args.config)
    modelpath = exp_dir + '/best.pth'

    class_names = load_class_names()
    num_classes = len(class_names)
    v2_info = separate_class(class_names)
    make_names = v2_info['make'].unique()
    num_makes = len(make_names)
    model_type_names = v2_info['model_type'].unique()
    num_types = len(model_type_names)

    model = construct_model(config, num_classes, num_makes, num_types)
    load_weight(model, modelpath, device)
    model = model.to(device)

    model.eval()
    img = Image.open(args.imgpath)
    img.show()
    img = img.convert('RGB')
    img = transforms.Resize(config['imgsize'])(img)
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )
    img = test_transform(img).float()
    img = img.to(device).unsqueeze(0)
    with torch.no_grad():
        pred, make_pred, model_type_pred = model(img)
        class_idx = pred.max(1)[1].item()
        cls = class_names[class_idx]
        print("Car Model:", cls)

        make_idx = make_pred.max(1)[1].item()
        make = make_names[make_idx]
        print("Car Make:", make)

        model_type_idx = model_type_pred.max(1)[1].item()
        model_type = model_type_names[model_type_idx]
        print("Car Type:", model_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Car Model, Make and Type on a single image')

    parser.add_argument('--config', required=True,
                        help='path to config.json')
    parser.add_argument('--imgpath', required=True,
                        help='path to image')
    parser.add_argument('--imgsize', default=400, type=int,
                        help='img size for testing (default: 400)')

    args = parser.parse_args()

    main(args)
